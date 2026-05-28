"""Profile-live command: profile against already-deployed inference endpoints.

Unlike the `profile` command which deploys its own temporary RBG instances
and requires external aiperf, this command profiles against existing endpoints
using a built-in HTTP streaming benchmark client.
"""

import asyncio
import logging

import click

from inference_ext_cli.profile.benchmark import (
    benchmark_decode,
    benchmark_ttft,
    get_server_info,
)
from inference_ext_cli.profile.defaults import DECODE_MEASUREMENT_OSL
from inference_ext_cli.profile.model_info import get_model_info
from inference_ext_cli.profile.output import generate_profiling_output
from inference_ext_cli.profile.profile_decode import get_num_request_range

logger = logging.getLogger(__name__)


async def run_profile_live(
    url: str,
    decode_info_url: str | None,
    model: str,
    prefill_num_gpus: int,
    decode_num_gpus: int,
    max_context_length: int | None,
    prefill_granularity: int,
    decode_granularity: int,
    output_dir: str,
    configmap_name: str,
    configmap_namespace: str,
    trust_remote_code: bool,
):
    """Run profiling against live endpoints.

    In PD-disagg mode, all requests go through the router URL.
    The decode_info_url is used only to query /server_info for KV cache size.
    """

    # --- Resolve max_context_length ---
    if max_context_length is None:
        logger.info(f"Fetching model info for {model}...")
        try:
            model_info = get_model_info(model, trust_remote_code=trust_remote_code)
            max_context_length = model_info.max_context_length
            logger.info(f"Model max_context_length: {max_context_length}")
        except Exception as e:
            raise click.ClickException(
                f"Cannot determine max_context_length: {e}. "
                "Use --max-context-length to set it manually."
            )

    if not max_context_length:
        raise click.ClickException(
            "Cannot determine max_context_length from model config. "
            "Use --max-context-length to set it manually."
        )

    # --- Get KV cache size from decode server ---
    info_url = decode_info_url or url
    logger.info(f"Querying server info at {info_url}...")
    server_info = await get_server_info(info_url)
    max_kv_tokens = 0
    if server_info:
        max_kv_tokens = (
            server_info.get("max_total_num_tokens", 0)
            or server_info.get("max_total_tokens", 0)
        )
        logger.info(f"Server max_kv_tokens: {max_kv_tokens}")

    if max_kv_tokens == 0:
        # Fallback: estimate from model context length
        max_kv_tokens = max_context_length
        logger.warning(
            f"Could not get KV cache size from server, "
            f"using max_context_length={max_kv_tokens} as fallback"
        )

    # --- Prefill ISL sweep ---
    logger.info("=" * 60)
    logger.info("Prefill ISL Sweep")
    logger.info("=" * 60)

    prefill_isl = []
    prefill_ttft_list = []
    prefill_thpt_per_gpu = []

    sweep_max_ctx = max_context_length - 512  # leave room for chat template
    if sweep_max_ctx <= 100:
        raise click.ClickException(
            f"max_context_length {max_context_length} too small for profiling"
        )

    step = max(1, (sweep_max_ctx - 100) // prefill_granularity)

    for isl in range(100, sweep_max_ctx, step):
        logger.info(f"  Profiling prefill ISL={isl}...")
        ttft = await benchmark_ttft(url, model, isl)

        if ttft is not None and ttft > 0:
            thpt = isl / ttft * 1000.0 / prefill_num_gpus
            prefill_isl.append(isl)
            prefill_ttft_list.append(round(ttft / 1000.0, 6))  # convert to seconds
            prefill_thpt_per_gpu.append(round(thpt, 1))
            logger.info(f"    TTFT={ttft:.2f}ms, throughput={thpt:.1f} tok/s/gpu")
        else:
            logger.warning(f"    Failed to measure TTFT at ISL={isl}, skipping")

    if len(prefill_isl) < 3:
        raise click.ClickException(
            f"Not enough prefill data points ({len(prefill_isl)}), need at least 3"
        )

    prefill_data = {
        "prefill_isl": prefill_isl,
        "prefill_ttft": prefill_ttft_list,
        "prefill_thpt_per_gpu": prefill_thpt_per_gpu,
    }
    logger.info(f"Prefill sweep complete: {len(prefill_isl)} data points")

    # --- Decode 2D sweep ---
    logger.info("=" * 60)
    logger.info("Decode 2D Sweep (ISL x Concurrency)")
    logger.info("=" * 60)

    x_kv_usage = []
    y_context_length = []
    z_itl = []
    z_thpt_per_gpu = []

    osl = DECODE_MEASUREMENT_OSL
    # Cap max concurrency for live profiling — PD-disagg KV transfer
    # can fail under high concurrent load, breaking the prefill instance.
    live_max_concurrency = 16
    step = max(1, (max_context_length - osl) // decode_granularity)

    for isl in range(100, max_context_length - osl, step):
        max_concurrency = min(max_kv_tokens // (isl + osl), live_max_concurrency)
        if max_concurrency <= 0:
            logger.warning(f"  ISL={isl}: KV cache too small, stopping sweep")
            break

        sweep_num_request = get_num_request_range(
            1,  # attention_dp_size
            max_concurrency,
            decode_granularity,
        )

        for num_request in sweep_num_request:
            logger.info(f"  Profiling decode ISL={isl}, concurrency={num_request}...")
            itl, thpt = await benchmark_decode(
                url, model, isl, osl, num_request,
            )

            if itl is not None and thpt is not None and itl > 0:
                kv_usage = (isl + osl / 2) * num_request / max_kv_tokens
                context_len = isl + osl / 2
                thpt_gpu = thpt / max(decode_num_gpus, 1)

                x_kv_usage.append(round(kv_usage, 4))
                y_context_length.append(round(context_len, 1))
                z_itl.append(round(itl / 1000.0, 6))  # convert to seconds
                z_thpt_per_gpu.append(round(thpt_gpu, 1))
                logger.info(
                    f"    ITL={itl:.3f}ms, thpt={thpt_gpu:.1f} tok/s/gpu, "
                    f"kv_usage={kv_usage:.3f}"
                )
            else:
                logger.warning(
                    f"    Failed to measure decode at ISL={isl}, n={num_request}"
                )

    if not x_kv_usage:
        raise click.ClickException("No decode data points collected")

    decode_data = {
        "x_kv_usage": x_kv_usage,
        "y_context_length": y_context_length,
        "z_itl": z_itl,
        "z_thpt_per_gpu": z_thpt_per_gpu,
        "max_kv_tokens": max_kv_tokens,
    }
    logger.info(f"Decode sweep complete: {len(x_kv_usage)} data points")

    # --- Generate output ---
    logger.info("=" * 60)
    logger.info("Generating output")
    logger.info("=" * 60)

    output_path = generate_profiling_output(
        prefill_data, decode_data,
        output_dir=output_dir,
        configmap_name=configmap_name,
        configmap_namespace=configmap_namespace,
    )
    logger.info(f"Profiling complete! Output saved to: {output_path}")
    return output_path


@click.command("profile-live")
@click.option("--url", required=True, help="Base URL of the inference endpoint (router URL for PD-disagg)")
@click.option("--decode-info-url", default=None, help="Optional URL of decode server for /server_info query (defaults to --url)")
@click.option("--model", required=True, help="Model name (HuggingFace ID or as served)")
@click.option("--prefill-num-gpus", default=1, type=int, help="GPUs per prefill engine")
@click.option("--decode-num-gpus", default=1, type=int, help="GPUs per decode engine")
@click.option("--max-context-length", default=None, type=int, help="Max context length (auto-detected if omitted)")
@click.option("--prefill-granularity", default=8, type=int, help="Number of ISL sweep points")
@click.option("--decode-granularity", default=4, type=int, help="Number of points per decode sweep dimension")
@click.option("--output-dir", default="./profiling-results", help="Output directory")
@click.option("--configmap-name", default="profiling-data", help="ConfigMap name")
@click.option("--configmap-namespace", default="default", help="ConfigMap namespace")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for model config")
def profile_live_command(
    url, decode_info_url, model,
    prefill_num_gpus, decode_num_gpus,
    max_context_length, prefill_granularity, decode_granularity,
    output_dir, configmap_name, configmap_namespace,
    trust_remote_code,
):
    """Profile against already-deployed inference endpoints.

    All benchmark requests are sent to the --url endpoint (typically the
    router URL in PD-disaggregated mode). Optionally use --decode-info-url
    to query KV cache size from the decode server directly.

    Runs ISL sweep for prefill and 2D (ISL x concurrency) sweep for decode
    using built-in HTTP streaming benchmark. Generates profiling ConfigMap
    YAML suitable for the RBG planner.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    asyncio.run(run_profile_live(
        url=url,
        decode_info_url=decode_info_url,
        model=model,
        prefill_num_gpus=prefill_num_gpus,
        decode_num_gpus=decode_num_gpus,
        max_context_length=max_context_length,
        prefill_granularity=prefill_granularity,
        decode_granularity=decode_granularity,
        output_dir=output_dir,
        configmap_name=configmap_name,
        configmap_namespace=configmap_namespace,
        trust_remote_code=trust_remote_code,
    ))

"""Main profile command orchestrating the full SLA profiling pipeline.

Ported from dynamo/benchmarks/profiler/profile_sla.py.
Deploys temporary RBG instances, runs AIPerf benchmarks, sweeps GPU configs,
and generates profiling ConfigMap.
"""

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import click
import numpy as np
import yaml

from inference_ext_cli.profile.aiperf import (
    get_decode_itl_and_thpt_per_gpu,
    get_prefill_ttft,
)
from inference_ext_cli.profile.config_modifiers import CONFIG_MODIFIERS
from inference_ext_cli.profile.defaults import DEFAULT_DEPLOYMENT_TIMEOUT, DEFAULT_ENGINE_PORT
from inference_ext_cli.profile.model_info import ModelInfo, get_model_info
from inference_ext_cli.profile.output import generate_profiling_output
from inference_ext_cli.profile.parallelization_mapping import (
    ParallelizationMapping,
    SubComponentType,
    apply_parallel_mapping_to_config,
    get_candidate_parallel_mappings,
)
from inference_ext_cli.profile.profile_decode import get_num_request_range, profile_decode
from inference_ext_cli.profile.profile_prefill import profile_prefill
from inference_ext_cli.profile.rbg_deployment import RBGDeploymentClient

logger = logging.getLogger(__name__)


@dataclass
class PrefillProfileData:
    """Container for prefill Phase 1 profiling results."""

    num_gpus: list[int] = field(default_factory=list)
    ttft: list[float] = field(default_factory=list)
    thpt_per_gpu: list[float] = field(default_factory=list)
    parallel_mapping_labels: list[str] = field(default_factory=list)
    parallel_mappings: list[ParallelizationMapping] = field(default_factory=list)

    def add_data(
        self, num_gpus: int, ttft: float, thpt_per_gpu: float,
        parallel_mapping_label: str, parallel_mapping: ParallelizationMapping,
    ):
        self.num_gpus.append(num_gpus)
        self.ttft.append(ttft)
        self.thpt_per_gpu.append(thpt_per_gpu)
        self.parallel_mapping_labels.append(parallel_mapping_label)
        self.parallel_mappings.append(parallel_mapping)


@dataclass
class DecodeProfileData:
    """Container for decode Phase 1 profiling results."""

    num_gpus: list[int] = field(default_factory=list)
    itl: list[float] = field(default_factory=list)
    thpt_per_gpu: list[float] = field(default_factory=list)
    concurrency: list[int] = field(default_factory=list)
    kv_cache_size: list[int] = field(default_factory=list)
    parallel_mapping_labels: list[str] = field(default_factory=list)
    parallel_mappings: list[ParallelizationMapping] = field(default_factory=list)

    def add_data(
        self, num_gpus: int, itl: float, thpt_per_gpu: float,
        concurrency: int, kv_cache_size: int,
        parallel_mapping_label: str, parallel_mapping: ParallelizationMapping,
    ):
        self.num_gpus.append(num_gpus)
        self.itl.append(itl)
        self.thpt_per_gpu.append(thpt_per_gpu)
        self.concurrency.append(concurrency)
        self.kv_cache_size.append(kv_cache_size)
        self.parallel_mapping_labels.append(parallel_mapping_label)
        self.parallel_mappings.append(parallel_mapping)


def _select_best_prefill(prefill_data: PrefillProfileData, ttft_sla: float) -> int:
    """Select best prefill config: within SLA, highest throughput/GPU."""
    if min(prefill_data.ttft) > ttft_sla:
        logger.warning("No config satisfies TTFT SLA, selecting lowest TTFT")
        return int(np.argmin(np.array(prefill_data.ttft)))

    valid_indices = [i for i, t in enumerate(prefill_data.ttft) if t <= ttft_sla]
    valid_thpts = [prefill_data.thpt_per_gpu[i] for i in valid_indices]
    return valid_indices[int(np.argmax(valid_thpts))]


def _select_best_decode(decode_data: DecodeProfileData, itl_sla: float) -> int:
    """Select best decode config: within SLA, highest throughput/GPU."""
    if min(decode_data.itl) > itl_sla:
        logger.warning("No config satisfies ITL SLA, selecting lowest ITL")
        return int(np.argmin(np.array(decode_data.itl)))

    valid_indices = [i for i, t in enumerate(decode_data.itl) if t <= itl_sla]
    valid_thpts = [decode_data.thpt_per_gpu[i] for i in valid_indices]
    return valid_indices[int(np.argmax(valid_thpts))]


async def run_profile(
    engine: str,
    model: str,
    engine_image: str,
    namespace: str,
    min_gpus: int,
    max_gpus: int,
    isl: int,
    osl: int,
    ttft_sla: float,
    itl_sla: float,
    max_context_length: Optional[int],
    prefill_interpolation_granularity: int,
    decode_interpolation_granularity: int,
    output_dir: str,
    configmap_name: str,
    configmap_namespace: str,
    num_gpus_per_node: int,
    deployment_timeout: int,
    dry_run: bool,
    trust_remote_code: bool,
):
    """Main profiling orchestration."""
    deployment_clients: list[RBGDeploymentClient] = []

    try:
        config_modifier = CONFIG_MODIFIERS[engine]

        # Get model info from HuggingFace
        logger.info(f"Fetching model info for {model}...")
        model_info = get_model_info(model, trust_remote_code=trust_remote_code)
        logger.info(
            f"Model: architecture={model_info.architecture}, "
            f"is_moe={model_info.is_moe}, "
            f"max_context_length={model_info.max_context_length}, "
            f"num_kv_heads={model_info.num_kv_heads}"
        )

        # Determine sweep max context length
        model_max_ctx = model_info.max_context_length
        if max_context_length and model_max_ctx:
            sweep_max_ctx = min(max_context_length, model_max_ctx)
        elif max_context_length:
            sweep_max_ctx = max_context_length
        elif model_max_ctx:
            sweep_max_ctx = model_max_ctx
        else:
            raise click.ClickException(
                "Cannot determine max_context_length from model config or CLI args"
            )
        logger.info(f"Using max context length: {sweep_max_ctx}")

        # GPU counts to profile (powers of 2)
        profile_num_gpus = [
            2**i for i in range(int(math.log2(max_gpus)) + 1)
            if min_gpus <= 2**i <= max_gpus
        ]
        logger.info(f"GPU counts to profile: {profile_num_gpus}")
        os.makedirs(output_dir, exist_ok=True)

        # ===== Phase 1: Parallelization Sweep =====
        logger.info("=" * 60)
        logger.info("Phase 1: Parallelization Sweep")
        logger.info("=" * 60)

        # --- Prefill sweep ---
        prefill_data = PrefillProfileData()
        logger.info("Profiling prefill configurations...")

        for num_gpus in profile_num_gpus:
            candidate_mappings = get_candidate_parallel_mappings(
                num_gpus, model_info, "prefill"
            )

            for mapping in candidate_mappings:
                logger.info(f"  Prefill: {num_gpus} GPUs, mapping={mapping.label()}")

                if dry_run:
                    logger.info("    [dry-run] Skipping deployment")
                    continue

                # Generate RBG spec with this mapping
                rbg_spec = config_modifier.generate_rbg_spec(
                    model=model, image=engine_image, num_gpus=num_gpus,
                    phase=SubComponentType.PREFILL, is_moe=model_info.is_moe,
                )
                rbg_spec = apply_parallel_mapping_to_config(
                    rbg_spec, mapping, SubComponentType.PREFILL,
                    config_modifier, num_gpus_per_node,
                )

                # Deploy and measure
                client = RBGDeploymentClient(
                    namespace=namespace, engine=engine, port=DEFAULT_ENGINE_PORT,
                )
                deployment_clients.append(client)

                await client.create_deployment(rbg_spec)
                await client.wait_for_ready(timeout=deployment_timeout)

                base_url = client.get_service_url()
                work_dir = f"{output_dir}/prefill_{num_gpus}gpus_{mapping.label().replace('=', '')}"
                os.makedirs(work_dir, exist_ok=True)

                ttft = get_prefill_ttft(
                    isl, f"{work_dir}/aiperf_isl{isl}",
                    model, model, base_url=base_url,
                    attention_dp_size=mapping.get_attn_dp_size(),
                )

                await client.delete_deployment()
                deployment_clients.remove(client)

                if ttft is not None:
                    thpt = isl / ttft / num_gpus * 1000 * mapping.get_attn_dp_size()
                    prefill_data.add_data(
                        num_gpus=num_gpus, ttft=ttft, thpt_per_gpu=thpt,
                        parallel_mapping_label=mapping.label(),
                        parallel_mapping=mapping,
                    )
                    logger.info(f"    TTFT={ttft:.2f}ms, throughput={thpt:.1f} tok/s/gpu")

        # --- Decode sweep ---
        decode_data = DecodeProfileData()
        logger.info("Profiling decode configurations...")

        for num_gpus in profile_num_gpus:
            candidate_mappings = get_candidate_parallel_mappings(
                num_gpus, model_info, "decode"
            )

            for mapping in candidate_mappings:
                logger.info(f"  Decode: {num_gpus} GPUs, mapping={mapping.label()}")

                if dry_run:
                    logger.info("    [dry-run] Skipping deployment")
                    continue

                rbg_spec = config_modifier.generate_rbg_spec(
                    model=model, image=engine_image, num_gpus=num_gpus,
                    phase=SubComponentType.DECODE, is_moe=model_info.is_moe,
                )
                rbg_spec = apply_parallel_mapping_to_config(
                    rbg_spec, mapping, SubComponentType.DECODE,
                    config_modifier, num_gpus_per_node,
                )

                client = RBGDeploymentClient(
                    namespace=namespace, engine=engine, port=DEFAULT_ENGINE_PORT,
                )
                deployment_clients.append(client)

                await client.create_deployment(rbg_spec)
                await client.wait_for_ready(timeout=deployment_timeout)

                # Get KV cache size from engine logs
                pod_logs = await client.get_pod_logs()
                attention_dp_size = mapping.get_attn_dp_size()
                max_kv_tokens = config_modifier.get_kv_cache_size_from_log(
                    pod_logs, attention_dp_size=attention_dp_size,
                )

                if max_kv_tokens == 0:
                    logger.warning("Could not detect KV cache size from logs, skipping")
                    await client.delete_deployment()
                    deployment_clients.remove(client)
                    continue

                max_concurrency = max_kv_tokens // (isl + osl)
                sweep_num_request = get_num_request_range(
                    attention_dp_size, max_concurrency, decode_interpolation_granularity,
                )

                base_url = client.get_service_url()
                work_dir = f"{output_dir}/decode_{num_gpus}gpus_{mapping.label().replace('=', '')}"
                os.makedirs(work_dir, exist_ok=True)

                for num_request in sweep_num_request:
                    artifact_dir = f"{work_dir}/aiperf_n{num_request}_isl{isl}_osl{osl}"
                    itl_val, thpt_val = get_decode_itl_and_thpt_per_gpu(
                        isl, osl, num_request, artifact_dir,
                        model, model, base_url=base_url,
                        num_gpus=num_gpus, attention_dp_size=attention_dp_size,
                    )

                    if itl_val is not None and thpt_val is not None:
                        decode_data.add_data(
                            num_gpus=num_gpus, itl=itl_val, thpt_per_gpu=thpt_val,
                            concurrency=num_request, kv_cache_size=max_kv_tokens,
                            parallel_mapping_label=mapping.label(),
                            parallel_mapping=mapping,
                        )
                        logger.info(
                            f"    n={num_request}: ITL={itl_val:.3f}ms, "
                            f"thpt={thpt_val:.1f} tok/s/gpu"
                        )

                await client.delete_deployment()
                deployment_clients.remove(client)

        # ===== Select best configs =====
        if dry_run:
            logger.info("[dry-run] Skipping config selection and Phase 2")
            return

        if not prefill_data.num_gpus:
            raise click.ClickException("No prefill results produced")
        if not decode_data.num_gpus:
            raise click.ClickException("No decode results produced")

        selected_prefill_idx = _select_best_prefill(prefill_data, ttft_sla)
        selected_decode_idx = _select_best_decode(decode_data, itl_sla)

        best_prefill_gpus = prefill_data.num_gpus[selected_prefill_idx]
        best_prefill_mapping = prefill_data.parallel_mappings[selected_prefill_idx]
        best_decode_gpus = decode_data.num_gpus[selected_decode_idx]
        best_decode_mapping = decode_data.parallel_mappings[selected_decode_idx]

        logger.info(
            f"Selected prefill: {best_prefill_mapping.label()} on {best_prefill_gpus} GPUs "
            f"(TTFT={prefill_data.ttft[selected_prefill_idx]:.2f}ms)"
        )
        logger.info(
            f"Selected decode: {best_decode_mapping.label()} on {best_decode_gpus} GPUs "
            f"(ITL={decode_data.itl[selected_decode_idx]:.2f}ms)"
        )

        # ===== Phase 2: Interpolation Sweep =====
        logger.info("=" * 60)
        logger.info("Phase 2: Interpolation Sweep")
        logger.info("=" * 60)

        # --- Prefill interpolation ---
        logger.info(f"Prefill ISL sweep with {best_prefill_gpus} GPUs, {best_prefill_mapping.label()}...")

        rbg_spec = config_modifier.generate_rbg_spec(
            model=model, image=engine_image, num_gpus=best_prefill_gpus,
            phase=SubComponentType.PREFILL, is_moe=model_info.is_moe,
        )
        rbg_spec = apply_parallel_mapping_to_config(
            rbg_spec, best_prefill_mapping, SubComponentType.PREFILL,
            config_modifier, num_gpus_per_node,
        )

        client = RBGDeploymentClient(
            namespace=namespace, engine=engine, port=DEFAULT_ENGINE_PORT,
        )
        deployment_clients.append(client)
        await client.create_deployment(rbg_spec)
        await client.wait_for_ready(timeout=deployment_timeout)

        prefill_work_dir = f"{output_dir}/selected_prefill_interpolation"
        base_url = client.get_service_url()
        prefill_results = profile_prefill(
            prefill_work_dir, model, model, base_url,
            best_prefill_gpus, sweep_max_ctx,
            prefill_interpolation_granularity,
            attention_dp_size=best_prefill_mapping.get_attn_dp_size(),
        )

        await client.delete_deployment()
        deployment_clients.remove(client)

        # --- Decode interpolation ---
        logger.info(f"Decode 2D sweep with {best_decode_gpus} GPUs, {best_decode_mapping.label()}...")

        rbg_spec = config_modifier.generate_rbg_spec(
            model=model, image=engine_image, num_gpus=best_decode_gpus,
            phase=SubComponentType.DECODE, is_moe=model_info.is_moe,
        )
        rbg_spec = apply_parallel_mapping_to_config(
            rbg_spec, best_decode_mapping, SubComponentType.DECODE,
            config_modifier, num_gpus_per_node,
        )

        client = RBGDeploymentClient(
            namespace=namespace, engine=engine, port=DEFAULT_ENGINE_PORT,
        )
        deployment_clients.append(client)
        await client.create_deployment(rbg_spec)
        await client.wait_for_ready(timeout=deployment_timeout)

        # Get KV cache size
        pod_logs = await client.get_pod_logs()
        attention_dp_size = best_decode_mapping.get_attn_dp_size()
        max_kv_tokens = config_modifier.get_kv_cache_size_from_log(
            pod_logs, attention_dp_size=attention_dp_size,
        )
        if max_kv_tokens == 0:
            raise click.ClickException("Could not detect KV cache size for decode interpolation")

        decode_work_dir = f"{output_dir}/selected_decode_interpolation"
        base_url = client.get_service_url()
        decode_results = profile_decode(
            decode_work_dir, model, model, base_url,
            best_decode_gpus, max_kv_tokens, sweep_max_ctx,
            decode_interpolation_granularity,
            attention_dp_size=attention_dp_size,
        )

        await client.delete_deployment()
        deployment_clients.remove(client)

        # ===== Generate output =====
        if prefill_results and decode_results:
            output_path = generate_profiling_output(
                prefill_results, decode_results,
                output_dir=output_dir,
                configmap_name=configmap_name,
                configmap_namespace=configmap_namespace,
            )
            logger.info(f"Profiling complete! ConfigMap saved to: {output_path}")
        else:
            logger.error("Profiling produced insufficient data for output generation")

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        raise
    finally:
        # Cleanup any remaining deployments
        for client in deployment_clients:
            try:
                await client.delete_deployment()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")


@click.command("profile")
@click.option("--engine", type=click.Choice(["sglang", "vllm"]), required=True, help="Inference engine type")
@click.option("--model", required=True, help="HuggingFace model ID or local path")
@click.option("--engine-image", required=True, help="Container image for the engine")
@click.option("--namespace", default="default", help="Kubernetes namespace for profiling")
@click.option("--min-gpus", default=1, type=int, help="Minimum GPUs per engine")
@click.option("--max-gpus", default=8, type=int, help="Maximum GPUs per engine")
@click.option("--isl", default=3000, type=int, help="Target input sequence length for SLA")
@click.option("--osl", default=500, type=int, help="Target output sequence length for SLA")
@click.option("--ttft-sla", default=200.0, type=float, help="Target TTFT in ms")
@click.option("--itl-sla", default=20.0, type=float, help="Target ITL in ms")
@click.option("--max-context-length", default=None, type=int, help="Override max context length")
@click.option("--prefill-interpolation-granularity", default=16, type=int, help="ISL sweep points")
@click.option("--decode-interpolation-granularity", default=6, type=int, help="Decode sweep points per dimension")
@click.option("--output-dir", default="./profiling-results", help="Output directory")
@click.option("--configmap-name", default="profiling-data", help="ConfigMap name")
@click.option("--configmap-namespace", default="default", help="ConfigMap namespace")
@click.option("--num-gpus-per-node", default=8, type=int, help="GPUs per node (for multinode)")
@click.option("--deployment-timeout", default=DEFAULT_DEPLOYMENT_TIMEOUT, type=int, help="Deployment ready timeout (s)")
@click.option("--dry-run", is_flag=True, help="Generate configs without deploying")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for model config")
def profile_command(
    engine, model, engine_image, namespace,
    min_gpus, max_gpus, isl, osl, ttft_sla, itl_sla,
    max_context_length, prefill_interpolation_granularity,
    decode_interpolation_granularity, output_dir,
    configmap_name, configmap_namespace, num_gpus_per_node,
    deployment_timeout, dry_run, trust_remote_code,
):
    """Run end-to-end SLA profiling pipeline.

    Deploys temporary RBG instances, runs AIPerf benchmarks at various GPU
    configurations, selects optimal prefill/decode parallelization, performs
    interpolation sweeps, and generates a Kubernetes ConfigMap with results.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    asyncio.run(run_profile(
        engine=engine,
        model=model,
        engine_image=engine_image,
        namespace=namespace,
        min_gpus=min_gpus,
        max_gpus=max_gpus,
        isl=isl,
        osl=osl,
        ttft_sla=ttft_sla,
        itl_sla=itl_sla,
        max_context_length=max_context_length,
        prefill_interpolation_granularity=prefill_interpolation_granularity,
        decode_interpolation_granularity=decode_interpolation_granularity,
        output_dir=output_dir,
        configmap_name=configmap_name,
        configmap_namespace=configmap_namespace,
        num_gpus_per_node=num_gpus_per_node,
        deployment_timeout=deployment_timeout,
        dry_run=dry_run,
        trust_remote_code=trust_remote_code,
    ))

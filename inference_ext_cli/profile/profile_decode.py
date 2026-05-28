"""Decode 2D sweep profiling logic.

Ported from dynamo/benchmarks/profiler/utils/profile_decode.py.
Sweeps ISL x concurrency to build decode performance surface.
"""

import logging
from typing import Callable, Optional, Tuple

import numpy as np

from inference_ext_cli.profile.aiperf import get_decode_itl_and_thpt_per_gpu
from inference_ext_cli.profile.defaults import DECODE_MAX_CONCURRENCY, DECODE_MEASUREMENT_OSL

logger = logging.getLogger(__name__)


def get_num_request_range(
    attn_dp_size: int, engine_max_concurrency: int, granularity: int
) -> list[int]:
    """Generate concurrency sweep range aligned to attention DP size.

    For MoE models with attention DP, num_request must be a multiple of attn_dp_size
    to ensure round-robin scheduling sends warmup and measurement to same DP rank.
    """
    max_concurrency = min(engine_max_concurrency, DECODE_MAX_CONCURRENCY)
    conc_per_dp = max_concurrency // attn_dp_size

    if conc_per_dp < granularity:
        return list(range(attn_dp_size, conc_per_dp * attn_dp_size + 1, attn_dp_size))
    else:
        step = (conc_per_dp - 1) * attn_dp_size / (granularity - 1)
        return [attn_dp_size + int(i * step) * attn_dp_size for i in range(granularity)]


def _profile_decode_helper(
    work_dir: str,
    num_gpus: int,
    max_kv_tokens: int,
    max_context_length: int,
    interpolation_granularity: int,
    get_itl_and_thpt_per_gpu: Callable[[int, int, int], Tuple[Optional[float], Optional[float]]],
    attention_dp_size: int,
) -> dict:
    """Run decode 2D sweep: ISL x concurrency → ITL, throughput.

    Returns:
        Dict with x_kv_usage, y_context_length, z_itl, z_thpt_per_gpu, max_kv_tokens.
    """
    x_kv_usage = []
    y_context_length = []
    z_itl = []
    z_thpt_per_gpu = []

    osl = DECODE_MEASUREMENT_OSL

    step = (max_context_length - osl) // interpolation_granularity
    for isl in range(100, max_context_length - osl, step):
        max_concurrency = max_kv_tokens // (isl + osl)
        if max_concurrency == 0:
            logger.warning(
                f"max_kv_tokens {max_kv_tokens} too small for "
                f"isl={isl} + osl={osl}, stopping sweep."
            )
            break

        sweep_num_request = get_num_request_range(
            attention_dp_size, max_concurrency, interpolation_granularity
        )

        for num_request in sweep_num_request:
            itl, thpt_per_gpu = get_itl_and_thpt_per_gpu(isl, osl, num_request)

            if itl is not None and thpt_per_gpu is not None:
                kv_usage = (isl + osl / 2) * num_request / max_kv_tokens
                context_len = isl + osl / 2
                x_kv_usage.append(kv_usage)
                y_context_length.append(context_len)
                z_itl.append(itl)
                z_thpt_per_gpu.append(thpt_per_gpu)
                logger.info(
                    f"  ISL={isl}, n={num_request}: "
                    f"ITL={itl:.3f}ms, thpt={thpt_per_gpu:.1f} tok/s/gpu, "
                    f"kv_usage={kv_usage:.3f}"
                )

    if not x_kv_usage:
        logger.warning("No decode data points collected")
        return {}

    save_path = f"{work_dir}/raw_data.npz"
    np.savez(
        save_path,
        x_kv_usage=np.array(x_kv_usage),
        y_context_length=np.array(y_context_length),
        z_itl=np.array(z_itl),
        z_thpt_per_gpu=np.array(z_thpt_per_gpu),
        max_kv_tokens=np.array([max_kv_tokens]),
    )
    logger.info(f"Saved decode data to {save_path}")

    return {
        "x_kv_usage": x_kv_usage,
        "y_context_length": y_context_length,
        "z_itl": z_itl,
        "z_thpt_per_gpu": z_thpt_per_gpu,
        "max_kv_tokens": max_kv_tokens,
    }


def profile_decode(
    work_dir: str,
    model_name: str,
    tokenizer: str,
    url: str,
    num_gpus: int,
    max_kv_tokens: int,
    max_context_length: int,
    interpolation_granularity: int,
    attention_dp_size: int,
) -> dict:
    """Run decode profiling 2D sweep using AIPerf.

    Args:
        work_dir: Directory for artifacts.
        model_name: Model name for AIPerf.
        tokenizer: Tokenizer name for AIPerf.
        url: Base URL of the engine.
        num_gpus: Number of GPUs.
        max_kv_tokens: Maximum KV cache tokens (from engine log).
        max_context_length: Maximum context length.
        interpolation_granularity: Number of points per dimension.
        attention_dp_size: Attention data parallelism size.

    Returns:
        Dict with profiling results.
    """
    def get_itl_and_thpt_per_gpu(
        isl: int, osl: int, num_request: int
    ) -> Tuple[Optional[float], Optional[float]]:
        artifact_dir = f"{work_dir}/aiperf_isl{isl}_osl{osl}_n{num_request}"
        return get_decode_itl_and_thpt_per_gpu(
            isl,
            osl,
            num_request,
            artifact_dir,
            model_name,
            tokenizer,
            base_url=url,
            num_gpus=num_gpus,
            attention_dp_size=attention_dp_size,
        )

    return _profile_decode_helper(
        work_dir,
        num_gpus,
        max_kv_tokens,
        max_context_length,
        interpolation_granularity,
        get_itl_and_thpt_per_gpu,
        attention_dp_size,
    )

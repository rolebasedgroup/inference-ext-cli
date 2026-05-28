"""Prefill ISL sweep profiling logic.

Ported from dynamo/benchmarks/profiler/utils/profile_prefill.py.
Sweeps ISL values and measures TTFT to build prefill performance profile.
"""

import logging
from typing import Callable, Optional

import numpy as np

from inference_ext_cli.profile.aiperf import get_prefill_ttft

logger = logging.getLogger(__name__)


def _profile_prefill_helper(
    work_dir: str,
    num_gpus: int,
    max_context_length: int,
    interpolation_granularity: int,
    get_ttft: Callable[[int], Optional[float]],
    attention_dp_size: int = 1,
) -> dict:
    """Run prefill ISL sweep and save results.

    Returns:
        Dict with prefill_isl, prefill_ttft, prefill_thpt_per_gpu arrays.
    """
    prefill_isl = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []

    # Leave room for chat template and system prompt
    max_context_length -= 512
    if max_context_length <= 100:
        raise ValueError(
            f"max_context_length {max_context_length + 512} is too small to profile prefill"
        )

    step = (max_context_length - 100) // interpolation_granularity
    for isl in range(100, max_context_length, step):
        ttft = get_ttft(isl)
        if ttft is not None:
            prefill_isl.append(isl)
            prefill_ttft.append(ttft)
            # Throughput = tokens / time / gpus, ttft is in ms
            prefill_thpt_per_gpu.append(
                isl / ttft / num_gpus * 1000 * attention_dp_size
            )
            logger.info(
                f"  ISL={isl}: TTFT={ttft:.2f}ms, "
                f"throughput={prefill_thpt_per_gpu[-1]:.1f} tok/s/gpu"
            )

    if len(prefill_isl) < 3:
        logger.warning("Not enough data points for interpolation (need at least 3)")
        return {}

    logger.info("Prefill profiling complete. Saving results...")

    prefill_isl_np = np.array(prefill_isl)
    prefill_ttft_np = np.array(prefill_ttft)
    prefill_thpt_per_gpu_np = np.array(prefill_thpt_per_gpu)

    save_path = f"{work_dir}/raw_data.npz"
    np.savez(
        save_path,
        prefill_isl=prefill_isl_np,
        prefill_ttft=prefill_ttft_np,
        prefill_thpt_per_gpu=prefill_thpt_per_gpu_np,
    )
    logger.info(f"Saved prefill data to {save_path}")

    return {
        "prefill_isl": prefill_isl,
        "prefill_ttft": prefill_ttft,
        "prefill_thpt_per_gpu": prefill_thpt_per_gpu,
    }


def profile_prefill(
    work_dir: str,
    model_name: str,
    tokenizer: str,
    url: str,
    num_gpus: int,
    max_context_length: int,
    interpolation_granularity: int,
    attention_dp_size: int = 1,
) -> dict:
    """Run prefill profiling sweep using AIPerf.

    Args:
        work_dir: Directory for artifacts.
        model_name: Model name for AIPerf.
        tokenizer: Tokenizer name for AIPerf.
        url: Base URL of the engine.
        num_gpus: Number of GPUs.
        max_context_length: Maximum context length of the model.
        interpolation_granularity: Number of ISL points to sample.
        attention_dp_size: Attention data parallelism size (for DEP).

    Returns:
        Dict with profiling results.
    """
    def get_ttft(isl: int) -> Optional[float]:
        artifact_dir = f"{work_dir}/aiperf_isl{isl}"
        return get_prefill_ttft(
            isl,
            artifact_dir,
            model_name,
            tokenizer,
            base_url=url,
            attention_dp_size=attention_dp_size,
        )

    return _profile_prefill_helper(
        work_dir,
        num_gpus,
        max_context_length,
        interpolation_granularity,
        get_ttft,
        attention_dp_size=attention_dp_size,
    )

"""AIPerf benchmark wrapper for profiling inference engines.

Ported from dynamo/benchmarks/profiler/utils/aiperf.py.
Builds AIPerf CLI commands, runs them, and parses JSON results.
"""

import json
import logging
import os
import random
import subprocess
from typing import Optional, Tuple

from inference_ext_cli.profile.defaults import (
    AIPERF_PREFILL_ATTN_DP_NUM_REQ_RATIO,
    AIPERF_PREFILL_BENCHMARK_OSL,
    AIPERF_WARMUP_REQUEST_PER_DP_RANK,
)

logger = logging.getLogger(__name__)


def _get_common_aiperf_cmd(
    artifact_dir: str,
    seed: int = 100,
    model: str = "model",
    tokenizer: str = "model",
    base_url: str = "http://localhost:8000",
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
) -> list[str]:
    return [
        "aiperf",
        "profile",
        "--model", model,
        "--tokenizer", tokenizer,
        "--endpoint-type", "chat",
        "--endpoint", "/v1/chat/completions",
        "--streaming",
        "--url", base_url,
        "--extra-inputs", "ignore_eos:true",
        "--extra-inputs", '{"nvext":{"ignore_eos":true}}',
        "--warmup-request-count", str(warmup_request_count),
        "--artifact-dir", artifact_dir,
        "--random-seed", str(seed),
        "--request-timeout-seconds", "1800",
    ]


def get_prefill_aiperf_cmd(
    isl: int,
    artifact_dir: str,
    seed: int = 100,
    model: str = "model",
    tokenizer: str = "model",
    osl: int = AIPERF_PREFILL_BENCHMARK_OSL,
    base_url: str = "http://localhost:8000",
    concurrency: int = 1,
    request_count: int = 1,
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
) -> list[str]:
    return _get_common_aiperf_cmd(
        artifact_dir, seed, model, tokenizer, base_url,
        warmup_request_count=warmup_request_count,
    ) + [
        "--synthetic-input-tokens-mean", str(isl),
        "--synthetic-input-tokens-stddev", "0",
        "--output-tokens-mean", str(osl),
        "--output-tokens-stddev", "0",
        "--extra-inputs", f"max_tokens:{osl}",
        "--extra-inputs", f"min_tokens:{osl}",
        "--concurrency", str(concurrency),
        "--request-count", str(request_count),
    ]


def get_decode_aiperf_cmd(
    isl: int,
    osl: int,
    artifact_dir: str,
    num_request: int,
    seed: int = 100,
    model: str = "model",
    tokenizer: str = "model",
    base_url: str = "http://localhost:8000",
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
) -> list[str]:
    return _get_common_aiperf_cmd(
        artifact_dir, seed, model, tokenizer, base_url,
        warmup_request_count=warmup_request_count,
    ) + [
        "--synthetic-input-tokens-mean", str(isl),
        "--synthetic-input-tokens-stddev", "0",
        "--output-tokens-mean", str(osl),
        "--output-tokens-stddev", "0",
        "--extra-inputs", f"max_tokens:{osl}",
        "--extra-inputs", f"min_tokens:{osl}",
        "--concurrency", str(num_request),
        "--num-dataset-entries", str(num_request),
        "--request-count", str(num_request),
    ]


def get_aiperf_result(artifact_dir: str) -> dict:
    """Find and parse the AIPerf JSON result file."""
    for root, _, files in os.walk(artifact_dir):
        if "profile_export_aiperf.json" in files:
            json_file_path = os.path.join(root, "profile_export_aiperf.json")
            with open(json_file_path, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"profile_export_aiperf.json not found in {artifact_dir}"
    )


def run_aiperf(cmd: list[str], artifact_dir: str) -> Optional[dict]:
    """Run an AIPerf command and return parsed results, or None on failure."""
    logger.info(f"Running AIPerf: {' '.join(cmd[:5])}...")
    logger.debug(f"Full command: {cmd}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        logger.info("AIPerf completed successfully")
        return get_aiperf_result(artifact_dir)
    else:
        logger.error(f"AIPerf failed (exit code {process.returncode})")
        logger.error(f"stderr: {stderr}")
        return None


def benchmark_prefill(
    isl: int,
    artifact_dir: str,
    model_name: str,
    tokenizer: str,
    base_url: str = "http://localhost:8000",
    concurrency: int = 1,
    request_count: int = 1,
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
) -> Optional[dict]:
    """Run prefill benchmark and return AIPerf result dict."""
    cmd = get_prefill_aiperf_cmd(
        isl, artifact_dir,
        model=model_name, tokenizer=tokenizer, base_url=base_url,
        concurrency=concurrency, request_count=request_count,
        warmup_request_count=warmup_request_count,
    )
    return run_aiperf(cmd, artifact_dir)


def benchmark_decode(
    isl: int,
    osl: int,
    num_request: int,
    artifact_dir: str,
    model_name: str,
    tokenizer: str,
    base_url: str = "http://localhost:8000",
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
) -> Optional[dict]:
    """Run decode benchmark with warmup pass then measurement pass."""
    seed = random.randint(0, 1000000)

    # Warmup pass: pre-compute prefill tokens with same seed
    warmup_cmd = get_decode_aiperf_cmd(
        isl, osl, f"{artifact_dir}_warmup", num_request,
        seed=seed, model=model_name, tokenizer=tokenizer,
        base_url=base_url, warmup_request_count=warmup_request_count,
    )
    warmup_process = subprocess.Popen(
        warmup_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    warmup_process.communicate()

    # Measurement pass: same prompts, should reuse KV cache
    measure_cmd = get_decode_aiperf_cmd(
        isl, osl, artifact_dir, num_request,
        seed=seed, model=model_name, tokenizer=tokenizer,
        base_url=base_url,
    )
    return run_aiperf(measure_cmd, artifact_dir)


def get_prefill_ttft(
    isl: int,
    artifact_dir: str,
    model_name: str,
    tokenizer: str,
    base_url: str = "http://localhost:8000",
    attention_dp_size: int = 1,
) -> Optional[float]:
    """Run prefill benchmark and extract TTFT in milliseconds."""
    if attention_dp_size > 1:
        total_concurrency = attention_dp_size * AIPERF_PREFILL_ATTN_DP_NUM_REQ_RATIO
        result = benchmark_prefill(
            isl, artifact_dir, model_name, tokenizer,
            base_url=base_url,
            concurrency=total_concurrency,
            request_count=total_concurrency,
            warmup_request_count=AIPERF_WARMUP_REQUEST_PER_DP_RANK * attention_dp_size,
        )
        if result is None:
            return None
        try:
            max_ttft = float(result["time_to_first_token"]["max"])
            max_ttft -= (
                float(result["inter_token_latency"]["avg"])
                * (AIPERF_PREFILL_BENCHMARK_OSL - 1)
                * (AIPERF_PREFILL_ATTN_DP_NUM_REQ_RATIO - 1)
            )
            return max_ttft / float(AIPERF_PREFILL_ATTN_DP_NUM_REQ_RATIO)
        except (KeyError, TypeError, ValueError):
            logger.warning("Failed to extract TTFT from DEP prefill result")
            return None

    result = benchmark_prefill(
        isl, artifact_dir, model_name, tokenizer, base_url=base_url,
    )
    if result is None:
        return None
    try:
        return float(result["time_to_first_token"]["avg"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Failed to extract TTFT from AIPerf result")
        return None


def get_decode_itl_and_thpt_per_gpu(
    isl: int,
    osl: int,
    num_request: int,
    artifact_dir: str,
    model_name: str,
    tokenizer: str,
    base_url: str = "http://localhost:8000",
    num_gpus: int = 1,
    attention_dp_size: int = 1,
) -> Tuple[Optional[float], Optional[float]]:
    """Run decode benchmark and extract (ITL ms, throughput/GPU)."""
    result = benchmark_decode(
        isl, osl, num_request, artifact_dir, model_name, tokenizer,
        base_url=base_url,
        warmup_request_count=AIPERF_WARMUP_REQUEST_PER_DP_RANK * attention_dp_size,
    )
    if result is None:
        return None, None
    try:
        itl = float(result["inter_token_latency"]["avg"])
        thpt_total = float(result["output_token_throughput"]["avg"])
        thpt_per_gpu = thpt_total / max(num_gpus, 1)
        return itl, thpt_per_gpu
    except (KeyError, TypeError, ValueError):
        logger.warning("Failed to extract decode metrics from AIPerf result")
        return None, None

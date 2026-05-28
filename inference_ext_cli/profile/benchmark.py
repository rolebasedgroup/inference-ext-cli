"""Built-in HTTP streaming benchmark for profiling inference endpoints.

Replaces external aiperf dependency with a lightweight httpx-based client
that measures TTFT and ITL from SSE streaming responses.
"""

import asyncio
import json
import logging
import time
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Timeout for individual requests (seconds)
REQUEST_TIMEOUT = 300.0


def _make_synthetic_prompt(target_isl: int) -> str:
    """Generate a synthetic prompt that approximates target_isl tokens.

    Uses repeated simple words. Rough approximation: 1 word ≈ 1.3 tokens.
    Exact token count doesn't matter — we measure actual tokens from response metadata.
    """
    words_needed = int(target_isl / 1.3)
    return "hi " * max(words_needed, 10)


def _make_chat_request_body(
    model: str, prompt: str, max_tokens: int, stream: bool = True,
) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }


async def _stream_request_ttft(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
) -> Optional[float]:
    """Send one streaming request and measure TTFT in milliseconds."""
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    t_start = time.monotonic()

    try:
        async with client.stream("POST", endpoint, json=body, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status_code != 200:
                logger.warning(f"Request failed with status {resp.status_code}")
                return None

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content")
                        # content can be "" (empty), None, or actual text
                        # TTFT is when we get the first non-empty content
                        if content is not None and content != "":
                            return (time.monotonic() - t_start) * 1000.0
                except json.JSONDecodeError:
                    continue

    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        logger.warning(f"Request error: {e}")
        return None

    return None


async def benchmark_ttft(
    url: str,
    model: str,
    isl: int,
    osl: int = 5,
    warmup: int = 2,
    runs: int = 3,
) -> Optional[float]:
    """Benchmark TTFT (Time to First Token) for a given ISL.

    Args:
        url: Base URL of the inference server (e.g. http://localhost:8000).
        model: Model name for the API request.
        isl: Target input sequence length in tokens.
        osl: Output sequence length (small, just need first token).
        warmup: Number of warmup requests.
        runs: Number of measurement requests.

    Returns:
        Average TTFT in milliseconds, or None on failure.
    """
    prompt = _make_synthetic_prompt(isl)
    body = _make_chat_request_body(model, prompt, max_tokens=osl)

    async with httpx.AsyncClient() as client:
        # Warmup
        for i in range(warmup):
            await _stream_request_ttft(client, url, body)

        # Measurement
        ttfts = []
        for i in range(runs):
            ttft = await _stream_request_ttft(client, url, body)
            if ttft is not None:
                ttfts.append(ttft)

    if not ttfts:
        return None

    avg_ttft = sum(ttfts) / len(ttfts)
    return avg_ttft


async def _stream_request_decode(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
) -> Tuple[list[float], int]:
    """Send one streaming request and collect per-token timestamps.

    Returns:
        (list of token arrival timestamps, total output tokens)
    """
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    timestamps = []
    total_tokens = 0

    try:
        async with client.stream("POST", endpoint, json=body, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status_code != 200:
                return [], 0

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content")
                        if content is not None and content != "":
                            timestamps.append(time.monotonic())
                            total_tokens += 1
                except json.JSONDecodeError:
                    continue

    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        logger.warning(f"Decode request error: {e}")

    return timestamps, total_tokens


async def benchmark_decode(
    url: str,
    model: str,
    isl: int,
    osl: int,
    num_request: int,
    warmup: int = 1,
) -> Tuple[Optional[float], Optional[float]]:
    """Benchmark decode performance: ITL and throughput.

    Sends num_request concurrent streaming requests and measures
    inter-token latency and output token throughput.

    Args:
        url: Base URL of the inference server.
        model: Model name.
        isl: Input sequence length.
        osl: Output sequence length.
        num_request: Number of concurrent requests.
        warmup: Number of sequential warmup requests before measurement.

    Returns:
        (avg_itl_ms, output_token_throughput) or (None, None) on failure.
    """
    prompt = _make_synthetic_prompt(isl)
    body = _make_chat_request_body(model, prompt, max_tokens=osl)

    async with httpx.AsyncClient() as client:
        # Warmup: sequential requests to fill KV cache
        for _ in range(warmup):
            await _stream_request_decode(client, url, body)

        # Measurement: concurrent requests
        t_wall_start = time.monotonic()
        tasks = [
            _stream_request_decode(client, url, body)
            for _ in range(num_request)
        ]
        results = await asyncio.gather(*tasks)
        t_wall_end = time.monotonic()

    # Compute ITL from per-request token timestamps
    all_itls = []
    total_output_tokens = 0
    for timestamps, n_tokens in results:
        total_output_tokens += n_tokens
        if len(timestamps) >= 2:
            # Skip first token (that's TTFT, not ITL)
            for i in range(1, len(timestamps) - 1):
                itl = (timestamps[i + 1] - timestamps[i]) * 1000.0
                all_itls.append(itl)

    if not all_itls or total_output_tokens == 0:
        return None, None

    avg_itl = sum(all_itls) / len(all_itls)
    wall_time = t_wall_end - t_wall_start
    throughput = total_output_tokens / wall_time if wall_time > 0 else 0.0

    return avg_itl, throughput


async def get_server_info(url: str) -> Optional[dict]:
    """Query SGLang /server_info endpoint for KV cache size etc."""
    endpoint = f"{url.rstrip('/')}/server_info"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(endpoint, timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
    except (httpx.ConnectError, httpx.ReadTimeout) as e:
        logger.warning(f"Failed to get server info from {url}: {e}")
    return None

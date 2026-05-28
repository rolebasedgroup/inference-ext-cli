"""vLLM RBG config modifier for profiling.

Generates RBG YAML specs for vLLM engine profiling and manipulates
container args for parallelization settings.
"""

import copy
import logging
import re

from inference_ext_cli.profile.config_modifiers.protocol import (
    append_argument,
    break_arguments,
    remove_valued_arguments,
    set_argument_value,
)
from inference_ext_cli.profile.defaults import DEFAULT_ENGINE_PORT
from inference_ext_cli.profile.parallelization_mapping import SubComponentType

logger = logging.getLogger(__name__)


class VLLMConfigModifier:
    """Config modifier for vLLM engine RBG deployments."""

    BACKEND = "vllm"

    @classmethod
    def generate_rbg_spec(
        cls,
        model: str,
        image: str,
        num_gpus: int,
        port: int = DEFAULT_ENGINE_PORT,
        phase: SubComponentType = SubComponentType.DECODE,
        is_moe: bool = False,
    ) -> dict:
        """Generate an RBG spec for a single vLLM engine role."""
        args = [
            "--model", model,
            "--tensor-parallel-size", str(num_gpus),
            "--port", str(port),
            "--host", "0.0.0.0",
        ]

        # Prefill: disable prefix caching for consistent TTFT measurement
        if phase == SubComponentType.PREFILL:
            args.append("--no-enable-prefix-caching")
        else:
            # Decode: enable prefix caching for KV reuse
            args.append("--enable-prefix-caching")

        return {
            "roles": [
                {
                    "name": "engine",
                    "replicas": 1,
                    "standalonePattern": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "engine",
                                        "image": image,
                                        "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
                                        "args": args,
                                        "ports": [{"containerPort": port}],
                                        "resources": {
                                            "limits": {
                                                "nvidia.com/gpu": str(num_gpus),
                                            }
                                        },
                                    }
                                ]
                            }
                        }
                    },
                }
            ]
        }

    @classmethod
    def get_container_args(cls, rbg_spec: dict) -> list[str]:
        """Extract engine container args from RBG spec."""
        container = rbg_spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        return break_arguments(container.get("args", []))

    @classmethod
    def set_container_args(cls, rbg_spec: dict, args: list[str]) -> dict:
        """Set engine container args in RBG spec."""
        rbg_spec = copy.deepcopy(rbg_spec)
        rbg_spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]["args"] = args
        return rbg_spec

    @classmethod
    def _set_gpu_resources(cls, rbg_spec: dict, num_gpus: int) -> dict:
        """Update GPU resource limit in the container spec."""
        rbg_spec = copy.deepcopy(rbg_spec)
        container = rbg_spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        container["resources"]["limits"]["nvidia.com/gpu"] = str(num_gpus)
        return rbg_spec

    @classmethod
    def set_config_tp_size(
        cls, rbg_spec: dict, tp_size: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Set TP size in vLLM args."""
        rbg_spec = cls._set_gpu_resources(rbg_spec, tp_size)
        args = cls.get_container_args(rbg_spec)

        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", str(tp_size))
        args = remove_valued_arguments(args, "--data-parallel-size")
        args = remove_valued_arguments(args, "--dp")
        args = remove_valued_arguments(args, "--data-parallel-size-local")
        if "--data-parallel-hybrid-lb" in args:
            args.remove("--data-parallel-hybrid-lb")
        if "--enable-expert-parallel" in args:
            args.remove("--enable-expert-parallel")

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def set_config_tep_size(
        cls, rbg_spec: dict, tep_size: int, num_gpus_per_node: int | None,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Set TEP in vLLM args.
        vLLM TEP: --tensor-parallel-size=tep_size --data-parallel-size=1 --enable-expert-parallel.
        EP is derived as TP * DP = tep_size.
        """
        rbg_spec = cls._set_gpu_resources(rbg_spec, tep_size)
        args = cls.get_container_args(rbg_spec)

        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", str(tep_size))
        args = remove_valued_arguments(args, "--dp")
        args = set_argument_value(args, "--data-parallel-size", "1")
        args = remove_valued_arguments(args, "--data-parallel-size-local")
        if "--data-parallel-hybrid-lb" in args:
            args.remove("--data-parallel-hybrid-lb")
        if "--enable-expert-parallel" not in args:
            args = append_argument(args, "--enable-expert-parallel")

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def set_config_dep_size(
        cls, rbg_spec: dict, dep_size: int, num_gpus_per_node: int | None,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Set DEP in vLLM args.
        vLLM DEP: --tensor-parallel-size=1 --data-parallel-size=dep_size --enable-expert-parallel.
        EP is derived as TP * DP = dep_size.
        """
        rbg_spec = cls._set_gpu_resources(rbg_spec, dep_size)
        args = cls.get_container_args(rbg_spec)

        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", "1")
        args = remove_valued_arguments(args, "--dp")
        args = set_argument_value(args, "--data-parallel-size", str(dep_size))

        # Hybrid load balancing for multinode DEP
        if num_gpus_per_node and dep_size > num_gpus_per_node:
            args = set_argument_value(
                args, "--data-parallel-size-local", str(num_gpus_per_node)
            )
            if "--data-parallel-hybrid-lb" not in args:
                args = append_argument(args, "--data-parallel-hybrid-lb")
        else:
            args = remove_valued_arguments(args, "--data-parallel-size-local")
            if "--data-parallel-hybrid-lb" in args:
                args.remove("--data-parallel-hybrid-lb")

        if "--enable-expert-parallel" not in args:
            args = append_argument(args, "--enable-expert-parallel")

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def set_prefill_config(
        cls,
        rbg_spec: dict,
        max_batch_size: int,
        max_num_tokens: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Configure prefill limits for vLLM.
        Uses --max-num-seqs and --max-num-batched-tokens.
        For DEP (DP > 1), uses per-GPU token limit to avoid OOM.
        """
        args = cls.get_container_args(rbg_spec)

        # Detect DP size from args
        dp_size = 1
        for i, arg in enumerate(args):
            if arg in ("--dp", "--data-parallel-size") and i + 1 < len(args):
                dp_size = int(args[i + 1])
                break

        per_gpu_max_tokens = max_num_tokens // dp_size if dp_size > 1 else max_num_tokens

        args = set_argument_value(args, "--max-num-seqs", str(max_batch_size))
        args = set_argument_value(args, "--max-num-batched-tokens", str(per_gpu_max_tokens))

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def get_kv_cache_size_from_log(cls, log: str, attention_dp_size: int = 1) -> int:
        """Parse KV cache token count from vLLM engine log."""
        for line in log.splitlines():
            if "Maximum concurrency for" in line:
                try:
                    part = line.strip().split("Maximum concurrency for ")[1]
                    token_count = int(part.split(" tokens per request: ")[0].replace(",", ""))
                    concurrency = float(part.split(" tokens per request: ")[1].rstrip("."))
                    kv_cache_per_rank = int(token_count * concurrency)
                    return kv_cache_per_rank * attention_dp_size
                except (IndexError, ValueError):
                    continue
        return 0

"""SGLang RBG config modifier for profiling.

Generates RBG YAML specs for SGLang engine profiling and manipulates
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


class SGLangConfigModifier:
    """Config modifier for SGLang engine RBG deployments."""

    BACKEND = "sglang"

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
        """Generate an RBG spec for a single SGLang engine role."""
        args = [
            "--model-path", model,
            "--tp", str(num_gpus),
            "--port", str(port),
            "--host", "0.0.0.0",
        ]

        # Prefill: disable radix cache for consistent TTFT measurement
        if phase == SubComponentType.PREFILL:
            args.append("--disable-radix-cache")
        else:
            # Decode: enable prefix caching for KV reuse
            if is_moe:
                args.extend(["--load-balance-method", "round_robin"])

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
                                        "command": ["python3", "-m", "sglang.launch_server"],
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
        """Set TP size in SGLang args."""
        rbg_spec = cls._set_gpu_resources(rbg_spec, tp_size)
        args = cls.get_container_args(rbg_spec)

        args = set_argument_value(args, "--tp", str(tp_size))
        args = remove_valued_arguments(args, "--tp-size")
        args = remove_valued_arguments(args, "--tensor-parallel-size")
        args = remove_valued_arguments(args, "--ep")
        args = remove_valued_arguments(args, "--ep-size")
        args = remove_valued_arguments(args, "--dp")
        args = remove_valued_arguments(args, "--dp-size")
        if "--enable-dp-attention" in args:
            args.remove("--enable-dp-attention")

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def set_config_tep_size(
        cls, rbg_spec: dict, tep_size: int, num_gpus_per_node: int | None,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Set TEP (tensor-expert parallelism) in SGLang args.
        TEP: --tp=tep_size --ep=tep_size, no DP.
        """
        rbg_spec = cls._set_gpu_resources(rbg_spec, tep_size)
        args = cls.get_container_args(rbg_spec)

        args = set_argument_value(args, "--tp", str(tep_size))
        args = remove_valued_arguments(args, "--tp-size")
        args = set_argument_value(args, "--ep", str(tep_size))
        args = remove_valued_arguments(args, "--ep-size")
        args = remove_valued_arguments(args, "--dp")
        args = remove_valued_arguments(args, "--dp-size")
        if "--enable-dp-attention" in args:
            args.remove("--enable-dp-attention")

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def set_config_dep_size(
        cls, rbg_spec: dict, dep_size: int, num_gpus_per_node: int | None,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Set DEP (data-expert parallelism) in SGLang args.
        DEP: --tp=dep_size --dp=dep_size --ep=dep_size --enable-dp-attention.
        """
        rbg_spec = cls._set_gpu_resources(rbg_spec, dep_size)
        args = cls.get_container_args(rbg_spec)

        args = set_argument_value(args, "--tp", str(dep_size))
        args = remove_valued_arguments(args, "--tp-size")
        args = set_argument_value(args, "--dp", str(dep_size))
        args = remove_valued_arguments(args, "--dp-size")
        args = set_argument_value(args, "--ep", str(dep_size))
        args = remove_valued_arguments(args, "--ep-size")
        if "--enable-dp-attention" not in args:
            args = append_argument(args, "--enable-dp-attention")

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def set_prefill_config(
        cls,
        rbg_spec: dict,
        max_batch_size: int,
        max_num_tokens: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Configure prefill limits: max running requests and chunked prefill size."""
        args = cls.get_container_args(rbg_spec)

        args = set_argument_value(args, "--max-running-requests", str(max_batch_size))
        args = set_argument_value(args, "--chunked-prefill-size", str(max_num_tokens))
        if "--enable-dp-lm-head" not in args:
            args = append_argument(args, "--enable-dp-lm-head")

        return cls.set_container_args(rbg_spec, args)

    @classmethod
    def get_kv_cache_size_from_log(cls, log: str, attention_dp_size: int = 1) -> int:
        """Parse KV cache token count from SGLang engine log."""
        for line in log.splitlines():
            if "KV Cache is allocated" in line and "#tokens:" in line:
                match = re.search(r"#tokens:\s*(\d+)", line)
                if match:
                    return int(match.group(1)) * attention_dp_size
        return 0

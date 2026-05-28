"""Base protocol for engine-specific RBG config modifiers."""

import re
from typing import Protocol

from inference_ext_cli.profile.parallelization_mapping import SubComponentType


def break_arguments(args: list[str] | None) -> list[str]:
    """Split combined arguments like '--tp=4' into ['--tp', '4']."""
    if not args:
        return []
    result = []
    for arg in args:
        if "=" in arg and arg.startswith("-"):
            parts = arg.split("=", 1)
            result.extend(parts)
        else:
            result.append(arg)
    return result


def set_argument_value(args: list[str], flag: str, value: str) -> list[str]:
    """Set the value of a flag argument. Adds it if not present."""
    args = list(args)
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            args[i + 1] = value
            return args
    args.extend([flag, value])
    return args


def remove_valued_arguments(args: list[str], flag: str) -> list[str]:
    """Remove a flag and its value from the args list."""
    result = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        result.append(arg)
    return result


def append_argument(args: list[str], arg: str | list[str]) -> list[str]:
    """Append argument(s) to the args list."""
    args = list(args)
    if isinstance(arg, list):
        args.extend(arg)
    else:
        args.append(arg)
    return args


class BaseConfigModifier(Protocol):
    """Protocol for engine-specific RBG config generation and manipulation."""

    @classmethod
    def generate_rbg_spec(
        cls,
        model: str,
        image: str,
        num_gpus: int,
        port: int,
        phase: SubComponentType,
        is_moe: bool,
    ) -> dict:
        """Generate an RBG spec dict for profiling with a single engine role."""
        ...

    @classmethod
    def get_container_args(cls, rbg_spec: dict) -> list[str]:
        """Extract engine container args from an RBG spec."""
        ...

    @classmethod
    def set_container_args(cls, rbg_spec: dict, args: list[str]) -> dict:
        """Set engine container args in an RBG spec."""
        ...

    @classmethod
    def set_config_tp_size(
        cls, rbg_spec: dict, tp_size: int, component_type: SubComponentType
    ) -> dict:
        """Set tensor parallelism size."""
        ...

    @classmethod
    def set_config_tep_size(
        cls, rbg_spec: dict, tep_size: int, num_gpus_per_node: int | None,
        component_type: SubComponentType,
    ) -> dict:
        """Set tensor-expert parallelism size."""
        ...

    @classmethod
    def set_config_dep_size(
        cls, rbg_spec: dict, dep_size: int, num_gpus_per_node: int | None,
        component_type: SubComponentType,
    ) -> dict:
        """Set data-expert parallelism size."""
        ...

    @classmethod
    def set_prefill_config(
        cls,
        rbg_spec: dict,
        max_batch_size: int,
        max_num_tokens: int,
        component_type: SubComponentType,
    ) -> dict:
        """Configure prefill-related limits."""
        ...

    @classmethod
    def get_kv_cache_size_from_log(cls, log: str, attention_dp_size: int) -> int:
        """Parse KV cache token count from engine pod log text."""
        ...

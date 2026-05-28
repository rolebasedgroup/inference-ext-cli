"""Parallelization mapping logic for GPU profiling sweeps.

Ported from dynamo/benchmarks/profiler/utils/config_modifiers/parallelization_mapping.py.
Generates candidate TP/TEP/DEP mappings verified against model constraints.
"""

import copy
import logging
from dataclasses import dataclass
from enum import Enum

from inference_ext_cli.profile.defaults import PREFILL_MAX_NUM_TOKENS
from inference_ext_cli.profile.model_info import MOE_ADDITIONAL_TP_ARCHITECTURES, ModelInfo

logger = logging.getLogger(__name__)


class ParallelizationStrategy(Enum):
    """Parallelization strategy types."""

    TP = "TP"
    TEP = "TEP"
    DEP = "DEP"


class SubComponentType(str, Enum):
    """Phase type for profiling."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ParallelizationMapping:
    """Represents a parallelization configuration (exactly one of tp/tep/dep is set)."""

    tp: int | None = None
    tep: int | None = None
    dep: int | None = None

    def label(self) -> str:
        if self.tp is not None:
            return f"{ParallelizationStrategy.TP.value}={self.tp}"
        if self.tep is not None:
            return f"{ParallelizationStrategy.TEP.value}={self.tep}"
        if self.dep is not None:
            return f"{ParallelizationStrategy.DEP.value}={self.dep}"
        return "default"

    def get_tp_size(self) -> int:
        """Effective TP size for KV heads splitting. Both TP and TEP split KV heads."""
        if self.tp is not None:
            return self.tp
        if self.tep is not None:
            return self.tep
        return 1

    def get_expert_split(self) -> int:
        """Effective expert split size. Both TEP and DEP split experts."""
        if self.tep is not None:
            return self.tep
        if self.dep is not None:
            return self.dep
        return 1

    def get_attn_dp_size(self) -> int:
        """Attention data parallelism size. Only DEP uses attention DP."""
        return self.dep if self.dep is not None else 1

    def get_num_gpus(self) -> int:
        """Total number of GPUs for this mapping."""
        if self.tp is not None:
            return self.tp
        if self.tep is not None:
            return self.tep
        if self.dep is not None:
            return self.dep
        raise ValueError("Invalid ParallelizationMapping: no strategy set")


def _check_divisibility(
    value: int | None,
    divisor: int,
    value_name: str,
    divisor_name: str,
    mapping_label: str,
) -> bool:
    """Check if value is divisible by divisor. Returns True if valid (or value is None)."""
    if value is None:
        logger.warning(
            f"Skipping {value_name} divisibility check for {mapping_label}: {value_name} is unknown"
        )
        return True

    if divisor > 1 and int(value) % divisor != 0:
        logger.warning(
            f"Invalid mapping {mapping_label}: {value_name}={value} not divisible by {divisor_name}={divisor}"
        )
        return False

    return True


def _validate_intermediate_size(
    mapping: ParallelizationMapping,
    intermediate_size: int | None,
    quant_block: int | None,
) -> bool:
    """Validate intermediate size and quantization block for TP/TEP strategies."""
    tp_size = mapping.get_tp_size()

    if not _check_divisibility(
        intermediate_size, tp_size, "intermediate_size", "tp_size", mapping.label()
    ):
        return False

    if intermediate_size is not None and quant_block is not None and tp_size > 1:
        per_shard = int(intermediate_size) // tp_size
        if not _check_divisibility(
            per_shard, quant_block, "per_shard", "quant_block", mapping.label()
        ):
            return False

    return True


def get_candidate_parallel_mappings(
    num_gpus: int, model_info: ModelInfo, phase: str
) -> list[ParallelizationMapping]:
    """Return verified candidate parallelization mappings for a GPU count and phase.

    Verification rules:
    - TP and TEP must divide num_kv_heads (if available)
    - TEP and DEP must divide num_experts (if available)
    - intermediate_size must be divisible by tp_size
    """
    is_moe = bool(model_info.is_moe)
    num_kv_heads = model_info.num_kv_heads
    num_experts = model_info.num_experts
    intermediate_size = model_info.intermediate_size
    quant_block = model_info.quantization_block_size

    candidates: list[ParallelizationMapping] = []
    if is_moe:
        candidates = [
            ParallelizationMapping(tep=num_gpus),
            ParallelizationMapping(dep=num_gpus),
        ]
        if model_info.architecture in MOE_ADDITIONAL_TP_ARCHITECTURES:
            candidates.append(ParallelizationMapping(tp=num_gpus))
    else:
        candidates = [ParallelizationMapping(tp=num_gpus)]

    verified: list[ParallelizationMapping] = []
    for m in candidates:
        if not _check_divisibility(
            num_kv_heads, m.get_tp_size(), "num_kv_heads", "tp_size", m.label()
        ):
            continue
        if not _check_divisibility(
            num_experts, m.get_expert_split(), "num_experts", "expert_split", m.label()
        ):
            continue
        if not _validate_intermediate_size(m, intermediate_size, quant_block):
            continue
        verified.append(m)

    return verified


def apply_parallel_mapping_to_config(
    base_config: dict,
    mapping: ParallelizationMapping,
    phase: SubComponentType,
    config_modifier,
    num_gpus_per_node: int | None,
) -> dict:
    """Apply parallelization mapping to an RBG config dict via config_modifier."""
    cfg = copy.deepcopy(base_config)

    if mapping.tp is not None:
        cfg = config_modifier.set_config_tp_size(cfg, mapping.tp, phase)
    elif mapping.tep is not None:
        cfg = config_modifier.set_config_tep_size(cfg, mapping.tep, num_gpus_per_node, phase)
    elif mapping.dep is not None:
        cfg = config_modifier.set_config_dep_size(cfg, mapping.dep, num_gpus_per_node, phase)
    else:
        raise ValueError(f"Invalid mapping: {mapping.label()}")

    if phase == SubComponentType.PREFILL:
        cfg = config_modifier.set_prefill_config(
            cfg,
            max_batch_size=mapping.get_attn_dp_size(),
            max_num_tokens=PREFILL_MAX_NUM_TOKENS * mapping.get_attn_dp_size(),
            component_type=phase,
        )

    return cfg

"""HuggingFace model metadata extraction.

Ported from dynamo/benchmarks/profiler/utils/model_info.py.
Fetches model architecture, size, context length, and MoE properties.
"""

from pathlib import Path
from typing import Optional, Union

from huggingface_hub import model_info
from pydantic import BaseModel
from transformers import AutoConfig

DTYPE_BYTES_MAP = {
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I4": 0.5,
}

CONTEXT_LENGTH_ATTRS = [
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "model_max_length",
    "sliding_window",
]

# Supported MoE architectures
MOE_ARCHITECTURES = {
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "Qwen3MoeForCausalLM",
}

# MoE architectures that also sweep TP in addition to TEP/DEP
MOE_ADDITIONAL_TP_ARCHITECTURES = {
    "Qwen3MoeForCausalLM",
}


def get_local_model_weight_size(model_path: Union[str, Path]) -> float:
    """Return model size in MB by scanning local directory."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise ValueError(f"Model path is not a directory: {model_path}")

    weight_extensions = [".safetensors", ".bin", ".pt", ".pth"]
    total_size_bytes = 0
    for file_path in model_path.rglob("*"):
        if file_path.is_file() and any(
            str(file_path).endswith(ext) for ext in weight_extensions
        ):
            total_size_bytes += file_path.stat().st_size

    return total_size_bytes / (1024**2)


def get_model_weight_size_from_hub(
    model_name: str, token: Optional[str] = None
) -> float:
    """Return model size in MB by querying Hugging Face Hub API."""
    try:
        info = model_info(model_name, token=token)

        weight_extensions = [".safetensors", ".bin", ".pt", ".pth"]
        total_size_bytes = 0

        if info.siblings is not None:
            for sibling in info.siblings:
                if any(sibling.rfilename.endswith(ext) for ext in weight_extensions):
                    if sibling.size is not None:
                        total_size_bytes += sibling.size

        if total_size_bytes == 0 and info.safetensors is not None:
            for dtype, param_count in info.safetensors.parameters.items():
                bytes_per_param = DTYPE_BYTES_MAP.get(dtype, 2)
                total_size_bytes += int(param_count * bytes_per_param)

        return total_size_bytes / (1024**2)
    except Exception as e:
        raise RuntimeError(f"Failed to get model info from Hub: {e}")


def get_model_weight_size(model_name_or_path: Union[str, Path]) -> float:
    """Return model size in MB (auto-detects local vs HF Hub)."""
    path = Path(model_name_or_path)
    if path.exists() and path.is_dir():
        return get_local_model_weight_size(model_name_or_path)
    else:
        return get_model_weight_size_from_hub(str(model_name_or_path))


class ModelInfo(BaseModel):
    model_size: float
    architecture: str
    is_moe: bool
    max_context_length: Optional[int] = None
    num_experts: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_kv_heads: Optional[int] = None
    quantization_block_size: Optional[int] = None


def get_model_info(
    model_name_or_path: Union[str, Path],
    trust_remote_code: bool = False,
) -> ModelInfo:
    """Fetch model metadata from HF config and compute weight size."""
    model_size = get_model_weight_size(model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    architecture = config.architectures[0]
    is_moe = architecture in MOE_ARCHITECTURES

    # Detect max context length
    max_context_length = None
    for attr in CONTEXT_LENGTH_ATTRS:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                max_context_length = value
                break

    # Detect number of experts for MoE models
    num_experts = None
    if is_moe:
        expert_attrs = ["n_routed_experts", "num_local_experts", "num_experts"]
        for attr in expert_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                if value is not None:
                    num_experts = value
                    break

    # Detect intermediate size (FFN hidden dimension)
    intermediate_size = None
    for attr in ["intermediate_size", "ffn_dim"]:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                intermediate_size = value
                break

    # Detect number of key-value heads (for GQA)
    num_kv_heads = None
    for attr in ["num_key_value_heads", "num_kv_heads"]:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                num_kv_heads = value
                break
    if num_kv_heads is None and hasattr(config, "num_attention_heads"):
        num_kv_heads = config.num_attention_heads

    # Detect quantization block size
    quantization_block_size = None
    if hasattr(config, "quantization_config"):
        quant_config = config.quantization_config
        if isinstance(quant_config, dict):
            quantization_block_size = (
                quant_config.get("weight_block_size")
                or quant_config.get("block_size")
                or quant_config.get("group_size")
                or quant_config.get("q_group_size")
            )
        elif quant_config is not None:
            for attr in ["weight_block_size", "block_size", "group_size", "q_group_size"]:
                if hasattr(quant_config, attr):
                    value = getattr(quant_config, attr)
                    if value is not None:
                        quantization_block_size = value
                        break

        if isinstance(quantization_block_size, list) and len(quantization_block_size) > 0:
            quantization_block_size = max(quantization_block_size)

    return ModelInfo(
        model_size=model_size,
        architecture=architecture,
        is_moe=is_moe,
        max_context_length=max_context_length,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        num_kv_heads=num_kv_heads,
        quantization_block_size=quantization_block_size,
    )

"""Output generation for profiling results.

Generates Kubernetes ConfigMap YAML containing profiling data,
suitable for consumption by the RBG planner.
"""

import json
import logging
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def npz_to_json(npz_path: str) -> dict:
    """Load an NPZ file and convert arrays to JSON-serializable lists."""
    data = np.load(npz_path)
    result = {}
    for key in data.files:
        arr = data[key]
        if arr.ndim == 0:
            result[key] = float(arr)
        elif arr.ndim == 1 and len(arr) == 1:
            result[key] = float(arr[0])
        else:
            result[key] = arr.tolist()
    return result


def generate_profiling_configmap(
    prefill_data: dict,
    decode_data: dict,
    name: str = "profiling-data",
    namespace: str = "default",
) -> dict:
    """Generate a Kubernetes ConfigMap containing profiling data.

    Args:
        prefill_data: Dict with prefill_isl, prefill_ttft, prefill_thpt_per_gpu.
        decode_data: Dict with x_kv_usage, y_context_length, z_itl, z_thpt_per_gpu, max_kv_tokens.
        name: ConfigMap name.
        namespace: ConfigMap namespace.

    Returns:
        ConfigMap dict ready for YAML serialization.
    """
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "data": {
            "prefill_raw_data.json": json.dumps(prefill_data, indent=2),
            "decode_raw_data.json": json.dumps(decode_data, indent=2),
        },
    }


def save_configmap_yaml(
    configmap: dict,
    output_path: str,
):
    """Save ConfigMap dict as YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(configmap, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved ConfigMap YAML to {output_path}")


def generate_profiling_output(
    prefill_data: dict,
    decode_data: dict,
    output_dir: str,
    configmap_name: str = "profiling-data",
    configmap_namespace: str = "default",
) -> str:
    """Generate profiling ConfigMap YAML and save to output directory.

    Args:
        prefill_data: Prefill profiling results dict.
        decode_data: Decode profiling results dict.
        output_dir: Directory to save outputs.
        configmap_name: Name for the ConfigMap resource.
        configmap_namespace: Namespace for the ConfigMap resource.

    Returns:
        Path to the generated ConfigMap YAML file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    configmap = generate_profiling_configmap(
        prefill_data, decode_data,
        name=configmap_name, namespace=configmap_namespace,
    )

    output_path = os.path.join(output_dir, "profiling-configmap.yaml")
    save_configmap_yaml(configmap, output_path)

    # Also save raw JSON files for reference
    prefill_json_path = os.path.join(output_dir, "prefill_raw_data.json")
    decode_json_path = os.path.join(output_dir, "decode_raw_data.json")

    with open(prefill_json_path, "w") as f:
        json.dump(prefill_data, f, indent=2)
    with open(decode_json_path, "w") as f:
        json.dump(decode_data, f, indent=2)

    logger.info(f"Saved raw JSON to {prefill_json_path} and {decode_json_path}")

    return output_path

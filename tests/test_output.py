"""Tests for output generation."""

import json
import os
import tempfile

import pytest

from inference_ext_cli.profile.output import (
    generate_profiling_configmap,
    generate_profiling_output,
)


class TestGenerateProfilingConfigmap:
    def test_basic_structure(self):
        prefill_data = {
            "prefill_isl": [100, 500, 1000],
            "prefill_ttft": [5.0, 20.0, 80.0],
            "prefill_thpt_per_gpu": [6000, 5000, 3000],
        }
        decode_data = {
            "x_kv_usage": [0.1, 0.5],
            "y_context_length": [512, 1024],
            "z_itl": [8.0, 15.0],
            "z_thpt_per_gpu": [1200, 900],
            "max_kv_tokens": 32768,
        }
        cm = generate_profiling_configmap(prefill_data, decode_data)

        assert cm["apiVersion"] == "v1"
        assert cm["kind"] == "ConfigMap"
        assert cm["metadata"]["name"] == "profiling-data"
        assert cm["metadata"]["namespace"] == "default"
        assert "prefill_raw_data.json" in cm["data"]
        assert "decode_raw_data.json" in cm["data"]

        # Verify JSON is valid
        prefill_parsed = json.loads(cm["data"]["prefill_raw_data.json"])
        assert prefill_parsed["prefill_isl"] == [100, 500, 1000]

    def test_custom_name_namespace(self):
        cm = generate_profiling_configmap(
            {"a": 1}, {"b": 2},
            name="my-profiling", namespace="inference",
        )
        assert cm["metadata"]["name"] == "my-profiling"
        assert cm["metadata"]["namespace"] == "inference"


class TestGenerateProfilingOutput:
    def test_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefill_data = {"prefill_isl": [100], "prefill_ttft": [5.0]}
            decode_data = {"z_itl": [8.0], "max_kv_tokens": 32768}

            output_path = generate_profiling_output(
                prefill_data, decode_data, output_dir=tmpdir,
            )

            assert os.path.exists(output_path)
            assert os.path.exists(os.path.join(tmpdir, "prefill_raw_data.json"))
            assert os.path.exists(os.path.join(tmpdir, "decode_raw_data.json"))

            # Verify YAML is valid
            import yaml
            with open(output_path) as f:
                cm = yaml.safe_load(f)
            assert cm["kind"] == "ConfigMap"

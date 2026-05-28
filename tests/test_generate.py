"""Tests for the unified generate command."""

import json
import os

import yaml
from click.testing import CliRunner

from inference_ext_cli.main import cli

SAMPLE_RBG = {
    "apiVersion": "workloads.x-k8s.io/v1alpha2",
    "kind": "RoleBasedGroup",
    "metadata": {
        "name": "sglang-pd-inference",
        "namespace": "default",
    },
    "spec": {
        "roles": [
            {
                "name": "prefill",
                "replicas": 2,
                "standalonePattern": {
                    "template": {
                        "spec": {
                            "containers": [
                                {"name": "sglang", "image": "sglang:latest"}
                            ]
                        }
                    }
                },
            },
            {
                "name": "decode",
                "replicas": 3,
                "standalonePattern": {
                    "template": {
                        "spec": {
                            "containers": [
                                {"name": "sglang", "image": "sglang:latest"}
                            ]
                        }
                    }
                },
            },
        ]
    },
}


def _create_prefill_json(path: str):
    data = {
        "prefill_isl": [128, 256, 512, 1024],
        "prefill_ttft": [0.01, 0.02, 0.04, 0.08],
        "prefill_thpt_per_gpu": [5000, 4000, 3000, 2000],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _create_decode_json(path: str):
    data = {
        "x_kv_usage": [0.1, 0.2, 0.3, 0.5],
        "y_context_length": [256, 512, 1024, 2048],
        "z_itl": [0.01, 0.015, 0.02, 0.03],
        "z_thpt_per_gpu": [1000, 900, 800, 600],
        "max_kv_tokens": 32768,
    }
    with open(path, "w") as f:
        json.dump(data, f)


class TestGenerateFromExistingRBG:
    """Tests for generate with --rbg-yaml input."""

    def test_json_profiling_source(self, tmp_path):
        rbg_path = str(tmp_path / "rbg.yaml")
        output_dir = str(tmp_path / "output")

        with open(rbg_path, "w") as f:
            yaml.dump(SAMPLE_RBG, f)

        prefill_path = str(tmp_path / "prefill.json")
        decode_path = str(tmp_path / "decode.json")
        _create_prefill_json(prefill_path)
        _create_decode_json(decode_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--rbg-yaml", rbg_path,
            "--enable-planner",
            "--planner-image", "ghcr.io/rbg/planner:v0.1",
            "--model-name", "Qwen/Qwen3-0.6B",
            "--profiling-source", "json",
            "--prefill-json", prefill_path,
            "--decode-json", decode_path,
            "--ttft-sla", "200",
            "--itl-sla", "20",
            "-o", output_dir,
        ])

        assert result.exit_code == 0, result.output

        # Check rbg.yaml output
        rbg_output = os.path.join(output_dir, "rbg.yaml")
        assert os.path.exists(rbg_output)
        with open(rbg_output) as f:
            rbg = yaml.safe_load(f)

        roles = rbg["spec"]["roles"]
        assert len(roles) == 3  # prefill + decode + planner
        planner_role = next(r for r in roles if r["name"] == "planner")
        container = planner_role["standalonePattern"]["template"]["spec"]["containers"][0]
        assert container["image"] == "ghcr.io/rbg/planner:v0.1"

        env_dict = {e["name"]: e["value"] for e in container["env"]}
        assert env_dict["RBG_NAME"] == "sglang-pd-inference"
        assert env_dict["MODEL_NAME"] == "Qwen/Qwen3-0.6B"
        assert env_dict["TTFT_SLA"] == "200.0"
        assert env_dict["ITL_SLA"] == "20.0"

        # Check ConfigMap output
        cm_output = os.path.join(output_dir, "profiling-configmap.yaml")
        assert os.path.exists(cm_output)
        with open(cm_output) as f:
            cm = yaml.safe_load(f)
        assert cm["kind"] == "ConfigMap"
        assert "prefill_raw_data.json" in cm["data"]

    def test_configmap_profiling_source(self, tmp_path):
        rbg_path = str(tmp_path / "rbg.yaml")
        output_dir = str(tmp_path / "output")

        with open(rbg_path, "w") as f:
            yaml.dump(SAMPLE_RBG, f)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--rbg-yaml", rbg_path,
            "--enable-planner",
            "--planner-image", "planner:latest",
            "--model-name", "test-model",
            "--profiling-source", "configmap",
            "--profiling-configmap", "my-existing-cm",
            "-o", output_dir,
        ])

        assert result.exit_code == 0, result.output

        # Check rbg.yaml references the existing ConfigMap
        rbg_output = os.path.join(output_dir, "rbg.yaml")
        with open(rbg_output) as f:
            rbg = yaml.safe_load(f)

        planner_role = next(r for r in rbg["spec"]["roles"] if r["name"] == "planner")
        volumes = planner_role["standalonePattern"]["template"]["spec"]["volumes"]
        assert volumes[0]["configMap"]["name"] == "my-existing-cm"

        # No ConfigMap file should be generated
        cm_output = os.path.join(output_dir, "profiling-configmap.yaml")
        assert not os.path.exists(cm_output)

    def test_without_planner(self, tmp_path):
        rbg_path = str(tmp_path / "rbg.yaml")
        output_dir = str(tmp_path / "output")

        with open(rbg_path, "w") as f:
            yaml.dump(SAMPLE_RBG, f)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--rbg-yaml", rbg_path,
            "-o", output_dir,
        ])

        assert result.exit_code == 0, result.output

        rbg_output = os.path.join(output_dir, "rbg.yaml")
        with open(rbg_output) as f:
            rbg = yaml.safe_load(f)

        # Should have only original roles, no planner
        roles = rbg["spec"]["roles"]
        assert len(roles) == 2
        assert all(r["name"] in ("prefill", "decode") for r in roles)

    def test_replaces_existing_planner(self, tmp_path):
        rbg_with_planner = {
            **SAMPLE_RBG,
            "spec": {
                "roles": list(SAMPLE_RBG["spec"]["roles"]) + [
                    {"name": "planner", "replicas": 1, "standalonePattern": {
                        "template": {"spec": {"containers": [{"name": "old"}]}}
                    }}
                ]
            },
        }

        rbg_path = str(tmp_path / "rbg.yaml")
        output_dir = str(tmp_path / "output")
        with open(rbg_path, "w") as f:
            yaml.dump(rbg_with_planner, f)

        prefill_path = str(tmp_path / "prefill.json")
        decode_path = str(tmp_path / "decode.json")
        _create_prefill_json(prefill_path)
        _create_decode_json(decode_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--rbg-yaml", rbg_path,
            "--enable-planner",
            "--planner-image", "planner:new",
            "--model-name", "model",
            "--profiling-source", "json",
            "--prefill-json", prefill_path,
            "--decode-json", decode_path,
            "-o", output_dir,
        ])

        assert result.exit_code == 0, result.output

        rbg_output = os.path.join(output_dir, "rbg.yaml")
        with open(rbg_output) as f:
            rbg = yaml.safe_load(f)

        planner_roles = [r for r in rbg["spec"]["roles"] if r["name"] == "planner"]
        assert len(planner_roles) == 1
        container = planner_roles[0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert container["image"] == "planner:new"


class TestGenerateFromScratch:
    """Tests for generate with --engine/--model/--engine-image."""

    def test_sglang_from_scratch(self, tmp_path):
        output_dir = str(tmp_path / "output")
        prefill_path = str(tmp_path / "prefill.json")
        decode_path = str(tmp_path / "decode.json")
        _create_prefill_json(prefill_path)
        _create_decode_json(decode_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--engine", "sglang",
            "--model", "Qwen/Qwen3-0.6B",
            "--engine-image", "lmsysorg/sglang:latest",
            "--prefill-tp", "2",
            "--decode-tp", "4",
            "--enable-planner",
            "--planner-image", "planner:v1",
            "--model-name", "Qwen/Qwen3-0.6B",
            "--profiling-source", "json",
            "--prefill-json", prefill_path,
            "--decode-json", decode_path,
            "-o", output_dir,
        ])

        assert result.exit_code == 0, result.output

        rbg_output = os.path.join(output_dir, "rbg.yaml")
        with open(rbg_output) as f:
            rbg = yaml.safe_load(f)

        assert rbg["apiVersion"] == "workloads.x-k8s.io/v1alpha2"
        assert rbg["kind"] == "RoleBasedGroup"

        roles = rbg["spec"]["roles"]
        assert len(roles) == 3  # prefill + decode + planner

        prefill_role = next(r for r in roles if r["name"] == "prefill")
        decode_role = next(r for r in roles if r["name"] == "decode")

        # Check GPU resources
        p_container = prefill_role["standalonePattern"]["template"]["spec"]["containers"][0]
        d_container = decode_role["standalonePattern"]["template"]["spec"]["containers"][0]
        assert p_container["resources"]["limits"]["nvidia.com/gpu"] == "2"
        assert d_container["resources"]["limits"]["nvidia.com/gpu"] == "4"

    def test_vllm_from_scratch(self, tmp_path):
        output_dir = str(tmp_path / "output")
        prefill_path = str(tmp_path / "prefill.json")
        decode_path = str(tmp_path / "decode.json")
        _create_prefill_json(prefill_path)
        _create_decode_json(decode_path)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--engine", "vllm",
            "--model", "meta-llama/Llama-3-8B",
            "--engine-image", "vllm/vllm-openai:latest",
            "--enable-planner",
            "--planner-image", "planner:v1",
            "--model-name", "meta-llama/Llama-3-8B",
            "--profiling-source", "json",
            "--prefill-json", prefill_path,
            "--decode-json", decode_path,
            "--metric-source", "vllm",
            "-o", output_dir,
        ])

        assert result.exit_code == 0, result.output

        rbg_output = os.path.join(output_dir, "rbg.yaml")
        with open(rbg_output) as f:
            rbg = yaml.safe_load(f)

        planner_role = next(r for r in rbg["spec"]["roles"] if r["name"] == "planner")
        container = planner_role["standalonePattern"]["template"]["spec"]["containers"][0]
        env_dict = {e["name"]: e["value"] for e in container["env"]}
        assert env_dict["METRIC_SOURCE"] == "vllm"


class TestGenerateErrors:
    """Tests for validation error cases."""

    def test_no_input_source(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "-o", "/tmp/x"])
        assert result.exit_code != 0

    def test_from_scratch_missing_model(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--engine", "sglang",
            "--engine-image", "img:v1",
            # missing --model
            "-o", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_planner_missing_image(self, tmp_path):
        rbg_path = str(tmp_path / "rbg.yaml")
        with open(rbg_path, "w") as f:
            yaml.dump(SAMPLE_RBG, f)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--rbg-yaml", rbg_path,
            "--enable-planner",
            # missing --planner-image
            "--model-name", "m",
            "--profiling-source", "configmap",
            "--profiling-configmap", "cm",
            "-o", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_json_source_missing_files(self, tmp_path):
        rbg_path = str(tmp_path / "rbg.yaml")
        with open(rbg_path, "w") as f:
            yaml.dump(SAMPLE_RBG, f)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--rbg-yaml", rbg_path,
            "--enable-planner",
            "--planner-image", "img",
            "--model-name", "m",
            "--profiling-source", "json",
            # missing --prefill-json and --decode-json
            "-o", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_configmap_source_missing_name(self, tmp_path):
        rbg_path = str(tmp_path / "rbg.yaml")
        with open(rbg_path, "w") as f:
            yaml.dump(SAMPLE_RBG, f)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "generate",
            "--rbg-yaml", rbg_path,
            "--enable-planner",
            "--planner-image", "img",
            "--model-name", "m",
            "--profiling-source", "configmap",
            # missing --profiling-configmap
            "-o", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

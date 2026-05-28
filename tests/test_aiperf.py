"""Tests for AIPerf command building and result parsing."""

import json
import os
import tempfile

import pytest

from inference_ext_cli.profile.aiperf import (
    get_aiperf_result,
    get_decode_aiperf_cmd,
    get_prefill_aiperf_cmd,
)


class TestPrefillAiperfCmd:
    def test_basic_cmd(self):
        cmd = get_prefill_aiperf_cmd(
            isl=1024, artifact_dir="/tmp/test", seed=42,
            model="test-model", tokenizer="test-tok",
        )
        assert "aiperf" == cmd[0]
        assert "profile" == cmd[1]
        assert "--model" in cmd
        assert "test-model" == cmd[cmd.index("--model") + 1]
        assert "--synthetic-input-tokens-mean" in cmd
        assert "1024" == cmd[cmd.index("--synthetic-input-tokens-mean") + 1]
        assert "--concurrency" in cmd
        assert "--request-count" in cmd

    def test_custom_concurrency(self):
        cmd = get_prefill_aiperf_cmd(
            isl=512, artifact_dir="/tmp/test",
            concurrency=4, request_count=8,
        )
        assert "4" == cmd[cmd.index("--concurrency") + 1]
        assert "8" == cmd[cmd.index("--request-count") + 1]

    def test_osl_in_cmd(self):
        cmd = get_prefill_aiperf_cmd(isl=256, artifact_dir="/tmp/test", osl=10)
        assert "--output-tokens-mean" in cmd
        assert "10" == cmd[cmd.index("--output-tokens-mean") + 1]


class TestDecodeAiperfCmd:
    def test_basic_cmd(self):
        cmd = get_decode_aiperf_cmd(
            isl=512, osl=256, artifact_dir="/tmp/test",
            num_request=16, seed=99,
        )
        assert "aiperf" == cmd[0]
        assert "--synthetic-input-tokens-mean" in cmd
        assert "512" == cmd[cmd.index("--synthetic-input-tokens-mean") + 1]
        assert "--output-tokens-mean" in cmd
        assert "256" == cmd[cmd.index("--output-tokens-mean") + 1]
        # Concurrency, dataset entries, and request count should all be num_request
        assert "16" == cmd[cmd.index("--concurrency") + 1]
        assert "16" == cmd[cmd.index("--num-dataset-entries") + 1]
        assert "16" == cmd[cmd.index("--request-count") + 1]


class TestGetAiperfResult:
    def test_finds_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "run1", "artifacts")
            os.makedirs(subdir)
            data = {"time_to_first_token": {"avg": 50.0, "max": 60.0}}
            with open(os.path.join(subdir, "profile_export_aiperf.json"), "w") as f:
                json.dump(data, f)

            result = get_aiperf_result(tmpdir)
            assert result["time_to_first_token"]["avg"] == 50.0

    def test_raises_when_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                get_aiperf_result(tmpdir)

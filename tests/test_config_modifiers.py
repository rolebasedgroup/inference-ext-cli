"""Tests for config modifiers (SGLang and vLLM)."""

import pytest

from inference_ext_cli.profile.config_modifiers.sglang import SGLangConfigModifier
from inference_ext_cli.profile.config_modifiers.vllm import VLLMConfigModifier
from inference_ext_cli.profile.parallelization_mapping import SubComponentType


class TestSGLangConfigModifier:
    def test_generate_rbg_spec_prefill(self):
        spec = SGLangConfigModifier.generate_rbg_spec(
            model="Qwen/Qwen3-0.6B", image="lmsysorg/sglang:latest",
            num_gpus=2, phase=SubComponentType.PREFILL,
        )
        assert len(spec["roles"]) == 1
        assert spec["roles"][0]["name"] == "engine"
        assert spec["roles"][0]["replicas"] == 1
        container = spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert container["image"] == "lmsysorg/sglang:latest"
        assert "--disable-radix-cache" in container["args"]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"

    def test_generate_rbg_spec_decode(self):
        spec = SGLangConfigModifier.generate_rbg_spec(
            model="test-model", image="img:v1", num_gpus=4,
            phase=SubComponentType.DECODE,
        )
        container = spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert "--disable-radix-cache" not in container["args"]

    def test_generate_rbg_spec_decode_moe(self):
        spec = SGLangConfigModifier.generate_rbg_spec(
            model="test-model", image="img:v1", num_gpus=4,
            phase=SubComponentType.DECODE, is_moe=True,
        )
        container = spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert "--load-balance-method" in container["args"]
        idx = container["args"].index("--load-balance-method")
        assert container["args"][idx + 1] == "round_robin"

    def test_set_config_tp_size(self):
        spec = SGLangConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = SGLangConfigModifier.set_config_tp_size(spec, 4)
        args = SGLangConfigModifier.get_container_args(spec)
        assert "--tp" in args
        assert args[args.index("--tp") + 1] == "4"
        # GPU resource should be updated
        container = spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"

    def test_set_config_tep_size(self):
        spec = SGLangConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = SGLangConfigModifier.set_config_tep_size(spec, 4, num_gpus_per_node=8)
        args = SGLangConfigModifier.get_container_args(spec)
        assert args[args.index("--tp") + 1] == "4"
        assert args[args.index("--ep") + 1] == "4"
        assert "--enable-dp-attention" not in args

    def test_set_config_dep_size(self):
        spec = SGLangConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = SGLangConfigModifier.set_config_dep_size(spec, 4, num_gpus_per_node=8)
        args = SGLangConfigModifier.get_container_args(spec)
        assert args[args.index("--tp") + 1] == "4"
        assert args[args.index("--dp") + 1] == "4"
        assert args[args.index("--ep") + 1] == "4"
        assert "--enable-dp-attention" in args

    def test_set_prefill_config(self):
        spec = SGLangConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = SGLangConfigModifier.set_prefill_config(
            spec, max_batch_size=2, max_num_tokens=65536,
        )
        args = SGLangConfigModifier.get_container_args(spec)
        assert args[args.index("--max-running-requests") + 1] == "2"
        assert args[args.index("--chunked-prefill-size") + 1] == "65536"
        assert "--enable-dp-lm-head" in args

    def test_get_kv_cache_size_from_log(self):
        log = """
INFO 2025-01-01 Loading model...
INFO 2025-01-01 KV Cache is allocated with #tokens: 16384
INFO 2025-01-01 Server ready
"""
        size = SGLangConfigModifier.get_kv_cache_size_from_log(log, attention_dp_size=1)
        assert size == 16384

    def test_get_kv_cache_size_from_log_with_dp(self):
        log = "KV Cache is allocated with #tokens: 8192\n"
        size = SGLangConfigModifier.get_kv_cache_size_from_log(log, attention_dp_size=4)
        assert size == 32768

    def test_get_kv_cache_size_from_log_not_found(self):
        log = "INFO Starting server...\n"
        size = SGLangConfigModifier.get_kv_cache_size_from_log(log)
        assert size == 0


class TestVLLMConfigModifier:
    def test_generate_rbg_spec_prefill(self):
        spec = VLLMConfigModifier.generate_rbg_spec(
            model="Qwen/Qwen3-0.6B", image="vllm/vllm-openai:latest",
            num_gpus=2, phase=SubComponentType.PREFILL,
        )
        container = spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert "--no-enable-prefix-caching" in container["args"]
        assert "--enable-prefix-caching" not in container["args"]
        assert "vllm.entrypoints.openai.api_server" in container["command"][2]

    def test_generate_rbg_spec_decode(self):
        spec = VLLMConfigModifier.generate_rbg_spec(
            model="test", image="img:v1", num_gpus=1,
            phase=SubComponentType.DECODE,
        )
        container = spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert "--enable-prefix-caching" in container["args"]
        assert "--no-enable-prefix-caching" not in container["args"]

    def test_set_config_tp_size(self):
        spec = VLLMConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = VLLMConfigModifier.set_config_tp_size(spec, 8)
        args = VLLMConfigModifier.get_container_args(spec)
        assert args[args.index("--tensor-parallel-size") + 1] == "8"
        container = spec["roles"][0]["standalonePattern"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "8"

    def test_set_config_tep_size(self):
        spec = VLLMConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = VLLMConfigModifier.set_config_tep_size(spec, 4, num_gpus_per_node=8)
        args = VLLMConfigModifier.get_container_args(spec)
        assert args[args.index("--tensor-parallel-size") + 1] == "4"
        assert args[args.index("--data-parallel-size") + 1] == "1"
        assert "--enable-expert-parallel" in args

    def test_set_config_dep_size(self):
        spec = VLLMConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = VLLMConfigModifier.set_config_dep_size(spec, 4, num_gpus_per_node=8)
        args = VLLMConfigModifier.get_container_args(spec)
        assert args[args.index("--tensor-parallel-size") + 1] == "1"
        assert args[args.index("--data-parallel-size") + 1] == "4"
        assert "--enable-expert-parallel" in args

    def test_set_prefill_config(self):
        spec = VLLMConfigModifier.generate_rbg_spec(
            model="m", image="i", num_gpus=1,
        )
        spec = VLLMConfigModifier.set_prefill_config(
            spec, max_batch_size=2, max_num_tokens=65536,
        )
        args = VLLMConfigModifier.get_container_args(spec)
        assert args[args.index("--max-num-seqs") + 1] == "2"
        assert args[args.index("--max-num-batched-tokens") + 1] == "65536"

    def test_get_kv_cache_size_from_log(self):
        log = "Maximum concurrency for 4096 tokens per request: 8.0\n"
        size = VLLMConfigModifier.get_kv_cache_size_from_log(log, attention_dp_size=1)
        assert size == 32768

    def test_get_kv_cache_size_from_log_with_dp(self):
        log = "Maximum concurrency for 2048 tokens per request: 4.0\n"
        size = VLLMConfigModifier.get_kv_cache_size_from_log(log, attention_dp_size=2)
        assert size == 16384

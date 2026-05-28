"""Tests for parallelization mapping logic."""

import pytest

from inference_ext_cli.profile.model_info import ModelInfo
from inference_ext_cli.profile.parallelization_mapping import (
    ParallelizationMapping,
    get_candidate_parallel_mappings,
)


class TestParallelizationMapping:
    def test_tp_label(self):
        m = ParallelizationMapping(tp=4)
        assert m.label() == "TP=4"

    def test_tep_label(self):
        m = ParallelizationMapping(tep=8)
        assert m.label() == "TEP=8"

    def test_dep_label(self):
        m = ParallelizationMapping(dep=2)
        assert m.label() == "DEP=2"

    def test_tp_sizes(self):
        m = ParallelizationMapping(tp=4)
        assert m.get_tp_size() == 4
        assert m.get_expert_split() == 1
        assert m.get_attn_dp_size() == 1
        assert m.get_num_gpus() == 4

    def test_tep_sizes(self):
        m = ParallelizationMapping(tep=4)
        assert m.get_tp_size() == 4
        assert m.get_expert_split() == 4
        assert m.get_attn_dp_size() == 1
        assert m.get_num_gpus() == 4

    def test_dep_sizes(self):
        m = ParallelizationMapping(dep=4)
        assert m.get_tp_size() == 1
        assert m.get_expert_split() == 4
        assert m.get_attn_dp_size() == 4
        assert m.get_num_gpus() == 4


class TestGetCandidateParallelMappings:
    def test_dense_model_only_tp(self):
        model_info = ModelInfo(
            model_size=1000.0, architecture="LlamaForCausalLM",
            is_moe=False, num_kv_heads=8,
        )
        mappings = get_candidate_parallel_mappings(4, model_info, "prefill")
        assert len(mappings) == 1
        assert mappings[0].tp == 4

    def test_dense_model_kv_head_divisibility(self):
        model_info = ModelInfo(
            model_size=1000.0, architecture="LlamaForCausalLM",
            is_moe=False, num_kv_heads=4,
        )
        # 8 GPUs doesn't divide 4 KV heads
        mappings = get_candidate_parallel_mappings(8, model_info, "decode")
        assert len(mappings) == 0

    def test_moe_model_tep_and_dep(self):
        model_info = ModelInfo(
            model_size=5000.0, architecture="DeepseekV3ForCausalLM",
            is_moe=True, num_kv_heads=8, num_experts=64,
        )
        mappings = get_candidate_parallel_mappings(4, model_info, "decode")
        labels = [m.label() for m in mappings]
        assert "TEP=4" in labels
        assert "DEP=4" in labels
        # DeepseekV3 doesn't have additional TP
        assert "TP=4" not in labels

    def test_moe_qwen3_additional_tp(self):
        model_info = ModelInfo(
            model_size=3000.0, architecture="Qwen3MoeForCausalLM",
            is_moe=True, num_kv_heads=8, num_experts=64,
        )
        mappings = get_candidate_parallel_mappings(4, model_info, "prefill")
        labels = [m.label() for m in mappings]
        assert "TEP=4" in labels
        assert "DEP=4" in labels
        assert "TP=4" in labels

    def test_expert_divisibility_filter(self):
        model_info = ModelInfo(
            model_size=5000.0, architecture="DeepseekV3ForCausalLM",
            is_moe=True, num_kv_heads=8, num_experts=64,
        )
        # 3 GPUs doesn't divide 64 experts evenly for TEP/DEP
        mappings = get_candidate_parallel_mappings(3, model_info, "decode")
        # TEP: tp_size=3, 8 kv heads not divisible by 3 → filtered
        # DEP: expert_split=3, 64 not divisible by 3 → filtered
        assert len(mappings) == 0

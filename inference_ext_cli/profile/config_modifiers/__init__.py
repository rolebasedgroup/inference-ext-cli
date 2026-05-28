"""Engine-specific config modifiers for RBG profiling."""

from inference_ext_cli.profile.config_modifiers.sglang import SGLangConfigModifier
from inference_ext_cli.profile.config_modifiers.vllm import VLLMConfigModifier

CONFIG_MODIFIERS = {
    "sglang": SGLangConfigModifier,
    "vllm": VLLMConfigModifier,
}

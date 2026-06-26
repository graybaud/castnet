"""CastNet Model Infrastructure — HuggingFace + DeepSeek adapters."""

from infrastructure.models.huggingface import HuggingFaceWeightProvider
from infrastructure.models.config import (
    detect_architecture,
    is_deepseek_v2,
    get_deepseek_v2_config,
    get_ffn_pairs,
    get_attention_layers,
)
from infrastructure.models.deepseek import (
    load_deepseek_v2_expert_weights,
    DeepSeekV2ExpertWrapper,
)

__all__ = [
    "HuggingFaceWeightProvider",
    "detect_architecture",
    "is_deepseek_v2",
    "get_deepseek_v2_config",
    "get_ffn_pairs",
    "get_attention_layers",
    "load_deepseek_v2_expert_weights",
    "DeepSeekV2ExpertWrapper",
]

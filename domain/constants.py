"""Domain constants — Pure business rules, zero framework imports."""

# FFN layer name patterns (used by is_ffn_layer)
FFN_PATTERNS = [
    "fc1", "fc2", "c_fc", "c_proj",
    "gate_proj", "up_proj", "down_proj",
    "dense_h_to_4h", "dense_4h_to_h",
    "mlp.fc1", "mlp.fc2", "mlp.c_fc", "mlp.c_proj",
]


def is_ffn_layer(name: str) -> bool:
    """Check if a module name corresponds to an FFN layer."""
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in FFN_PATTERNS)

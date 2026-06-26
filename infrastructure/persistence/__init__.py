"""CastNet Persistence — Score/mask I/O adapters."""

from infrastructure.persistence.safetensors_persister import (
    SafetensorsScorePersister,
    SafetensorsMaskPersister,
)
from infrastructure.persistence.score_io import (
    infer_method_from_filename,
    load_scores,
)

__all__ = [
    "SafetensorsScorePersister",
    "SafetensorsMaskPersister",
    "infer_method_from_filename",
    "load_scores",
]

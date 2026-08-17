"""Public API for MultiFlow.

The package intentionally operates on paired latent states.  Dataset-specific
RNA/ATAC encoders and decoders remain explicit inputs to an analysis so their
scale, feature order, and training scope can be audited independently.
"""

from ._version import __version__
from .checkpoint import load_checkpoint, save_checkpoint
from .legacy import (
    LegacyCheckpointError,
    load_legacy_checkpoint,
    migrate_legacy_checkpoint,
)
from .models import (
    BidirectionalCrossAttention,
    CellStateFlow,
    ConditionalConcatFlow,
    PerturbationFlow,
    build_model,
)
from .normalization import LatentStandardizer
from .sampling import sample_paired_latents
from .training import TrainingConfig, TrainingResult, fit, seed_everything

# Concise public names.  The longer class names remain available so published
# checkpoints and ablations are unambiguous.
MultiFlow = CellStateFlow
MultiFlowPerturbation = PerturbationFlow
MultiFlowConcat = ConditionalConcatFlow

__all__ = [
    "__version__",
    "BidirectionalCrossAttention",
    "CellStateFlow",
    "ConditionalConcatFlow",
    "LatentStandardizer",
    "LegacyCheckpointError",
    "MultiFlow",
    "MultiFlowConcat",
    "MultiFlowPerturbation",
    "PerturbationFlow",
    "TrainingConfig",
    "TrainingResult",
    "build_model",
    "fit",
    "load_checkpoint",
    "load_legacy_checkpoint",
    "migrate_legacy_checkpoint",
    "sample_paired_latents",
    "save_checkpoint",
    "seed_everything",
]

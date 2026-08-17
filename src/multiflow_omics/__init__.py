"""Public API for MultiFlow."""

from ._version import __version__
from .checkpoint import load_checkpoint, save_checkpoint
from .h5mu import (
    read_paired_latents,
    validate_h5mu,
    write_generated_h5mu,
    write_toy_h5mu,
)
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
    "read_paired_latents",
    "sample_paired_latents",
    "save_checkpoint",
    "seed_everything",
    "validate_h5mu",
    "write_generated_h5mu",
    "write_toy_h5mu",
]

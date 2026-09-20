"""Signs of tokenization awareness in Qwen3-4B.

A layer-0 transcoder feature that fires on compound words whose
tokenization is unstable under a leading whitespace.

See README.md for the pipeline and docs/ for the write-up.
"""

from tokenization_awareness.config import (
    DEFAULT_COMPOUNDS_PATH,
    FEATURE_INDEX,
    LAYER,
    MODEL_NAME,
    TRANSCODER_REPO,
)
from tokenization_awareness.dataset import (
    CompoundGroups,
    build_groups,
    filter_closed_compounds,
    load_compounds,
    split_by_token_count,
)

__all__ = [
    "MODEL_NAME",
    "TRANSCODER_REPO",
    "LAYER",
    "FEATURE_INDEX",
    "DEFAULT_COMPOUNDS_PATH",
    "CompoundGroups",
    "build_groups",
    "filter_closed_compounds",
    "load_compounds",
    "split_by_token_count",
]

__version__ = "1.0.0"

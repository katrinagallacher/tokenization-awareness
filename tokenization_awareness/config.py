"""Constants for the experiment.

Everything that identifies *which* model, *which* transcoder and *which*
feature is under study lives here, so a re-run against a different feature
or layer is a one-line change.
"""

from __future__ import annotations

from pathlib import Path

# --- Model under study -------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B"

# Michael Hanna's transcoders for Qwen3-4B.
# https://huggingface.co/mwhanna/qwen3-4b-transcoders
TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"

# The layer whose MLP the transcoder decomposes, and the feature index within
# that transcoder's ~164k-wide sparse feature space.
LAYER = 0
FEATURE_INDEX = 118230

#: Neuronpedia page for the feature this repo is about.
FEATURE_URL = "https://www.neuronpedia.org/qwen3-4b/0-transcoder-hp/118230"

#: The attribution graph the feature was first spotted in.
CIRCUIT_TRACER_URL = (
    "https://www.neuronpedia.org/qwen3-4b/graph"
    "?slug=lovewasabattlefi-1760639948021&pruningThreshold=0.8&densityThreshold=0.99"
)

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_COMPOUNDS_PATH = PROJECT_ROOT / "data" / "compounds.txt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"

# --- Analysis ----------------------------------------------------------------

#: Group names, in the order they are reported and plotted throughout.
GROUP_NAMES = ("true_single", "conditional_single", "true_multi")

#: Human-readable labels for those groups.
GROUP_LABELS = {
    "true_single": "True single-token",
    "conditional_single": "Conditional single-token",
    "true_multi": "True multi-token",
}

#: Plot colours, matching the figures in the write-up.
GROUP_COLORS = {
    "true_single": "#FF1744",
    "conditional_single": "#FF6F00",
    "true_multi": "#00E5FF",
}

#: Language code passed to ``wordfreq.zipf_frequency``.
FREQUENCY_LANGUAGE = "en"

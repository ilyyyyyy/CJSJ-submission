"""Settings for the analysis."""

import numpy as np

DATA_DIR = "data"
OUTPUT_DIR = "output"

# Data files, one per question pair, plus the test-retest baseline
PAIR_FILES = {1: "pair1_ethical_integrity.csv",
              2: "pair2_consensus_deferral.csv",
              3: "pair3_foundations.csv"}
BASELINE_FILE = "baseline.csv"
HUMAN_MEANS_FILE = "human_group_means.csv"     # made by make_human_means.py

# Column headers in the data files
COLS = {"sample": "Passage", "version": "Genre", "position": "Position",
        "question": "Question Type", "rating": "Rating", "replicate": "Replicate"}

# Labels in the data files -> names used in the analysis
VERSION_MAP = {"original": "original", "fable": "fable"}
POSITION_MAP = {"unanchored": "unanchored", "anchored": "anchored"}
QUESTION_MAP = {"endorsement": "agreement", "ethical": "ethical", "consensus": "consensus",
                "verdict": "verdict", "care": "care", "fairness": "fairness",
                "loyalty": "loyalty", "authority": "authority", "purity": "purity"}

VERSIONS = ["original", "fable"]
FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "purity"]

# A rating has moved when two ratings differ by at least this many points.
MOVE_CUTOFF = 1.0
# The pooled ethical rating averages three replicates, so its cutoff is 1/sqrt(3).
POOLED_CUTOFF = MOVE_CUTOFF / np.sqrt(3)

# Figures
TEAL, BROWN = "#0F6B62", "#9C5B38"
JITTER = 0.06                 # small offset so tied ratings stay visible

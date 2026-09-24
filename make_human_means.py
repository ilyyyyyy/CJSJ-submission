"""Turn the raw human survey responses into the group means the analysis uses.

Run this once, on your own computer:

    python make_human_means.py

It reads the three human survey tabs, downloaded as .csv, from the folder
private_human_data/ (which .gitignore keeps out of the repository), and writes
data/human_group_means.csv. Only that file of means is published.
"""

import glob
import os

import pandas as pd

PRIVATE_DIR = "private_human_data"
OUT = os.path.join("data", "human_group_means.csv")

# Word in each downloaded filename -> question name used in the analysis
TABS = {"endorsement": "agreement", "ethical": "ethical", "consensus": "consensus"}


def find(word):
    hits = [f for f in glob.glob(os.path.join(PRIVATE_DIR, "*.csv")) if word in f.lower()]
    if len(hits) != 1:
        raise IOError(f"Expected one file containing '{word}' in {PRIVATE_DIR}, found {hits}.")
    return hits[0]


frames = []
for word, question in TABS.items():
    raw = pd.read_csv(find(word))
    frames.append(pd.DataFrame({
        "question": question,
        "sample": raw["Passage"].astype(str).str.extract(r"(\d+)")[0].astype(int),
        "version": raw["Genre"].astype(str).str.strip().str.lower(),
        "rating": pd.to_numeric(raw["Rating"], errors="coerce"),
    }))

means = (pd.concat(frames)
         .groupby(["question", "sample", "version"])
         .rating.agg(mean="mean", n="count").reset_index())
means["mean"] = means["mean"].round(4)
means.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {len(means)} rows (3 questions x 20 samples x 2 versions = 120 expected).")

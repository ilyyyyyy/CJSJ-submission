"""Reading the data files and writing outputs."""

import os

import pandas as pd

import config as cfg


def _read(name):
    return pd.read_csv(os.path.join(cfg.DATA_DIR, name))


def _label(series, mapping, what, name):
    s = series.astype(str).str.strip().str.lower()
    out = s.map(mapping)
    bad = sorted(set(s[out.isna()]))
    if bad:
        raise ValueError(f"{name}: unrecognised {what} {bad}. Add it to the map in config.py.")
    return out


def load_llm():
    """One row per model rating: pair, sample, version, question, position, rating."""
    frames = []
    for pair, name in cfg.PAIR_FILES.items():
        raw = _read(name)
        frames.append(pd.DataFrame({
            "pair": pair,
            "sample": raw[cfg.COLS["sample"]].astype(int),
            "version": _label(raw[cfg.COLS["version"]], cfg.VERSION_MAP, "version", name),
            "question": _label(raw[cfg.COLS["question"]], cfg.QUESTION_MAP, "question", name),
            "position": _label(raw[cfg.COLS["position"]], cfg.POSITION_MAP, "position", name),
            "rating": pd.to_numeric(raw[cfg.COLS["rating"]], errors="coerce"),
        }))
    return pd.concat(frames, ignore_index=True)


def load_baseline():
    """Test-retest baseline: sample, version, replicate (1 or 2), rating."""
    raw = _read(cfg.BASELINE_FILE)
    return pd.DataFrame({
        "sample": raw[cfg.COLS["sample"]].astype(int),
        "version": _label(raw[cfg.COLS["version"]], cfg.VERSION_MAP, "version", cfg.BASELINE_FILE),
        "replicate": pd.to_numeric(raw[cfg.COLS["replicate"]]).astype(int),
        "rating": pd.to_numeric(raw[cfg.COLS["rating"]], errors="coerce"),
    })


def load_human_means():
    """Human group means per question, sample and version, or None if absent."""
    path = os.path.join(cfg.DATA_DIR, cfg.HUMAN_MEANS_FILE)
    return pd.read_csv(path) if os.path.exists(path) else None


def ensure_dirs():
    for sub in ("figures", "tables"):
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, sub), exist_ok=True)


def _fmt(x):
    """Whole numbers without decimals, others to two places, missing as blank."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return "" if x != x else f"{round(float(x), 2) + 0.0:g}"   # + 0.0 turns -0 into 0
    return x


def save_table(df, name):
    path = os.path.join(cfg.OUTPUT_DIR, "tables", name + ".csv")
    df.map(_fmt).to_csv(path, index=False)
    print("  table :", path)


def load_human():
    """Same name as in the author's local copy, so run.py is identical in both."""
    return load_human_means()

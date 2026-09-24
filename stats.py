"""Every number the paper reports. No plotting, no file reading."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import config as cfg


def tau_b(x, y):
    """Kendall's tau-b, ignoring missing pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or len(set(x[ok])) < 2 or len(set(y[ok])) < 2:
        return np.nan
    return kendalltau(x[ok], y[ok]).statistic


def moved(a, b, cutoff=cfg.MOVE_CUTOFF):
    """Number of text samples on which two ratings differ by at least the cutoff."""
    d = np.abs(np.asarray(b, float) - np.asarray(a, float))
    return int((d[~np.isnan(d)] >= cutoff).sum())


def wide(llm, position):
    """One row per (pair, sample, version), one column per question.
    In pair 3 the unanchored verdict and unanchored foundations come from
    different conversations, which is what the position label records."""
    d = llm[llm.position == position]
    return (d.pivot_table(index=["pair", "sample", "version"], columns="question",
                          values="rating", aggfunc="first").reset_index())


def pair_data(w, pair, version):
    return w[(w.pair == pair) & (w.version == version)].set_index("sample")


def ethical_replicates(un):
    """The three unanchored ethical ratings, one per question pair, and their mean
    (the pooled ethical rating). Pair 3 asks the ethicality question as the verdict."""
    cols = {1: "ethical", 2: "ethical", 3: "verdict"}
    parts = [un[un.pair == p][["sample", "version", q]].rename(columns={q: f"pair{p}"})
             for p, q in cols.items()]
    m = parts[0].merge(parts[1], on=["sample", "version"]).merge(parts[2], on=["sample", "version"])
    m["pooled"] = m[["pair1", "pair2", "pair3"]].mean(axis=1)
    return m


def composite(df):
    """Mean of the five foundation ratings; missing if any foundation is missing."""
    f = df[cfg.FOUNDATIONS]
    return f.mean(axis=1).where(f.notna().all(axis=1))


def baseline(base):
    """Per version: the ceiling (tau-b between replicates) and how many samples
    moved under repetition alone."""
    w = base.pivot_table(index=["sample", "version"], columns="replicate",
                         values="rating", aggfunc="first").reset_index()
    out = {}
    for v in cfg.VERSIONS:
        d = w[w.version == v]
        out[v] = {"ceiling": tau_b(d[1], d[2]), "moved": moved(d[1], d[2]), "n": len(d)}
    return out


def anchoring(un, an):
    """Anchoring statistic A per text sample:
        A = (anchored Q - unanchored Q) / (unanchored P - unanchored Q)
    Pairs 1 and 2: both directions. Pair 3: each foundation anchored on the verdict.
    Returns one row per sample per direction; A is missing where Q and P began equal."""
    directions = [(1, "ethical", "agreement"), (1, "agreement", "ethical"),
                  (2, "ethical", "consensus"), (2, "consensus", "ethical")]
    directions += [(3, f, "verdict") for f in cfg.FOUNDATIONS]
    rows = []
    for pair, q, p in directions:
        for v in cfg.VERSIONS:
            u, a = pair_data(un, pair, v), pair_data(an, pair, v)
            for s in u.index.intersection(a.index):
                q0, q1, p0 = u.at[s, q], a.at[s, q], u.at[s, p]
                if pd.isna(q0) or pd.isna(q1) or pd.isna(p0):
                    continue
                gap = p0 - q0
                rows.append({"pair": pair, "question": q, "anchor": p, "version": v,
                             "sample": s, "change": q1 - q0,
                             "A": (q1 - q0) / gap if gap != 0 else np.nan})
    return pd.DataFrame(rows)


def pooled_closure(start, end, anchor):
    """Total distance closed toward the anchor over total distance available,
    summed across text samples before dividing. 0 = no movement toward the
    anchor overall; 1 = full arrival; negative = moved away overall."""
    before = (anchor - start).abs()
    after = (anchor - end).abs()
    ok = before.notna() & after.notna()
    return (before[ok] - after[ok]).sum() / before[ok].sum()


def pair3_order(un, an):
    """Third question pair under the order manipulation, per version:
    - verdict anchored on the composite: samples moved and pooled closure
      (the composite is a mean, so A's denominator can be fractional);
    - the five foundations anchored on the verdict: pooled closure, all five
      combined, so the two directions are measured the same way."""
    out = {}
    for v in cfg.VERSIONS:
        u, a = pair_data(un, 3, v), pair_data(an, 3, v)
        s = u.index.intersection(a.index)
        u, a = u.loc[s], a.loc[s]
        verdict_start, verdict_end = u["verdict"], a["verdict"]
        starts = pd.concat([u[f] for f in cfg.FOUNDATIONS])
        ends = pd.concat([a[f] for f in cfg.FOUNDATIONS])
        anchors = pd.concat([u["verdict"]] * len(cfg.FOUNDATIONS))
        out[v] = {
            "verdict moved": moved(verdict_start, verdict_end),
            "verdict pooled closure": pooled_closure(verdict_start, verdict_end, composite(u)),
            "foundations pooled closure": pooled_closure(starts, ends, anchors),
        }
    return out


def claude_gaps(un):
    """Gap between the two unanchored ratings of pairs 1 and 2, per sample."""
    g1 = un[un.pair == 1].assign(gap=lambda d: d.agreement - d.ethical)
    g2 = un[un.pair == 2].assign(gap=lambda d: d.ethical - d.consensus)
    return pd.concat([g1, g2])[["pair", "sample", "version", "gap"]]


def human_gaps(means):
    """Human ratings are between-subjects, so each gap is taken between group means.
    means: one row per question, sample and version, with a 'mean' column."""
    m = means.pivot_table(index=["sample", "version"], columns="question", values="mean")
    g1 = (m.agreement - m.ethical).rename("gap").reset_index().assign(pair=1)
    g2 = (m.ethical - m.consensus).rename("gap").reset_index().assign(pair=2)
    return pd.concat([g1, g2])[["pair", "sample", "version", "gap"]]

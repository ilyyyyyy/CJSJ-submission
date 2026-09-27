"""Produce every table and figure the paper reports.

    python run.py
"""

import numpy as np
import pandas as pd

import config as cfg
import dataio as io
import plots as pl
import stats as st

O, F = cfg.VERSIONS


def row(measure, values):
    return {"measure": measure, O: values[O], F: values[F]}


def by_version(fn):
    return {v: fn(v) for v in cfg.VERSIONS}


def table2(un, base):
    """Relationships between unanchored ratings, by version."""
    def tau(pair, a, b):
        return by_version(lambda v: st.tau_b(*st.pair_data(un, pair, v)[[a, b]].T.values))

    def integrity_signs(v):
        g = (lambda d: d.agreement - d.ethical)(st.pair_data(un, 1, v))
        return f"{(g < 0).sum()} / {(g == 0).sum()} / {(g > 0).sum()}"

    def consensus_matches(v):
        d = st.pair_data(un, 2, v)
        return int((d.ethical == d.consensus).sum())

    rows = [
        row("ceiling: test-retest tau-b", by_version(lambda v: base[v]["ceiling"])),
        row("agreement x ethical tau-b", tau(1, "agreement", "ethical")),
        row("ethical x consensus tau-b", tau(2, "ethical", "consensus")),
    ]
    rows += [row(f"{f} x verdict tau-b", tau(3, f, "verdict")) for f in cfg.FOUNDATIONS]
    rows += [
        row("agreement below / equal to / above ethical", by_version(integrity_signs)),
        row("ethical equal to consensus", by_version(consensus_matches)),
    ]
    return pd.DataFrame(rows)


def table3(un, an, base):
    """Order manipulation, one row per direction, five columns:
    moved, and median A with the number of samples it rests on in brackets.
    "Q anchored on P" means rating Q asked after question P. The first row is movement
    under repetition alone, the threshold every movement count is read against."""
    def median_n(d):
        n = int(d.A.notna().sum())
        return f"{d.A.median():.2f} ({n})" if n else "— (0)"

    A = st.anchoring(un, an)
    rows = [{"direction": "repetition only",
             **{f"{v}: {c}": (base[v]["moved"] if c == "moved" else "—")
                for v in cfg.VERSIONS for c in ("moved", "median A (n)")}}]
    for (pair, q, p), g in A.groupby(["pair", "question", "anchor"], sort=False):
        r = {"direction": f"{q} anchored on {p}"}
        for v in cfg.VERSIONS:
            d = g[g.version == v]
            r[f"{v}: moved"] = int((d.change.abs() >= cfg.MOVE_CUTOFF).sum())
            r[f"{v}: median A (n)"] = median_n(d)
        rows.append(r)

    # Verdict anchored on the composite: movement only; A is not computed for this
    # direction (fractional denominator), and its pooled closure is in Table 4.
    p3 = st.pair3_order(un, an)
    rows.append({"direction": "verdict anchored on composite",
                 **{f"{v}: {c}": (p3[v]["verdict moved"] if c == "moved" else "—")
                    for v in cfg.VERSIONS for c in ("moved", "median A (n)")}})
    return pd.DataFrame(rows)


def table4(un, an):
    """Third question pair order effect summary: pooled closure in both directions."""
    p3 = st.pair3_order(un, an)
    return pd.DataFrame([
        row("five foundations anchored on verdict: pooled closure",
            by_version(lambda v: p3[v]["foundations pooled closure"])),
        row("verdict anchored on composite: pooled closure",
            by_version(lambda v: p3[v]["verdict pooled closure"])),
    ])


def table5(un, reps):
    """Genre manipulation: text samples moved from original to fable, per question.
    Laid out as two side-by-side blocks to keep the table square."""
    def moved(pair, q):
        d = un[un.pair == pair].pivot_table(index="sample", columns="version", values=q)
        return st.moved(d[O], d[F])

    p = reps.pivot_table(index="sample", columns="version", values="pooled")
    counts = [("ethical (pooled; cutoff 0.577)", st.moved(p[O], p[F], cfg.POOLED_CUTOFF)),
              ("agreement", moved(1, "agreement")),
              ("consensus", moved(2, "consensus"))]
    counts += [(f, moved(3, f)) for f in cfg.FOUNDATIONS]
    half = (len(counts) + 1) // 2
    left, right = counts[:half], counts[half:] + [("", "")] * (2 * half - len(counts))
    return pd.DataFrame([{"question": a[0], "moved": a[1], "question ": b[0], "moved ": b[1]}
                         for a, b in zip(left, right)])


def human_positions(cg, hg):
    """Human comparison: per text sample, is Claude's gap below, above or on the
    human gap? Returns the counts and the merged gaps used for Figures 3 and 4."""
    m = cg.merge(hg, on=["pair", "sample", "version"], suffixes=("_claude", "_human"))
    rows = []
    for (pair, v), d in m.groupby(["pair", "version"]):
        diff = d.gap_claude - d.gap_human
        rows.append({"question pair": {1: "ethical integrity", 2: "consensus deferral"}[pair],
                     "version": v, "below": int((diff < 0).sum()),
                     "above": int((diff > 0).sum()), "on": int((diff == 0).sum())})
    return pd.DataFrame(rows), m


def replicate_agreement(reps, base):
    """Agreement among the three unanchored ethical ratings, one per question pair,
    against the test-retest baseline. Backs the Discussion's limitation."""
    rows = []
    for v in cfg.VERSIONS:
        d = reps[reps.version == v]
        for a, b in [("pair1", "pair2"), ("pair1", "pair3"), ("pair2", "pair3")]:
            rows.append({"version": v, "comparison": f"{a} vs {b}",
                         "tau_b": st.tau_b(d[a], d[b]), "ceiling": base[v]["ceiling"],
                         "moved": st.moved(d[a], d[b]), "baseline moved": base[v]["moved"]})
    return pd.DataFrame(rows)


def main():
    io.ensure_dirs()
    print("Loading...")
    llm, base_raw, hum = io.load_llm(), io.load_baseline(), io.load_human()
    un, an = st.wide(llm, "unanchored"), st.wide(llm, "anchored")
    base = st.baseline(base_raw)
    reps = st.ethical_replicates(un)

    print("Tables...")
    io.save_table(table2(un, base), "table2_relationships")
    io.save_table(table3(un, an, base), "table3_order")
    io.save_table(table4(un, an), "table4_pair3_order_summary")
    io.save_table(table5(un, reps), "table5_genre")
    positions, merged = human_positions(st.claude_gaps(un), st.human_gaps(hum))
    io.save_table(positions, "human_positions")
    io.save_table(replicate_agreement(reps, base), "discussion_ethical_replicates")

    print("Figures...")
    panels = []
    for v in cfg.VERSIONS:
        d = st.pair_data(un, 1, v)
        panels.append((v.capitalize(), d.ethical, d.agreement,
                       f"Correlation {st.tau_b(d.ethical, d.agreement):.2f}, "
                       f"ceiling {base[v]['ceiling']:.2f}"))
    pl.paired_scatter(panels, "Unanchored ethical rating",
                      "Unanchored agreement rating", "figure1_integrity")

    taus = {v: {f: st.tau_b(*st.pair_data(un, 3, v)[[f, "verdict"]].T.values)
                for f in cfg.FOUNDATIONS} for v in cfg.VERSIONS}
    pl.foundation_bars(taus, "figure2_foundation_profile")

    for pair, fig, label in [(1, 3, "integrity"), (2, 4, "consensus")]:
        panels = []
        for v in cfg.VERSIONS:
            d = merged[(merged.pair == pair) & (merged.version == v)]
            panels.append((v.capitalize(), d.gap_human, d.gap_claude, None))
        pl.paired_scatter(panels, "Human gap between group means", "Claude's gap between unanchored ratings",
                          f"figure{fig}_human_{label}", rating_scale=False)
    print("Done. Outputs are in", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()

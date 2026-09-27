"""The paper's figures, drawn at their printed size in Times New Roman, 8 pt.

Text width for the two-column template is about 7.0 in; one column is about 3.4 in.
Figures are drawn at the width they will occupy, so labels print at 8 pt without
being scaled down on insertion.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg

TEXT_WIDTH, COLUMN_WIDTH = 7.0, 3.4          # inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})


def _save(fig, name):
    path = os.path.join(cfg.OUTPUT_DIR, "figures", name + ".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figure:", path)


def _note(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", color="#444444",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.85))


def _clean(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok], y[ok]


def paired_scatter(panels, xlabel, ylabel, name, rating_scale=True):
    """Two scatter panels (original, fable) at single-column width, each
    with the equality line y = x. panels: [(title, x, y, note_or_None), ...].
    rating_scale fixes both axes to 1-6; otherwise both panels share limits
    fitted to the data (used for gaps)."""
    data = [(t, *_clean(x, y), note) for t, x, y, note in panels]
    if rating_scale:
        lo, hi = 0.5, 6.5
    else:
        allv = np.concatenate([np.concatenate([x, y]) for _, x, y, _ in data])
        lo, hi = allv.min() - 0.5, allv.max() + 0.5

    # One column wide. "stacked": original above fable, larger panels.
    # "side_by_side": both panels in a row, more compact but smaller.
    if cfg.FIGURE_LAYOUT == "stacked":
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 1.9), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.6), sharey=True)
    rng = np.random.default_rng(0)
    jit = lambda a: a + rng.uniform(-cfg.JITTER, cfg.JITTER, len(a))
    for ax, (title, x, y, note) in zip(axes, data):
        ax.scatter(jit(x), jit(y), s=18, alpha=0.75, color=cfg.TEAL,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=cfg.BROWN, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        if rating_scale:
            ax.set_xticks(range(1, 7)); ax.set_yticks(range(1, 7))
        ax.set_aspect("equal")
        ax.set_title(title)
        if cfg.FIGURE_LAYOUT != "stacked" or ax is axes[-1]:
            ax.set_xlabel(xlabel)
        if cfg.FIGURE_LAYOUT == "stacked":
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if note:
            _note(ax, note)
    if cfg.FIGURE_LAYOUT != "stacked":
        axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    _save(fig, name)


def foundation_bars(taus, name):
    """Correlation of each foundation with the verdict, original and fable side by
    side, one column wide. taus: {version: {foundation: tau}}"""
    xs, width = np.arange(len(cfg.FOUNDATIONS)), 0.38
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.7))
    for i, v in enumerate(cfg.VERSIONS):
        ax.bar(xs + (i - 0.5) * width, [taus[v][f] for f in cfg.FOUNDATIONS],
               width, label=v, color=[cfg.TEAL, cfg.BROWN][i], alpha=0.9)
    lowest = min(min(taus[v].values()) for v in cfg.VERSIONS)
    if lowest < 0:
        print(f"  warning: a foundation correlation is negative ({lowest:.2f}) "
              f"and falls below the 0-1 axis")
    ax.set_xticks(xs); ax.set_xticklabels(cfg.FOUNDATIONS)
    ax.set_ylabel("Correlation with verdict")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, name)

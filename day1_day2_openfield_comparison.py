#!/usr/bin/env python3
"""
Day 1 (20260421) vs Day 2 (20260422) open-field comparison.

Reads the two per-cohort summaries written by openfield_analysis.py and compares
the same mice across the two recording days on four measures:

    total_distance_cm         total distance (cm)
    pct_time_center           % time in center
    pct_time_immobile         % time immobile
    pct_time_wall_following   % time wall-following

Page 1 (whole session): each metric is one panel -- day 1 vs day 2 group mean
+/- SEM, WT and Df(h16p12)/+ as separate bars, with every animal as a point and
a per-genotype paired t-test p-value.

Time-course PDFs (SEPARATE files, one per width in BIN_MINUTES, e.g.
<out>_10min.pdf and <out>_5min.pdf): the same metrics chopped into intervals,
laid out as a Day 1 block and a Day 2 block (rows = metric, x = interval), each
panel the WT / Df(h16p12)/+ grouped-bar plot for that day. Per-frame data for
the bins is read from the per_animal/*_track.csv files next to each summary.

Inputs default to (both under .../202604_moseq-pilot):
    day 1: 20260421_openfield_out/summary_metrics_openfield.csv
    day 2: 20260422_openfield_out/summary_metrics_openfield.csv
Both must have been produced by the CURRENT openfield_analysis.py (older runs
lack the immobile / wall-following columns); if a column is missing the script
says so and tells you to re-run the analysis.

Usage:
    /opt/miniconda3/envs/DEEPLABCUT/bin/python day1_day2_openfield_comparison.py
    ... day1_day2_openfield_comparison.py --day1 <csv> --day2 <csv> -o out.pdf
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DATA_ROOT = "/Volumes/dorothea_1T/202604_moseq-pilot"
DAY1_CSV = os.path.join(DATA_ROOT, "20260421_openfield_out", "summary_metrics_openfield.csv")
DAY2_CSV = os.path.join(DATA_ROOT, "20260422_openfield_out", "summary_metrics_openfield.csv")
OUT_PDF = os.path.join(DATA_ROOT, "day1_day2_openfield_comparison.pdf")

GROUP_COLORS = {"WT": "#4477AA", "Df(h16p12)/+": "#EE6677", "?": "#999999"}

# For the time-binned page: recording rate (must match openfield_analysis.py) and
# the interval width. Per-frame data comes from the per_animal/*_track.csv files.
FPS = 30.0
BIN_MINUTES = (10.0, 5.0)     # one separate binned PDF is written per interval width
LINE_BIN_MINUTES = (5.0,)     # these widths also get a line-graph version

METRICS = [
    ("total_distance_cm", "total distance (cm)"),
    ("pct_time_center", "% time in center"),
    ("pct_time_immobile", "% time immobile"),
    ("pct_time_wall_following", "% time wall-following"),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_summary(path: str, day: str) -> pd.DataFrame:
    """Load one cohort summary, keep QC-passed animals, index by mouse id."""
    if not os.path.exists(path):
        sys.exit(f"{day}: summary not found: {path}\n"
                 f"Run openfield_analysis.py for that cohort first.")
    df = pd.read_csv(path)
    need = ["mouse", "genotype"] + [c for c, _ in METRICS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"{day}: {os.path.basename(path)} is missing column(s): "
                 f"{', '.join(missing)}\nRe-run the current openfield_analysis.py "
                 f"on that cohort so these metrics are computed.")
    if "qc_pass" in df.columns:
        keep = df["qc_pass"].astype(str).str.lower().isin(["true", "1"])
        df = df[keep]
    df["mouse"] = df["mouse"].astype(str)
    return df.set_index("mouse")


def pair_days(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    """Return one row per mouse present (and QC-passed) on BOTH days."""
    common = sorted(set(d1.index) & set(d2.index), key=lambda s: (len(s), s))
    if not common:
        sys.exit("No mouse ids are shared between the two days (after QC).")
    rows = []
    for mid in common:
        r = {"mouse": mid,
             "genotype": d1.loc[mid, "genotype"] if d1.loc[mid, "genotype"] != "?"
             else d2.loc[mid, "genotype"]}
        for col, _ in METRICS:
            r[f"{col}__d1"] = float(d1.loc[mid, col])
            r[f"{col}__d2"] = float(d2.loc[mid, col])
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _mean_sem(vals: np.ndarray) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    sem = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan
    return float(vals.mean()), sem


def _paired_p(v1: np.ndarray, v2: np.ndarray) -> float | None:
    """Paired t-test p-value on animals finite on both days, or None."""
    ok = np.isfinite(v1) & np.isfinite(v2)
    if ok.sum() < 2:
        return None
    try:
        from scipy.stats import ttest_rel
    except Exception:
        return None
    return float(ttest_rel(v1[ok], v2[ok]).pvalue)


def _genotype_order(genos: np.ndarray) -> list:
    ordered = [g for g in ("WT", "Df(h16p12)/+") if g in set(genos)]
    return ordered or sorted(set(genos))


def _draw_panel(ax, v1, v2, genos, ylabel_text=None, title_text=None, show_p=True):
    """day 1 vs day 2 grouped bars (WT / Df(h16p12)/+) with SEM and animal points.

    v1, v2 are per-animal metric values on day 1 / day 2; genos the matching
    genotype labels. Reused for the whole-session page and each 10-min bin.
    """
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    genos = np.asarray(genos)
    day_vals = {0: v1, 1: v2}
    geno_order = _genotype_order(genos)
    # dodge > bar_w so the two genotype bars are fully separated (no overlap)
    dodge = 0.36          # WT vs Df(h16p12)/+ separation within a day
    bar_w = 0.30

    def geno_x(gi: int) -> float:
        return (gi - (len(geno_order) - 1) / 2.0) * dodge

    # one WT bar and one Df(h16p12)/+ bar per day (mean +/- SEM)
    for di in (0, 1):
        for gi, g in enumerate(geno_order):
            vals = day_vals[di][genos == g]
            mean, sem = _mean_sem(vals)
            x = di + geno_x(gi)
            ax.bar(x, mean, width=bar_w, color=GROUP_COLORS.get(g, "#999999"),
                   edgecolor="k", alpha=0.55, zorder=1)
            if np.isfinite(sem):
                ax.errorbar(x, mean, yerr=sem, color="k", capsize=4, lw=1.3, zorder=2)

    # per-animal x jitter within its genotype's bar (deterministic)
    xoff = np.zeros(len(genos))
    for gi, g in enumerate(geno_order):
        idxs = [i for i in range(len(genos)) if genos[i] == g]
        for j, i in enumerate(idxs):
            xoff[i] = geno_x(gi) + (j - (len(idxs) - 1) / 2.0) * 0.06

    # individual animals: one point per day (no connecting line)
    for i in range(len(genos)):
        ax.scatter([0 + xoff[i], 1 + xoff[i]], [v1[i], v2[i]], color="k", s=22, zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["day 1", "day 2"])
    if ylabel_text:
        ax.set_ylabel(ylabel_text)
    if title_text:
        ax.set_title(title_text, fontsize=10)

    if show_p:                       # paired t-test (day 1 vs day 2) within genotype
        ytxt = 0.98
        for g in geno_order:
            mask = genos == g
            p = _paired_p(v1[mask], v2[mask])
            if p is not None:
                ax.text(0.5, ytxt, f"{g}: paired t p = {p:.3f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=8, color=GROUP_COLORS.get(g, "#333"))
                ytxt -= 0.06


def metric_panel(ax, paired: pd.DataFrame, col: str, label: str):
    _draw_panel(ax, paired[f"{col}__d1"].to_numpy(float),
                paired[f"{col}__d2"].to_numpy(float),
                paired["genotype"].to_numpy(),
                ylabel_text=label, title_text=label, show_p=True)


def _genotype_legend(fig, genos):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GROUP_COLORS.get(g, "#999999"), alpha=0.55,
                     edgecolor="k", label=g) for g in _genotype_order(genos)]
    handles.append(plt.Line2D([0], [0], marker="o", ls="", color="k", label="animal"))
    # Anchor to the right of the figure (vertically centered), clear of the
    # top suptitle; bbox_inches="tight" on save keeps it in frame.
    fig.legend(handles=handles, title="genotype", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)


# --------------------------------------------------------------------------- #
# Time-binned data (per-frame, from the per_animal/*_track.csv files)
# --------------------------------------------------------------------------- #
def _track_dir(summary_csv: str) -> str:
    return os.path.join(os.path.dirname(summary_csv), "per_animal")


def _find_track(pdir: str, mouse: str):
    hits = sorted(glob.glob(os.path.join(pdir, f"*_m{mouse}_track.csv")))
    return hits[0] if hits else None


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy() if name in df.columns else np.full(len(df), np.nan)


def _binned_metrics(df: pd.DataFrame, n_bins: int, bin_min: float) -> dict:
    """Compute the four metrics within each bin_min-minute interval of one track."""
    fpb = int(round(bin_min * 60 * FPS))
    binof = np.arange(len(df)) // fpb
    step = _col(df, "step_cm")
    valid = np.isfinite(_col(df, "r_px"))
    zone = _col(df, "zone").astype(str)
    wall = _col(df, "wall_follow")
    state = _col(df, "state").astype(str)
    out = {}
    for b in range(n_bins):
        mb = binof == b
        vb = mb & valid
        wb = mb & np.isfinite(wall)
        sb = mb & (state != "nan") & (state != "")     # frames with a valid state
        out[b] = {
            "total_distance_cm": float(np.nansum(step[mb])) if mb.any() else np.nan,
            "pct_time_center": 100.0 * np.mean(zone[vb] == "center") if vb.any() else np.nan,
            "pct_time_wall_following": 100.0 * np.nanmean(wall[wb]) if wb.any() else np.nan,
            "pct_time_immobile": 100.0 * np.mean(state[sb] == "immobile") if sb.any() else np.nan,
        }
    return out


def load_tracks(paired: pd.DataFrame, day1_csv: str, day2_csv: str):
    """Load per-animal track CSVs for both days; return {mouse: df} pairs + mice."""
    d1_dir, d2_dir = _track_dir(day1_csv), _track_dir(day2_csv)
    t1, t2, missing = {}, {}, []
    for mid in paired["mouse"].astype(str):
        p1, p2 = _find_track(d1_dir, mid), _find_track(d2_dir, mid)
        if p1 and p2:
            t1[mid] = pd.read_csv(p1, index_col=0)
            t2[mid] = pd.read_csv(p2, index_col=0)
        else:
            missing.append(mid)
    if missing:
        print(f"[warn] no per_animal track csv for mice {missing} -- excluded from "
              f"the binned pages", file=sys.stderr)
    mice = [m for m in paired["mouse"].astype(str) if m in t1 and m in t2]
    return t1, t2, mice


def compute_binned(paired: pd.DataFrame, t1: dict, t2: dict, mice: list, bin_min: float):
    """Return {(metric, bin): (v1, v2, genos)} and n_bins for a given bin width."""
    if not mice:
        return None, 0
    geno_by = dict(zip(paired["mouse"].astype(str), paired["genotype"]))
    fpb = bin_min * 60 * FPS
    n_bins = max(int(np.ceil(len(t[m]) / fpb)) for t in (t1, t2) for m in mice)
    b1 = {m: _binned_metrics(t1[m], n_bins, bin_min) for m in mice}
    b2 = {m: _binned_metrics(t2[m], n_bins, bin_min) for m in mice}
    genos = np.array([geno_by[m] for m in mice])
    binned = {}
    for col, _ in METRICS:
        for b in range(n_bins):
            binned[(col, b)] = (np.array([b1[m][b][col] for m in mice], float),
                                np.array([b2[m][b][col] for m in mice], float),
                                genos)
    return binned, n_bins


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def whole_session_page(pdf, paired: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (col, label) in zip(axes.flat, METRICS):
        metric_panel(ax, paired, col, label)
    fig.suptitle(f"Open-field, whole session: day 1 (20260421) vs day 2 (20260422)   "
                 f"n = {len(paired)} paired mice")
    fig.tight_layout()
    _genotype_legend(fig, paired["genotype"].to_numpy())
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _grouped_bars(ax, groups_vals, genos, xtick_labels, ylabel=None, title=None):
    """Grouped WT / Df(h16p12)/+ bars (SEM + points) at each x-group.

    groups_vals is a list (one per x-group, e.g. one per 10-min interval) of
    per-animal value arrays; genos gives each animal's genotype.
    """
    genos = np.asarray(genos)
    geno_order = _genotype_order(genos)
    dodge, bar_w = 0.36, 0.30

    def geno_x(gi):
        return (gi - (len(geno_order) - 1) / 2.0) * dodge

    for xi, vals_all in enumerate(groups_vals):
        vals_all = np.asarray(vals_all, float)
        for gi, g in enumerate(geno_order):
            mean, sem = _mean_sem(vals_all[genos == g])
            x = xi + geno_x(gi)
            ax.bar(x, mean, width=bar_w, color=GROUP_COLORS.get(g, "#999999"),
                   edgecolor="k", alpha=0.55, zorder=1)
            if np.isfinite(sem):
                ax.errorbar(x, mean, yerr=sem, color="k", capsize=4, lw=1.3, zorder=2)
        # individual animals, jittered within their genotype bar
        for gi, g in enumerate(geno_order):
            idxs = [i for i in range(len(genos)) if genos[i] == g]
            for j, i in enumerate(idxs):
                xo = geno_x(gi) + (j - (len(idxs) - 1) / 2.0) * 0.06
                ax.scatter([xi + xo], [vals_all[i]], color="k", s=18, zorder=4)

    ax.set_xticks(range(len(groups_vals)))
    if len(groups_vals) > 3:                     # avoid crowding with 5-min bins
        ax.set_xticklabels(xtick_labels, rotation=40, ha="right", fontsize=8)
    else:
        ax.set_xticklabels(xtick_labels, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)


def _grouped_lines(ax, groups_vals, genos, xtick_labels, ylabel=None, title=None):
    """One line per genotype across x-groups: mean +/- SEM markers + animal dots.

    Same inputs/positions as _grouped_bars, but drawn as connected lines instead
    of bars. Genotypes are nudged apart slightly so points don't overlap.
    """
    genos = np.asarray(genos)
    geno_order = _genotype_order(genos)
    dodge = 0.08
    xbase = np.arange(len(groups_vals))
    for gi, g in enumerate(geno_order):
        off = (gi - (len(geno_order) - 1) / 2.0) * dodge
        x = xbase + off
        color = GROUP_COLORS.get(g, "#999999")
        means, sems = [], []
        for xi, vals_all in enumerate(groups_vals):
            vals_all = np.asarray(vals_all, float)
            gv = vals_all[genos == g]
            mean, sem = _mean_sem(gv)
            means.append(mean)
            sems.append(sem if np.isfinite(sem) else 0.0)
            # individual animals at this interval (faint, genotype-colored)
            for j, val in enumerate(gv):
                jx = x[xi] + (j - (len(gv) - 1) / 2.0) * 0.03
                ax.scatter([jx], [val], color=color, s=14, alpha=0.45, zorder=2)
        ax.errorbar(x, means, yerr=sems, color=color, marker="o", capsize=3,
                    lw=1.8, zorder=3, label=g)

    ax.set_xticks(xbase)
    if len(groups_vals) > 3:
        ax.set_xticklabels(xtick_labels, rotation=40, ha="right", fontsize=8)
    else:
        ax.set_xticklabels(xtick_labels, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)


def binned_page(pdf, binned: dict, n_bins: int, mice: list, bin_min: float,
                draw_fn=_grouped_bars, kind: str = "bars", share_y: bool = True):
    """Day 1 block and Day 2 block side by side; within each, metric vs interval.

    Rows = metric, columns = day block. Each panel's x-axis is the bin_min-minute
    intervals, with WT / Df(h16p12)/+ bars (SEM + animal points). With share_y,
    the Day 1 and Day 2 panels of a metric share one y-axis so the two days are
    read off the same scale.
    """
    genos = binned[(METRICS[0][0], 0)][2]
    days = [("Day 1", 0), ("Day 2", 1)]
    xlabels = [f"{int(b * bin_min)}–{int((b + 1) * bin_min)}" for b in range(n_bins)]

    panel_w = 2.0 + 0.9 * n_bins                 # wider when there are more bins
    fig, axes = plt.subplots(len(METRICS), len(days),
                             figsize=(panel_w * len(days), 3.4 * len(METRICS)),
                             squeeze=False, gridspec_kw={"wspace": 0.28})
    for r, (col, label) in enumerate(METRICS):
        for c, (dname, didx) in enumerate(days):
            groups = [binned[(col, b)][didx] for b in range(n_bins)]
            draw_fn(axes[r][c], groups, genos, xlabels,
                    ylabel=(label if c == 0 else None),
                    title=(dname if r == 0 else None))
    if share_y:            # same y-scale for Day 1 and Day 2 within each metric
        for r in range(len(METRICS)):
            lims = [ax.get_ylim() for ax in axes[r]]
            lo, hi = min(l for l, _ in lims), max(h for _, h in lims)
            for ax in axes[r]:
                ax.set_ylim(lo, hi)

    for c in range(len(days)):
        axes[-1][c].set_xlabel("interval (min)")

    fig.suptitle(f"Open-field by {int(bin_min)}-min interval, {kind}  "
                 f"(n = {len(mice)} mice)")
    fig.tight_layout()
    _genotype_legend(fig, genos)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(paired: pd.DataFrame, out_pdf: str):
    """Whole-session comparison (page 1) -> its own PDF."""
    with PdfPages(out_pdf) as pdf:
        whole_session_page(pdf, paired)


def build_binned_pdf(binned: dict, n_bins: int, mice: list, out_pdf: str,
                     bin_min: float, draw_fn=_grouped_bars, kind: str = "bars",
                     share_y: bool = True):
    """Time-binned comparison -> a separate PDF (bars or lines)."""
    with PdfPages(out_pdf) as pdf:
        binned_page(pdf, binned, n_bins, mice, bin_min, draw_fn=draw_fn, kind=kind,
                    share_y=share_y)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day1", default=DAY1_CSV, help="day-1 summary csv")
    p.add_argument("--day2", default=DAY2_CSV, help="day-2 summary csv")
    p.add_argument("-o", "--out", default=OUT_PDF, help="whole-session output pdf")
    args = p.parse_args()

    d1 = load_summary(args.day1, "day 1")
    d2 = load_summary(args.day2, "day 2")
    paired = pair_days(d1, d2)

    base = os.path.splitext(args.out)[0]
    csv_out = base + "_data.csv"
    paired.to_csv(csv_out, index=False)
    build_pdf(paired, args.out)

    print(f"{len(paired)} paired mice: {', '.join(paired['mouse'])}")
    print(f"Wrote {args.out}\n      {csv_out}")

    # one separate binned PDF per interval width (e.g. 10-min and 5-min); widths
    # in LINE_BIN_MINUTES additionally get a line-graph version.
    t1, t2, mice = load_tracks(paired, args.day1, args.day2)
    for bin_min in BIN_MINUTES:
        binned, n_bins = compute_binned(paired, t1, t2, mice, bin_min)
        if not binned:
            continue
        out = f"{base}_{int(bin_min)}min.pdf"
        build_binned_pdf(binned, n_bins, mice, out, bin_min)
        print(f"      {out}  ({n_bins} x {int(bin_min)}-min intervals, bars)")
        if bin_min in LINE_BIN_MINUTES:
            out_line = f"{base}_{int(bin_min)}min_line.pdf"
            build_binned_pdf(binned, n_bins, mice, out_line, bin_min,
                             draw_fn=_grouped_lines, kind="lines")
            print(f"      {out_line}  ({n_bins} x {int(bin_min)}-min intervals, lines)")


if __name__ == "__main__":
    main()

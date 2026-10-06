#!/usr/bin/env python3
"""
Open-field (round bucket) analysis of DeepLabCut torso tracking.

For every 20260421_m3* folder under DATA_ROOT this script picks the CSV from the
most recent shuffle, calibrates pixels to cm from a hand-measured bucket
diameter, and computes distance travelled, center/periphery occupancy, and a
per-frame behavioral state (immobile / moving-center / moving-periphery) from
the torso body part.

Coordinates stay in the raw video pixel frame throughout; the bucket geometry is
applied per video (arena center, radius, scale) and cm are used only for
distances and speeds. No cross-animal common reference frame is constructed.

Outputs (into OUT_DIR):
  summary_metrics_openfield.csv    one row per animal, all metrics
  per_animal/<mouse>_track.csv     per-frame cleaned torso track + state
  openfield_figures.pdf            per animal: trajectory (+ % time caption),
                                   occupancy heatmap, behavior ethogram

Arena geometry comes from a SEPARATE csv (never edit the DLC csv -- it is
overwritten every time you re-analyze). One row per video:

    video_id,hx1,hy1,hx2,hy2,vx1,vy1,vx2,vy2
    20260421_m367,102.0,455.0,610.0,455.0,356.0,201.0,356.0,709.0

  h* = the two endpoints of the HORIZONTAL line drawn across the bucket in ImageJ
  v* = the two endpoints of the VERTICAL line
  video_id = the animal folder name (or the mouse id, e.g. "m367" / "367")

Generate a pre-filled template to edit:
    python openfield_analysis.py --make-template

Usage (both cohorts live under DATA_ROOT = .../202604_moseq-pilot):
    # day 1 (20260421) -> 20260421_openfield_out (defaults):
    /opt/miniconda3/envs/DEEPLABCUT/bin/python openfield_analysis.py
    # day 2 (20260422) -> 20260422_openfield_out:
    ... openfield_analysis.py --pattern '20260422_m*' \
            -o /Volumes/dorothea_1T/202604_moseq-pilot/20260422_openfield_out
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Both cohorts' animal folders (20260421_m* and 20260422_m*) live under this
# parent. The default run does the 20260421 (day-1) cohort; pass --pattern and
# -o for the 20260422 (day-2) cohort (see the commands in the header).
DATA_ROOT = "/Volumes/dorothea_1T/202604_moseq-pilot"
OUT_DIR = os.path.join(DATA_ROOT, "20260421_openfield_out")
ARENA_CSV = "/Volumes/dorothea_1T/arena_coords.csv"     # shared by both cohorts

# Which animal folders to analyze, and which DLC output within them. The csv
# pattern pins the Jul16 training set and the snapshot_best exports, so older
# runs are never picked up even with --recursive.
FOLDER_PATTERN = "20260421_m*"
CSV_PATTERN = "depthDLC_Resnet50_20260421-m3*Jul16shuffle*_snapshot_best-*.csv"

CM_PER_IN = 2.54
BUCKET_DIAMETER_IN = 17.0                       # real bucket diameter, inches
BUCKET_DIAMETER_CM = BUCKET_DIAMETER_IN * CM_PER_IN   # all distances reported in cm
FPS = 30.0                  # Kinect v2 depth camera
PCUTOFF = 0.7               # DLC likelihood threshold (matches config.yaml)
SMOOTH_WIN = 5              # rolling-median window, frames
BODYPART = "torso"          # body part used for locomotion and zone occupancy

# Gaps of low-confidence frames longer than this are left as NaN instead of
# being interpolated across. Without a cap, a video the model failed on gets
# "filled in" into a smooth, nearly-stationary track that looks like a real
# (immobile) animal rather than like missing data.
MAX_GAP_FRAMES = 15         # 0.5 s at 30 fps

# QC gate. An animal above this fraction of sub-threshold frames is reported
# but marked qc_pass = False and excluded from the group figure.
MAX_LOWCONF_FRAC = 0.5

# Center zone = concentric circle with 50% of the ARENA RADIUS, matching
# CENTER_FRAC in dlc_limb_analysis.py. Note this covers 25% of the floor area,
# which is the chance level the occupancy index is measured against.
CENTER_FRAC = 0.5

# Displacements below this (in cm) are treated as tracking jitter, not
# locomotion, and excluded from total distance. ~1 mm at 30 fps.
MIN_STEP_CM = 0.1

# --- behavior classification (per frame) ---------------------------------- #
# Each valid frame is assigned a behavioral state by torso speed (cm/s):
#   immobile : speed <= IMMOBILE_MAX_CM_S                       (resting)
#   walking  : WALK_MIN_CM_S <= speed < RUN_MIN_CM_S            (slow locomotion)
#   running  : speed >= RUN_MIN_CM_S                            (fast locomotion)
# The (8, 9) gap is closed: walking now runs right up to the running threshold,
# so every speed >= WALK_MIN is classified. A small (2, 3) gap remains by design;
# in practice the MIN_STEP_CM jitter floor makes the lowest non-zero speed 3 cm/s,
# so almost no frame lands there. Frames without valid tracking stay unclassified.
IMMOBILE_MAX_CM_S = 2.0     # immobile if speed <= this
WALK_MIN_CM_S = 3.0         # walking if WALK_MIN <= speed < RUN_MIN
RUN_MIN_CM_S = 9.0          # running if speed >= this
# Behavioral runs shorter than this are absorbed into the neighbouring state, so
# single-frame speed flicker around a threshold does not fragment the ethogram
# or inflate the bout counts. 5 frames = 1/6 s at 30 fps.
MIN_BOUT_FRAMES = 5

# Ethogram row order (bottom to top), per-frame integer code, labels and colors.
# Colors are fixed per behavior (not per genotype) so the ethogram reads by
# state; the palette runs cool->warm with increasing speed.
STATES = ["immobile", "walking", "running"]
STATE_CODE = {"immobile": 0, "walking": 1, "running": 2}
STATE_LABELS = {"immobile": "immobile", "walking": "walking", "running": "running"}
STATE_COLORS = {"immobile": "#BBBBBB", "walking": "#4C9F70", "running": "#E15759"}

# The ethogram is drawn as two stacked halves (first 15 min above the last),
# so individual bouts stay legible instead of merging when zoomed out.
ETHOGRAM_SPLIT_MIN = 15.0

# Wall-following (thigmotaxis): torso in the outer band of the arena, i.e. radial
# position >= this fraction of the arena radius. Summarised in WALL_BIN_MIN bins.
WALL_FOLLOW_FRAC = 0.8
WALL_BIN_MIN = 5.0

# Body length = distance between these two tracked points, per frame (cm).
BODYLEN_PART_A = "nose"
BODYLEN_PART_B = "tail"

GENOTYPE = {
    "367": "Df(h16p12)/+", "375": "Df(h16p12)/+", "390": "Df(h16p12)/+",
    "368": "WT", "376": "WT", "389": "WT",
}
GROUP_COLORS = {"WT": "#4477AA", "Df(h16p12)/+": "#EE6677", "?": "#999999"}


# --------------------------------------------------------------------------- #
# (1) Locate the most recent shuffle's CSV in each animal folder
# --------------------------------------------------------------------------- #
def _shuffle_rank(path: str) -> tuple:
    """Sort key that puts the newest DLC run last.

    Ranks by shuffle number first, then snapshot number, then mtime, so
    shuffle3_snapshot_best-200 beats shuffle2_snapshot_best-50.
    """
    name = os.path.basename(path)
    m = re.search(r"shuffle(\d+)", name)
    shuffle = int(m.group(1)) if m else -1
    m = re.search(r"snapshot[-_](?:best[-_])?(\d+)", name)
    snapshot = int(m.group(1)) if m else -1
    return (shuffle, snapshot, os.path.getmtime(path))


def find_animals(root: str, pattern: str = FOLDER_PATTERN,
                 csv_pattern: str = CSV_PATTERN,
                 recursive: bool = False) -> dict[str, str]:
    """Return {folder_name: most_recent_shuffle_csv} for each matching folder.

    Only the top level of each animal folder is searched by default; the
    prelim-analysis subfolders hold CSVs from a different (older) model and
    would otherwise be mixed in.
    """
    animals: dict[str, str] = {}
    for folder in sorted(glob.glob(os.path.join(root, pattern))):
        if not os.path.isdir(folder):
            continue
        csv_glob = f"**/{csv_pattern}" if recursive else csv_pattern
        csvs = glob.glob(os.path.join(folder, csv_glob), recursive=recursive)
        csvs = [c for c in csvs if "_full" not in c and "_meta" not in c]
        if not csvs:
            print(f"  [skip] no DLC csv in {os.path.basename(folder)}", file=sys.stderr)
            continue
        animals[os.path.basename(folder)] = max(csvs, key=_shuffle_rank)
    return animals


def mouse_id(folder_name: str) -> str:
    m = re.search(r"_m(\d+)", folder_name)
    return m.group(1) if m else folder_name


# --------------------------------------------------------------------------- #
# Arena geometry: center, radius, and the pixel -> cm scale
# --------------------------------------------------------------------------- #
class Arena:
    """Bucket geometry for one video, derived from two ImageJ diameter lines."""

    def __init__(self, row: pd.Series):
        h1 = np.array([row["hx1"], row["hy1"]], float)
        h2 = np.array([row["hx2"], row["hy2"]], float)
        v1 = np.array([row["vx1"], row["vy1"]], float)
        v2 = np.array([row["vx2"], row["vy2"]], float)

        # Center = mean of the four endpoints. For two true diameters this is
        # the intersection point, and averaging is robust to sloppy clicking.
        self.cx, self.cy = np.mean([h1, h2, v1, v2], axis=0)

        self.d_h_px = float(np.hypot(*(h2 - h1)))
        self.d_v_px = float(np.hypot(*(v2 - v1)))
        self.diameter_px = (self.d_h_px + self.d_v_px) / 2.0
        self.radius_px = self.diameter_px / 2.0

        # QC: how much the two hand-drawn diameters disagree, as % of the mean.
        # A perfect circle measured well is < ~2%. Large values mean a mis-drawn
        # line or genuine camera-angle distortion, and the scale is unreliable.
        self.asymmetry_pct = abs(self.d_h_px - self.d_v_px) / self.diameter_px * 100

        # Circumference, for cross-checking against an ImageJ oval measurement.
        self.circumference_px = np.pi * self.diameter_px
        self.circumference_cm = np.pi * BUCKET_DIAMETER_CM

        # Scale. Note the pi cancels, so the diameter and circumference routes
        # give the identical number; both are kept for transparency.
        self.cm_per_px = BUCKET_DIAMETER_CM / self.diameter_px
        self.cm_per_px_via_circumference = self.circumference_cm / self.circumference_px

        self.radius_cm = self.radius_px * self.cm_per_px   # == 21.59 by construction
        self.area_cm2 = np.pi * self.radius_cm ** 2

        # Center zone: 50% of the arena radius (dlc_limb_analysis.py convention).
        self.center_radius_px = CENTER_FRAC * self.radius_px
        self.center_radius_cm = CENTER_FRAC * self.radius_cm
        self.center_area_frac = CENTER_FRAC ** 2              # 0.25 of the floor

    def describe(self) -> dict:
        return {
            "arena_center_x_px": self.cx,
            "arena_center_y_px": self.cy,
            "arena_diameter_h_px": self.d_h_px,
            "arena_diameter_v_px": self.d_v_px,
            "arena_diameter_px": self.diameter_px,
            "arena_diameter_asymmetry_pct": self.asymmetry_pct,
            "arena_circumference_px": self.circumference_px,
            "arena_circumference_cm": self.circumference_cm,
            "scale_cm_per_px": self.cm_per_px,
            "scale_px_per_cm": 1.0 / self.cm_per_px,
            "arena_radius_px": self.radius_px,
            "arena_radius_cm": self.radius_cm,
            "arena_area_cm2": self.area_cm2,
            "center_frac_of_arena_radius": CENTER_FRAC,
            "center_radius_px": self.center_radius_px,
            "center_radius_cm": self.center_radius_cm,
            "center_area_frac_of_arena": self.center_area_frac,
        }


def load_arena_table(path: str) -> pd.DataFrame:
    """Read arena_coords.csv and index it by every plausible key for a video."""
    if not os.path.exists(path):
        sys.exit(
            f"Arena coordinate file not found: {path}\n"
            f"Create it with:  python {os.path.basename(__file__)} --make-template"
        )
    df = pd.read_csv(path)
    need = ["video_id", "hx1", "hy1", "hx2", "hy2", "vx1", "vy1", "vx2", "vy2"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing column(s): {', '.join(missing)}")
    df = df.dropna(subset=need[1:])
    # Strip whitespace: ids edited in Excel often pick up a trailing space, which
    # would silently fail to match the folder name.
    return df.set_index(df["video_id"].astype(str).str.strip())


def arena_for(folder_name: str, table: pd.DataFrame) -> Arena | None:
    """Match a folder to its arena row by folder name, m<id>, or bare id."""
    mid = mouse_id(folder_name)
    for key in (folder_name, f"m{mid}", mid):
        if key in table.index:
            row = table.loc[key]
            return Arena(row.iloc[0] if isinstance(row, pd.DataFrame) else row)
    return None


# --------------------------------------------------------------------------- #
# Loading and cleaning the tracking data
# --------------------------------------------------------------------------- #
def load_bodypart(csv_path: str, bodypart: str) -> pd.DataFrame:
    """Read one body part out of a DLC csv as columns x, y, likelihood (pixels)."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    scorer = df.columns.levels[0][0]
    available = list(dict.fromkeys(df.columns.get_level_values(1)))
    if bodypart not in available:
        raise KeyError(f"'{bodypart}' not in {csv_path}; body parts are {available}")
    sub = df[scorer][bodypart].copy()
    sub.columns = [str(c) for c in sub.columns]
    return sub[["x", "y", "likelihood"]].astype(float)


def _long_gap_mask(bad: np.ndarray, max_gap: int) -> np.ndarray:
    """True on every frame belonging to a run of >max_gap consecutive bad frames."""
    out = np.zeros(len(bad), bool)
    if not bad.any():
        return out
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start > max_gap:
            out[start:end] = True
    return out


def clean_track(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop low-confidence points, interpolate SHORT gaps, smooth.

    Points below PCUTOFF are set to NaN rather than trusted, then filled by
    linear interpolation and a rolling median that removes single-frame jitter
    without rounding off real turns. Runs of bad frames longer than
    MAX_GAP_FRAMES are restored to NaN afterwards: interpolating a long gap
    invents a straight-line path the animal never took, and on a video the
    model failed outright it manufactures a plausible-looking flat track.
    """
    out = raw.copy()
    bad = (out["likelihood"] < PCUTOFF).to_numpy()
    out.loc[bad, ["x", "y"]] = np.nan
    out[["x", "y"]] = (
        out[["x", "y"]]
        .interpolate(method="linear", limit_direction="both")
        .rolling(SMOOTH_WIN, center=True, min_periods=1)
        .median()
    )
    out.loc[_long_gap_mask(bad, MAX_GAP_FRAMES), ["x", "y"]] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _nanstat(values, fn) -> float:
    """Apply a nan-aware reducer, returning NaN instead of warning on all-NaN."""
    arr = np.asarray(values, dtype=float)
    return float(fn(arr)) if np.isfinite(arr).any() else np.nan


def _runs(codes: np.ndarray):
    """Yield (start, end, value) for each maximal run of equal values in codes."""
    n = len(codes)
    if n == 0:
        return
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    for s, e in zip(starts, ends):
        yield int(s), int(e), codes[s]


def _despeckle(codes: np.ndarray, min_len: int, invalid=-1) -> np.ndarray:
    """Absorb valid runs shorter than min_len into the preceding valid state.

    Invalid frames (code == invalid) are never merged and never used as fill, so
    true tracking gaps stay gaps. A leading short run is filled from the next
    valid run instead. Operates left-to-right so merged runs can themselves be
    extended by the following short run.
    """
    out = codes.copy()
    for s, e, val in _runs(codes):
        if val == invalid or (e - s) >= min_len:
            continue
        prev = out[s - 1] if s > 0 else invalid
        if prev != invalid:
            out[s:e] = prev            # extend the previous valid state forward
        else:
            nxt = codes[e] if e < len(codes) else invalid
            out[s:e] = nxt             # no valid predecessor: borrow the successor
    return out


def _bouts(codes: np.ndarray, value) -> list:
    """Return the lengths (in frames) of every run equal to value."""
    return [e - s for s, e, v in _runs(codes) if v == value]


def analyze(folder: str, csv_path: str, arena: Arena) -> tuple[dict, pd.DataFrame]:
    mid = mouse_id(folder)
    raw = load_bodypart(csv_path, BODYPART)
    trk = clean_track(raw)

    n_frames = len(trk)
    shuffle_m = re.search(r"shuffle(\d+)", csv_path)
    metrics: dict = {
        "mouse": mid,
        "folder": folder,
        "genotype": GENOTYPE.get(mid, "?"),
        "dlc_csv": os.path.basename(csv_path),
        "shuffle": shuffle_m.group(1) if shuffle_m else "",
        "n_frames": n_frames,
        "fps": FPS,
        "duration_s": n_frames / FPS,
        "bodypart": BODYPART,
        "pcutoff": PCUTOFF,
        "lowconf_frac": float((raw["likelihood"] < PCUTOFF).mean()),
        "mean_likelihood": float(raw["likelihood"].mean()),
    }
    metrics.update(arena.describe())

    # --- radial position in the video's own pixel frame -------------------- #
    trk["r_px"] = np.hypot(trk["x"] - arena.cx, trk["y"] - arena.cy)

    # QC only: points beyond the wall mean a bad arena measurement or a
    # mistracked frame. Expressed as a fraction of arena radius so it is
    # readable without knowing this video's pixel scale.
    trk["r_frac"] = trk["r_px"] / arena.radius_px       # 0 = center, 1 = wall
    r_frac = trk["r_frac"]
    metrics["frac_outside_arena"] = float((r_frac > 1.05).mean())
    metrics["max_r_frac_of_arena_radius"] = _nanstat(r_frac, np.nanmax)
    metrics["mean_r_frac_of_arena_radius"] = _nanstat(r_frac, np.nanmean)
    # Wall-following (thigmotaxis): fraction of valid frames in the outer band.
    wall = (r_frac >= WALL_FOLLOW_FRAC) & r_frac.notna()
    trk["wall_follow"] = np.where(r_frac.notna(), wall, np.nan)
    metrics["pct_time_wall_following"] = (
        100.0 * float(wall.sum()) / float(r_frac.notna().sum())
        if r_frac.notna().any() else np.nan)

    # --- body length (distance between two tracked points, per frame) ------- #
    try:
        pa = clean_track(load_bodypart(csv_path, BODYLEN_PART_A))
        pb = clean_track(load_bodypart(csv_path, BODYLEN_PART_B))
        trk["body_length_cm"] = np.hypot(pa["x"] - pb["x"],
                                         pa["y"] - pb["y"]) * arena.cm_per_px
    except KeyError:
        trk["body_length_cm"] = np.nan
    metrics["median_body_length_cm"] = _nanstat(trk["body_length_cm"], np.nanmedian)

    # --- total distance travelled ------------------------------------------ #
    # Per-frame displacement in pixels, converted to cm with this video's scale.
    # Steps under MIN_STEP_CM are jitter and would otherwise accumulate into a
    # large phantom distance over ~54k frames.
    step_px = np.hypot(trk["x"].diff(), trk["y"].diff())
    step_cm = step_px * arena.cm_per_px
    step_clean = step_cm.where(step_cm >= MIN_STEP_CM, 0.0)
    trk["step_px"] = step_px
    trk["step_cm"] = step_clean
    trk["speed_cm_s"] = step_clean * FPS

    metrics["total_distance_px"] = float(np.nansum(step_px))
    metrics["total_distance_cm"] = float(np.nansum(step_clean))
    metrics["total_distance_m"] = metrics["total_distance_cm"] / 100.0
    metrics["total_distance_raw_cm"] = float(np.nansum(step_cm))  # pre jitter filter
    metrics["mean_speed_cm_s"] = _nanstat(trk["speed_cm_s"], np.nanmean)
    metrics["median_speed_cm_s"] = _nanstat(trk["speed_cm_s"], np.nanmedian)

    # --- center vs periphery ----------------------------------------------- #
    in_center = trk["r_px"] <= arena.center_radius_px
    valid = trk["r_px"].notna()
    trk["zone"] = np.where(in_center, "center", "periphery")
    trk.loc[~valid, "zone"] = np.nan

    n_valid = int(valid.sum())
    n_center = int((in_center & valid).sum())
    metrics["n_valid_frames"] = n_valid
    metrics["time_center_s"] = n_center / FPS
    metrics["time_periphery_s"] = (n_valid - n_center) / FPS
    metrics["pct_time_center"] = 100.0 * n_center / n_valid if n_valid else np.nan
    metrics["pct_time_periphery"] = 100.0 - metrics["pct_time_center"]

    # A step is credited to the zone the animal ARRIVES in, so each of the
    # N-1 displacements is assigned to exactly one zone and the two sum to the
    # total distance.
    arrive_center = in_center.to_numpy()[1:]
    steps = step_clean.to_numpy()[1:]
    d_center = float(np.nansum(steps[arrive_center]))
    d_periph = float(np.nansum(steps[~arrive_center]))
    d_total = d_center + d_periph
    metrics["dist_center_cm"] = d_center
    metrics["dist_periphery_cm"] = d_periph
    metrics["pct_dist_center"] = 100.0 * d_center / d_total if d_total else np.nan
    metrics["pct_dist_periphery"] = 100.0 - metrics["pct_dist_center"]

    # Occupancy relative to chance: 1.0 means the animal spends exactly as much
    # time in the center as an animal moving at random would. With a 50%-radius
    # center the chance level is 25% of time, not 50%.
    metrics["center_occupancy_index"] = (
        (metrics["pct_time_center"] / 100.0) / arena.center_area_frac
    )

    # Entries into the center: transitions periphery -> center.
    z = in_center.to_numpy()
    metrics["center_entries"] = int(np.sum(~z[:-1] & z[1:]))

    # --- behavior classification (immobile / walking / running) ------------- #
    # Code each frame by torso speed: -1 invalid, 0 immobile, 1 walking, 2 running.
    speed = trk["speed_cm_s"].to_numpy()
    valid_state = valid.to_numpy() & np.isfinite(speed)
    code = np.full(len(trk), -1, int)             # -1 stays for the (2,3)/(8,9) gaps
    code[valid_state & (speed <= IMMOBILE_MAX_CM_S)] = STATE_CODE["immobile"]
    code[valid_state & (speed >= WALK_MIN_CM_S)
                     & (speed < RUN_MIN_CM_S)] = STATE_CODE["walking"]
    code[valid_state & (speed >= RUN_MIN_CM_S)] = STATE_CODE["running"]
    code = _despeckle(code, MIN_BOUT_FRAMES)      # merge sub-flicker runs

    code_to_state = {v: k for k, v in STATE_CODE.items()}
    code_to_state[-1] = np.nan
    trk["state_code"] = code
    trk["state"] = pd.Series(code, index=trk.index).map(code_to_state)

    n_state_valid = int((code != -1).sum())
    metrics["n_state_frames"] = n_state_valid
    for name, val in STATE_CODE.items():
        n = int((code == val).sum())
        bouts = _bouts(code, val)
        metrics[f"time_{name}_s"] = n / FPS
        metrics[f"pct_time_{name}"] = 100.0 * n / n_state_valid if n_state_valid else np.nan
        metrics[f"n_bouts_{name}"] = len(bouts)
        metrics[f"mean_bout_{name}_s"] = (np.mean(bouts) / FPS) if bouts else np.nan
    # Convenience roll-up: walking + running.
    metrics["pct_time_mobile"] = (metrics["pct_time_walking"]
                                  + metrics["pct_time_running"])

    # --- QC verdict --------------------------------------------------------- #
    # A model that failed on this video produces sub-threshold likelihoods and
    # body parts placed outside the bucket. Such an animal still gets a row, so
    # the failure is visible, but is flagged and kept out of the group figure.
    reasons = []
    if metrics["lowconf_frac"] > MAX_LOWCONF_FRAC:
        reasons.append(f"{metrics['lowconf_frac']*100:.1f}% of frames below p={PCUTOFF}")
    if metrics["frac_outside_arena"] > 0.05:
        reasons.append(f"{metrics['frac_outside_arena']*100:.1f}% of frames outside the arena")
    if n_valid < 0.5 * n_frames:
        reasons.append(f"only {100*n_valid/n_frames:.1f}% of frames usable")
    metrics["qc_pass"] = not reasons
    metrics["qc_note"] = "; ".join(reasons)

    return metrics, trk


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _draw_arena(ax, arena: Arena):
    """Draw the arena wall and center-zone circles in the video's pixel frame."""
    # Arena wall: solid black, so the center circle reads as the inner boundary.
    ax.add_patch(plt.Circle((arena.cx, arena.cy), arena.radius_px,
                            fill=False, lw=2.0, color="k", zorder=5))
    # Center zone: a bold lime dashed line chosen to contrast with BOTH the pink
    # (KO) and blue (WT) trajectory colors and the black wall, with a white halo
    # underneath so it stays visible where the trajectory is densest.
    center = plt.Circle((arena.cx, arena.cy), arena.center_radius_px,
                        fill=False, edgecolor="#00DD00", lw=2.5, ls="--", zorder=6)
    center.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])
    ax.add_patch(center)
    pad = arena.radius_px * 1.08
    ax.set_xlim(arena.cx - pad, arena.cx + pad)
    ax.set_ylim(arena.cy + pad, arena.cy - pad)   # image convention: y downward
    ax.set_aspect("equal")


def _draw_ethogram_rows(ax, code: np.ndarray, xmin_min: float, xmax_min: float):
    """Draw the 3 behavior rows on ax for the time window [xmin_min, xmax_min].

    One full-height bar per bout (immobile bottom, walking middle, running top);
    bouts outside the window are clipped by the x-limits. Gaps = unclassified.
    """
    min_per_frame = 1.0 / FPS / 60.0
    for row, state in enumerate(STATES):                 # bottom -> top
        segs = [(s * min_per_frame, (e - s) * min_per_frame)
                for s, e, v in _runs(code) if v == STATE_CODE[state]]
        if segs:                                 # broken_barh errors on empty input
            ax.broken_barh(segs, (row - 0.45, 0.9), facecolors=STATE_COLORS[state])
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels([STATE_LABELS[s] for s in STATES], fontsize=9)
    ax.set_ylim(-0.6, len(STATES) - 0.4)
    ax.set_xlim(xmin_min, xmax_min)


def _ethogram_pct_text(ax, m: dict, y=-0.85):
    """Write the % of classified time per state below an ethogram axis."""
    label_color = {"immobile": "#777777",
                   "walking": STATE_COLORS["walking"],
                   "running": STATE_COLORS["running"]}
    # Anchor each label to a different edge (left / center / right) so they cannot
    # overlap regardless of how narrow the axis is.
    for (x, ha), state in zip(((0.0, "left"), (0.5, "center"), (1.0, "right")), STATES):
        ax.text(x, y, f"{STATE_LABELS[state]}: {m[f'pct_time_{state}']:.1f}%",
                transform=ax.transAxes, ha=ha, va="top",
                fontsize=9, fontweight="bold", color=label_color[state])


def per_animal_page(pdf, m: dict, trk: pd.DataFrame, arena: Arena):
    fig = plt.figure(figsize=(17, 5.5))
    # Nested layout: a left block (trajectory + occupancy, kept close together)
    # and a right block (two stacked ethogram halves). The tiny wspace inside the
    # left block removes the wide blank gap between trajectory and occupancy.
    outer = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.3], wspace=0.24)
    left = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)
    right = outer[1].subgridspec(2, 1, hspace=0.55)
    ax_traj = fig.add_subplot(left[0, 0])
    ax_occ = fig.add_subplot(left[0, 1])
    ax_eth1 = fig.add_subplot(right[0, 0])
    ax_eth2 = fig.add_subplot(right[1, 0])

    color = GROUP_COLORS.get(m["genotype"], "#999999")
    title = (f"m{m['mouse']} ({m['genotype']}) - shuffle {m['shuffle']} - "
             f"{m['duration_s']/60:.1f} min - {m['total_distance_cm']:.0f} cm travelled")
    if not m["qc_pass"]:
        title += f"\nFAILED QC: {m['qc_note']}"
        color = "#999999"
    fig.suptitle(title, fontsize=12,
                 color="#CC3311" if not m["qc_pass"] else "black")

    ax_traj.plot(trk["x"], trk["y"], lw=0.3, color=color, alpha=0.7, zorder=2)
    _draw_arena(ax_traj, arena)
    ax_traj.set_title("trajectory (torso)")
    ax_traj.set_xlabel("x (pixels)")
    ax_traj.set_ylabel("y (pixels)")
    # Zone occupancy percentages, written below the trajectory.
    ax_traj.text(0.5, -0.16,
                 f"center: {m['pct_time_center']:.1f}% time     "
                 f"periphery: {m['pct_time_periphery']:.1f}% time",
                 transform=ax_traj.transAxes, ha="center", va="top",
                 fontsize=11, fontweight="bold")

    # Occupancy heatmap, matching dlc_limb_analysis.py: magma hist2d with a
    # frame-count colorbar, equal aspect, image y-orientation, no overlay.
    ok = trk[["x", "y"]].dropna()
    hb = ax_occ.hist2d(ok["x"], ok["y"], bins=60, cmap="magma")
    hb[3].set_rasterized(True)     # bake the mesh so PDF shows no vector seams
    fig.colorbar(hb[3], ax=ax_occ, label="frames", fraction=0.046, pad=0.04)
    ax_occ.grid(False)             # keep the heatmap clean, as in dlc_limb_analysis
    ax_occ.set_aspect("equal")
    ax_occ.invert_yaxis()
    ax_occ.set_title("occupancy")
    ax_occ.set_xlabel("x (pixels)")
    ax_occ.tick_params(labelleft=False)   # y is same pixel scale as the trajectory

    # Behavior ethogram, split into first-15-min (top) and last-15-min (bottom)
    # halves so bouts stay legible; percentages (whole session) sit below.
    code = trk["state_code"].to_numpy()
    end2 = max(2 * ETHOGRAM_SPLIT_MIN, len(code) / FPS / 60.0)
    _draw_ethogram_rows(ax_eth1, code, 0.0, ETHOGRAM_SPLIT_MIN)
    ax_eth1.set_title("behavior ethogram")
    _draw_ethogram_rows(ax_eth2, code, ETHOGRAM_SPLIT_MIN, end2)
    ax_eth2.set_xlabel("time (min)")
    _ethogram_pct_text(ax_eth2, m, y=-0.85)

    pdf.savefig(fig, bbox_inches="tight")   # tight bbox keeps the % caption in frame
    plt.close(fig)


def _passed_tracks(tracks: dict, rows: list):
    """Yield (metrics, trk) for every QC-passed animal, in genotype/mouse order."""
    by_folder = {r["folder"]: r for r in rows}
    order = sorted(tracks, key=lambda f: (by_folder[f]["genotype"], by_folder[f]["mouse"])
                   if f in by_folder else (f,))
    for folder in order:
        m = by_folder.get(folder)
        if m and m["qc_pass"]:
            yield m, tracks[folder][0]


def _metric_bar(ax, summary: pd.DataFrame, col: str, label: str):
    """Group-mean bar with individual animals overlaid as points."""
    groups = sorted(summary["genotype"].unique())
    for i, g in enumerate(groups):
        vals = summary.loc[summary["genotype"] == g, col].astype(float)
        ax.bar(i, vals.mean(), color=GROUP_COLORS.get(g, "#999999"),
               edgecolor="k", width=0.6)
        ax.scatter(np.full(len(vals), i), vals, color="k", zorder=3, s=18)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_title(label, fontsize=10)


def _wallfollow_panel(ax, tracks: dict, rows: list):
    """% time wall-following in WALL_BIN_MIN bins, one line per genotype."""
    data, max_bins = [], 0
    for m, trk in _passed_tracks(tracks, rows):
        rf = trk["r_frac"].to_numpy()
        valid = np.isfinite(rf)
        wall = rf >= WALL_FOLLOW_FRAC
        binidx = np.floor((np.arange(len(rf)) / FPS / 60.0) / WALL_BIN_MIN).astype(int)
        nb = int(binidx.max()) + 1 if len(binidx) else 0
        pct = np.full(nb, np.nan)
        for b in range(nb):
            sel = (binidx == b) & valid
            if sel.any():
                pct[b] = 100.0 * wall[sel].mean()
        data.append((m["genotype"], pct))
        max_bins = max(max_bins, nb)

    centers = (np.arange(max_bins) + 0.5) * WALL_BIN_MIN
    genos = sorted({gg for gg, _ in data})
    # Dodge each genotype sideways so points/error bars don't stack at the same x.
    dodge = 0.12 * WALL_BIN_MIN
    for gi, g in enumerate(genos):
        rows_g = [p for gg, p in data if gg == g]
        mat = np.full((len(rows_g), max_bins), np.nan)
        for r, p in enumerate(rows_g):
            mat[r, :len(p)] = p
        x = centers + (gi - (len(genos) - 1) / 2.0) * dodge
        color = GROUP_COLORS.get(g, "#999999")
        n = np.sum(np.isfinite(mat), axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        sem[n < 2] = np.nan                      # SEM undefined for a single animal
        # individual animals (one faint point per mouse per bin)
        for r in range(mat.shape[0]):
            ax.scatter(x, mat[r], color=color, s=12, alpha=0.4, zorder=2)
        # group mean +/- SEM
        ax.errorbar(x, mean, yerr=sem, color=color, marker="o", capsize=3,
                    lw=1.8, zorder=3, label=g)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("% time wall-following")
    ax.set_ylim(0, 100)
    ax.set_title(f"wall-following (r ≥ {WALL_FOLLOW_FRAC:g}R), {WALL_BIN_MIN:g}-min bins "
                 f"(mean ± SEM, points = animals)", fontsize=9)
    ax.legend(fontsize=8, title="genotype")


def _bodylen_median_panel(ax, summary: pd.DataFrame):
    """One open circle per mouse (its median body length); bar = group median."""
    groups = sorted(summary["genotype"].unique())
    rng = np.random.RandomState(0)
    for i, g in enumerate(groups):
        vals = (summary.loc[summary["genotype"] == g, "median_body_length_cm"]
                .astype(float).dropna().to_numpy())
        if not len(vals):
            continue
        jit = (rng.rand(len(vals)) - 0.5) * 0.15
        ax.scatter(np.full(len(vals), i) + jit, vals, facecolors="none",
                   edgecolors=GROUP_COLORS.get(g, "#999999"), s=60, linewidths=1.5,
                   zorder=3)
        med = np.median(vals)
        ax.plot([i - 0.25, i + 0.25], [med, med], color="k", lw=2, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("median body length (cm)")
    ax.set_title("body length: per-animal medians", fontsize=9)


def _bodylen_z_panel(ax, tracks: dict, rows: list):
    """Within-animal z-scored body length, pooled per genotype as a density curve."""
    try:
        from scipy.stats import gaussian_kde
        have_kde = True
    except Exception:
        have_kde = False
    pooled: dict[str, list] = {}
    for m, trk in _passed_tracks(tracks, rows):
        bl = trk["body_length_cm"].to_numpy()
        finite = bl[np.isfinite(bl)]
        if finite.size < 10 or finite.std() == 0:
            continue
        pooled.setdefault(m["genotype"], []).append((finite - finite.mean()) / finite.std())

    grid = np.linspace(-4, 4, 200)
    rng = np.random.RandomState(0)
    for g in sorted(pooled):
        z = np.concatenate(pooled[g])
        color = GROUP_COLORS.get(g, "#999999")
        if have_kde and z.size > 5:
            zs = rng.choice(z, 20000, replace=False) if z.size > 20000 else z
            ax.plot(grid, gaussian_kde(zs)(grid), color=color, label=g)
        else:
            ax.hist(z, bins=60, density=True, histtype="step", color=color, label=g)
    ax.set_xlabel("body length (within-animal z-score)")
    ax.set_ylabel("density")
    ax.set_title("body length: within-animal z-scored", fontsize=9)
    ax.legend(fontsize=8, title="genotype")


def group_page(pdf, summary: pd.DataFrame, tracks: dict, rows: list):
    fields = [
        ("total_distance_cm", "total distance (cm)"),
        ("pct_time_center", "% time in center"),
        ("pct_time_immobile", "% time immobile"),
        ("center_entries", "center entries"),
    ]
    fig = plt.figure(figsize=(22, 10))
    # Two independent rows so each can set its own generous column spacing; this
    # keeps the axis/scale labels from colliding with the neighbouring panel.
    outer = fig.add_gridspec(2, 1, hspace=0.55)
    top = outer[0].subgridspec(1, 4, wspace=0.55)
    bot = outer[1].subgridspec(1, 3, wspace=0.45)
    # Top row: the four group-metric bars.
    for i, (col, label) in enumerate(fields):
        _metric_bar(fig.add_subplot(top[0, i]), summary, col, label)
    # Bottom row: the three new panels.
    _wallfollow_panel(fig.add_subplot(bot[0, 0]), tracks, rows)
    _bodylen_median_panel(fig.add_subplot(bot[0, 1]), summary)
    _bodylen_z_panel(fig.add_subplot(bot[0, 2]), tracks, rows)
    fig.suptitle("group comparison (points = individual animals)")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Template generation
# --------------------------------------------------------------------------- #
def make_template(root: str, pattern: str, csv_pattern: str, path: str):
    if os.path.exists(path):
        sys.exit(f"{path} already exists -- refusing to overwrite it.")
    rows = [{"video_id": name, "hx1": "", "hy1": "", "hx2": "", "hy2": "",
             "vx1": "", "vy1": "", "vx2": "", "vy2": "", "notes": ""}
            for name in sorted(find_animals(root, pattern, csv_pattern))]
    if not rows:
        sys.exit(f"No folders matching {pattern!r} found under {root}")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote template with {len(rows)} rows: {path}\n"
          f"Fill in the ImageJ endpoints of the horizontal (h) and vertical (v) "
          f"diameter lines, in pixels, then re-run without --make-template.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=DATA_ROOT,
                   help="folder containing the animal subfolders")
    p.add_argument("-o", "--out", default=None, help="output folder")
    p.add_argument("-a", "--arena", default=None, help="path to arena_coords.csv")
    p.add_argument("--pattern", default=FOLDER_PATTERN,
                   help=f"glob for animal folders (default: {FOLDER_PATTERN})")
    p.add_argument("--csv-pattern", default=CSV_PATTERN,
                   help=f"glob for the DLC csv (default: {CSV_PATTERN})")
    p.add_argument("--recursive", action="store_true",
                   help="also search subfolders (e.g. prelim-analysis) for DLC csvs")
    p.add_argument("--make-template", action="store_true",
                   help="write an empty arena_coords.csv and exit")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    arena_csv = args.arena or ARENA_CSV        # one shared arena file for both cohorts
    out_dir = args.out or os.path.join(root, os.path.basename(OUT_DIR))

    if args.make_template:
        make_template(root, args.pattern, args.csv_pattern, arena_csv)
        return

    os.makedirs(os.path.join(out_dir, "per_animal"), exist_ok=True)
    arena_table = load_arena_table(arena_csv)

    animals = find_animals(root, args.pattern, args.csv_pattern,
                           recursive=args.recursive)
    if not animals:
        sys.exit(f"No DLC csvs found in {args.pattern!r} folders under {root}")
    print(f"Found {len(animals)} animal folder(s) matching {args.pattern!r}")

    # Flag arena rows that match no folder on disk -- almost always a typo in
    # video_id, which would otherwise just look like a missing animal.
    folder_keys: set[str] = set()
    for folder in animals:
        mid = mouse_id(folder)
        folder_keys |= {folder, f"m{mid}", mid}
    unused = [k for k in arena_table.index if k not in folder_keys]
    if unused:
        print(f"  [warn] {os.path.basename(arena_csv)} rows matching no folder: "
              f"{', '.join(unused)}", file=sys.stderr)

    rows, tracks = [], {}
    for folder, csv_path in animals.items():
        arena = arena_for(folder, arena_table)
        if arena is None:
            print(f"  [skip] {folder}: no row in {os.path.basename(arena_csv)}",
                  file=sys.stderr)
            continue
        print(f"  {folder}: {os.path.basename(csv_path)}")
        if arena.asymmetry_pct > 5:
            print(f"    [warn] the two diameters differ by "
                  f"{arena.asymmetry_pct:.1f}% -- check the ImageJ lines",
                  file=sys.stderr)
        m, trk = analyze(folder, csv_path, arena)
        rows.append(m)
        tracks[folder] = (trk, arena)
        trk.to_csv(os.path.join(out_dir, "per_animal", f"{folder}_track.csv"))
        if m["qc_pass"]:
            print(f"    {m['total_distance_cm']:.0f} cm travelled, "
                  f"{m['pct_time_center']:.1f}% time in center, "
                  f"{m['pct_time_immobile']:.1f}% immobile")
        else:
            print(f"    [FAILED QC] {m['qc_note']} -- metrics not usable",
                  file=sys.stderr)

    if not rows:
        sys.exit("No animals could be analyzed -- is arena_coords.csv filled in?")

    summary = pd.DataFrame(rows).sort_values(["genotype", "mouse"])
    summary_path = os.path.join(out_dir, "summary_metrics_openfield.csv")
    summary.to_csv(summary_path, index=False)

    pdf_path = os.path.join(out_dir, "openfield_figures.pdf")
    passed = summary[summary["qc_pass"]]
    with PdfPages(pdf_path) as pdf:
        for folder, (trk, arena) in tracks.items():
            m = next(r for r in rows if r["folder"] == folder)
            per_animal_page(pdf, m, trk, arena)
        if passed["genotype"].nunique() > 1:
            group_page(pdf, passed, tracks, rows)

    failed = summary[~summary["qc_pass"]]
    if len(failed):
        print(f"\n{len(failed)} of {len(summary)} animals FAILED QC and are "
              f"excluded from the group figure:", file=sys.stderr)
        for _, r in failed.iterrows():
            print(f"  m{r['mouse']} (shuffle {r['shuffle']}): {r['qc_note']}",
                  file=sys.stderr)

    print(f"\nWrote {summary_path}\n      {pdf_path}")


if __name__ == "__main__":
    main()

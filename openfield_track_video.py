#!/usr/bin/env python3
"""
Ink-drawing style track video of the torso path inside the bucket.

For each animal folder this renders a video that looks like a pen drawing: the
arena as a black circle, the torso trajectory accumulating inside it stroke by
stroke, and a filled dot marking where the animal is right now. The pen takes
the animal's genotype color -- blue for WT, red for Df(h16p12)/+, matching the
line graphs -- so the video is readable next to the rest of the figures.
Nothing is ever erased, so the final frame is the whole session's path.

Everything is read through openfield_analysis.py, so the geometry and the
tracking clean-up are identical to the analysis:

    torso x/y   the DLC csv of the most recent shuffle in the animal folder,
                low-confidence points dropped, short gaps interpolated, long
                gaps left as NaN (the pen lifts -- the trail breaks)
    arena       arena_coords.csv, the two hand-drawn ImageJ diameters

The trail is cumulative by default, which fills the circle in over a long
session; --trail-sec N instead shows a rolling window of the last N seconds, so
the drawing keeps breathing (and the recent path stays readable) for the whole
recording.

Playback speed
--------------
The output frame rate is  FPS * speed / stride, so with the defaults
(30 fps source, speed 2, stride 2) the video is a 30 fps file that plays the
session at 2x real time. Frames skipped by stride are still drawn into the
trail before being skipped, so decimating never shortens the path.

Usage:
    # day 1 cohort -> 20260421_openfield_out/track_videos/
    /Applications/miniconda3/envs/DEEPLABCUT/bin/python openfield_track_video.py

    # day 2 cohort
    ... openfield_track_video.py --pattern '20260422_m*' \
            -o /Volumes/dorothea_1T/202604_moseq-pilot/20260422_openfield_out/track_videos

    # one animal, first 2 minutes only (quick look at the style)
    ... openfield_track_video.py --mouse 367 --end-min 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openfield_analysis as of      # arena geometry + DLC loading/cleaning

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required:  pip install opencv-python")

# --- look of the drawing --------------------------------------------------- #
# The pen color is the animal's genotype color, the same blue / red used for WT
# and Df(h16p12)/+ everywhere else (openfield_analysis.GROUP_COLORS), so a track
# video sits next to the line graphs without a second color code to learn.
ARENA_BGR = (0, 0, 0)            # black bucket perimeter
PAPER_BGR = (250, 250, 248)      # off-white paper
HUD_BGR = (140, 140, 140)
FALLBACK_BGR = (153, 153, 153)   # unknown genotype ("?")

CANVAS_PX = 720                  # output frame is CANVAS_PX x CANVAS_PX
ARENA_FRAC = 0.44                # arena radius as a fraction of the frame size
ARENA_LW = 3                     # circle outline thickness
TRAIL_LW = 2                     # trajectory stroke thickness
DOT_R = 6                        # current-position dot radius

SPEED = 10.0                      # playback multiplier, can increase number for 
                                 # faster videos
STRIDE = 5                       # render every Nth frame (trail keeps all),
                                 # change number to 1/2 of the speed


# --------------------------------------------------------------------------- #
# Geometry: video pixels -> canvas pixels
# --------------------------------------------------------------------------- #
def make_projector(arena, size: int, arena_frac: float):
    """Return (project, canvas_center, canvas_radius) for one arena.

    The bucket is mapped to a fixed circle in the middle of the frame, so every
    animal's video is drawn at the same size no matter how big the bucket was
    in its own recording. y is left pointing down (video convention).
    """
    c = size / 2.0
    r_canvas = arena_frac * size
    k = r_canvas / arena.radius_px

    def project(x: np.ndarray, y: np.ndarray):
        return (x - arena.cx) * k + c, (y - arena.cy) * k + c

    return project, c, r_canvas


def ink_for(mouse: str) -> tuple[int, int, int]:
    """Pen color (BGR) for a mouse: its genotype's plotting color."""
    hexcode = of.GROUP_COLORS.get(of.GENOTYPE.get(mouse, "?"))
    if not hexcode:
        return FALLBACK_BGR
    r, g, b = (int(hexcode.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def valid_runs(ok: np.ndarray) -> np.ndarray:
    """[start, end) index pairs of each run of consecutively tracked frames."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(np.int8), [0]))))
    return np.stack([edges[0::2], edges[1::2]], axis=1) if edges.size else np.empty((0, 2), int)


def _blank_canvas(size: int, center: float, radius: float) -> np.ndarray:
    """Paper with the arena circle already drawn on it."""
    img = np.full((size, size, 3), PAPER_BGR, np.uint8)
    cv2.circle(img, (int(round(center)), int(round(center))), int(round(radius)),
               ARENA_BGR, ARENA_LW, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Video writing
# --------------------------------------------------------------------------- #
class VideoOut:
    """Frame sink: ffmpeg (H.264) when available, else OpenCV's mp4v."""

    def __init__(self, path: str, size: int, fps: float):
        self.path = path
        self.proc = None
        self.writer = None
        ffmpeg = shutil.which("ffmpeg") or os.path.join(
            os.path.dirname(sys.executable), "ffmpeg")
        if os.path.exists(ffmpeg):
            cmd = [ffmpeg, "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{size}x{size}", "-r", f"{fps:g}", "-i", "-",
                   "-an", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "20", "-pix_fmt", "yuv420p", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps, (size, size))
            if not self.writer.isOpened():
                sys.exit(f"could not open a video writer for {path}")

    def write(self, frame: np.ndarray):
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                sys.exit(f"ffmpeg failed writing {self.path}")
        else:
            self.writer.release()


# --------------------------------------------------------------------------- #
# Rendering one animal
# --------------------------------------------------------------------------- #
def render(folder: str, csv_path: str, arena, out_path: str, args) -> tuple[int, float]:
    """Draw the whole (or clipped) session and return (frames written, seconds)."""
    trk = of.clean_track(of.load_bodypart(csv_path, args.bodypart))
    x = trk["x"].to_numpy(float)
    y = trk["y"].to_numpy(float)

    i0 = int(round(args.start_min * 60 * of.FPS))
    i1 = len(x) if args.end_min is None else int(round(args.end_min * 60 * of.FPS))
    i0, i1 = max(0, i0), min(len(x), i1)
    if i1 - i0 < 2:
        print(f"  [skip] {folder}: nothing to draw in that time window", file=sys.stderr)
        return 0, 0.0
    x, y = x[i0:i1], y[i0:i1]

    project, center, radius = make_projector(arena, args.size, ARENA_FRAC)
    px, py = project(x, y)
    ok = np.isfinite(px) & np.isfinite(py)
    ipx = np.where(ok, px, 0).astype(np.int32)
    ipy = np.where(ok, py, 0).astype(np.int32)

    base = _blank_canvas(args.size, center, radius)
    if args.center_circle:                       # the 50%-radius center zone
        cv2.circle(base, (int(round(center)), int(round(center))),
                   int(round(of.CENTER_FRAC * radius)), (205, 205, 205), 1, cv2.LINE_AA)

    # window > 0 redraws the last N seconds every frame; window == 0 keeps one
    # canvas and just extends it, which is both cheaper and fully cumulative
    window = int(round(args.trail_sec * of.FPS))
    pts = np.stack([ipx, ipy], axis=1)
    runs = valid_runs(ok) if window else None
    trail = base if window else base.copy()

    out_fps = of.FPS * args.speed / args.stride
    vid = VideoOut(out_path, args.size, out_fps)
    mid = of.mouse_id(folder)
    ink = ink_for(mid)
    label = f"m{mid}  {of.GENOTYPE.get(mid, '?')}"
    written = 0
    try:
        for i in range(len(x)):
            # extend the trail: only between two consecutive valid frames, so a
            # tracking gap lifts the pen instead of drawing a shortcut across
            if not window and i and ok[i] and ok[i - 1]:
                cv2.line(trail, (ipx[i - 1], ipy[i - 1]), (ipx[i], ipy[i]),
                         ink, args.trail_width, cv2.LINE_AA)

            if i % args.stride:                  # drawn into the trail, not emitted
                continue

            if window:
                frame = base.copy()
                lo = max(0, i - window)
                for a, b in runs[(runs[:, 1] > lo) & (runs[:, 0] <= i)]:
                    seg = pts[max(a, lo):min(b, i + 1)]
                    if len(seg) > 1:
                        cv2.polylines(frame, [seg], False, ink,
                                      args.trail_width, cv2.LINE_AA)
            else:
                frame = trail.copy()
            if ok[i]:
                cv2.circle(frame, (ipx[i], ipy[i]), args.dot_radius, ink, -1,
                           cv2.LINE_AA)
            if args.hud:
                t = (i0 + i) / of.FPS
                cv2.putText(frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, HUD_BGR, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(t // 60):02d}:{int(t % 60):02d}",
                            (args.size - 86, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, HUD_BGR, 1, cv2.LINE_AA)
            vid.write(frame)
            written += 1
    finally:
        vid.close()
    return written, written / out_fps


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=of.DATA_ROOT, help="parent of the animal folders")
    p.add_argument("--pattern", default=of.FOLDER_PATTERN, help="animal folder glob")
    p.add_argument("--csv-pattern", default=of.CSV_PATTERN, help="DLC csv glob")
    p.add_argument("--arena", default=of.ARENA_CSV, help="arena_coords.csv")
    p.add_argument("-o", "--out", default=None,
                   help="output folder (default <cohort>_openfield_out/track_videos)")
    p.add_argument("--mouse", action="append", default=None,
                   help="only this mouse id (repeatable), e.g. --mouse 367")
    p.add_argument("--bodypart", default=of.BODYPART, help="body part to draw")
    p.add_argument("--speed", type=float, default=SPEED, help="playback multiplier")
    p.add_argument("--stride", type=int, default=STRIDE,
                   help="emit every Nth frame (the trail still uses every frame)")
    p.add_argument("--size", type=int, default=CANVAS_PX, help="frame size, px")
    p.add_argument("--trail-width", type=int, default=TRAIL_LW, help="stroke thickness")
    p.add_argument("--dot-radius", type=int, default=DOT_R, help="current-position dot")
    p.add_argument("--trail-sec", type=float, default=0.0,
                   help="show only the last N seconds of path (0 = cumulative)")
    p.add_argument("--center-circle", action="store_true",
                   help="also outline the 50%%-radius center zone")
    p.add_argument("--no-hud", dest="hud", action="store_false",
                   help="drop the mouse id / clock overlay (bare drawing)")
    p.add_argument("--start-min", type=float, default=0.0, help="clip start, minutes")
    p.add_argument("--end-min", type=float, default=None, help="clip end, minutes")
    args = p.parse_args()

    if args.stride < 1:
        sys.exit("--stride must be >= 1")
    if args.speed <= 0:
        sys.exit("--speed must be > 0")
    args.size += args.size % 2               # H.264 needs even dimensions

    animals = of.find_animals(args.root, args.pattern, args.csv_pattern)
    if not animals:
        sys.exit(f"no animal folders matching {args.pattern} under {args.root}")
    if args.mouse:
        want = {m.lstrip("m") for m in args.mouse}
        animals = {f: c for f, c in animals.items() if of.mouse_id(f) in want}
        if not animals:
            sys.exit(f"none of {sorted(want)} matched {args.pattern}")

    arena_table = of.load_arena_table(args.arena)

    out_dir = args.out
    if out_dir is None:
        cohort = sorted(animals)[0].split("_")[0]
        out_dir = os.path.join(args.root, f"{cohort}_openfield_out", "track_videos")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{len(animals)} animal(s) -> {out_dir}")
    print(f"playback {args.speed:g}x  (output {of.FPS * args.speed / args.stride:g} fps)")
    for folder, csv_path in sorted(animals.items()):
        arena = of.arena_for(folder, arena_table)
        if arena is None:
            print(f"  [skip] {folder}: no row in {os.path.basename(args.arena)}",
                  file=sys.stderr)
            continue
        out_path = os.path.join(out_dir, f"{folder}_track_{args.speed:g}x.mp4")
        print(f"  {folder} ...", end="", flush=True)
        n, secs = render(folder, csv_path, arena, out_path, args)
        if n:
            print(f" {n} frames, {secs / 60:.1f} min -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()

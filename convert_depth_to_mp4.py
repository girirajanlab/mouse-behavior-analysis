#!/usr/bin/env python
"""Convert a MoSeq raw depth.dat to an 8-bit H.264 .mp4 that DeepLabCut can
ingest. Resolution / endianness are read from the session's metadata.json and
the frame rate from depth_ts.txt, so it works on any session folder. The source
depth.dat is opened read-only and never modified.

Usage:
    python convert_depth_to_mp4.py [SESSION_DIR_OR_DAT] [-o OUTPUT.mp4]

Examples:
    python convert_depth_to_mp4.py                       # current folder
    python convert_depth_to_mp4.py /data/m370_5_1_26     # a session folder
    python convert_depth_to_mp4.py /data/s/depth.dat -o /tmp/out.mp4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np

# Use ffmpeg from the active environment / PATH (portable across machines).
# Override with the FFMPEG env var if it lives somewhere unusual.
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def resolve_paths(target):
    """Accept a session folder OR a path to depth.dat; return (dat, meta, ts)."""
    if os.path.isdir(target):
        d = target
    else:
        d = os.path.dirname(target) or "."
    dat = os.path.join(d, "depth.dat")
    if not os.path.isfile(dat):
        sys.exit(f"[error] no depth.dat found at {dat}")
    return dat, os.path.join(d, "metadata.json"), os.path.join(d, "depth_ts.txt")


def read_resolution(meta_path):
    """W, H and numpy dtype from metadata.json (falls back to Kinect v2)."""
    if not os.path.isfile(meta_path):
        print(f"[warn] no metadata.json; assuming 512x424 UInt16 LE", flush=True)
        return 512, 424, np.dtype("<u2")
    meta = json.load(open(meta_path))
    w, h = meta["DepthResolution"]
    little = meta.get("IsLittleEndian", True)
    if "16" not in meta.get("DepthDataType", "UInt16[]"):
        sys.exit(f"[error] unexpected DepthDataType {meta.get('DepthDataType')!r}")
    return int(w), int(h), np.dtype("<u2" if little else ">u2")


def read_fps(ts_path, default=30):
    """Median frame rate from depth_ts.txt timestamps (ms in column 1)."""
    if not os.path.isfile(ts_path):
        return default
    try:
        t = np.loadtxt(ts_path, usecols=0)
        if t.size < 2:
            return default
        dt = np.median(np.diff(t))          # ms per frame
        fps = round(1000.0 / dt)
        return fps if fps > 0 else default
    except Exception as e:
        print(f"[warn] could not parse {ts_path} ({e}); using {default} fps", flush=True)
        return default


def main():
    ap = argparse.ArgumentParser(description="MoSeq depth.dat -> DeepLabCut .mp4")
    ap.add_argument("target", nargs="?", default=".",
                    help="session folder or path to depth.dat (default: current dir)")
    ap.add_argument("-o", "--output", help="output .mp4 (default: depth.mp4 next to depth.dat)")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality, lower=better (default 18)")
    args = ap.parse_args()

    dat, meta_path, ts_path = resolve_paths(args.target)
    dst = args.output or os.path.join(os.path.dirname(dat) or ".", "depth.mp4")
    W, H, dtype = read_resolution(meta_path)
    fps = read_fps(ts_path)

    frame_bytes = W * H * dtype.itemsize
    total = os.path.getsize(dat) // frame_bytes
    print(f"[info] {dat} -> {dst}", flush=True)
    print(f"[info] {total} frames, {W}x{H}, {dtype.str}, {fps} fps", flush=True)

    # ---- Pass 1: sample frames for the normalization range over valid pixels.
    # Kinect encodes "no measurement" as 0; ignore those so contrast tracks the
    # real depth range (arena floor -> animal).
    n_sample = min(600, total)
    sample_idx = np.linspace(0, total - 1, n_sample).astype(np.int64)
    lo_vals, hi_vals = [], []
    with open(dat, "rb") as f:
        for i in sample_idx:
            f.seek(int(i) * frame_bytes)
            frame = np.frombuffer(f.read(frame_bytes), dtype=dtype)
            valid = frame[frame > 0]
            if valid.size:
                lo_vals.append(np.percentile(valid, 0.5))
                hi_vals.append(np.percentile(valid, 99.5))
    if not hi_vals:
        sys.exit("[error] no valid (non-zero) depth pixels found")
    vmin = float(np.min(lo_vals))
    vmax = float(np.max(hi_vals))
    if vmax <= vmin:
        vmax = vmin + 1
    scale = 255.0 / (vmax - vmin)
    print(f"[info] normalization range (valid depth): vmin={vmin:.0f} vmax={vmax:.0f} mm", flush=True)

    # ---- Pass 2: stream every frame, normalize, pipe raw gray8 to ffmpeg/libx264.
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", dst,
    ]
    print("[info] " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    CHUNK = 64  # frames per read
    with open(dat, "rb") as f:
        done = 0
        while True:
            raw = f.read(frame_bytes * CHUNK)
            if not raw:
                break
            nfr = len(raw) // frame_bytes
            arr = np.frombuffer(raw[: nfr * frame_bytes], dtype=dtype).astype(np.float32)
            arr = (arr - vmin) * scale
            np.clip(arr, 0, 255, out=arr)
            proc.stdin.write(arr.astype(np.uint8).tobytes())
            done += nfr
            if done % (CHUNK * 50) == 0 or done == total:
                print(f"[progress] {done}/{total} ({100*done/max(total,1):.1f}%)", flush=True)

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        sys.exit(f"[error] ffmpeg exited with code {ret}")
    print(f"[done] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()

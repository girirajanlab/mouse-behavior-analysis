"""
Find every depthDLC_Resnet50_2026090809Sep14shuffle1_snapshot_best-100_p60_labeled.mp4
under a root directory (searching all subfolders) and write a 10x sped-up copy,
depthDLC_shuffle1_10x.mp4, into the same folder as each source video.

Speed-up method: keep every 10th frame and write at the original frame rate,
so playback is 10x faster and the output file is ~10x smaller.

Usage:
    python3 speedup_videos.py [ROOT_DIR] [--overwrite]
"""

import argparse
import sys
from pathlib import Path

import cv2

SOURCE_NAME = "depthDLC_Resnet50_2026090809Sep14shuffle1_snapshot_best-100_p60_labeled.mp4"
OUTPUT_NAME = "depthDLC_shuffle1_10x.mp4"
SPEED = 10


def speed_up(src: Path, dst: Path, speed: int = SPEED) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = dst.with_name(dst.stem + ".partial.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot create {dst}")

    idx = written = 0
    try:
        while True:
            # grab() skips decoding-to-array for frames we discard, which is faster
            if not cap.grab():
                break
            if idx % speed == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                writer.write(frame)
                written += 1
            idx += 1
    finally:
        cap.release()
        writer.release()

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise IOError(f"No frames read from {src}")

    tmp.replace(dst)  # only appears under the final name once complete
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=".", help="Directory to search (default: current directory)")
    parser.add_argument("--overwrite", action="store_true", help="Re-create outputs that already exist")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    # Skip macOS "._" resource-fork files that appear on external drives
    sources = sorted(p for p in root.rglob(SOURCE_NAME) if not p.name.startswith("._"))

    if not sources:
        print(f"No '{SOURCE_NAME}' files found under {root}")
        return

    print(f"Found {len(sources)} video(s) under {root}\n")
    failures = 0
    for i, src in enumerate(sources, 1):
        dst = src.parent / OUTPUT_NAME
        print(f"[{i}/{len(sources)}] {src.parent}")
        if dst.exists() and not args.overwrite:
            print("    skipped (output already exists; use --overwrite to redo)")
            continue
        try:
            n = speed_up(src, dst)
            print(f"    saved {dst.name} ({n} frames)")
        except Exception as e:
            failures += 1
            print(f"    FAILED: {e}")

    print(f"\nDone. {len(sources) - failures} succeeded, {failures} failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

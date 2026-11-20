"""Labelled images of pieces standing OFF CENTRE, to teach the cameras offsets.

  python offset_capture.py "h5 R a 15"                       one piece, 15 mm toward a
  python offset_capture.py "c3 R a 10" "e4 P robot 5" "g5 K h 15"   several at once
  directions: a / h (toward that file), 8 / 1 (toward that rank), none (centred)

Each capture is data_offsets/<stamp>_<camera>.jpg + <stamp>.json with the square,
piece and the true offset in mm: dx along a->h (+ toward h), dy along the files
(+ toward rank 8 / the robot). tolerance_test.py saves one too before every test.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

import config

OUT = os.path.join(config.HERE, "data_offsets")
# toward the a-file / h-file / rank 8 (the robot) / rank 1 (the front camera)
DIRS = {"a": (-1, 0), "h": (1, 0), "8": (0, 1), "1": (0, -1), "none": (0, 0),
        "robot": (0, 1), "away": (0, -1)}          # old names (= toward rank 8 / rank 1)


def offset_mm(direction, mm):
    sx, sy = DIRS[direction]
    return sx * float(mm), sy * float(mm)


def save(frames, square, piece, direction, mm, note="", extra=None):
    """frames: {camera: image}. One piece (square/piece/direction/mm), or several
    via extra=[(square, piece, direction, mm), ...]. Returns the stamp."""
    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for cam, f in frames.items():
        cv2.imwrite(os.path.join(OUT, f"{stamp}_{cam}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    pieces = [(square, piece, direction, mm)] if square else []
    pieces += list(extra or [])
    items = []
    for sq, pc, d, m in pieces:
        dx, dy = offset_mm(d, m)
        items.append({"square": sq, "piece": pc, "direction": d, "mm": float(m), "dx_mm": dx, "dy_mm": dy})
    rec = {"pieces": items, "cameras": sorted(frames), "note": note,
           "calibration": {k: config.CAMERAS[k]["file"] for k in frames}}
    if len(items) == 1:
        rec.update(items[0])
    with open(os.path.join(OUT, stamp + ".json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return stamp


def parse(spec):
    """'h5 R a 10' -> ('h5', 'R', 'a', 10.0)"""
    sq, pc, d, m = spec.replace(",", " ").split()
    assert d in DIRS, f"direction must be one of {list(DIRS)}"
    return sq.lower(), pc, d, float(m)


def main():
    pieces = [parse(s) for s in sys.argv[1:]]
    from cameras import open_views
    views = open_views()
    frames = {}
    for v in views:
        for _ in range(4):        # skip frames damaged on the shared USB link (garbled bands)
            f = v.settled_frame(15)
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
            if np.abs(np.diff(g.mean(axis=1))).max() < 15.0:
                break
            print(f"{v.key}: damaged frame, taking another", flush=True)
        frames[v.key] = f
    stamp = save(frames, None, None, None, None, "offset_capture.py", extra=pieces)
    n = sum(len(json.load(open(os.path.join(OUT, f))).get("pieces", [1]))
            for f in os.listdir(OUT) if f.endswith(".json"))
    print(f"saved {stamp}: " + "; ".join(f"{p} {s} {m:g} mm {d}" for s, p, d, m in pieces)
          + f"  ({n} labelled pieces in total)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

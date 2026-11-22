"""Re-align a camera that was nudged, WITHOUT emptying the board (2026-11-22).

  python camera_realign.py front            measure + save if it clearly improves
  python camera_realign.py front --dry-run  measure only

The stored empty-board top-down picture (made at calibration) is the reference.
The live top-down view (current calibration, pieces on the board) is matched to it
on the tile edges (OpenCV ECC, homography, started from the best whole-pixel shift).
The warp found moves the 49 inner tile corners and the 4 outer corners to where the
board now appears in the image; a new geometry is fitted from them exactly as
calibrate_camera.py does. It is saved only if the tile-alignment check rises above
ACCEPT (the old file is kept in config/old/). Pieces hide part of the tile edges,
so a move of more than ~1/3 square may not be recovered: then empty the board and
run calibrate_camera.py.
"""
import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

import config
from board_vision import BoardGeometry
from cameras import open_views

ACCEPT = 0.6


def edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
    e = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.GaussianBlur(e / (e.max() + 1e-6), (7, 7), 0)


def best_shift(ref, cur, r=24, step=2):
    """Whole-pixel translation of `cur` that best matches `ref` (top-down pixels)."""
    best = (-2.0, 0, 0)
    a = ref - ref.mean()
    for dx in range(-r, r + 1, step):
        for dy in range(-r, r + 1, step):
            b = np.roll(np.roll(cur, -dy, 0), -dx, 1)
            b = b - b.mean()
            c = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if c > best[0]:
                best = (c, dx, dy)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camera", choices=list(config.CAMERAS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    v = open_views([a.camera])[0]
    time.sleep(2)
    frame = v.settled_frame()
    geo = v.geo
    if geo is None or getattr(geo, "reference", None) is None:
        sys.exit("camera not calibrated / no empty-board reference")
    before = geo.tile_alignment(frame)
    W, n = geo.WARP, 8 * geo.WARP
    ref, cur = edges(geo.reference), edges(geo.topdown(frame))
    c0, dx, dy = best_shift(ref, cur)
    # ECC: ref(x) ~ cur(warp . x)  -> a board point drawn at x in the reference top-down
    # is now at warp . x in the current top-down
    warp = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float32)
    try:
        _, warp = cv2.findTransformECC(ref, cur, warp, cv2.MOTION_HOMOGRAPHY,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error as e:
        v.release()
        sys.exit(f"ECC did not converge ({e}) - empty the board and run calibrate_camera.py")
    M = cv2.getPerspectiveTransform(geo.corners_px, np.float32([[0, n], [n, n], [n, 0], [0, 0]]))
    to_img = np.linalg.inv(M) @ warp                      # reference top-down -> image now

    def img_pts(xy_board):
        td = np.float32([[x * W, n - y * W] for x, y in xy_board]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(td, to_img).reshape(-1, 2)

    corners = img_pts([(0, 0), (8, 0), (8, 8), (0, 8)])
    inner = img_pts([(x, y) for y in range(1, 8) for x in range(1, 8)])
    new = BoardGeometry(corners.tolist(), tuple(geo.frame_size), geo.markers, inner_px=inner.tolist())
    new.reference = geo.reference
    after = new.tile_alignment(frame)
    shift = float(np.median(np.hypot(*(corners - geo.corners_px).T)))
    print(f"{a.camera}: alignment {before:.2f} -> {after:.2f}; whole-pixel shift ({dx},{dy}) top-down px "
          f"(corr {c0:.2f}); corners moved {shift:.1f} px in the image")
    if a.dry_run:
        print("dry run: nothing saved")
    elif after >= ACCEPT and after > before:
        os.makedirs(os.path.join(config.CONFIG_DIR, "old"), exist_ok=True)
        f = v.cfg["file"]
        shutil.copy(f, os.path.join(config.CONFIG_DIR, "old",
                                    os.path.basename(f)[:-5] + time.strftime("_pre_realign_%Y%m%d-%H%M%S.json")))
        new.save(f)          # the empty-board reference file is kept as it is
        print(f"saved {f}")
    else:
        print(f"NOT saved (needs >= {ACCEPT} and better than before) - empty the board and run calibrate_camera.py")
    v.release()


if __name__ == "__main__":
    main()

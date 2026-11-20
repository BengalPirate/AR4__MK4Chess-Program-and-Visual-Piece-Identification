"""Live view of every board camera side by side, with each camera's saved grid,
for aiming the cameras. Q or ESC quits.

  python live_view.py [--snapshot PREFIX] [--no-grid]
With --snapshot each camera's latest frame is written to PREFIX_<camera>.jpg once a second.
"""
import argparse
import time

import cv2
import numpy as np

from cameras import open_views, tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot")
    ap.add_argument("--no-grid", action="store_true", help="hide the saved grids (e.g. while re-aiming)")
    a = ap.parse_args()
    views = open_views()
    cv2.namedWindow("AR4 cameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AR4 cameras", 1600, 520)
    last = 0
    while True:
        imgs, names = [], []
        for v in views:
            f, _ = v.get()
            if f is None:
                f = np.zeros((720, 1280, 3), np.uint8)
            elif a.snapshot and time.time() - last > 1.0:
                cv2.imwrite(f"{a.snapshot}_{v.key}.jpg", f)
            img = f.copy()
            h, w = img.shape[:2]
            if v.geo is not None and not a.no_grid:
                v.geo.draw(img)
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
            cv2.drawMarker(img, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
            imgs.append(img)
            names.append(f"{v.key}  {v.device}" + ("" if v.geo else "  (not calibrated)"))
        if a.snapshot and time.time() - last > 1.0:
            last = time.time()
        cv2.imshow("AR4 cameras", tile(imgs, names))
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break


if __name__ == "__main__":
    main()

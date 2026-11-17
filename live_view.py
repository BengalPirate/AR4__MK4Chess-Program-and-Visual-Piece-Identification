"""Live camera view with the board grid (if calibrated) and a framing guide.

  python live_view.py [--snapshot PATH] [--no-grid]    Q or ESC quits
The window shows the frame edges in red where the board or pieces would be cut
off. With --snapshot the latest frame is also written to PATH once a second.
"""
import argparse
import time

import cv2

from board_vision import BoardGeometry, open_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot")
    ap.add_argument("--no-grid", action="store_true", help="hide the saved grid (e.g. while re-aiming)")
    a = ap.parse_args()
    cap, dev = open_camera()
    geo = None if a.no_grid else BoardGeometry.load()
    cv2.namedWindow("AR4 camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AR4 camera", 1280, 720)
    last = 0
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        if a.snapshot and time.time() - last > 1.0:
            cv2.imwrite(a.snapshot, f)
            last = time.time()
        img = f.copy()
        h, w = img.shape[:2]
        if geo is not None:
            geo.draw(img)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)
        cv2.putText(img, f"{dev}  {w}x{h}   whole board + tops of far pieces must be inside",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("AR4 camera", img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break


if __name__ == "__main__":
    main()

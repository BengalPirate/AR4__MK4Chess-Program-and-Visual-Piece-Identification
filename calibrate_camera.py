"""Calibrate the board in the camera image (board must be EMPTY).

  python calibrate_camera.py               auto: finds the tile corners, saves
                                           config/board_image.json + empty reference
  python calibrate_camera.py --rank-near-camera 8   if rank 8 is printed nearest the camera
Writes results/calibration.png showing the 64 squares with a1 and h8 marked.
"""
import argparse
import os
import time

import cv2

import config
from board_vision import BoardGeometry, auto_calibrate, detect_markers, open_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-near-camera", type=int, default=1, choices=(1, 8))
    ap.add_argument("--image", help="use a saved image instead of the camera")
    a = ap.parse_args()
    if a.image:
        frame = cv2.imread(a.image)
    else:
        cap, _ = open_camera()
        for _ in range(15):
            ok, frame = cap.read()
        time.sleep(0.2)
        ok, frame = cap.read()
    found = auto_calibrate(frame, a.rank_near_camera)
    if found is None:
        raise SystemExit("Board pattern not found - is the board empty and fully in view?")
    corners, inner = found
    geo = BoardGeometry(corners, (frame.shape[1], frame.shape[0]), detect_markers(frame), inner)
    corners = geo.corners_px.tolist()
    geo.save()
    geo.set_reference(frame)
    img = frame.copy()
    geo.draw(img)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, "calibration.png")
    cv2.imwrite(out, img)
    print(f"corners a1,h1,h8,a8 = {[[round(v) for v in c] for c in corners]}")
    print(f"focal {geo.K[0, 0]:.0f} px, lens k1 {0 if geo.dist is None else geo.dist.ravel()[0]:.3f}, "
          f"fit error {geo.reproj_px if geo.reproj_px is None else round(geo.reproj_px, 2)} px, "
          f"camera at {geo.cam_in_board.round(1)} squares")
    print(f"saved {config.BOARD_IMAGE_FILE}, empty reference, and {out}")


if __name__ == "__main__":
    main()

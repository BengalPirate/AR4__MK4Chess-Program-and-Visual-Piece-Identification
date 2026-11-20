"""Calibrate the board in every camera's image (board must be EMPTY).

  python calibrate_camera.py                 all attached cameras (config.CAMERAS)
  python calibrate_camera.py --camera side   one camera
  python calibrate_camera.py --camera side --image empty.jpg   from a saved frame
Each camera: finds the 49 inner tile corners, fits its camera model, saves its
calibration file + empty-board reference, and writes results/calibration_<camera>.png
with the 64 squares, a1 and h8 marked (check them!).
"""
import argparse
import os

import cv2

import config
from cameras import View


def report(v, geo, frame):
    img = frame.copy()
    geo.draw(img)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, f"calibration_{v.key}.png")
    cv2.imwrite(out, img)
    print(f"[{v.key}] corners a1,h1,h8,a8 = {[[round(c) for c in p] for p in geo.corners_px.tolist()]}")
    print(f"[{v.key}] focal {geo.K[0, 0]:.0f} px, lens k1 {0 if geo.dist is None else geo.dist.ravel()[0]:.3f}, "
          f"fit error {geo.reproj_px if geo.reproj_px is None else round(geo.reproj_px, 2)} px, "
          f"camera at {geo.cam_in_board.round(1)} squares (x = files, y = ranks, z = up)")
    print(f"[{v.key}] saved {v.cfg['file']}, empty reference, and {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", choices=list(config.CAMERAS))
    ap.add_argument("--image", help="use a saved image instead of the camera (needs --camera)")
    a = ap.parse_args()
    keys = [a.camera] if a.camera else list(config.CAMERAS)
    failed = []
    for key in keys:
        try:
            v = View(key, start=not a.image)
        except Exception as e:      # noqa: BLE001
            print(f"[{key}] camera not available: {e}")
            failed.append(key)
            continue
        frame = cv2.imread(a.image) if a.image else v.settled_frame()
        geo = v.calibrate(frame)
        if geo is None:
            print(f"[{key}] board pattern not found - is the board empty and fully in view?")
            cv2.imwrite(os.path.join(config.RESULTS_DIR, f"calibration_{key}_failed.png"), frame)
            failed.append(key)
        else:
            report(v, geo, frame)
        v.release()
    if failed:
        raise SystemExit(f"not calibrated: {', '.join(failed)}")


if __name__ == "__main__":
    main()

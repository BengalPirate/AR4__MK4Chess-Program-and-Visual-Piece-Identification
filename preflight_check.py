"""Check that everything needed for a game is present and working. NOTHING MOVES.

  python preflight_check.py

Checks, in order: Annin kinematics module, Stockfish, classifier, calibration
files, both cameras (found, alignment with the current board), the arm controller
(found, pose = PARK), the gripper Nano (found, resting current). Prints OK / FIX
for each with what to do. Run it on a new computer after cloning, and before a game.
"""
import json
import os
import shutil
import sys
import time

import numpy as np

import config

results = []


def report(name, ok, msg):
    results.append(ok)
    print(f"[{'OK ' if ok else 'FIX'}] {name}: {msg}", flush=True)


def main():
    try:
        import kin
        kin.tip([0, 0, 0, 0, 0, 0])
        report("kinematics", True, f"Annin robot_kinematics from {kin.HMI_DIR}")
    except Exception as e:      # noqa: BLE001
        report("kinematics", False, f"{e} - build ARrobots/robot_kinematics (README Setup), set AR4_HMI_DIR")
        sys.exit(1)

    sf = shutil.which("stockfish") or next((p for p in config.STOCKFISH_PATHS if os.path.exists(p)), None)
    report("stockfish", sf is not None, sf or "not found - apt install stockfish (else a weak built-in engine plays)")

    try:
        from pieces import PieceClassifier
        clf = PieceClassifier()
        report("classifier", True, str(clf.info))
    except Exception as e:      # noqa: BLE001
        clf = None
        report("classifier", False, f"{e} - models/piece_cnn.pt missing?")

    for f in ("board_robot.json", "board_image.json", "board_image_side.json", "joint_limits.json"):
        p = os.path.join(config.CONFIG_DIR, f)
        report(f"config/{f}", os.path.exists(p), "present" if os.path.exists(p) else "MISSING")
    br = json.load(open(os.path.join(config.CONFIG_DIR, "board_robot.json")))
    park = np.array(br["park"]["joints"], float)

    try:
        from cameras import open_views
        views = open_views(list(config.CAMERAS))
        time.sleep(3)
        for v in views:
            f = v.settled_frame()
            al = v.geo.tile_alignment(f) if v.geo is not None else None
            ok = al is not None and al >= config.MIN_TILE_ALIGNMENT
            report(f"camera {v.key}", ok, f"{v.device}, alignment {al:.2f} (need >= {config.MIN_TILE_ALIGNMENT})"
                   + ("" if ok else " - small nudge: python camera_realign.py " + v.key
                      + "; moved a lot: empty board + python calibrate_camera.py"))
        missing = set(config.CAMERAS) - {v.key for v in views}
        for k in missing:
            report(f"camera {k}", False, "not found - plug it in (found by name: " + config.CAMERAS[k]["name"] + ")")
        for v in views:
            v.release()
    except Exception as e:      # noqa: BLE001
        report("cameras", False, str(e))

    try:
        from ar4link import AR4
        arm = AR4(safe=True, j5_window=config.J5_WINDOW)
        js = np.array(list(arm.pose().joints), float)
        if np.abs(js - park).max() < 0.5:
            report("arm", True, f"{arm.port}, at PARK")
        elif np.abs(js).max() < 0.5:
            report("arm", False, "reads all joints 0: just calibrated (then: python go_park.py) or the "
                                 "controller restarted (then: recalibrate, or restore_park.py if the arm "
                                 "has not moved since it was parked)")
        else:
            report("arm", False, f"reads {np.round(js, 1).tolist()}, not PARK - python go_park.py --check")
        arm.close()
    except Exception as e:      # noqa: BLE001
        report("arm", False, f"{e} - USB plugged in? E-stop released? user in group dialout?")

    try:
        from gripper import IDLE_MAX_AMPS, Gripper
        g = Gripper()
        amps = g.idle_amps()
        report("gripper", True, f"{g.port if hasattr(g, 'port') else 'Nano'}, resting {amps:.2f} A "
                                f"(open-check limit {IDLE_MAX_AMPS} A)")
    except Exception as e:      # noqa: BLE001
        report("gripper", False, f"{e} - Nano plugged in?")

    print("\nREADY TO PLAY" if all(results) else "\nNot ready - fix the items marked FIX")


if __name__ == "__main__":
    main()

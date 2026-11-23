"""Move the arm to PARK after a fresh joint calibration (or from wherever it was parked).

  python go_park.py           check, then move to PARK at 10 %
  python go_park.py --check   only check, nothing moves

Use after the Annin HMI auto-calibration (J1-J4 + J6) and the J5 zero in
manual_calib.py: the arm then stands at the calibrated zero pose (upright, gripper
pointing forward) and the game expects it at PARK (config/board_robot.json).

Refuses unless the controller reads the calibrated zero pose (all joints within
0.5 deg of 0) or a pose within 2 deg of PARK - anything else means the angles may
not match the real arm (e.g. the controller restarted). The joint path is
simulated first; the jaw tips must stay above MIN_TIP_Z mm (the board is ~97 mm).
"""
import argparse
import sys

import numpy as np

import config
import kin
from ar4link import AR4
from planner import BoardRobot

MIN_TIP_Z = 140.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--speed", type=int, default=10)
    a = ap.parse_args()
    park = np.array(BoardRobot().park["joints"], float)
    arm = AR4(safe=True, j5_window=config.J5_WINDOW)
    js = np.array(list(arm.pose().joints), float)
    print(f"controller reports {np.round(js, 2).tolist()}")
    if np.abs(js - park).max() < 0.05:
        print("Already at PARK.")
        return
    if not (np.abs(js).max() < 0.5 or np.abs(js - park).max() < 2.0):
        sys.exit("Not the calibrated zero pose and not near PARK - the angles may not match the arm. "
                 "Recalibrate (HMI J1-J4 + J6, J5 in manual_calib.py) before moving.")
    zs = [kin.tip(js + (park - js) * t)[2] for t in np.linspace(0, 1, 201)]
    print(f"path to PARK: lowest jaw-tip height {min(zs):.0f} mm (must be > {MIN_TIP_Z:.0f})")
    if min(zs) < MIN_TIP_Z:
        sys.exit("Path too low - not moving.")
    if a.check:
        print("--check: nothing moved.")
        return
    print(f"Moving to PARK at {a.speed} % - hand on the E-stop")
    arm.move_joints(list(park), a.speed)
    print(f"At PARK: {arm.pose()}")


if __name__ == "__main__":
    main()

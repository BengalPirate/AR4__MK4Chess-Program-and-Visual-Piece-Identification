"""Tell a freshly (re)started arm controller that the arm is standing at PARK.

  python restore_park.py            check, and restore if the controller was reset
  python restore_park.py --check    only report, change nothing

NOTHING MOVES. The Teensy 4.1 is powered over USB: unplugging it (or moving the
cable to another computer) restarts it, and it then reports every joint as 0. If
the arm itself was NOT moved in between - it was left parked, motor power on or
off but nobody pushed it - the joint calibration is still physically valid, and
it is enough to send the controller the park angles (SP command), exactly as the
J5 zero in manual_calib.py does. Used 2026-11-22 after a USB reset; the hover
check afterwards was centred.

Refuses unless the controller reads either the reset pose (all joints 0) or
already the park pose. If the arm WAS moved, do not use this: recalibrate
(Annin HMI auto-cal J1-J4 + J6, then J5 by eye in manual_calib.py).

After restoring, check before playing: python hover_test.py e4 --above 40
(10 %: config.SPEED_PCT) and look that the jaws are centred over e4.
"""
import argparse
import sys

import numpy as np

import config
from ar4link import AR4
from planner import BoardRobot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report, change nothing")
    a = ap.parse_args()
    park = np.array(BoardRobot().park["joints"], float)
    arm = AR4(safe=True, j5_window=config.J5_WINDOW)
    js = np.array(list(arm.pose().joints), float)
    print(f"controller on {arm.port}")
    print(f"  reports  {np.round(js, 2).tolist()}")
    print(f"  PARK     {np.round(park, 3).tolist()}  (config/board_robot.json)")
    if np.abs(js - park).max() < 0.05:
        print("Already at PARK - nothing to do.")
        return
    if np.abs(js).max() > 0.05:
        sys.exit("The controller reads neither the reset pose (all 0) nor PARK: its angles are from "
                 "something else. Do NOT restore - recalibrate (HMI auto-cal J1-J4 + J6, J5 in "
                 "manual_calib.py).")
    print("Controller was reset (all joints 0).")
    if a.check:
        print("--check: nothing changed. Run without --check if the arm has NOT moved since it was parked.")
        return
    reply = arm._cmd("SP" + "".join(f"{L}{v:.3f}" for L, v in zip("ABCDEF", park)) + "G0.000H0.000I0.000")
    if "Done" not in reply:
        sys.exit(f"Set position failed ({reply.strip()})")
    p = arm.pose()
    print(f"Restored: {p}")
    arm.sync_app_config()        # the stock HMI would otherwise send its own saved angles on startup
    print("Now verify with a hover over an empty square at 10 % before playing "
          "(python hover_test.py e4 --above 40).")


if __name__ == "__main__":
    main()

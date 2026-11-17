"""Jog ONE joint by a relative amount (recovery tool, e.g. unwinding J1's cables).

  python jog_joint.py J1 -10          move J1 by -10 deg at 10 % speed
  python jog_joint.py J1 -10 --speed 5

Only the named joint changes; the others are sent their current angles. Steps are
limited to 30 deg. Works even when the controller's absolute angles are wrong
(uncalibrated), because the move is relative to what the controller believes now.
J5 is never jogged here.
"""
import argparse

from ar4link import AR4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joint", choices=["J1", "J2", "J3", "J4", "J6"])
    ap.add_argument("deg", type=float)
    ap.add_argument("--speed", type=int, default=10)
    a = ap.parse_args()
    if abs(a.deg) > 30:
        raise SystemExit("Refused: at most 30 deg per jog")
    arm = AR4(safe=True)
    before = arm.pose()
    js = list(before.joints)
    i = int(a.joint[1]) - 1
    js[i] += a.deg
    print(f"before: {before}")
    after = arm.move_joints(js, a.speed, allow_j5_outside=True)
    print(f"after:  {after}")


if __name__ == "__main__":
    main()

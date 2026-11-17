"""Which squares can the arm reach with the gripper down and J5 in its safe window?

  python reach_check.py                      uses config/board_robot.json (taught board)
  python reach_check.py --near 240 --pitch 47 --length 120
                                             hypothetical board: rank-1 centres at X=near,
                                             board surface at base Z=0, gripper length (mm)
Prints an 8x8 map: '.' reachable at grasp and travel height, 'g' grasp fails,
't' travel height fails, 'X' both.
"""
import argparse
import json
import os
import tempfile

import chess

import config
import kin
from planner import BoardRobot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float)
    ap.add_argument("--pitch", type=float, default=47.0)
    ap.add_argument("--length", type=float, default=120.0)
    ap.add_argument("--yaw", type=float, default=config.GRIPPER_YAW_DEG)
    a = ap.parse_args()
    if a.near is None:
        br = BoardRobot()
        yaw = br.yaw if br.yaw is not None else a.yaw
    else:
        half, far = 3.5 * a.pitch, a.near + 7 * a.pitch
        z = a.length
        cal = {"squares": {"a1": {"x": a.near, "y": half, "z": z}, "h1": {"x": a.near, "y": -half, "z": z},
                           "h8": {"x": far, "y": -half, "z": z}, "a8": {"x": far, "y": half, "z": z}}}
        path = os.path.join(tempfile.mkdtemp(), "b.json")
        with open(path, "w") as f:
            json.dump(cal, f)
        br, yaw = BoardRobot(path), a.yaw
    seed = [0, 30, 0, 0, 60, 0]
    fails = 0
    print(f"travel heights: empty {config.travel_z()} mm, carrying king {config.travel_z('K')} mm")
    for r in range(7, -1, -1):
        row = ""
        for f in range(8):
            x, y, z = br.square_xyz(chess.square(f, r))
            g = kin.ik_down((x, y, z + min(config.GRASP_Z.values())), yaw, seed) is None
            t = kin.ik_down((x, y, z + config.travel_z("K")), yaw, seed) is None
            row += "X" if g and t else "g" if g else "t" if t else "."
            fails += g or t
        print(f" {r + 1}  {' '.join(row)}")
    print("    a b c d e f g h")
    print(f"{fails} square(s) not fully reachable")


if __name__ == "__main__":
    main()

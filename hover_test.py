"""Slow check of the taught board and the grip heights on real pieces.

  python hover_test.py d4 e8 a1                 hover 15 mm above each square's grip
                                                height (piece type read from --piece)
  python hover_test.py e2 --piece P --above 0   go down to the exact grip height
  python hover_test.py e2 --piece P --grip      grip, lift 20 mm, put back, release

Close robot_setup.py / the HMI first (one program per arm port). Every move uses
the same code as a real game (ik_down: straight down, or tilted on b8-g8), at
config.SPEED_PCT. The script stops at each square and waits for Enter; Ctrl+C or
the E-stop to abort. Ends at PARK.
"""
import argparse
import sys
import time

import chess

import config
from ar4link import AR4
from gripper import Gripper
from motion import BoardLock, Executor
from planner import BoardRobot


def wait(msg, secs):
    if sys.stdin.isatty():
        input(msg)
    else:                       # run from a tool without a keyboard: just pause
        print(msg.strip(), f"(pausing {secs:g} s)", flush=True)
        time.sleep(secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("squares", nargs="+")
    ap.add_argument("--piece", default="P", choices=list("PNBRQK"))
    ap.add_argument("--above", type=float, default=15.0, help="mm above the grip height to stop at")
    ap.add_argument("--grip", action="store_true", help="grip the piece, lift 20 mm, put it back")
    ap.add_argument("--squeeze", type=int, help="override config.GRIP_SQUEEZE_DEG for this piece")
    ap.add_argument("--tilt", type=float, help="force this gripper tilt (deg) instead of the rank-8 default")
    ap.add_argument("--yaw", type=float, help="jaw rotation (deg) instead of the taught one")
    ap.add_argument("--height", type=float, help="grip height (mm above the board) instead of config.GRASP_Z")
    ap.add_argument("--pause", type=float, default=6.0, help="seconds to pause per square when not run from a terminal")
    a = ap.parse_args()
    arm = AR4(safe=True, j5_window=config.J5_WINDOW)
    grip = Gripper()
    br = BoardRobot()
    lock = BoardLock()
    ex = Executor(arm, grip, br, lock)
    if not lock.acquire():
        raise SystemExit("Board interlock held by another program")
    zg = config.GRASP_Z[a.piece] if a.height is None else a.height
    ex.yaw = ex.yaw_for(a.piece) if a.yaw is None else a.yaw
    t_empty = config.travel_z()
    try:
        grip.open(config.GRIP_OPEN_PIECE[a.piece])
        for name in a.squares:
            x, y, zb = br.square_xyz(chess.parse_square(name))
            ex.travel()
            ex.min_tilt = ex.tilt_for(name) if a.tilt is None else a.tilt
            ex._goto((x, y, zb + t_empty))
            stop = zb + zg + (0 if a.grip else a.above)
            ex._vertical(x, y, zb + t_empty, stop)
            import kin
            js = ex._joints()
            print(f"{name}: jaw tips {stop - zb:.0f} mm above the board "
                  f"({a.piece} grip height {zg:.0f} mm), tilt {kin.tilt_deg(js):.0f} deg")
            if a.grip:
                sq = config.GRIP_SQUEEZE_DEG[a.piece] if a.squeeze is None else a.squeeze
                ok = grip.hold(config.GRIP_HOLD_DEG[a.piece])
                print(f"  {'piece felt' if ok else 'NOTHING FELT (closed with no load)'}; squeeze {sq}; "
                      f"(angle, amps): {grip.trace}", flush=True)
                if ok:
                    ex._vertical(x, y, stop, stop + 20)
                    wait("  lifted 20 mm - Enter to put it back ", a.pause)
                    ex._vertical(x, y, stop + 20, stop + config.PLACE_EXTRA_Z)
                grip.open(config.GRIP_OPEN_PIECE[a.piece])
            else:
                wait("  check the height / centring, Enter to continue ", a.pause)
            ex._vertical(x, y, zb + (zg if a.grip else stop - zb), zb + t_empty)
        ex.go_park()
    finally:
        lock.release()


if __name__ == "__main__":
    main()

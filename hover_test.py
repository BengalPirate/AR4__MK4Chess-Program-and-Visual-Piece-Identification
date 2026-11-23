"""Slow check of the taught board and the grip heights on real pieces.

  python hover_test.py d4 e8 a1 tray:0          hover 15 mm above each square's (or tray slot's) grip
                                                height (piece type read from --piece)
  python hover_test.py e2 --piece P --above 0   go down to the exact grip height
  python hover_test.py e2 --piece P --grip      grip, lift 20 mm, put back, release
  python hover_test.py a8 e8 h8 --pointer 40    pointed tool 40 mm below the jaw tips:
                                                the point stops 10 mm over each centre

Close robot_setup.py / the HMI first (one program per arm port). Every move uses
the same code as a real game (ik_down: straight down, or tilted on b8-g8), at
--speed percent (default 10). The script stops at each square and waits for Enter; Ctrl+C or
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
    ap.add_argument("--pointer", type=float, default=0.0,
                    help="pointed tool in the jaws, sticking out this many mm below the jaw tips: "
                         "the POINT is brought to --above mm over the square centre (default and minimum 10)")
    ap.add_argument("--pause", type=float, default=6.0, help="seconds to pause per square when not run from a terminal")
    ap.add_argument("--speed", type=int, default=10, help="percent speed for every move (default 10 - a check, not a game)")
    a = ap.parse_args()
    config.SPEED_PCT = config.TRAVEL_SPEED_PCT = a.speed
    if a.pointer:
        # Planning heights are FLANGE heights (straight down), with "jaw tips touching"
        # as the reference: the pointer's length must be ADDED to every height, not
        # only to the tool length (which the tilted maths uses). Grip height 0 = board.
        # (2026-11-17: the first version only lengthened the tool and drove the
        # point ~70 mm into the board on a1/h1.)
        config.TOOL_LENGTH_MM += a.pointer
        config.TRAVEL_CLEARANCE += a.pointer
        a.height = a.pointer
        if a.above == 15.0:
            a.above = 10.0
        if a.above < 10.0:
            raise SystemExit("--pointer: stop at least 10 mm above the board")
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
        if not a.pointer:               # with a pointer the jaws stay closed on it
            grip.open(config.GRIP_OPEN_PIECE[a.piece])
        for name in a.squares:
            x, y, zb = (br.tray_xyz(int(name[5:])) if name.startswith("tray:")
                        else br.square_xyz(chess.parse_square(name)))
            ex.travel()
            ex.min_tilt = ex.tilt_for(name) if a.tilt is None else a.tilt
            if ex.min_tilt > 0 and ex.tilt_for(name) == 0:      # same tilted correction as a game
                dx, dy, dz = br.tilt_correction(x, y)
                x, y, zb = x + dx, y + dy, zb + dz
            ex._approach((x, y, zb + t_empty))
            stop = zb + zg + (0 if a.grip else a.above)
            ex._vertical(x, y, zb + t_empty, stop)
            import kin
            js = ex._joints()
            print(f"{name}: {'POINT' if a.pointer else 'jaw tips'} {stop - zb - a.pointer:.0f} mm above the board "
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

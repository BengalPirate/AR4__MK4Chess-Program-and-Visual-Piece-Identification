"""Measure where the jaws meet each piece type (after changing the foam pads).

  python grip_measure.py e2:P g1:N f1:B h1:R d1:Q e1:K      (10 % speed)

For each square: the arm comes down over the piece exactly as for a real grip
(same approach, same grip height, gripper straight down / tilted like a game), the
jaws close 1 degree at a time while the supply current is logged, and closing
stops at the first of: current above STOP_AMPS, or the piece's configured hold
angle. Then the jaws open, the arm rises, and the next square follows. Nothing is
lifted. Ends at PARK. Results: results/grip_measure_<date>.csv and printed:
contact angle = first step with a clear current rise above the free level.
Keep a hand on the E-stop.
"""
import argparse
import csv
import os
import time

import chess

import config
from ar4link import AR4
from gripper import Gripper
from motion import BoardLock, Executor
from planner import BoardRobot

STOP_AMPS = 0.7          # stop closing here: firm contact (STALL_AMPS is 0.8)
RISE_AMPS = 0.25         # contact = current this much above the free level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="square:piece, e.g. e2:P")
    ap.add_argument("--speed", type=int, default=10)
    a = ap.parse_args()
    config.SPEED_PCT = config.TRAVEL_SPEED_PCT = a.speed
    arm = AR4(safe=True, j5_window=config.J5_WINDOW)
    grip = Gripper()
    br = BoardRobot()
    lock = BoardLock()
    ex = Executor(arm, grip, br, lock)
    if not lock.acquire():
        raise SystemExit("Board interlock held by another program")
    out = os.path.join(config.RESULTS_DIR, f"grip_measure_{time.strftime('%Y%m%d')}.csv")
    new = not os.path.exists(out)
    rows = []
    t_empty = config.travel_z()
    try:
        for item in a.items:
            name, piece = item.split(":")
            piece = piece.upper()
            x, y, zb = br.square_xyz(chess.parse_square(name))
            ex.yaw = ex.yaw_for(piece)
            ex.travel()
            ex.min_tilt = ex.tilt_for(name)
            grip.open(config.GRIP_OPEN_PIECE[piece])
            free = grip.idle_amps()
            ex._approach((x, y, zb + config.travel_z(piece)))
            ex._vertical(x, y, zb + config.travel_z(piece), zb + config.GRASP_Z[piece])
            trace, contact = [], None
            for deg in range(grip.angle - 1, config.GRIP_HOLD_DEG[piece] - 1, -1):
                grip._cmd(f"SV0P{deg}", expect_line=False)
                grip.angle = deg
                time.sleep(0.35)
                amps = sorted(grip.current() for _ in range(3))[1]
                trace.append((deg, round(amps, 2)))
                if contact is None and amps > free + RISE_AMPS:
                    contact = deg
                if amps > STOP_AMPS:
                    break
            print(f"{name} {piece}: free {free:.2f} A, contact at {contact} deg, "
                  f"stopped at {grip.angle} deg; trace {trace}", flush=True)
            rows.append([time.strftime("%F %T"), name, piece, f"{ex.min_tilt:g}", f"{free:.2f}",
                         contact, grip.angle, " ".join(f"{d}:{c}" for d, c in trace)])
            grip.open(config.GRIP_OPEN_PIECE[piece])
            ex._vertical(x, y, zb + config.GRASP_Z[piece], zb + t_empty)
        ex.go_park()
    finally:
        lock.release()
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "square", "piece", "tilt", "free_amps", "contact_deg", "stop_deg", "trace"])
            w.writerows(rows)


if __name__ == "__main__":
    main()

"""Re-centre one piece on its square with the cameras (the set-down check on its own).

  python recentre.py b6 N --tilt 30        simulate, then on the arm with --go
  python recentre.py b6 N --tilt 30 --go

The arm steps aside, reads the piece's offset (offset_model), and if it is more than
config.PLACE_TOL_MM off re-picks it where it stands and sets it down counter-shifted,
up to config.PLACE_FIX_TRIES times (motion.Executor._check_placement). --tilt: the
tilt the piece was last set down with (30 for pieces carried from rank 8), so the
counter-shift uses the same kind of set-down. Ends at PARK.
"""
import argparse
import os
import sys

import config
from motion import BoardLock, DryArm, DryGripper, Executor
from planner import BoardRobot, Op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("square")
    ap.add_argument("piece")
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    br = BoardRobot()
    op = br.locate(Op("place", a.square, a.piece))
    # simulation: a reader that reports the piece 7 mm off, then centred
    seq = [{"dx_mm": -4.0, "dy_mm": 7.0}, {"dx_mm": 0.5, "dy_mm": 0.5}]
    sim = Executor(DryArm(br.park["joints"]), DryGripper(), br, BoardLock(config.LOCK_FILE + ".dry"),
                   log=lambda *x: None, offset_reader=lambda sq: seq.pop(0) if seq else None)
    sim.last_place_tilt = a.tilt
    sim.min_tilt = a.tilt
    sim._check_placement(op)
    sim.go_park()
    print(f"simulated OK: {len(sim.arm.log)} joint moves", flush=True)
    if not a.go:
        return
    from ar4link import AR4
    from grip_check import open_checker
    from gripper import Gripper
    from offset_model import OffsetReader
    check = open_checker()
    ex = Executor(AR4(safe=True, j5_window=config.J5_WINDOW), Gripper(), br, BoardLock(),
                  pick_check=check, offset_reader=OffsetReader(check.views))
    try:
        if not ex.lock.acquire():
            raise SystemExit("board interlock held")
        ex.last_place_tilt = a.tilt
        ex.min_tilt = a.tilt
        ex._check_placement(op)
        ex.go_park()
        print("done", flush=True)
    except Exception as e:      # noqa: BLE001
        print(f"STOPPED: {e}", flush=True)
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

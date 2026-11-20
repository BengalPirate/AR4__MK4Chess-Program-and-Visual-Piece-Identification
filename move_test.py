"""Arm test: move single pieces exactly as a game would (stops midway included).

  python move_test.py a8a4 --board "r6r/8/8/8/8/8/8/8"          plan + dry run only
  python move_test.py a8a4 --board "r6r/8/8/8/8/8/8/8" --go     ... then move the arm

--board is the PHYSICAL position (FEN board part) before the move: the piece on
the source square is the one carried, and every other piece marks a square the
stop-midway search must avoid. No chess rules are applied (any square to any
empty square; a piece on the destination is captured into --tray-slot first).
The plan is built by planner.add_regrips (the same code a game
uses), checked for reach, and dry-run on a simulated arm before anything moves.
Speed config.SPEED_PCT; ends at PARK. Keep a hand on the E-stop.
"""
import argparse

import chess

import config
import kin
from chess_logic import Game
from motion import BoardLock, DryArm, DryGripper, Executor
from planner import BoardRobot, Op, add_regrips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("move", help="source and destination, e.g. a8a4")
    ap.add_argument("--board", required=True, help="physical position before the move (FEN board part)")
    ap.add_argument("--go", action="store_true", help="execute on the arm after the dry run")
    ap.add_argument("--tray-slot", type=int, default=0, help="tray slot for a captured piece")
    ap.add_argument("--no-check", action="store_true", help="no camera check / retry after each pick")
    ap.add_argument("--no-shift", action="store_true", help="no camera offset reading / pick shift before each pick")
    a = ap.parse_args()
    src, dst = a.move[:2], a.move[2:4]
    g = Game(a.board.split()[0] + " w - - 0 1")
    pc = g.board.piece_at(chess.parse_square(src))
    if pc is None:
        raise SystemExit(f"no piece on {src} in --board")
    br = BoardRobot()
    ops = []
    victim = g.board.piece_at(chess.parse_square(dst))
    if victim is not None:          # capture: the piece on dst goes to the tray first
        ops += [br.locate(Op("pick", dst, victim.symbol(), "capture")),
                br.locate(Op("place", f"tray:{a.tray_slot}", victim.symbol(), "to captured tray"))]
    ops += [br.locate(Op("pick", src, pc.symbol())), br.locate(Op("place", dst, pc.symbol()))]
    ops = add_regrips(ops, g, br)
    print("plan:")
    for o in ops:
        print("   ", o, "" if o.tilt is None else f"(picked at tilt {o.tilt:g})")
    if any(o.kind == "pause" for o in ops):
        raise SystemExit("plan needs the operator - not run")
    dry = Executor(DryArm(br.park["joints"]), DryGripper(), br, BoardLock(config.LOCK_FILE + ".dry"),
                   log=lambda *x: None)
    dry.run(ops, park=True)
    js = [j for _, j in dry.arm.log]
    print(f"dry run OK: {len(js)} joint moves, tilt up to {max(kin.tilt_deg(j) for j in js):.0f} deg, "
          f"J4 {min(j[3] for j in js):.0f}..{max(j[3] for j in js):.0f}, "
          f"J5 {min(j[4] for j in js):.0f}..{max(j[4] for j in js):.0f}", flush=True)
    if not a.go:
        return
    from ar4link import AR4
    from gripper import Gripper
    check = reader = None
    if not (a.no_check and a.no_shift):
        from grip_check import open_checker
        check = open_checker()        # cameras confirm each pick; a miss is retried 5 mm along a-h
    if not a.no_shift and config.OFFSET_CORRECTION:
        from offset_model import OffsetReader
        reader = OffsetReader(check.views)    # piece more than 10 mm off -> pick shifted 15 mm that way
    ex = Executor(AR4(safe=True, j5_window=config.J5_WINDOW), Gripper(), br, BoardLock(),
                  pick_check=None if a.no_check else check, offset_reader=reader)
    try:
        ex.run(ops, park=True)
        print("done", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()           # print it now: the os._exit below would swallow it
        raise
    finally:
        if check is not None:      # camera threads block in read(); leave without tearing them down
            import os
            import sys
            sys.stdout.flush()
            os._exit(0 if sys.exc_info()[0] is None else 1)


if __name__ == "__main__":
    main()

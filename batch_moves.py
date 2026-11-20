"""Arm test: a batch of single-piece moves, exactly as a game moves pieces.

  python batch_moves.py --board "RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr" a7a5 b7b5 ...        dry run
  python batch_moves.py --board ... a7a5 b7b5 ... --go                                               on the arm

--board is the PHYSICAL position (FEN board part) before the first move; it is
updated after every move. No chess rules (any square to any EMPTY square). Each
move is planned like a game move (planner.add_regrips: stops midway when needed),
the whole batch is dry-run on a simulated arm first, then executed with the
cameras on: offset-shifted picks, the lift-and-confirm check (stepping aside when
the square is unclear), camera-aimed retries and the set-down check. Stops at the
first problem (arm left in place); ends at PARK.
"""
import argparse
import os
import sys
import time

import chess

import config
from chess_logic import Game
from motion import BoardLock, DryArm, DryGripper, Executor, MotionError
from planner import BoardRobot, Op, add_regrips


def plan(br, board_fen, move):
    """(ops, board after) for one move on the physical board."""
    src, dst = move[:2], move[2:4]
    b = chess.Board(board_fen + " w - - 0 1")
    pc = b.piece_at(chess.parse_square(src))
    if pc is None:
        raise SystemExit(f"{move}: no piece on {src}")
    if b.piece_at(chess.parse_square(dst)) is not None:
        raise SystemExit(f"{move}: {dst} is occupied")
    ops = [br.locate(Op("pick", src, pc.symbol())), br.locate(Op("place", dst, pc.symbol()))]
    ops = add_regrips(ops, Game(board_fen + " w - - 0 1"), br)
    if any(o.kind == "pause" for o in ops):
        raise SystemExit(f"{move}: plan needs the operator - not run")
    b.remove_piece_at(chess.parse_square(src))
    b.set_piece_at(chess.parse_square(dst), pc)
    return ops, b.board_fen()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moves", nargs="+")
    ap.add_argument("--board", required=True)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    br = BoardRobot()
    plans, fen = [], a.board.split()[0]
    for m in a.moves:
        ops, fen = plan(br, fen, m)
        plans.append((m, ops))
    dry = Executor(DryArm(br.park["joints"]), DryGripper(), br, BoardLock(config.LOCK_FILE + ".dry"),
                   log=lambda *x: None)
    for m, ops in plans:
        dry.run(ops, park=False)
    dry.go_park()
    print(f"dry run OK: {len(plans)} moves, {len(dry.arm.log)} joint moves; board after: {fen}", flush=True)
    if not a.go:
        return
    from ar4link import AR4
    from grip_check import open_checker
    from gripper import Gripper
    from offset_model import OffsetReader
    check = open_checker()
    ex = Executor(AR4(safe=True, j5_window=config.J5_WINDOW), Gripper(), br, BoardLock(),
                  pick_check=check, offset_reader=OffsetReader(check.views))
    done = 0
    t0 = time.time()
    try:
        for m, ops in plans:
            print(f"== {m} ({done + 1}/{len(plans)})", flush=True)
            ex.run(ops, park=False)
            done += 1
        ex.go_park()
        print(f"batch done: {done}/{len(plans)} moves in {(time.time() - t0) / 60:.1f} min", flush=True)
    except MotionError as e:
        print(f"STOPPED at {plans[done][0]} after {done} moves: {e}", flush=True)
    except Exception as e:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED at {plans[done][0]} after {done} moves: {e}", flush=True)
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

"""Arm test: move pieces from the start position N ranks forward (default 2).

  python shift_test.py                   all four groups, in a safe order
  python shift_test.py --only white      white pawns then white back rank
  python shift_test.py --only black
  python shift_test.py --dry             plan + reachability only, no motion

White moves toward the arm (rank up), black away from it (rank down). Pawns go
first so every destination is empty. Same pick / place code as a game
(motion.Executor), at config.SPEED_PCT. Stops at the first problem (piece not
felt, unreachable, E-stop). Ends at PARK.
"""
import argparse

import chess

import config
from motion import BoardLock, DryArm, DryGripper, Executor
from planner import BoardRobot, Op


def moves(only, n):
    b = chess.Board()
    groups = {"white": [(1, +n), (0, +n)], "black": [(6, -n), (7, -n)]}
    out = []
    for side in (["white", "black"] if only is None else [only]):
        for rank, step in groups[side]:
            for f in range(8):
                sq = chess.square(f, rank)
                out.append((chess.square_name(sq), chess.square_name(chess.square(f, rank + step)),
                            b.piece_at(sq).symbol()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["white", "black"])
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--count", type=int, help="only the first N moves")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N moves (already done)")
    a = ap.parse_args()
    br = BoardRobot()
    if a.dry:
        arm, grip = DryArm(br.park["joints"]), DryGripper()
    else:
        from ar4link import AR4
        from gripper import Gripper
        arm, grip = AR4(safe=True, j5_window=config.J5_WINDOW), Gripper()
    ex = Executor(arm, grip, br, BoardLock())
    plan = moves(a.only, a.ranks)[a.skip:][:a.count]
    ops = []
    for src, dst, p in plan:
        ops += [br.locate(Op("pick", src, p)), br.locate(Op("place", dst, p))]
    ex.preflight(ops)
    print(f"{len(plan)} moves, all points reachable", flush=True)
    for i, (src, dst, p) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {p} {src} -> {dst}", flush=True)
        ex.run(ops[2 * i - 2: 2 * i], park=(i == len(plan)))
    print("done", flush=True)


if __name__ == "__main__":
    main()

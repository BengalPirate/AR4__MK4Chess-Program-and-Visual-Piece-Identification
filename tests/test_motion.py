"""Simulated pick-and-place: gripper always points down with the jaws at a fixed
direction, J5 stays in its safe window, the path stays near straight lines, and
unreachable plans are refused before moving.
Uses Annin's kinematics (needs the AR4 HMI 6.7 folder, see kin.py)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
kin = pytest.importorskip("kin")
import config  # noqa: E402
from chess_logic import Game  # noqa: E402
from motion import BoardLock, DryArm, DryGripper, Executor, MotionError  # noqa: E402
from planner import BoardRobot, plan_move  # noqa: E402

YAW = 45.0
GRIPPER_L = 120.0     # assumed flange-to-fingertip length for the simulation


def make_robot(tmp_path, near_x=235.0, pitch=47.0):
    # Robot plays White from its side: rank 1 nearest the robot, a-file on the robot's left (+Y)
    far_x = near_x + 7 * pitch
    half = 3.5 * pitch
    z = GRIPPER_L     # flange height when the fingertips touch the board (board at base z = 0)
    cal = {"squares": {"a1": {"x": near_x, "y": half, "z": z}, "h1": {"x": near_x, "y": -half, "z": z},
                       "h8": {"x": far_x, "y": -half, "z": z}, "a8": {"x": far_x, "y": half, "z": z}},
           "tray": {"x": near_x, "y": -half - 1.5 * pitch, "z": z - 10},
           "reserve": {"x": near_x - 10, "y": half + 1.5 * pitch, "z": z - 10},
           "yaw": YAW, "park": {"joints": [0, 20, 10, 0, 60, 0]}}
    p = tmp_path / "board_robot.json"
    p.write_text(json.dumps(cal))
    return BoardRobot(str(p))


def run(tmp_path, br, game, uci):
    arm = DryArm(br.park["joints"])
    ex = Executor(arm, DryGripper(), br, BoardLock(str(tmp_path / "lock")), log=lambda *_: None)
    ops = [br.locate(o) for o in plan_move(game, game.parse_move(uci))]
    ex.run(ops)
    return arm


def test_gripper_down_fixed_jaw_direction_j5_window(tmp_path):
    br = make_robot(tmp_path)
    g = Game()
    arm = run(tmp_path, br, g, "e2e4")
    assert len(arm.log) > 10
    lo, hi = config.J5_WINDOW
    for _, js in arm.log[:-1]:              # last move is PARK (joint target from calibration)
        assert kin.tool_axis_down(js)
        assert lo <= js[4] <= hi
        # jaw direction over the board: J6 - J1 constant (= the pawn's yaw; the
        # two-jaw gripper is symmetric, so +-180 is the same direction)
        yaw = config.GRIP_YAW_PIECE.get("P", YAW)
        assert ((js[5] - js[0] - yaw + 90) % 180) - 90 == pytest.approx(0, abs=0.01)


def test_descent_is_vertical(tmp_path):
    br = make_robot(tmp_path)
    arm = run(tmp_path, br, Game(), "g1f3")
    x, y, _ = br.square_xyz(6)              # g1
    near = [kin.fk(js)[:3] for _, js in arm.log]
    low = [p for p in near if p[2] < br.board_z(x, y) + config.travel_z() - 1 and
           np.hypot(p[0] - x, p[1] - y) < 30]
    assert low, "no descent at g1"
    assert max(np.hypot(p[0] - x, p[1] - y) for p in low) < 2.0


def test_unreachable_plan_refused_before_moving(tmp_path):
    br = make_robot(tmp_path, near_x=30)   # board pushed into the base
    arm = DryArm(br.park["joints"])
    ex = Executor(arm, DryGripper(), br, BoardLock(str(tmp_path / "lock")), log=lambda *_: None)
    g = Game()
    ops = [br.locate(o) for o in plan_move(g, g.parse_move("e2e4"))]
    with pytest.raises(MotionError, match="out of reach"):
        ex.run(ops)
    assert arm.log == []                    # nothing was sent


def test_interlock_blocks_second_arm(tmp_path):
    br = make_robot(tmp_path)
    other = BoardLock(str(tmp_path / "lock"))
    assert other.acquire()
    arm = DryArm(br.park["joints"])
    ex = Executor(arm, DryGripper(), br, BoardLock(str(tmp_path / "lock")), log=lambda *_: None)
    g = Game()
    with pytest.raises(MotionError, match="interlock"):
        ex.run([br.locate(o) for o in plan_move(g, g.parse_move("e2e4"))])
    other.release()


def test_j5_outside_window_refused(tmp_path):
    br = make_robot(tmp_path)
    arm = DryArm([0, 20, 10, 0, -101.0, 0])  # J5 folded past the measured safe range
    with pytest.raises(MotionError, match="J5"):
        Executor(arm, DryGripper(), br, BoardLock(str(tmp_path / "lock")))


def test_missed_grip_is_retried_5mm_along_a_to_h(tmp_path):
    import kin
    br = make_robot(tmp_path)
    g = Game()
    seen = iter(["occupied", "empty"])       # first grip misses, the retry takes it
    ex = Executor(DryArm(br.park["joints"] if br.park else [0, 30, 0, 0, 60, 0]), DryGripper(), br,
                  BoardLock(str(tmp_path / "l")), log=lambda *a: None, pick_check=lambda sq: next(seen))
    ops = [br.locate(o) for o in plan_move(g, g.parse_move("e2e4"))]
    ex.run(ops, park=False)
    assert [k for k, _ in ex.grip.log].count("hold") == 2          # missed once, retried once
    # the arm visited the square centre and a point GRIP_RETRY_MM further along a->h
    e2 = np.array(ops[0].xyz[:2])
    along = br.file_step[:2] / np.linalg.norm(br.file_step[:2])
    xy = [np.array(kin.vfk(js)[:2]) for _, js in ex.arm.log]
    want = e2 + config.GRIP_RETRY_FIRST * config.GRIP_RETRY_MM * along
    assert any(np.linalg.norm(p - e2) < 0.3 for p in xy)
    assert any(np.linalg.norm(p - want) < 0.3 for p in xy)


def test_grip_missed_twice_stops(tmp_path):
    br = make_robot(tmp_path)
    g = Game()
    ex = Executor(DryArm(br.park["joints"] if br.park else [0, 30, 0, 0, 60, 0]), DryGripper(), br,
                  BoardLock(str(tmp_path / "l")), log=lambda *a: None, pick_check=lambda sq: "occupied")
    with pytest.raises(MotionError, match="missed"):
        ex.run([br.locate(o) for o in plan_move(g, g.parse_move("e2e4"))], park=False)
    assert [k for k, _ in ex.grip.log].count("hold") == 1 + config.GRIP_RETRIES


def test_undecided_pick_is_retried_not_stopped(tmp_path):
    """2026-11-22 (h4 capture): a pick the cameras cannot confirm (undecided even from
    aside) is retried - with no contact felt the jaws open just above the piece's top
    over its square - instead of stopping the game."""
    import kin
    br = make_robot(tmp_path)
    g = Game()
    seen = iter([None, None, "empty"])        # over the square, from aside, then the retry
    logs = []
    ex = Executor(DryArm(br.park["joints"] if br.park else [0, 30, 0, 0, 60, 0]), DryGripper(), br,
                  BoardLock(str(tmp_path / "l")), log=logs.append, pick_check=lambda sq: next(seen))
    ops = [br.locate(o) for o in plan_move(g, g.parse_move("e2e4"))]
    ex.run(ops, park=False)
    assert [k for k, _ in ex.grip.log].count("hold") == 2
    assert any("no contact felt" in m for m in logs)
    # it opened over e2 no lower than 5 mm above a standing pawn's top
    e2 = np.array(ops[0].xyz[:2])
    zb = ops[0].xyz[2]
    lows = [kin.vfk(js)[2] for _, js in ex.arm.log
            if np.linalg.norm(np.array(kin.vfk(js)[:2]) - e2) < 0.5]
    assert min(lows) >= zb + config.GRASP_Z["P"] - 0.5      # the real grip, nothing lower


def test_undecided_pick_still_stops_after_all_tries(tmp_path):
    br = make_robot(tmp_path)
    g = Game()
    ex = Executor(DryArm(br.park["joints"] if br.park else [0, 30, 0, 0, 60, 0]), DryGripper(), br,
                  BoardLock(str(tmp_path / "l")), log=lambda *a: None, pick_check=lambda sq: None)
    with pytest.raises(MotionError, match="undecided"):
        ex.run([br.locate(o) for o in plan_move(g, g.parse_move("e2e4"))], park=False)
    assert [k for k, _ in ex.grip.log].count("hold") == 1 + config.GRIP_RETRIES

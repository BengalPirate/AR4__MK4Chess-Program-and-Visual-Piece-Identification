"""Chess logic, move inference and pick-and-place planning tests (no hardware).

Run:  python -m pytest -q tests
"""
import json
import os
import sys

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from chess_logic import LABEL_INDEX, LABELS, Engine, Game, square_labels  # noqa: E402
from planner import BoardRobot, plan_move  # noqa: E402


def play(game, *moves):
    for t in moves:
        game.push(game.parse_move(t))


# ---------------------------------------------------------------- rules
def test_turn_tracking_and_illegal_moves():
    g = Game()
    assert g.board.turn == chess.WHITE
    with pytest.raises(ValueError, match="turn"):
        g.parse_move("e7e5")                      # black piece on white's turn
    with pytest.raises(ValueError, match="cannot move"):
        g.parse_move("e2e5")                      # pawn cannot jump three
    with pytest.raises(ValueError, match="no piece"):
        g.parse_move("e3e4")
    with pytest.raises(ValueError, match="not a move"):
        g.parse_move("hello")
    play(g, "e2e4")
    assert g.board.turn == chess.BLACK
    play(g, "Nf6")                                # SAN accepted too
    assert g.board.turn == chess.WHITE


def test_pinned_piece_rejected():
    g = Game("4k3/8/8/8/1b6/8/3N4/4K3 w - - 0 1")   # knight d2 pinned by bishop b4
    with pytest.raises(ValueError, match="check"):
        g.parse_move("d2f3")


def test_castling_rights_preserved_beyond_image():
    g = Game()
    play(g, "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6")
    assert g.board.has_kingside_castling_rights(chess.WHITE)
    play(g, "h1g1", "f8c5", "g1h1", "d7d6")       # rook leaves and returns: right is gone
    # the image looks like the rook never moved, but the game state remembers
    assert not g.board.has_kingside_castling_rights(chess.WHITE)
    with pytest.raises(ValueError):
        g.parse_move("e1g1")


def test_en_passant_window():
    g = Game()
    play(g, "e2e4", "a7a6", "e4e5", "d7d5")
    assert g.board.ep_square == chess.D6
    m = g.parse_move("e5d6")
    assert g.board.is_en_passant(m)
    play(g, "h2h3", "h7h6")                       # one move later the right has expired
    g2 = Game()
    play(g2, "e2e4", "a7a6", "e4e5", "d7d5", "h2h3", "h7h6")
    with pytest.raises(ValueError):
        g2.parse_move("e5d6")


def test_promotion_defaults_to_queen_and_tray():
    g = Game("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    m = g.parse_move("b7b8")
    assert m.promotion == chess.QUEEN
    m = g.parse_move("b7b8n")
    assert m.promotion == chess.KNIGHT


def test_undo_restores_tray():
    g = Game()
    play(g, "e2e4", "d7d5", "e4d5")
    assert g.tray == ["p"]
    g.undo()
    assert g.tray == []


def test_engine_returns_legal_move():
    e = Engine()
    try:
        g = Game()
        m = e.choose(g.board)
        assert m in g.board.legal_moves
        # mate in one is found
        g = Game("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        assert e.choose(g.board) == chess.Move.from_uci("a1a8")
    finally:
        e.close()


# ---------------------------------------------------------------- vision -> move
def fake_logp(labels, conf=0.9, noisy=()):
    """Classifier output that puts `conf` on the true label (uniform elsewhere)."""
    p = np.full((64, len(LABELS)), (1 - conf) / (len(LABELS) - 1))
    for sq, c in enumerate(labels):
        p[sq][LABEL_INDEX[c]] = conf
    for sq, c in noisy:                         # square read as something else
        p[sq][:] = (1 - conf) / (len(LABELS) - 1)
        p[sq][LABEL_INDEX[c]] = conf
    return np.log(p)


def test_infer_move_simple():
    g = Game()
    after = chess.Board()
    after.push_uci("e2e4")
    inf = g.infer_move(fake_logp(square_labels(after)))
    assert inf["move"] == chess.Move.from_uci("e2e4")
    assert g.confident(inf)


def test_infer_no_move():
    g = Game()
    inf = g.infer_move(fake_logp(square_labels(g.board)))
    assert inf["move"] is None


def test_infer_survives_one_misread_square():
    g = Game()
    after = chess.Board()
    after.push_uci("g1f3")
    # the knight on f3 is misread as a bishop; still the best legal explanation
    inf = g.infer_move(fake_logp(square_labels(after), noisy=[(chess.F3, "B")]))
    assert inf["move"] == chess.Move.from_uci("g1f3")
    assert not g.confident(inf)                 # ... but flagged for confirmation


def test_infer_castling_and_capture():
    g = Game()
    play(g, "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6")
    after = g.board.copy()
    after.push_uci("e1g1")
    inf = g.infer_move(fake_logp(square_labels(after)))
    assert inf["move"] == chess.Move.from_uci("e1g1")
    g2 = Game()
    play(g2, "e2e4", "d7d5")
    after = g2.board.copy()
    after.push_uci("e4d5")
    assert g2.infer_move(fake_logp(square_labels(after)))["move"] == chess.Move.from_uci("e4d5")


def test_inconsistent_board_is_not_confident():
    g = Game()
    labels = square_labels(g.board)
    labels[chess.E2] = "."
    labels[chess.D2] = "."                      # two pawns vanished: no legal move does that
    inf = g.infer_move(fake_logp(labels))
    assert not g.confident(inf)


# ---------------------------------------------------------------- planning
@pytest.fixture
def robot(tmp_path):
    cal = {"squares": {"a1": {"x": 250, "y": 140, "z": 100}, "h1": {"x": 250, "y": -140, "z": 100},
                       "h8": {"x": 530, "y": -140, "z": 100}, "a8": {"x": 530, "y": 140, "z": 100}},
           "tray": {"x": 250, "y": -220, "z": 70}, "reserve": {"x": 600, "y": 0, "z": 70}}
    p = tmp_path / "board_robot.json"
    p.write_text(json.dumps(cal))
    return BoardRobot(str(p))


def test_square_mapping(robot, monkeypatch):
    monkeypatch.setattr(config, "RANK_OFFSET_MM", {})   # pure corner interpolation
    monkeypatch.setattr(config, "SQUARE_OFFSET_MM", {})
    monkeypatch.setattr(config, "AIM_CORRECTION", False)  # measured map of the real arm, not this test board
    assert robot.square_xyz(chess.A1) == pytest.approx((250, 140, 100))
    assert robot.square_xyz(chess.H8) == pytest.approx((530, -140, 100))
    x, y, z = robot.square_xyz(chess.E4)
    assert x == pytest.approx(250 + 3 * 40) and y == pytest.approx(140 - 4 * 40)
    assert robot.pitch == pytest.approx(40)


def test_tray_slots_step_away_from_board(robot):
    s0, s8 = robot.tray_xyz(0), robot.tray_xyz(8)
    assert s8[1] < s0[1]                        # second column further from the board


def kinds(ops):
    return [(o.kind, o.where) for o in ops]


def test_plan_quiet_move():
    g = Game()
    assert kinds(plan_move(g, g.parse_move("e2e4"))) == [("pick", "e2"), ("place", "e4")]


def test_plan_capture():
    g = Game()
    play(g, "e2e4", "d7d5")
    ops = plan_move(g, g.parse_move("e4d5"))
    assert kinds(ops) == [("pick", "d5"), ("place", "tray:0"), ("pick", "e4"), ("place", "d5")]


def test_plan_en_passant():
    g = Game()
    play(g, "e2e4", "a7a6", "e4e5", "d7d5")
    ops = plan_move(g, g.parse_move("e5d6"))
    assert kinds(ops) == [("pick", "d5"), ("place", "tray:0"), ("pick", "e5"), ("place", "d6")]


def test_plan_castling():
    g = Game()
    play(g, "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6")
    ops = plan_move(g, g.parse_move("O-O"))
    assert kinds(ops) == [("pick", "e1"), ("place", "g1"), ("pick", "h1"), ("place", "f1")]


def test_promotion_without_tray_piece_keeps_pawn_as_standin():
    g = Game("4k3/8/8/8/8/8/1p6/4K3 b - - 0 1")
    ops = plan_move(g, g.parse_move("b2b1q"))
    assert kinds(ops) == [("pick", "b2"), ("place", "b1")]       # no swap
    assert ops[0].piece == "p"                                     # gripped as a pawn
    g.push(g.parse_move("b2b1q"))
    assert g.board.piece_at(chess.B1).symbol() == "q"              # it IS a queen in the game
    assert g.standins == {chess.B1: "p"} and g.tray == []
    assert g.physical_labels()[chess.B1] == "p"                    # cameras expect a pawn
    g.push(g.parse_move("e1e2"))
    ops = plan_move(g, g.parse_move("b1h7"))                       # moves as a queen
    assert ops[0].piece == "p"                                     # still gripped as a pawn
    after = Game(g.board.fen())
    after.standins = dict(g.standins)
    after.push(after.parse_move("b1h7"))
    inf = g.infer_move(fake_logp(after.physical_labels()))         # vision sees a pawn move b1->h7
    assert inf["move"] == chess.Move.from_uci("b1h7")
    g.push(g.parse_move("b1h7"))
    assert g.standins == {chess.H7: "p"}
    # a captured stand-in goes to the tray as a pawn; undo restores it
    g2 = Game("4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1")
    g2.push(g2.parse_move("b2b1q"))
    ops = plan_move(g2, g2.parse_move("a1b1"))
    assert kinds(ops)[:2] == [("pick", "b1"), ("place", "tray:0")] and ops[0].piece == "p"
    g2.push(g2.parse_move("a1b1"))
    assert g2.tray == ["p"] and g2.standins == {}
    g2.undo()
    assert g2.standins == {chess.B1: "p"} and g2.tray == []


def test_plan_promotion_swaps_from_tray():
    g = Game("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    g.tray = ["N"]                              # a captured white knight is available
    ops = plan_move(g, g.parse_move("b7b8n"))
    assert kinds(ops) == [("pick", "b7"), ("place", "tray:1"), ("pick", "tray:0"), ("place", "b8")]
    g.push(g.parse_move("b7b8n"))
    assert g.tray == [None, "P"] and g.standins == {}


def test_promotion_swaps_captured_queen_back_and_tracks_tray():
    # Black's queen was captured earlier (tray slot 0); Black then promotes on a1.
    g = Game("4k3/8/8/8/8/8/p7/4K2R b - - 0 1")
    g.tray = ["q", "P"]
    ops = plan_move(g, g.parse_move("a2a1q"))
    assert kinds(ops) == [("pick", "a2"), ("place", "tray:2"), ("pick", "tray:0"), ("place", "a1")]
    g.push(g.parse_move("a2a1q"))
    assert g.tray == [None, "P", "p"] and g.standins == {}      # queen out, pawn in
    g.push(g.parse_move("e1e2"))
    ops = plan_move(g, g.parse_move("a1h1"))     # next capture fills the emptied slot 0
    assert kinds(ops)[:2] == [("pick", "h1"), ("place", "tray:0")]
    g.push(g.parse_move("a1h1"))
    assert g.tray == ["R", "P", "p"]
    g.undo()
    g.undo()
    g.undo()
    assert g.tray == ["q", "P"]                  # undo restores everything


def test_capture_promotion_with_no_queen_in_tray():
    g = Game("1n2k3/P7/8/8/8/8/8/R3K3 w - - 0 1")
    ops = plan_move(g, g.parse_move("a7b8q"))    # captures a knight while promoting
    assert kinds(ops) == [("pick", "b8"), ("place", "tray:0"), ("pick", "a7"), ("place", "b8")]
    g.push(g.parse_move("a7b8q"))
    assert g.tray == ["n"] and g.standins == {chess.B8: "P"}


def test_tilted_carry_gets_a_stop_midway():
    from planner import add_regrips

    class FakeBR:                       # 50 mm squares, board Z 100
        def square_xyz(self, sq):
            return (500.0 - 50 * chess.square_rank(sq), 50.0 * chess.square_file(sq), 100.0)

    def located(ops):
        for o in ops:
            o.xyz = FakeBR().square_xyz(chess.parse_square(o.where))
        return ops
    g = Game("r3k3/8/8/8/8/8/8/4K3 b - - 0 1")
    ops = add_regrips(located(plan_move(g, g.parse_move("a8a1"))), g, FakeBR(),
                      reach=lambda xyz, p, tilt, place: True)
    assert [(o.kind, o.where) for o in ops] == [("pick", "a8"), ("place", "a4"), ("pick", "a4"), ("place", "a1")]
    ops = add_regrips(located(plan_move(g, g.parse_move("a8a5"))), g, FakeBR(),
                      reach=lambda xyz, p, tilt, place: True)
    assert len(ops) == 2                 # reachable tilted, not rank 1-2: no stop
    g = Game("r3k3/8/8/8/8/8/8/4K3 b - - 0 1")
    ops = add_regrips(located(plan_move(g, g.parse_move("a8a5"))), g, FakeBR(),
                      reach=lambda xyz, p, tilt, place: tilt == 0 or xyz[0] < 280)   # a5 (X 300) not tilted
    assert [(o.kind, o.where) for o in ops][1:3] in ([("place", "a6"), ("pick", "a6")], [("place", "a7"), ("pick", "a7")])


def test_moves_to_rank8_stop_only_from_ranks_1_2():
    from planner import add_regrips

    class FakeBR:
        def square_xyz(self, sq):
            return (500.0 - 50 * chess.square_rank(sq), 50.0 * chess.square_file(sq), 100.0)

    def run(fen, uci):
        g = Game(fen)
        ops = plan_move(g, g.parse_move(uci))
        for o in ops:
            o.xyz = FakeBR().square_xyz(chess.parse_square(o.where))
        return add_regrips(ops, g, FakeBR(), reach=lambda xyz, p, tilt, place: True)
    ops = run("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1a8")       # rank 1 -> 8: stop midway
    assert [(o.kind, o.where, o.tilt) for o in ops] == [
        ("pick", "a1", None), ("place", "a4", None), ("pick", "a4", 30.0), ("place", "a8", None)]
    ops = run("4k3/8/8/8/R7/8/8/4K3 w - - 0 1", "a4a8")       # rank 4 -> 8: picked tilted, no stop
    assert [(o.kind, o.where, o.tilt) for o in ops] == [("pick", "a4", 30.0), ("place", "a8", None)]


def test_plan_rejects_illegal():
    g = Game()
    with pytest.raises(ValueError):
        plan_move(g, chess.Move.from_uci("e2e5"))


class LocBR:
    """50 mm squares, board Z 100, with locate (robot squares)."""
    def square_xyz(self, sq):
        return (500.0 - 50 * chess.square_rank(sq), 50.0 * chess.square_file(sq), 100.0)

    def locate(self, o):
        o.xyz = self.square_xyz(chess.parse_square(o.where))
        return o


def test_files_mirrored_for_the_game(monkeypatch):
    """Chess squares map to robot squares with the files mirrored: chess e1 (white
    king) is robot d1; the engine's e2e4 is picked on robot d2 and placed on robot d4."""
    from planner import robot_name, robot_sq, to_robot
    monkeypatch.setattr(config, "CHESS_FILES_MIRRORED", True)
    assert robot_name("e1") == "d1" and robot_name("a8") == "h8" and robot_name("tray:2") == "tray:2"
    assert robot_sq(chess.E1) == chess.D1
    g = Game()
    ops = to_robot(plan_move(g, g.parse_move("e2e4")), g.board, LocBR())
    assert [(o.kind, o.where) for o in ops] == [("pick", "d2"), ("place", "d4")]


def test_illegal_move_taken_back(monkeypatch):
    """The piece moved illegally is picked where it stands and put back on its square."""
    from planner import plan_restore
    monkeypatch.setattr(config, "CHESS_FILES_MIRRORED", True)
    g = Game()
    ops = plan_restore(g.board, chess.B1, chess.B3, "N", LocBR())   # knight b1 -> b3: illegal
    assert [(o.kind, o.where, o.piece) for o in ops] == [("pick", "g3", "N"), ("place", "g1", "N")]

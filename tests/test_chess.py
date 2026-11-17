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


def test_plan_promotion_uses_reserve_or_tray():
    g = Game("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    ops = plan_move(g, g.parse_move("b7b8q"))
    assert kinds(ops) == [("pick", "b7"), ("place", "tray:0"), ("pick", "reserve"), ("place", "b8")]
    g.tray = ["N"]                              # a captured white knight is available
    ops = plan_move(g, g.parse_move("b7b8n"))
    assert ("pick", "tray:0") in kinds(ops)
    g.tray = []
    ops = plan_move(g, g.parse_move("b7b8r"))   # no spare rook anywhere -> operator
    assert ops[-1].kind == "pause"


def test_plan_rejects_illegal():
    g = Game()
    with pytest.raises(ValueError):
        plan_move(g, chess.Move.from_uci("e2e5"))

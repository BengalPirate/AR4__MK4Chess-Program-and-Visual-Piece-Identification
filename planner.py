"""Chess move -> robot pick-and-place plan, and the board-to-robot mapping.

Robot calibration (config/board_robot.json, written by board_teach.py): the
gripper fingertips were jogged down until they touched the board at the centre
of a1, h1, h8 and a8 (robot base frame, mm, flange position with the gripper
pointing straight down, jaws at a fixed direction). Every square centre is a bilinear blend of
those four points, so a board that is slightly rotated or not perfectly square
to the robot still maps correctly. The board surface is the plane through the
four touch heights. Captured pieces go to a tray of slots beside the board
(taught origin, 8 slots per column, one square pitch apart).

Every plan is a list of Op steps. Each pick/place expands, in motion.py, to:
  travel height -> above target -> slow vertical descent -> grip/release ->
  vertical retreat -> travel height.
"""
import json
import os
from dataclasses import dataclass, field

import chess
import numpy as np

import config


@dataclass
class Op:
    kind: str                 # "pick", "place" or "pause"
    where: str                # "e2", "tray:3", "reserve" or "" for pause
    piece: str = ""           # FEN letter of the piece being handled
    note: str = ""
    xyz: tuple = field(default=None, repr=False)   # filled by BoardRobot.locate

    def __str__(self):
        if self.kind == "pause":
            return f"PAUSE  {self.note}"
        s = f"{self.kind.upper():5s}  {self.piece} @ {self.where:7s}"
        if self.xyz is not None:
            s += f"  (X {self.xyz[0]:.1f} Y {self.xyz[1]:.1f} board Z {self.xyz[2]:.1f})"
        return s + (f"  - {self.note}" if self.note else "")


def plan_move(game, move):
    """Ops that make `move` physically, before it is pushed to the game.

    Captures:   the captured piece goes to the next tray slot first.
    En passant: the captured pawn is on the moving pawn's start rank, not on the
                destination square.
    Castling:   king first (two squares), then the rook jumps over it.
    Promotion:  the pawn goes to the tray; the new piece comes from the tray if a
                piece of that kind was captured earlier, else from the reserve
                spot (operator must put a spare queen there) - or a pause if the
                promotion piece is not a queen and none is in the tray.
    """
    b = game.board
    if move not in b.legal_moves:
        raise ValueError(game.why_illegal(move))
    name = chess.square_name
    pc = b.piece_at(move.from_square).symbol()
    ops = []
    tray_next = len(game.tray)
    cap_sq = game.captured_square(move)
    if cap_sq is not None:
        victim = b.piece_at(cap_sq).symbol()
        note = "en passant capture" if b.is_en_passant(move) else "capture"
        ops += [Op("pick", name(cap_sq), victim, note),
                Op("place", f"tray:{tray_next}", victim, "to captured tray")]
        tray_next += 1
    if b.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) == 6
        r_from = chess.square(7 if kingside else 0, rank)
        r_to = chess.square(5 if kingside else 3, rank)
        rook = b.piece_at(r_from).symbol()
        ops += [Op("pick", name(move.from_square), pc, "castling: king"),
                Op("place", name(move.to_square), pc),
                Op("pick", name(r_from), rook, "castling: rook"),
                Op("place", name(r_to), rook)]
    elif move.promotion:
        new = chess.Piece(move.promotion, b.turn).symbol()
        ops += [Op("pick", name(move.from_square), pc, "promotion: pawn off the board"),
                Op("place", f"tray:{tray_next}", pc)]
        # captured pieces of the right kind and colour can come back from the tray
        src = None
        for i, s in enumerate(game.tray):
            if s == new:
                src = f"tray:{i}"
                break
        if src is None and new.upper() == "Q":
            src = "reserve"
        if src is None:
            ops.append(Op("pause", "", new, f"operator: place a {new} on {name(move.to_square)}"))
        else:
            ops += [Op("pick", src, new, "promotion piece"),
                    Op("place", name(move.to_square), new)]
    else:
        ops += [Op("pick", name(move.from_square), pc),
                Op("place", name(move.to_square), pc)]
    return ops


class BoardRobot:
    """Square name / tray slot -> robot X, Y and board-surface Z (mm)."""

    def __init__(self, path=config.BOARD_ROBOT_FILE):
        if not os.path.exists(path):
            raise RuntimeError(f"{path} not found - teach the board with board_teach.py")
        with open(path) as f:
            self.cal = json.load(f)
        sq = self.cal["squares"]
        missing = [s for s in ("a1", "h1", "h8", "a8") if s not in sq]
        if missing:
            raise RuntimeError(f"board_robot.json is missing taught squares {missing}")
        self.c = {k: np.array([sq[k]["x"], sq[k]["y"], sq[k]["z"]]) for k in ("a1", "h1", "h8", "a8")}
        pts = np.array(list(self.c.values()))
        A = np.c_[pts[:, 0], pts[:, 1], np.ones(4)]
        self.plane = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]   # z = a x + b y + c
        self.file_step = ((self.c["h1"] - self.c["a1"]) + (self.c["h8"] - self.c["a8"])) / 14
        self.rank_step = ((self.c["a8"] - self.c["a1"]) + (self.c["h8"] - self.c["h1"])) / 14
        self.pitch = float(np.linalg.norm(self.file_step[:2]) + np.linalg.norm(self.rank_step[:2])) / 2
        self.yaw = self.cal.get("yaw")
        self.park = self.cal.get("park")

    def board_z(self, x, y):
        a, b, c = self.plane
        return a * x + b * y + c

    def square_xyz(self, square):
        f, r = chess.square_file(square), chess.square_rank(square)
        u, v = f / 7, r / 7
        c = self.c
        p = (1 - u) * (1 - v) * c["a1"] + u * (1 - v) * c["h1"] + u * v * c["h8"] + (1 - u) * v * c["a8"]
        dx, dy = config.SQUARE_OFFSET_MM.get(chess.square_name(square),
                                             config.RANK_OFFSET_MM.get(r + 1, (0.0, 0.0)))
        p = p + np.array([dx, dy, 0.0])
        return (float(p[0]), float(p[1]), float(self.board_z(p[0], p[1])))

    def tray_xyz(self, slot):
        t = self.cal.get("tray")
        if not t:
            raise RuntimeError("No tray taught - teach a tray origin in board_teach.py")
        o = np.array([t["x"], t["y"]])
        # columns run parallel to the files, stepping away from the board
        centre = (self.c["a1"] + self.c["h8"])[:2] / 2
        away = np.sign(np.dot(o - centre, self.file_step[:2])) or 1.0
        p = o + (slot % 8) * self.rank_step[:2] + (slot // 8) * away * self.file_step[:2]
        return (float(p[0]), float(p[1]), float(t["z"]))   # tray surface (mat) height

    def reserve_xyz(self):
        r = self.cal.get("reserve")
        if not r:
            raise RuntimeError("No reserve spot taught (spare queen for promotion)")
        return (r["x"], r["y"], r["z"])

    def locate(self, op):
        if op.kind == "pause":
            return op
        if op.where.startswith("tray:"):
            op.xyz = self.tray_xyz(int(op.where[5:]))
        elif op.where == "reserve":
            op.xyz = self.reserve_xyz()
        else:
            op.xyz = self.square_xyz(chess.parse_square(op.where))
        return op

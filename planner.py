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
import math
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
    tilt: float = None        # pick only: tilt to pick at instead of the square's own (stop midway)

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
    Promotion:  if the tray holds that exact piece (captured earlier), the pawn
                goes to the tray and that piece comes back onto the square; else
                the pawn just moves and stays as a stand-in for the new piece
                (Game.standins). Tray slots come from Game.tray_after, the same
                bookkeeping Game.push applies.
    Stand-ins are handled as the pawn they physically are (grip height, tray).
    """
    b = game.board
    if move not in b.legal_moves:
        raise ValueError(game.why_illegal(move))
    name = chess.square_name
    pc = game.physical(move.from_square)       # a stand-in pawn is gripped as a pawn
    _, _, steps = game.tray_after(move)       # the same tray bookkeeping Game.push applies
    ins = [st for st in steps if st[0] == "in"]
    ops = []
    cap_sq = game.captured_square(move)
    if cap_sq is not None:
        _, slot, victim = ins.pop(0)
        note = "en passant capture" if b.is_en_passant(move) else "capture"
        ops += [Op("pick", name(cap_sq), victim, note),
                Op("place", f"tray:{slot}", victim, "to captured tray")]
    if b.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) == 6
        r_from = chess.square(7 if kingside else 0, rank)
        r_to = chess.square(5 if kingside else 3, rank)
        rook = game.physical(r_from)
        ops += [Op("pick", name(move.from_square), pc, "castling: king"),
                Op("place", name(move.to_square), pc),
                Op("pick", name(r_from), rook, "castling: rook"),
                Op("place", name(r_to), rook)]
    elif move.promotion and ins:                # swap: pawn to the tray, piece back from it
        _, slot, _ = ins.pop(0)
        _, src, new = next(st for st in steps if st[0] == "out")
        ops += [Op("pick", name(move.from_square), pc, "promotion: pawn off the board"),
                Op("place", f"tray:{slot}", pc)]
        where = "reserve" if src == "reserve" else f"tray:{src}"
        ops += [Op("pick", where, new, "promotion piece"),
                Op("place", name(move.to_square), new)]
    else:
        note = ""
        if move.promotion:
            new = chess.Piece(move.promotion, b.turn).symbol()
            note = f"promotion: the pawn stays as a stand-in {new} (none in the tray)"
        ops += [Op("pick", name(move.from_square), pc, note),
                Op("place", name(move.to_square), pc)]
    return ops


def tilt_for(where):
    """Minimum gripper tilt (deg) for a pick/place location: rank 8 is always worked
    tilted back (config.RANK8_TILT_DEG); everything else straight down."""
    return config.RANK8_TILT_DEG if len(where) == 2 and where[1] == "8" else 0.0


def _robot_reach(br, xyz, piece, tilt, place):
    """Can the gripper reach this pick/place at this tilt, at every height it uses
    (travel with and without the piece, grip / release)?"""
    import kin
    x, y, zb = xyz
    p = piece.upper()
    yaw = br.jaw_yaw + config.GRIP_YAW_PIECE.get(p, 0.0)
    seed = br.park["joints"] if br.park else [0, 30, 0, 0, 60, 0]
    grip = config.GRASP_Z[p] + (config.PLACE_EXTRA_Z if place else 0.0)
    lo, hi = config.J5_WINDOW
    m = config.TILT_J5_MARGIN_DEG if tilt > 0 else 0.0
    for dz in (config.travel_z(p), config.travel_z(), grip):
        js = kin.ik_down((x, y, zb + dz), yaw, seed, tilt)
        if js is None or not lo + m <= js[4] <= hi - m:
            return False
    return True


def add_regrips(ops, game, br, reach=None):
    """A piece picked tilted (rank 8) is carried at that tilt - it sits in the jaws
    at that angle, so it is also set down at it. Before ranks 1-2
    (config.REGRIP_RANKS), or where the destination cannot be reached at that tilt
    with J5 at least TILT_J5_MARGIN_DEG inside its window, insert a stop midway: set the
    piece down tilted on a square that is empty at that moment, then pick it up
    again straight down and carry it on straight down. Ops must be located
    (BoardRobot.locate). The stop square is the empty square of ranks 3-7 nearest
    the middle of the carry that is reachable both ways. The reverse too: a piece
    picked straight and going to rank 8 is set down straight midway and picked up
    again tilted, so it is never set down tipped in the jaws."""
    reach = reach or (lambda xyz, piece, tilt, place: _robot_reach(br, xyz, piece, tilt, place))
    occupied = {sq for sq in range(64) if game.board.piece_at(sq) is not None}
    # a piece on ranks 3-7 going to rank 8 is simply picked tilted (no stop needed)
    for i, op in enumerate(ops[:-1]):
        nxt = ops[i + 1]
        if (op.kind == "pick" and op.tilt is None and len(op.where) == 2 and nxt.kind == "place"
                and tilt_for(nxt.where) > 0 and tilt_for(op.where) == 0
                and int(op.where[1]) not in config.REGRIP_RANKS
                and reach(op.xyz, op.piece, tilt_for(nxt.where), False)):
            op.tilt = tilt_for(nxt.where)
    out, carry_tilt, src_xy = [], 0.0, None
    for op in ops:
        sq = chess.parse_square(op.where) if len(op.where) == 2 else None
        if op.kind == "pick":
            if sq is not None:
                occupied.discard(sq)
            carry_tilt = tilt_for(op.where) if op.tilt is None else op.tilt
            src_xy = np.array(op.xyz[:2]) if op.xyz is not None else None
        elif op.kind == "place":
            tilt = max(tilt_for(op.where), carry_tilt)
            far = len(op.where) == 2 and int(op.where[1]) in config.REGRIP_RANKS
            if carry_tilt == 0 and tilt_for(op.where) > 0:
                # picked straight, going to rank 8 (worked tilted): set it down straight
                # midway and pick it up again tilted, so it is not set down tipped
                st = _stop_square(br, occupied, sq, src_xy, op, reach, place_tilt=0.0,
                                  pick_tilt=tilt_for(op.where))
                if st is None:
                    out.append(Op("pause", "", op.piece, f"operator: no free square to tilt the "
                                                         f"gripper on the way to {op.where}"))
                else:
                    name, xyz = st
                    out += [Op("place", name, op.piece, "stop midway: set down straight", xyz=xyz),
                            Op("pick", name, op.piece, "stop midway: picked up tilted", xyz=xyz,
                               tilt=tilt_for(op.where))]
            elif carry_tilt > 0 and (far or not reach(op.xyz, op.piece, tilt, True)):
                st = _stop_square(br, occupied, sq, src_xy, op, reach, place_tilt=carry_tilt, pick_tilt=0.0)
                if st is None:
                    out.append(Op("pause", "", op.piece, f"operator: no free square to straighten the "
                                                         f"gripper on the way to {op.where}"))
                else:
                    name, xyz = st
                    out += [Op("place", name, op.piece, "stop midway: set down tilted", xyz=xyz),
                            Op("pick", name, op.piece, "stop midway: picked up straight down", xyz=xyz,
                               tilt=0.0)]
                    carry_tilt = 0.0
            if sq is not None:
                occupied.add(sq)
            carry_tilt = 0.0
        out.append(op)
    return out


def _stop_square(br, occupied, dest_sq, src_xy, op, reach, place_tilt, pick_tilt):
    """(name, xyz) of the empty square of ranks 3-7 nearest the middle of the carry
    where the piece can be set down at place_tilt and picked up at pick_tilt."""
    mid = (src_xy + np.array(op.xyz[:2])) / 2 if src_xy is not None else np.array(op.xyz[:2])
    best = None
    for cand in range(chess.A3, chess.H7 + 1):
        if cand in occupied or cand == dest_sq:
            continue
        xyz = br.square_xyz(cand)
        d = float(np.linalg.norm(np.array(xyz[:2]) - mid))
        if best is not None and d >= best[0]:
            continue
        if reach(xyz, op.piece, place_tilt, True) and reach(xyz, op.piece, pick_tilt, False):
            best = (d, chess.square_name(cand), xyz)
    return None if best is None else best[1:]


def plan_robot_move(game, move, br):
    """plan_move, located on the robot, with the midway stops add_regrips needs."""
    return add_regrips([br.locate(o) for o in plan_move(game, move)], game, br)


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
        self._jaw_yaw = None

        self.park = self.cal.get("park")

    @property
    def jaw_yaw(self):
        """Jaw rotation (deg, the `yaw` of kin.ik_down) that puts the jaws exactly
        parallel to the board's square lines - measured from the taught corners, so
        a board rotated a few degrees on the table is followed. Closest to 0 of the
        parallel settings (the jaws are symmetric, so +-90 / 180 grip alike)."""
        if self._jaw_yaw is None:
            import kin
            d = self.rank_step[:2] / np.linalg.norm(self.rank_step[:2])
            j0, j1 = kin._jaw_ref(0.0)[:2], kin._jaw_ref(10.0)[:2]
            sign = 1.0 if (j0[0] * j1[1] - j0[1] * j1[0]) > 0 else -1.0   # +yaw turns the jaws this way
            ang = math.degrees(math.atan2(j0[0] * d[1] - j0[1] * d[0], j0 @ d))  # j0 -> d, ccw
            ang = (ang + 45.0) % 90.0 - 45.0                                  # nearest parallel line
            self._jaw_yaw = round(sign * ang, 2)
        return self._jaw_yaw

    def board_uv(self, x, y):
        """Approximate board coordinates (file, rank index 0..7) of a robot X, Y."""
        A = np.c_[self.file_step[:2], self.rank_step[:2]]
        return np.linalg.solve(A, np.array([x, y]) - self.c["a1"][:2])

    def tilt_correction(self, x, y):
        """(dx, dy, dz) to add to a straight-down target when the gripper is TILTED
        there (not rank 8, which has its own tilted corners): the arm model's error
        between the two poses, measured by teaching a8/h8/a4/h4 tilted. Interpolated
        along the files, and between ranks 4 and 8 (held at the rank-4 value below
        rank 4). Zero until a4 and h4 are taught tilted."""
        st = self.cal.get("squares_tilted", {})
        if not all(k in st for k in ("a8", "h8", "a4", "h4")):
            return np.zeros(3)

        def corr(k):
            sq = chess.parse_square(k)
            straight = np.array(self._straight_xyz(sq))
            return np.array([st[k]["x"], st[k]["y"], st[k]["z"]]) - straight
        u, v = self.board_uv(x, y)
        fu = float(np.clip(u / 7, -0.2, 1.3))
        fv = float(np.clip((v - 3) / 4, 0.0, 1.0))
        c4 = (1 - fu) * corr("a4") + fu * corr("h4")
        c8 = (1 - fu) * corr("a8") + fu * corr("h8")
        return (1 - fv) * c4 + fv * c8

    def board_z(self, x, y):
        a, b, c = self.plane
        return a * x + b * y + c

    def _straight_xyz(self, square):
        """Straight-down target: bilinear between the four straight-taught corners."""
        f, r = chess.square_file(square), chess.square_rank(square)
        u, v = f / 7, r / 7
        c = self.c
        p = (1 - u) * (1 - v) * c["a1"] + u * (1 - v) * c["h1"] + u * v * c["h8"] + (1 - u) * v * c["a8"]
        return (float(p[0]), float(p[1]), float(self.board_z(p[0], p[1])))

    def square_xyz(self, square):
        f, r = chess.square_file(square), chess.square_rank(square)
        st = self.cal.get("squares_tilted", {})
        if r == 7 and "a8" in st and "h8" in st:
            # rank 8 taught in its tilted working pose: interpolate between those
            a8, h8 = (np.array([st[k]["x"], st[k]["y"], st[k]["z"]]) for k in ("a8", "h8"))
            p = a8 + (f / 7) * (h8 - a8)
            dx, dy = config.TILTED_SQUARE_OFFSET_MM.get(chess.square_name(square), (0.0, 0.0))
            rx, ry = getattr(config, "RANK8_TILTED_OFFSET_MM", (0.0, 0.0))
            cx, cy = self._aim_fix(square, along_file=False)
            return (float(p[0] + dx + rx + cx), float(p[1] + dy + ry + cy), float(p[2]))
        u, v = f / 7, r / 7
        c = self.c
        p = (1 - u) * (1 - v) * c["a1"] + u * (1 - v) * c["h1"] + u * v * c["h8"] + (1 - u) * v * c["a8"]
        dx, dy = config.SQUARE_OFFSET_MM.get(chess.square_name(square),
                                             config.RANK_OFFSET_MM.get(r + 1, (0.0, 0.0)))
        p = p + np.array([dx, dy, 0.0])
        cx, cy = self._aim_fix(square)
        p = p + np.array([cx, cy, 0.0])
        return (float(p[0]), float(p[1]), float(self.board_z(p[0], p[1])))

    def _aim_fix(self, square, along_file=True):
        """Robot-frame (dx, dy) that cancels the MEASURED aim error at this square
        (config/aim_correction.json, board mm [toward h, toward rank 8])."""
        if not getattr(config, "AIM_CORRECTION", False):
            return 0.0, 0.0
        if not hasattr(self, "_aim_err"):
            try:
                with open(config.AIM_CORRECTION_FILE) as fh:
                    self._aim_err = json.load(fh)["error_mm"]
            except (OSError, ValueError, KeyError):
                self._aim_err = {}
        e = self._aim_err.get(chess.square_name(square))
        if not e:
            return 0.0, 0.0
        uh = self.file_step[:2] / np.linalg.norm(self.file_step[:2])
        u8 = self.rank_step[:2] / np.linalg.norm(self.rank_step[:2])
        k = float(getattr(config, "AIM_CORRECTION_SCALE", 1.0))
        v = -k * (e[0] * uh + (e[1] if along_file else 0.0) * u8)
        return float(v[0]), float(v[1])

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

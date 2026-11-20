"""Did the pick actually take the piece? Ask the cameras, two ways.

1. The GRIPPER, in the air (the operator's idea). The arm photographs its empty
   jaws at the lifted height on the way in, then grips and returns to exactly the
   same pose. In the region just below the jaw tips - where a held piece hangs,
   along the tool axis (tilted on rank 8) - the two pictures can only differ by
   the piece. Clear difference = held; none = the jaws are empty.
2. The SQUARE: the piece's square should now look empty (classifier, both views
   fused; occupancy is its most reliable output).

3. The GRIP CURRENT: a rise at the end of the close means the jaws met something.

motion.Executor.pick combines them: the jaws hold a piece if the current rose
(it overrules the side camera) or the side camera says so; they are empty only if
the side camera says so AND the current did not rise. Only the SQUARE seen empty
confirms a pick; undecided -> re-checked from over the destination. Held -> go on; empty and the
piece still on its square -> missed; empty and the square empty -> the piece was
knocked away -> stop for the operator.
"""
import numpy as np
import cv2

import config
from chess_logic import LABELS
from pieces import fuse

EMPTY = LABELS.index(".")
# Which camera's below-the-jaws difference decides, and its thresholds. Measured
# 2026-11-17 (rook, tilted at h5/h8): side camera held 15.0-17.2, empty 3.7-8.6;
# the front camera overlapped (held 9.5-15.6, empty 7.7-14.9), so it is only logged.
DECIDING_CAMERA = "side"
HELD_DIFF = 12.0            # at or above: a piece hangs in the jaws
EMPTY_DIFF = 10.0           # at or below: the jaws are empty
# gripper current at the end of the close: at or above = closed on something.
# 2026-11-19 (rook): good grips ended at 0.81-1.51 A; two jaw-on-the-crown misses
# (h8 crush, h7 launch) ended at 0.46 A and passed the old 0.45 A mark.
GRIP_RISE_AMPS = 0.65


class PickChecker:
    def __init__(self, views, clf, frames=2):
        self.views, self.clf, self.k = views, clf, frames

    # ------------------------------------------------------------ the square
    def p_empty(self, square):
        """Probability (both cameras fused) that the square is empty now."""
        per = []
        for v in self.views:
            fr = v.frames(self.k)
            if not fr or v.geo is None:
                continue
            per.append(self.clf.logp([[v.geo.crops(f)[square]] for f in fr]))
        if not per:
            return None
        return float(np.exp(fuse(per)[0, EMPTY]))

    def __call__(self, square):
        """'empty' (piece gone), 'occupied' (still there) or None (undecided)."""
        p = self.p_empty(square)
        if p is None:
            return None
        return "empty" if p > 0.7 else "occupied" if p < 0.3 else None

    # ------------------------------------------------------------ the gripper
    def snapshot(self):
        """Current frame of every camera (the arm must be standing still)."""
        out = {}
        for v in self.views:
            fr = v.frames(2)
            if fr:
                out[v.key] = fr[-1]
        return out

    def _region(self, v, br, tip, axis, hang_mm):
        """Image box (x0, y0, x1, y1) of the space a held piece fills below the tips."""
        A = np.c_[br.file_step[:2], br.rank_step[:2]]
        pts = []
        for s in np.linspace(4.0, max(6.0, hang_mm - 4.0), 5):
            p = np.asarray(tip) + s * axis
            u, w = np.linalg.solve(A, p[:2] - br.c["a1"][:2])
            h = (p[2] - (br.board_z(p[0], p[1]) - config.TOOL_LENGTH_MM)) / config.SQUARE_MM
            for du in (-0.18, 0.18):
                pts.append([u + 0.5 + du, w + 0.5, h])
        px = v.geo.project(pts)
        x0, y0 = np.floor(px.min(0)).astype(int)
        x1, y1 = np.ceil(px.max(0)).astype(int)
        H, W = v.geo.frame_size[1], v.geo.frame_size[0]
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 6 or y1 - y0 < 6:
            return None
        return x0, y0, x1, y1

    def held(self, before, br, tip, axis, hang_mm):
        """True (piece in the jaws), False (jaws empty) or None (cannot tell), plus
        the per-camera differences for the log. tip: jaw tip position (robot mm,
        true tip, not planning coords); axis: unit tool axis pointing down."""
        after = self.snapshot()
        diffs = {}
        for v in self.views:
            if v.key not in before or v.key not in after or v.geo is None:
                continue
            box = self._region(v, br, tip, axis, hang_mm)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            lab = [cv2.cvtColor(cv2.GaussianBlur(im[y0:y1, x0:x1], (5, 5), 0).astype(np.float32) / 255,
                                cv2.COLOR_BGR2Lab) for im in (before[v.key], after[v.key])]
            diffs[v.key] = float(np.mean(np.linalg.norm(lab[0] - lab[1], axis=2)))
        d = diffs.get(DECIDING_CAMERA)
        if d is None:
            return None, diffs
        if d >= HELD_DIFF:
            return True, diffs
        if d <= EMPTY_DIFF:
            return False, diffs
        return None, diffs


# per piece type (light pawns never reach the rook-based 0.65 A; 0.45 is PROVISIONAL
# until results/grips.csv has real pawn grips)
GRIP_RISE_AMPS_PIECE = {"P": 0.45}


def grip_rise(trace, piece=None):
    """Did the gripper current rise at the end of the close (jaws met something)?"""
    tail = [a for _, a in (trace or [])[-3:]]
    lim = GRIP_RISE_AMPS_PIECE.get((piece or "").upper(), GRIP_RISE_AMPS)
    return bool(tail) and max(tail) >= lim


def open_checker():
    """PickChecker on all attached cameras with the current classifier."""
    from cameras import open_views
    from pieces import PieceClassifier
    return PickChecker(open_views(), PieceClassifier())

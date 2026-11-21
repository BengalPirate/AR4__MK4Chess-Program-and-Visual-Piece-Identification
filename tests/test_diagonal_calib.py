"""Corner-on (diagonal) camera calibration: synthetic board warped to the 2026-11-20
front-camera geometry; the found a1/h1/h8/a8 must match the true corners."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board_vision import auto_calibrate  # noqa: E402

TRUE = {"a1": (370, 410), "h1": (770, 700), "h8": (1060, 370), "a8": (660, 195)}


def synthetic(corners):
    S = 60
    board = np.full((8 * S + 2 * S, 8 * S + 2 * S), 200, np.uint8)       # light margin
    for x in range(8):
        for y in range(8):
            dark = (x + y) % 2 == 0                                      # a1 is dark
            r0 = S + (7 - y) * S
            board[r0:r0 + S, S + x * S:S + x * S + S] = 40 if dark else 220
    src = np.float32([[S, S + 8 * S], [S + 8 * S, S + 8 * S], [S + 8 * S, S], [S, S]])  # a1 h1 h8 a8
    dst = np.float32([corners[k] for k in ("a1", "h1", "h8", "a8")])
    H = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(board, H, (1280, 720), borderValue=120)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_diagonal_h1_nearest():
    found = auto_calibrate(synthetic(TRUE), "h1")
    assert found is not None
    for k, (fx, fy) in zip(("a1", "h1", "h8", "a8"), found[0]):
        assert np.hypot(fx - TRUE[k][0], fy - TRUE[k][1]) < 4, (k, fx, fy)


def test_diagonal_a8_nearest():
    # the same picture rotated 180 deg: a8 is now the lowest corner
    rot = {k: (1280 - x, 720 - y) for k, (x, y) in TRUE.items()}
    found = auto_calibrate(cv2.rotate(synthetic(TRUE), cv2.ROTATE_180), "h1")
    assert found is not None
    found_a8 = auto_calibrate(cv2.rotate(synthetic(TRUE), cv2.ROTATE_180), "a8")
    for k, (fx, fy) in zip(("a1", "h1", "h8", "a8"), found_a8[0]):
        assert np.hypot(fx - rot[k][0], fy - rot[k][1]) < 4, (k, fx, fy)

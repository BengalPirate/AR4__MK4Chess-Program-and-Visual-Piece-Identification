"""Where does a piece really stand on its square? Both cameras, no markers.

Each calibrated camera knows where every point of the board plane appears in its
image. The empty-board reference (stored top-down at calibration) is warped back
into the camera image, so every live pixel that looks at the board can be compared
with the empty board: what differs is a piece. Pieces are round, so a piece's
outline is symmetric about its own vertical axis. Just above the base (a few mm
up, where it stands against the board) the midpoint of the outline on an image
row is where that axis crosses the row.

One camera only fixes the piece's position ACROSS its line of sight (the depth
is poorly defined): the front camera, looking along the files, measures the
offset along the ranks; the side camera, looking along the ranks, the other
direction. All rows of both cameras are combined in one small least-squares fit
of the offset (du, dv) in squares from the square centre. Rows where the outline
is too wide (merged with a neighbour in front) or too narrow are dropped; with too
few good rows from either camera the result says so instead of guessing.

  python centering.py e2 d7 ...      live: offsets of the pieces on those squares,
                                     writes results/centering_<camera>.png
"""
import chess
import cv2
import numpy as np

import config

BASE_RADIUS_SQ = 0.33          # piece base radius in squares (~15 mm)
ROW_HEIGHTS_SQ = (0.08, 0.13, 0.18, 0.23, 0.28)   # rows just above the base (4-13 mm)
GRID_BAND_SQ = 0.05            # ignore +-this (squares) around the board's grid lines
MIN_EDGE = 12.0                # minimum outline edge strength (Sobel units)
MIN_ROWS_PER_VIEW = 2


class EmptyView:
    """One camera's empty board, aligned pixel for pixel with its live image.

    Every image pixel is traced back to the board plane through the full camera
    model (lens distortion included), so the stored top-down empty board can be
    resampled exactly into the camera's view. Pieces are found by their OUTLINES:
    vertical edges in the live image that the empty board does not have. (Colour
    does not work: ivory pieces and the beige squares look alike, and the room
    light drifts.) The board's own grid lines are masked out, as their positions
    are known exactly."""

    def __init__(self, geo, frame_shape):
        h, w = frame_shape[:2]
        self.geo = geo
        R, _ = cv2.Rodrigues(geo.rvec)
        t = np.asarray(geo.tvec, float).reshape(3)
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        n = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2), geo.K,
                                geo.dist).reshape(-1, 2)
        rays = np.c_[n, np.ones(len(n))] @ R
        cam = -R.T @ t
        s = -cam[2] / rays[:, 2]
        P = cam + rays * s[:, None]                    # board point (squares) seen by each pixel
        W = geo.WARP
        mx = (P[:, 0] * W).reshape(h, w).astype(np.float32)
        my = ((8 - P[:, 1]) * W).reshape(h, w).astype(np.float32)
        self.image = cv2.remap(geo.reference, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bx, by = P[:, 0].reshape(h, w), P[:, 1].reshape(h, w)
        inside = (s.reshape(h, w) > 0) & (bx > 0.03) & (bx < 7.97) & (by > 0.03) & (by < 7.97)
        near_line = (np.abs(bx - np.round(bx)) < GRID_BAND_SQ) | (np.abs(by - np.round(by)) < GRID_BAND_SQ)
        self.usable = inside & ~near_line

    def edges(self, frame):
        """Strength of vertical outline edges not present on the empty board."""
        g = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        live = np.abs(cv2.Sobel(g(frame), cv2.CV_32F, 1, 0, ksize=3))
        empty = cv2.dilate(np.abs(cv2.Sobel(g(self.image), cv2.CV_32F, 1, 0, ksize=3)), np.ones((3, 5), np.uint8))
        return np.clip(live - 1.5 * empty, 0, None) * self.usable


def _rows(view, edges, sq):
    """[(z, image x of the piece axis, image y)] for one square: on each row just
    above the base, the centre about which the outline edges are most symmetric."""
    geo = view.geo
    f, r = chess.square_file(sq) + 0.5, chess.square_rank(sq) + 0.5
    out = []
    for z in ROW_HEIGHTS_SQ:
        c = geo.project([[f, r, z]])[0]
        y = int(round(c[1]))
        if not 1 <= y < edges.shape[0] - 1:
            continue
        e = geo.project([[f - BASE_RADIUS_SQ, r, z], [f + BASE_RADIUS_SQ, r, z],
                         [f, r - BASE_RADIUS_SQ, z], [f, r + BASE_RADIUS_SQ, z]])
        half = max(abs(e[1, 0] - e[0, 0]), abs(e[3, 0] - e[2, 0])) / 2      # base half-width (px)
        x0 = c[0]
        lo, hi = int(x0 - 2.5 * half), int(x0 + 2.5 * half) + 1
        if lo < 0 or hi > edges.shape[1]:
            continue
        prof = edges[y - 1:y + 2, lo:hi].mean(axis=0)
        if prof.max() < MIN_EDGE:
            continue
        best = None
        ks = np.arange(int(0.45 * half), int(1.5 * half) + 1)
        for cc in np.arange(x0 - 0.9 * half, x0 + 0.9 * half + 0.01, 0.5):
            i = cc - lo
            left, right = np.interp(i - ks, np.arange(len(prof)), prof), np.interp(i + ks, np.arange(len(prof)), prof)
            score = float(np.max(np.minimum(left, right)))       # one edge on each side, mirrored
            if best is None or score > best[0]:
                best = (score, cc)
        if best[0] >= MIN_EDGE:
            out.append((z, best[1], y))
    return out


def measure(views_frames, sq, empties=None):
    """views_frames: [(key, geometry, frame)]. Returns dict(ok, du, dv (squares, along
    files / ranks), dx_mm, dy_mm, rows {key: n}, why)."""
    eqs = []
    rows_used = {}
    for key, geo, frame in views_frames:
        ev = (empties or {}).get(key) or EmptyView(geo, frame.shape)
        rows = _rows(ev, ev.edges(frame), sq)
        rows_used[key] = len(rows)
        if len(rows) >= MIN_ROWS_PER_VIEW:
            eqs += [(geo, z, mx) for z, mx, _ in rows]
    good = [k for k, n in rows_used.items() if n >= MIN_ROWS_PER_VIEW]
    res = {"ok": False, "du": 0.0, "dv": 0.0, "dx_mm": 0.0, "dy_mm": 0.0, "rows": rows_used}
    if len(good) < 2:
        res["why"] = "base not clearly visible in both cameras " + str(rows_used)
        return res
    f, r = chess.square_file(sq) + 0.5, chess.square_rank(sq) + 0.5

    def resid(p):
        return np.array([geo.project([[f + p[0], r + p[1], z]])[0][0] - mx for geo, z, mx in eqs])
    p = np.zeros(2)
    for _ in range(6):                                   # Gauss-Newton, 2 unknowns
        r0 = resid(p)
        J = np.stack([(resid(p + d) - r0) / 1e-3 for d in (np.array([1e-3, 0]), np.array([0, 1e-3]))], 1)
        step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        p = p + step
        if np.abs(step).max() < 1e-4:
            break
    rms = float(np.sqrt(np.mean(resid(p) ** 2)))
    res.update(ok=rms < 4.0, du=float(p[0]), dv=float(p[1]),
               dx_mm=float(p[0] * config.SQUARE_MM), dy_mm=float(p[1] * config.SQUARE_MM), rms_px=rms)
    if not res["ok"]:
        res["why"] = f"cameras disagree (fit error {rms:.1f} px)"
    return res


def draw(img, geo, sq, res, key):
    f, r = chess.square_file(sq) + 0.5, chess.square_rank(sq) + 0.5
    c = geo.project([[f, r, 0]])[0]
    m = geo.project([[f + res["du"], r + res["dv"], 0]])[0]
    cv2.drawMarker(img, (int(c[0]), int(c[1])), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.circle(img, (int(m[0]), int(m[1])), 6, (0, 0, 255) if res["ok"] else (0, 165, 255), 2)
    cv2.putText(img, f"{chess.square_name(sq)} {res['dx_mm']:+.0f},{res['dy_mm']:+.0f}mm",
                (int(c[0]) - 40, int(c[1]) + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def main():
    import os
    import sys
    from cameras import open_views
    sqs = [chess.parse_square(s) for s in sys.argv[1:]] or [chess.A8, chess.H8]
    views = open_views()
    frames = {v.key: v.settled_frame(20) for v in views}
    vf = [(v.key, v.geo, frames[v.key]) for v in views]
    empties = {v.key: EmptyView(v.geo, frames[v.key].shape) for v in views}
    imgs = {v.key: frames[v.key].copy() for v in views}
    for sq in sqs:
        res = measure(vf, sq, empties)
        print(f"{chess.square_name(sq)}: " + (f"off centre {res['dx_mm']:+.1f} mm along the rank (toward h +), "
                                               f"{res['dy_mm']:+.1f} mm along the file (toward rank 8 +), "
                                               f"fit {res.get('rms_px', 0):.1f} px, rows {res['rows']}"
                                               if res["ok"] else f"NOT MEASURED - {res['why']}"))
        for v in views:
            draw(imgs[v.key], v.geo, sq, res, v.key)
    for v in views:
        cv2.imwrite(os.path.join(config.RESULTS_DIR, f"centering_{v.key}.png"), imgs[v.key])


if __name__ == "__main__":
    main()

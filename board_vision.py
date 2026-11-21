"""Locate the chessboard in the camera image and cut one crop per square.

Calibration (config/board_image.json): the operator clicks the four outer corners
of the 8x8 playing area in the live image, in the order a1, h1, h8, a8 (the
corner of each of those squares that is also a corner of the board). That fixes
the board orientation, so no assumption about which side the camera is on.
When the board is empty, OpenCV's chessboard detector refines the corners to
sub-pixel accuracy from the 7x7 inner corners.

Geometry: board coordinates are in square units, (0,0) = outer corner of a1,
x along the files (a->h), y along the ranks (1->8), z up. A camera model is
recovered from the four corners: the focal length from the homography (Zhang's
orthogonality constraint, principal point at the image centre), then the pose
with solvePnP. With it each square's "column" - the square plus the space above
it up to PIECE_BOX_HEIGHT - is projected into the image, and its bounding box is
the crop the classifier sees. Pieces are tall, so a crop of the square alone
would show mostly the bottom of the piece.

Using the tiles themselves (no markers needed):
- Board finding: the checker pattern's 7x7 inner corners are found by OpenCV's
  chessboard detector on the empty board (refine_corners_empty_board).
- Empty-board reference: at calibration the empty board is photographed; each
  square's top-down appearance (warped through the board homography) is the
  reference. change_scores() compares a live frame with it - an occupancy cue
  independent of the classifier, and against a known light/dark background.
- Camera bump check: the tile edges are a permanent reference. tile_alignment()
  correlates the live gradient image of the board with the reference along the
  grid lines; squares hidden by pieces barely matter since most grid lines stay
  visible. If the board has moved in the image the score drops and the system
  pauses. (ArUco markers, if still visible, are checked too.)
"""
import json
import os
import time

import chess
import cv2
import numpy as np

import config


def find_camera(name=config.CAMERA_NAME):
    """The /dev/videoN capture node of the named camera (direct USB passthrough
    device preferred over the Parallels-virtualised copy, which gives no frames)."""
    import glob
    matches = []
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            dev_name = open(os.path.join(d, "name")).read().strip()
            index = int(open(os.path.join(d, "index")).read())
        except OSError:
            continue
        if name in dev_name and index == 0:
            matches.append((":" in dev_name, "/dev/" + os.path.basename(d)))
    if not matches:
        raise RuntimeError(f"No camera named {name!r} found")
    return sorted(matches)[0][1]


def open_camera(device=None, size=config.FRAME_SIZE):
    device = device or find_camera()
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {device}")
    return cap, device


def detect_markers(frame):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(frame)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in config.MARKER_IDS:
                out[int(i)] = c.reshape(4, 2).mean(axis=0).tolist()
    return out


def focal_from_homography(H, cx, cy):
    """Focal length (px) from a plane->image homography, square pixels, known
    principal point (Zhang 2000: the first two rotation columns are orthogonal)."""
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    h = T @ H
    h1, h2 = h[:, 0], h[:, 1]
    num = -(h1[0] * h2[0] + h1[1] * h2[1])
    den = h1[2] * h2[2]
    if abs(den) < 1e-12 or num / den <= 0:
        return None
    return float(np.sqrt(num / den))


class BoardGeometry:
    def __init__(self, corners_px, frame_size, markers=None, inner_px=None):
        """corners_px: image points of the outer corners of a1, h1, h8, a8.
        inner_px: optional 7x7 inner tile corners (row-major, board y = 1..7 outer
        loop, x = 1..7 inner) - when given, the camera model (focal length and
        radial lens distortion) is fitted to all 49 points."""
        self.corners_px = np.float32(corners_px)
        self.inner_px = None if inner_px is None else np.float32(inner_px).reshape(-1, 2)
        self.markers = markers or {}
        self.dist = None
        w, h = frame_size
        self.frame_size = (w, h)
        board_pts = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
        self.H = cv2.getPerspectiveTransform(board_pts, self.corners_px)   # board -> image
        cx, cy = w / 2, h / 2
        f = focal_from_homography(self.H, cx, cy)
        if f is None or not 0.3 * w < f < 3 * w:
            f = 0.75 * w             # C930e: ~ 82 deg horizontal field of view
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        obj = np.float32([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0]])
        ok, self.rvec, self.tvec = cv2.solvePnP(obj, self.corners_px, self.K, None,
                                                flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise RuntimeError("Could not recover the camera pose from the corners")
        self.reproj_px = None
        if self.inner_px is not None and len(self.inner_px) == 49:
            obj49 = np.float32([[x, y, 0] for y in range(1, 8) for x in range(1, 8)])
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
                     cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
                     cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
            err, K, dist, rv, tv = cv2.calibrateCamera([obj49], [self.inner_px], (w, h),
                                                       self.K.copy(), np.zeros(5), flags=flags)
            if 0.3 * w < K[0, 0] < 3 * w:
                self.K, self.dist, self.rvec, self.tvec = K, dist, rv[0], tv[0]
                self.reproj_px = float(err)
                # outer corners from the fitted model (consistent with the grid)
                self.corners_px = np.float32(self.project([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0]],
                                                          _raw=True))
        # Camera is above the board: +z (up) must point toward the camera
        R, _ = cv2.Rodrigues(self.rvec)
        cam_in_board = (-R.T @ self.tvec).ravel()
        self.z_sign = 1.0 if cam_in_board[2] > 0 else -1.0
        self.cam_in_board = cam_in_board * np.array([1, 1, self.z_sign])
        self.boxes = [self._square_box(sq) for sq in range(64)]

    # ---- persistence
    def save(self, path=config.BOARD_IMAGE_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(path, "w") as f:
            json.dump({"corners_px": self.corners_px.tolist(), "frame_size": self.frame_size,
                       "inner_px": None if self.inner_px is None else self.inner_px.tolist(),
                       "reprojection_px": self.reproj_px,
                       "markers": self.markers, "focal_px": float(self.K[0, 0]),
                       "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "order": "outer corners of a1, h1, h8, a8"}, f, indent=2)

    @classmethod
    def load(cls, path=config.BOARD_IMAGE_FILE):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        geo = cls(d["corners_px"], tuple(d["frame_size"]), d.get("markers"), d.get("inner_px"))
        geo.path = path
        return geo

    def _ref_path(self, path):
        return path or getattr(self, "path", config.BOARD_IMAGE_FILE).replace(".json", "_empty.png")

    # ---- projection
    def project(self, pts3, _raw=False):
        pts3 = np.float32(pts3)
        if not _raw:
            pts3 = pts3 * np.float32([1, 1, self.z_sign])
        px, _ = cv2.projectPoints(pts3, self.rvec, self.tvec, self.K, self.dist)
        return px.reshape(-1, 2)

    def square_polygon(self, sq, z=0.0):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return self.project([[f, r, z], [f + 1, r, z], [f + 1, r + 1, z], [f, r + 1, z]])

    def square_center(self, sq, z=0.0):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return self.project([[f + 0.5, r + 0.5, z]])[0]

    def _square_box(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        Hh = config.PIECE_BOX_HEIGHT
        m = 0.08          # slight inset: neighbours' edges stay out of the crop
        pts = [[x, y, z] for x in (f + m, f + 1 - m) for y in (r + m, r + 1 - m) for z in (0, Hh)]
        px = self.project(pts)
        w, h = self.frame_size
        x0, y0 = np.floor(px.min(axis=0)).astype(int)
        x1, y1 = np.ceil(px.max(axis=0)).astype(int)
        return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))

    def crops(self, frame):
        """64 crops (index = square), each resized to CROP_SIZE."""
        out = []
        for x0, y0, x1, y1 in self.boxes:
            c = frame[y0:y1, x0:x1]
            if c.size == 0:
                c = np.zeros((8, 8, 3), np.uint8)
            out.append(cv2.resize(c, config.CROP_SIZE, interpolation=cv2.INTER_AREA))
        return out

    # ---- tiles: empty-board reference and bump check
    WARP = 40           # px per square in the top-down view

    def topdown(self, frame):
        """Top-down 8x8 view of the board plane (a1 at bottom-left), WARP px per square."""
        n = 8 * self.WARP
        dst = np.float32([[0, n], [n, n], [n, 0], [0, 0]])   # a1, h1, h8, a8
        M = cv2.getPerspectiveTransform(self.corners_px, dst)
        return cv2.warpPerspective(frame, M, (n, n))

    def set_reference(self, empty_frame, path=None):
        td = self.topdown(empty_frame)
        self.reference = td
        cv2.imwrite(self._ref_path(path), td)

    def load_reference(self, path=None):
        img = cv2.imread(self._ref_path(path))
        self.reference = img
        return img is not None

    def change_scores(self, frame):
        """Per-square difference from the empty board (0 = looks empty), index = square.
        Uses the inner 60% of each square's top-down patch (piece bases sit there)."""
        if getattr(self, "reference", None) is None:
            return None
        td = cv2.GaussianBlur(self.topdown(frame), (5, 5), 0).astype(np.float32)
        ref = cv2.GaussianBlur(self.reference, (5, 5), 0).astype(np.float32)
        lab_d = np.abs(cv2.cvtColor(td / 255, cv2.COLOR_BGR2Lab) - cv2.cvtColor(ref / 255, cv2.COLOR_BGR2Lab))
        out = np.zeros(64)
        W, m = self.WARP, int(self.WARP * 0.2)
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            y0 = (7 - r) * W
            patch = lab_d[y0 + m:y0 + W - m, f * W + m:f * W + W - m]
            out[sq] = float(np.median(patch.sum(axis=2)))
        return out

    def tile_alignment(self, frame):
        """How well the live board's tile edges line up with the reference (0..1).
        ~1 when nothing moved; falls sharply if the camera or board shifts by a few px."""
        if getattr(self, "reference", None) is None:
            return None

        def edges(img):
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
            gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
            return np.abs(gx), np.abs(gy)

        lx, ly = edges(self.topdown(frame))
        rx, ry = edges(self.reference)
        n, W = 8 * self.WARP, self.WARP
        mask = np.zeros((n, n), bool)
        for k in range(1, 8):              # bands along the 7 inner grid lines each way
            mask[:, k * W - 2:k * W + 3] = True
            mask[k * W - 2:k * W + 3, :] = True
        a = np.concatenate([lx[mask], ly[mask]])
        b = np.concatenate([rx[mask], ry[mask]])
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # ---- validity
    def camera_moved(self, frame):
        """Max marker shift (px) since calibration, or None if no markers to compare."""
        if not self.markers:
            return None
        now = detect_markers(frame)
        shifts = [np.hypot(*(np.array(now[int(k)]) - np.array(v)))
                  for k, v in self.markers.items() if int(k) in now]
        return max(shifts) if shifts else None

    def draw(self, img, labels=None, colors=None):
        for sq in range(64):
            poly = self.square_polygon(sq).astype(np.int32)
            col = (colors[sq] if colors else (200, 200, 200))
            cv2.polylines(img, [poly], True, col, 1, cv2.LINE_AA)
        a1 = self.square_center(chess.A1)
        h8 = self.square_center(chess.H8)
        cv2.putText(img, "a1", tuple(int(v) for v in a1 + [-10, 5]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)
        cv2.putText(img, "h8", tuple(int(v) for v in h8 + [-10, 5]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)
        if labels:
            for sq, (txt, col) in labels.items():
                c = self.square_center(sq, 0.3)
                cv2.putText(img, txt, (int(c[0]) - 8, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, txt, (int(c[0]) - 8, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, col, 2, cv2.LINE_AA)


def refine_corners_empty_board(frame, rough):
    """With an empty board, fit the 7x7 inner corners and extrapolate the outer
    corners. Returns refined corners (a1, h1, h8, a8) or None."""
    pts = find_inner_corners(frame)
    if pts is None:
        return None
    # Fit board->image homography from the rough corners, and use it to label
    # each detected inner corner with its board coordinate (1..7, 1..7).
    Hr = cv2.getPerspectiveTransform(np.float32([[0, 0], [8, 0], [8, 8], [0, 8]]), np.float32(rough))
    Hinv = np.linalg.inv(Hr)
    bp = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    grid = np.round(bp)
    if np.abs(bp - grid).max() > 0.35 or grid.min() < 1 or grid.max() > 7:
        return None
    H, _ = cv2.findHomography(np.float32(grid), np.float32(pts))
    out = cv2.perspectiveTransform(np.float32([[[0, 0]], [[8, 0]], [[8, 8]], [[0, 8]]]), H)
    return out.reshape(4, 2).tolist()


def find_inner_corners(frame):
    """The 49 inner tile corners of an empty board (49 x 2), or None. Tries the
    sector-based detector first, then the classic one (copes better with the
    fold line of a folding board), refined to sub-pixel accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, pts = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, pts = cv2.findChessboardCorners(gray, (7, 7), cv2.CALIB_CB_ADAPTIVE_THRESH |
                                            cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
        pts = cv2.cornerSubPix(gray, pts, (7, 7), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
    return pts.reshape(-1, 2)


def _auto_calibrate_corner(frame, corner):
    """auto_calibrate for a camera looking at the board across a CORNER (diagonal
    view, 2026-11-20): `corner` is the board corner nearest the camera, i.e. the
    lowest of the four in the image. The camera sees the board from above, so the
    image is not mirrored: a1 -> h1 -> h8 -> a8 runs with negative signed area in
    image pixels (y down). That fixes the handedness; the named corner fixes the
    rotation. Row/column order of OpenCV's grid does not matter."""
    pts = find_inner_corners(frame)
    if pts is None:
        return None
    g = pts.reshape(7, 7, 2)
    src0 = [(j + 1, i + 1) for i in range(7) for j in range(7)]
    dst = [g[i, j] for i in range(7) for j in range(7)]

    def outer(src):
        H, _ = cv2.findHomography(np.float32(src), np.float32(dst))
        return cv2.perspectiveTransform(np.float32([[[0, 0]], [[8, 0]], [[8, 8]], [[0, 8]]]), H).reshape(4, 2)

    def area(c):
        return sum(c[k][0] * c[(k + 1) % 4][1] - c[(k + 1) % 4][0] * c[k][1] for k in range(4))

    if area(outer(src0)) > 0:                       # mirrored assignment: swap x and y
        src0 = [(y, x) for x, y in src0]
    rot = [lambda x, y: (x, y), lambda x, y: (8 - y, x), lambda x, y: (8 - x, 8 - y), lambda x, y: (y, 8 - x)]
    want = ["a1", "h1", "h8", "a8"].index(corner)
    for r in rot:
        src = [r(x, y) for x, y in src0]
        c = outer(src)
        if int(np.argmax(c[:, 1])) == want:         # the named corner is lowest in the image
            break
    else:
        return None
    lookup = {xy: p for xy, p in zip(src, dst)}
    inner = [lookup[(x, y)] for y in range(1, 8) for x in range(1, 8)]
    return c.tolist(), np.float32(inner).tolist()


def auto_calibrate(frame, near="rank1"):
    """Find the board on an EMPTY board with no clicks.

    The 7x7 inner tile corners are found by OpenCV; the outer corners are
    extrapolated one square out. Orientation: `near` is the board edge nearest
    the camera (bottom of the image): "rank1", "rank8", "fileA" or "fileH"
    (1 / 8 accepted for the ranks). Standing at that edge, the rest follows from
    the board's handedness (from rank 1, file a is on the left; from file a,
    rank 1 is on the right). Returns corners (a1, h1, h8, a8) and the 49 inner
    corners, or None if the pattern is not found."""
    near = {1: "rank1", 8: "rank8"}.get(near, near)
    if near in ("a1", "h1", "h8", "a8"):
        return _auto_calibrate_corner(frame, near)
    if near not in ("rank1", "rank8", "fileA", "fileH"):
        raise ValueError(f"near must be rank1/rank8/fileA/fileH or a corner a1/h1/h8/a8, not {near!r}")
    pts = find_inner_corners(frame)
    if pts is None:
        return None
    g = pts.reshape(7, 7, 2)
    # Make rows run from the bottom of the image (camera side) to the top
    if g[0, :, 1].mean() < g[-1, :, 1].mean():
        g = g[::-1]
    # Within a row, columns left -> right in the image
    if g[:, 0, 0].mean() > g[:, -1, 0].mean():
        g = g[:, ::-1]
    # Board coordinates of the inner corners: row i counts away from the camera,
    # column j left -> right in the image.
    src, dst = [], []
    for i in range(7):
        for j in range(7):
            x, y = {"rank1": (j + 1, i + 1), "rank8": (7 - j, 7 - i),
                    "fileA": (i + 1, 7 - j), "fileH": (7 - i, j + 1)}[near]
            src.append((x, y))
            dst.append(g[i, j])
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst))
    out = cv2.perspectiveTransform(np.float32([[[0, 0]], [[8, 0]], [[8, 8]], [[0, 8]]]), H)
    # inner corners in row-major board order (y = 1..7, x = 1..7)
    lookup = {xy: p for xy, p in zip(src, dst)}
    inner = [lookup[(x, y)] for y in range(1, 8) for x in range(1, 8)]
    return out.reshape(4, 2).tolist(), np.float32(inner).tolist()

"""How far off centre does a piece stand? Learned from the labelled data_offsets/.

For every labelled piece, each camera gives one patch: the board plane around the
square (the square plus half a square on every side) warped to a fixed top-down
view, so the same offset always moves the piece's base the same way in the patch.
ResNet-18 features of both cameras' patches -> ridge regression of the offset
(dx along a->h, + toward h; dy along the files, + toward rank 8), in mm.

  python offset_model.py            leave-one-capture-out test + train on all, save the model
  python offset_model.py --test     test only

The pick rule this feeds (operator's rule): if the piece is more than 10 mm off
centre in x or y, shift the pick 15 mm in that direction.
"""
import glob
import json
import os
import pickle
import sys

import chess
import cv2
import numpy as np

import config
from board_vision import BoardGeometry

DATA = os.path.join(config.HERE, "data_offsets")
MODEL = os.path.join(config.MODEL_DIR, "offset_model.pkl")
PATCH = 224
MARGIN = 0.5            # squares around the square
SHIFT_ABOVE_MM = 10.0   # operator's rule: shift when more than this off
SHIFT_MM = 15.0
# The model under-estimates large offsets (strong shrinkage with little data), so its
# OUTPUT is compared with a lower threshold. Leave-one-photo-out, 76 pieces from
# 11 photos (11-19), per axis: 7 mm -> no shift on any of the 102 cases <=5 mm off
# and none the wrong way; 33 of the 35 offsets >=15 mm shifted (the other 2 fall
# back to the grip retry).
PRED_THRESHOLD_MM = 7.0
CAMS = ("front", "side")


def patch(frame, geo, square):
    """Top-down patch of the board plane around the square (fixed orientation:
    a-file left, rank 8 up, for every camera)."""
    sq = chess.parse_square(square)
    f, r = chess.square_file(sq), chess.square_rank(sq)
    lo, hi = -MARGIN, 1 + MARGIN
    src = geo.project([[f + lo, r + hi, 0], [f + hi, r + hi, 0], [f + hi, r + lo, 0], [f + lo, r + lo, 0]])
    dst = np.float32([[0, 0], [PATCH, 0], [PATCH, PATCH], [0, PATCH]])
    H = cv2.getPerspectiveTransform(np.float32(src), dst)
    return cv2.warpPerspective(frame, H, (PATCH, PATCH), borderMode=cv2.BORDER_REPLICATE)


_geos = {}


def geometry(path):
    if path not in _geos:
        _geos[path] = BoardGeometry.load(path)
    return _geos[path]


def load():
    """[(stamp, square, piece, dx, dy, {cam: patch})]"""
    out = []
    for js in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        rec = json.load(open(js))
        stamp = os.path.basename(js)[:-5]
        frames = {c: cv2.imread(os.path.join(DATA, f"{stamp}_{c}.jpg")) for c in CAMS}
        if any(f is None for f in frames.values()):
            continue
        cal = rec.get("calibration", {})
        for p in rec.get("pieces", []):
            pats = {c: patch(frames[c], geometry(cal.get(c, config.CAMERAS[c]["file"])), p["square"]) for c in CAMS}
            out.append((stamp, p["square"], p["piece"], p["dx_mm"], p["dy_mm"], pats))
    return out


def spatial_features(patches):
    """ResNet-18 up to layer3, pooled to a 7x7 grid: keeps WHERE things are in the
    patch (the global average of the classifier's features throws that away)."""
    import torch
    from pieces import _MEAN, _STD, _network
    net = _network()
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in patches]).astype(np.float32) / 255
    t = torch.from_numpy(np.ascontiguousarray(((x - _MEAN) / _STD).transpose(0, 3, 1, 2)))
    with torch.no_grad():
        h = net.maxpool(net.relu(net.bn1(net.conv1(t))))
        h = net.layer3(net.layer2(net.layer1(h)))
        h = torch.nn.functional.adaptive_avg_pool2d(h, 7)
    return h.flatten(1).numpy()


def featurize(items):
    return np.hstack([spatial_features([it[5][c] for it in items]) for c in CAMS])


class OffsetModel:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.ym = Y.mean(0)
        # dual form: few samples, many features -> solve n x n, not d x d
        self.W = Z.T @ np.linalg.solve(Z @ Z.T + self.alpha * np.eye(len(Z)), Y - self.ym)
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.W + self.ym


def shift(d, above=SHIFT_ABOVE_MM):
    """Operator's rule on one axis (mm): the pick shift."""
    return SHIFT_MM * np.sign(d) * (abs(d) > above)


def evaluate(items, X, alpha):
    Y = np.array([[it[3], it[4]] for it in items])
    stamps = sorted({it[0] for it in items})
    P = np.zeros_like(Y)
    for s in stamps:                       # leave one capture (photo) out
        tr = np.array([it[0] != s for it in items])
        P[~tr] = OffsetModel(alpha).fit(X[tr], Y[tr]).predict(X[~tr])
    err = np.abs(P - Y)
    rule_true = np.vectorize(shift)(Y)
    rule_pred = np.vectorize(lambda d: shift(d, PRED_THRESHOLD_MM))(P)
    return P, Y, err, (rule_true == rule_pred).mean(0)


def main():
    items = load()
    print(f"{len(items)} labelled pieces from {len({i[0] for i in items})} photos", flush=True)
    X = featurize(items)
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        P, Y, err, rule = evaluate(items, X, alpha)
        print(f"alpha {alpha:>7g}: mean error x {err[:, 0].mean():.1f} mm, y {err[:, 1].mean():.1f} mm; "
              f"shift decision right x {100 * rule[0]:.0f}%, y {100 * rule[1]:.0f}%")
        if best is None or err.mean() < best[0]:
            best = (err.mean(), alpha, P, Y)
    _, alpha, P, Y = best
    base = np.abs(Y - Y.mean(0)).mean(0)
    print(f"(guessing the average every time: x {base[0]:.1f} mm, y {base[1]:.1f} mm)")
    print(f"\nbest alpha {alpha:g}, per piece (true -> predicted, mm, x toward h / y toward rank 8):")
    for it, p, y in zip(items, P, Y):
        print(f"  {it[0]} {it[2]} {it[1]}: ({y[0]:+4.0f},{y[1]:+4.0f}) -> ({p[0]:+5.1f},{p[1]:+5.1f})")
    if "--test" in sys.argv:
        return
    import offset_model                              # pickle under the module name, not __main__
    m = offset_model.OffsetModel(alpha).fit(X, Y)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(MODEL, "wb") as fh:
        pickle.dump(m, fh)
    print("saved", MODEL)


class OffsetReader:
    """Live: where does the piece on a square stand, and how should the pick shift?

    reader(square) -> {"dx_mm", "dy_mm", "shift_x_mm", "shift_y_mm"} (x along a->h,
    + toward h; y along the files, + toward rank 8) or None if a camera has no
    frame. Call it while the arm is NOT between the cameras and the square."""

    def __init__(self, views, model=None):
        self.views = {v.key: v for v in views}
        self.model = model or load_model()

    def __call__(self, square):
        pats = {}
        for c in CAMS:
            v = self.views.get(c)
            fr = v.frames(2) if v is not None else None
            if not fr or v.geo is None:
                return None
            pats[c] = patch(fr[-1], v.geo, square)
        X = np.hstack([spatial_features([pats[c]]) for c in CAMS])
        dx, dy = (float(a) for a in self.model.predict(X)[0])
        return {"dx_mm": dx, "dy_mm": dy,
                "shift_x_mm": float(shift(dx, PRED_THRESHOLD_MM)), "shift_y_mm": float(shift(dy, PRED_THRESHOLD_MM))}


def load_model():
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            return OffsetModel if name == "OffsetModel" else super().find_class(mod, name)
    with open(MODEL, "rb") as fh:
        return U(fh).load()


if __name__ == "__main__":
    main()

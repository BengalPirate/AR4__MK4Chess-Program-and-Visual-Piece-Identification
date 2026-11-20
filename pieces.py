"""Piece classifier: square crop -> empty / piece type and colour (13 classes).

Method (transfer learning, small data):
  1. Feature extractor: ResNet-18 pretrained on ImageNet (torchvision,
     BSD-3-Clause; He et al. 2016), final layer removed -> 512-d feature per crop.
     Its weights are frozen; nothing is fine-tuned, so training takes seconds on
     the VM's CPU.
  2. Classifier: multinomial logistic regression (scikit-learn, BSD-3-Clause)
     on the standardised features, trained on crops of OUR board and pieces,
     each labelled from a known position (see collect / correction in
     chess_demo.py). Horizontal mirror images are added as extra samples.
The output is a probability for each of the 13 labels on every square, which
the game logic combines with the legal moves (chess_logic.Game.infer_move).
"""
import glob
import json
import os

import cv2
import numpy as np

import config
from chess_logic import LABELS

_net = None
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _network():
    global _net
    if _net is None:
        import torch
        import torchvision
        os.environ.setdefault("TORCH_HOME", os.path.join(config.MODEL_DIR, "torch"))
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        m.eval()
        _net = m
    return _net


def features(crops, flip=False):
    """List of BGR crops (CROP_SIZE) -> N x 512 feature array."""
    import torch
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255
    if flip:
        x = x[:, :, ::-1, :]
    x = (x - _MEAN) / _STD
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
    with torch.no_grad():
        out = []
        for i in range(0, len(t), 64):
            out.append(_network()(t[i:i + 64]).numpy())
    return np.concatenate(out)


CNN_FILE = os.path.join(config.MODEL_DIR, "piece_cnn.pt")


class PieceClassifier:
    """Uses the fine-tuned network (finetune.py -> models/piece_cnn.pt) when it
    exists, otherwise ResNet features + logistic regression (train.py)."""

    def __init__(self, path=config.CLASSIFIER_FILE, cnn_path=CNN_FILE):
        self.cnn = None
        if cnn_path and os.path.exists(cnn_path):
            import torch
            from finetune import make_net
            d = torch.load(cnn_path, map_location="cpu")
            net = make_net()
            net.load_state_dict(d["state_dict"])
            self.cnn = net.eval()
            self.info = {"boards": 0, "kind": "fine-tuned ResNet-18", "trained": d["trained"],
                         "photos": d.get("photos"), "boards_used": d.get("boards", 0)}
            return
        import joblib
        if not os.path.exists(path):
            raise RuntimeError("No trained classifier - collect labelled boards and run train.py")
        d = joblib.load(path)
        self.model, self.info = d["model"], d["info"]

    def logp(self, crops_list):
        """crops_list: list of frames' 64-crop lists. Averages the probabilities
        over the frames; returns 64 x 13 log-probabilities in LABELS order."""
        probs = None
        for crops in crops_list:
            p = self._proba_crops(crops)
            probs = p if probs is None else probs + p
        probs /= len(crops_list)
        return np.log(np.clip(probs, 1e-6, 1.0))

    def _proba_crops(self, crops):
        if self.cnn is None:
            return self._proba(features(crops))
        import torch
        from finetune import to_tensor
        with torch.no_grad():
            return torch.softmax(self.cnn(to_tensor(crops)), 1).numpy()

    def _proba(self, feats):
        p = np.zeros((len(feats), len(LABELS)))
        p[:, [LABELS.index(c) for c in self.model.classes_]] = self.model.predict_proba(feats)
        return p


def fuse(view_logps, weights=None):
    """Combine per-view 64 x 13 log-probabilities into one: weighted geometric
    mean of the probabilities, renormalised per square. With equal weights a
    square both cameras agree on stays confident; where they disagree it becomes
    uncertain, so the move check pauses instead of trusting either view."""
    if len(view_logps) == 1:
        return view_logps[0]
    w = np.float64(weights if weights is not None else [1.0] * len(view_logps))
    w = w / w.sum()
    lp = sum(wi * l for wi, l in zip(w, view_logps))
    lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
    return np.maximum(lp, np.log(1e-6))


def scan_views(clf, views_frames):
    """views_frames: list of (view_key, geometry, frames). Returns the fused
    64 x 13 log-probabilities and the per-view ones {key: logp}."""
    per = {k: clf.logp([geo.crops(f) for f in frames]) for k, geo, frames in views_frames}
    keys = list(per)
    fused = fuse([per[k] for k in keys], [config.VIEW_WEIGHT.get(k, 1.0) for k in keys])
    return fused, per


# ---------------------------------------------------------------- dataset
def save_sample(frame, geometry, labels, source, out_dir=config.DATA_DIR, camera=None, stamp=None):
    """Store a labelled board image: the frame, the board corners and 64 labels.
    Several cameras' frames of the same position share the stamp, with the camera
    name appended (<stamp>_side.jpg); the front camera keeps the plain name."""
    import time
    os.makedirs(out_dir, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    name = stamp if camera in (None, "front") else f"{stamp}_{camera}"
    cv2.imwrite(os.path.join(out_dir, name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump({"labels": "".join(labels), "corners_px": geometry.corners_px.tolist(),
                   "inner_px": None if geometry.inner_px is None else geometry.inner_px.tolist(),
                   "frame_size": list(geometry.frame_size), "source": source,
                   "camera": camera or "front", "group": stamp}, f)
    return stamp


def sample_camera(name):
    """Camera of a stored sample from its file name."""
    return name.rsplit("_", 1)[1] if "_" in name else "front"


def load_samples(data_dir=config.DATA_DIR):
    """Yield (name, frame, BoardGeometry, labels) for every stored sample."""
    from board_vision import BoardGeometry
    for js in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(js) as f:
            d = json.load(f)
        img = cv2.imread(js[:-5] + ".jpg")
        if img is None:
            continue
        yield os.path.basename(js[:-5]), img, \
            BoardGeometry(d["corners_px"], tuple(d["frame_size"]), inner_px=d.get("inner_px")), \
            list(d["labels"])

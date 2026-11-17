"""Train the piece classifier from the phone photos (photo_dataset.py) and test it.

Data split (no leakage between sets):
  Consecutive photos of a piece are near-duplicates (same pose, a few degrees
  apart), so a plain random split by photo would let the test set "see" its own
  twins in training and overstate accuracy. Photos are therefore grouped in runs
  of 10 consecutive image numbers per class, and whole groups are assigned at
  random (fixed seed) to train 70% / validation 15% / test 15%, per class.
  - train: fit the classifier (with augmented crops)
  - validation: choose the regularisation C
  - test: touched once, for the reported photo accuracy
  External test: labelled WEBCAM boards in data_webcam_test/ (other camera,
  real square crops from board_vision) - this is the number that matters for
  the robot, since phone photos and webcam crops look different.

The crop for a piece mimics BoardGeometry.crops: one square wide, ~2.2 squares
of height above the base, piece base near the bottom, resized to CROP_SIZE.
Empty-square samples are cut from the same photos away from the piece.

Final model = train + validation photos (+ any webcam boards in data/), saved to
models/piece_classifier.joblib in the format pieces.PieceClassifier loads.

  python train_photos.py            # split, tune, test, save model + results/accuracy_photos.md
"""
import json
import os
import random
import re
import time

import cv2
import numpy as np

import config
from chess_logic import LABELS
from photo_dataset import BOXES, CLASS_LETTER, IMG, best_box
from pieces import features, load_samples

MIN_DET_SCORE = 0.03
WEBCAM_TEST_DIR = os.path.join(config.HERE, "data_webcam_test")
SEED = 7


# ---------------------------------------------------------------- crops
def piece_crop(img, box, rng, jitter):
    """Crop shaped like a webcam square crop around a detected piece."""
    _, x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    if jitter:
        s = bw * rng.uniform(1.15, 1.6)                # square width vs piece width
        cx += rng.uniform(-0.1, 0.1) * s
        top = y0 - bh * rng.uniform(0.0, 0.5)
        bottom = y1 + s * rng.uniform(0.05, 0.35)
    else:
        s, top, bottom = bw * 1.35, y0 - bh * 0.2, y1 + bw * 1.35 * 0.2
    # never narrower than half the crop height (tall king vs. squat pawn)
    s = max(s, (bottom - top) / 2.6)
    xa, xb = int(max(0, cx - s / 2)), int(min(W, cx + s / 2))
    ya, yb = int(max(0, top)), int(min(H, bottom))
    return img[ya:yb, xa:xb], (xa, ya, xb, yb)


def empty_crop(img, box, like, rng):
    """Same-size crop elsewhere in the photo, not overlapping the piece."""
    _, x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    w, h = like[2] - like[0], like[3] - like[1]
    for _ in range(30):
        xa, ya = rng.uniform(0, max(1, W - w)), rng.uniform(0, max(1, H - h))
        if xa + w < x0 - 5 or xa > x1 + 5 or ya + h < y0 - 5 or ya > y1 + 5:
            return img[int(ya):int(ya + h), int(xa):int(xa + w)]
    return None


def to_input(crop, rng, augment):
    """Resize to CROP_SIZE; when augmenting, simulate the webcam (low resolution,
    blur, exposure / colour shifts, mirror)."""
    if augment:
        cw = int(rng.uniform(30, 110))                 # webcam square crops are ~40-90 px wide
        ch = max(8, int(cw * crop.shape[0] / max(1, crop.shape[1])))
        crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            crop = cv2.flip(crop, 1)
        f = crop.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
        f *= np.float32([rng.uniform(0.9, 1.1) for _ in range(3)])
        crop = np.clip(f, 0, 255).astype(np.uint8)
    return cv2.resize(crop, config.CROP_SIZE, interpolation=cv2.INTER_AREA if not augment else cv2.INTER_LINEAR)


# ---------------------------------------------------------------- split
def split(boxes):
    groups = {}
    for key, e in boxes.items():
        if best_box(e)[0] < MIN_DET_SCORE:
            continue
        cls, name = key.split("/")
        n = int(re.findall(r"\d+", name)[-1])
        groups.setdefault((cls, n // 10), []).append(key)
    rng = random.Random(SEED)
    sets = {"train": [], "val": [], "test": []}
    for cls in sorted(CLASS_LETTER):
        g = sorted(k for k in groups if k[0] == cls)
        rng.shuffle(g)
        n_val = n_test = max(1, round(0.15 * len(g)))
        for i, k in enumerate(g):
            part = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            sets[part] += groups[k]
    return sets


def build(keys, boxes, n_aug, n_empty, rng):
    crops, labels = [], []
    for key in keys:
        img = cv2.imread(os.path.join(IMG, key))
        box = best_box(boxes[key])
        letter = CLASS_LETTER[key.split("/")[0]]
        for i in range(max(1, n_aug)):
            c, rect = piece_crop(img, box, rng, jitter=n_aug > 0)
            if c.size == 0:
                continue
            crops.append(to_input(c, rng, augment=n_aug > 0))
            labels.append(letter)
            for _ in range(n_empty if i == 0 else 0):
                e = empty_crop(img, box, rect, rng)
                if e is not None and e.size:
                    crops.append(to_input(e, rng, augment=n_aug > 0))
                    labels.append(".")
    return crops, np.array(labels)


def feats(crops):
    return features(crops) if crops else np.zeros((0, 512), np.float32)


def fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced")).fit(X, y)


def webcam_test():
    """Labelled webcam boards (frame + corners + 64 labels), same format as data/."""
    X, y = [], []
    for _, img, geo, labels in load_samples(WEBCAM_TEST_DIR):
        X.append(features(geo.crops(img)))
        y += labels
    return (np.concatenate(X), np.array(y)) if X else (None, None)


def main():
    from train import report, save_confusion
    import joblib
    t0 = time.time()
    boxes = json.load(open(BOXES))
    sets = split(boxes)
    rng = np.random.default_rng(SEED)
    print({k: len(v) for k, v in sets.items()}, "photos")
    tr_c, tr_y = build(sets["train"], boxes, n_aug=4, n_empty=2, rng=rng)
    va_c, va_y = build(sets["val"], boxes, n_aug=0, n_empty=1, rng=rng)
    te_c, te_y = build(sets["test"], boxes, n_aug=0, n_empty=1, rng=rng)
    print(f"crops: train {len(tr_y)}, val {len(va_y)}, test {len(te_y)}; extracting features ...", flush=True)
    Xtr, Xva, Xte = feats(tr_c), feats(va_c), feats(te_c)
    # validation-style crops are un-augmented; evaluate them also at webcam resolution
    best = None
    for C in (0.05, 0.1, 0.3, 1.0):
        acc = float(np.mean(fit(Xtr, tr_y, C).predict(Xva) == va_y))
        print(f"  C={C}: validation accuracy {acc:.1%}")
        if best is None or acc > best[1]:
            best = (C, acc)
    C = best[0]
    clf = fit(Xtr, tr_y, C)
    out = [f"# Piece recognition accuracy (phone-photo training)\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')}. "
           f"Photos: train {len(sets['train'])}, validation {len(sets['val'])}, test {len(sets['test'])} "
           f"(split by groups of 10 consecutive shots). C={C} chosen on validation "
           f"({best[1]:.1%}).\n"]
    txt, cm = report(te_y, clf.predict(Xte), "Held-out phone photos (test split)")
    out.append(txt)
    print(txt)
    save_confusion(cm, os.path.join(config.RESULTS_DIR, "confusion_photos.png"))

    Xw, yw = webcam_test()
    if Xw is not None:
        txt, cm = report(yw, clf.predict(Xw), "Webcam boards (data_webcam_test/, never trained on)")
        out.append(txt)
        print(txt)
        save_confusion(cm, os.path.join(config.RESULTS_DIR, "confusion_webcam.png"))

    # final model: train + validation photos, plus labelled webcam boards from data/ if any
    X, y = np.concatenate([Xtr, Xva]), np.concatenate([tr_y, va_y])
    web = list(load_samples())
    for _, img, geo, labels in web:
        X = np.concatenate([X, features(geo.crops(img))])
        y = np.concatenate([y, labels])
    final = fit(X, y, C)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump({"model": final, "info": {"boards": len(web), "photos": len(sets["train"]) + len(sets["val"]),
                                          "trained": time.strftime("%Y-%m-%d %H:%M"), "classes": sorted(set(y)),
                                          "source": "train_photos.py"}}, config.CLASSIFIER_FILE)
    with open(os.path.join(config.RESULTS_DIR, "accuracy_photos.md"), "w") as f:
        f.write("\n".join(out))
    print(f"saved {config.CLASSIFIER_FILE} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""Train the piece classifier from the labelled boards in data/ and report accuracy.

  python train.py                 train on all samples, save models/piece_classifier.joblib
  python train.py --holdout 0.25  also evaluate on the last 25% of boards (by time),
                                  which were not used for training

Evaluation is per board image, held out as whole boards (squares of one photo are
never split between training and test, which would overstate accuracy).
Writes results/accuracy.md and results/confusion.png.
"""
import argparse
import os
import time

import numpy as np

import config
from chess_logic import LABELS
from pieces import features, load_samples


def build(samples):
    X, y, groups = [], [], []
    for gi, (name, img, geo, labels) in enumerate(samples):
        crops = geo.crops(img)
        for flip in (False, True):
            X.append(features(crops, flip=flip))
            y += labels
            groups += [gi] * 64
    return np.concatenate(X), np.array(y), np.array(groups)


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000,
                                                             class_weight="balanced"))
    return clf.fit(X, y)


def report(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix
    acc = float(np.mean(y_true == y_pred))
    occ_true, occ_pred = y_true != ".", y_pred != "."
    occ = float(np.mean(occ_true == occ_pred))
    pcs = occ_true
    col = float(np.mean([t.isupper() == p.isupper() for t, p in zip(y_true[pcs], y_pred[pcs])
                         if p != "."])) if pcs.any() else float("nan")
    typ = float(np.mean([t.upper() == p.upper() for t, p in zip(y_true[pcs], y_pred[pcs])]))
    lines = [f"### {title}", "",
             f"- squares: {len(y_true)} ({int(pcs.sum())} occupied)",
             f"- exact label (empty / type+colour): **{acc:.1%}**",
             f"- occupancy: {occ:.1%}",
             f"- colour, on occupied squares detected as occupied: {col:.1%}",
             f"- piece type, on occupied squares: {typ:.1%}", ""]
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    lines += ["| true \\ pred | " + " | ".join(LABELS) + " |",
              "|---" * (len(LABELS) + 1) + "|"]
    for lab, row in zip(LABELS, cm):
        lines.append(f"| **{lab}** | " + " | ".join(str(v) if v else "" for v in row) + " |")
    return "\n".join(lines) + "\n", cm


def main():
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=float, default=0.0)
    a = ap.parse_args()
    samples = list(load_samples())
    if not samples:
        raise SystemExit("No samples in data/ - capture labelled boards first (chess_demo.py, key B)")
    print(f"{len(samples)} labelled boards")
    t0 = time.time()
    out = [f"# Piece recognition accuracy\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} "
           f"from {len(samples)} labelled board images.\n"]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    if a.holdout > 0 and len(samples) >= 4:
        n_test = max(1, int(round(len(samples) * a.holdout)))
        train_s, test_s = samples[:-n_test], samples[-n_test:]
        Xtr, ytr, _ = build(train_s)
        Xte, yte, gte = build(test_s)
        keep = np.arange(len(yte)) % 128 < 64        # evaluate unflipped crops only
        clf = fit(Xtr, ytr)
        pred = clf.predict(Xte[keep])
        txt, cm = report(yte[keep], pred, f"Held-out boards ({n_test} images, never trained on)")
        out.append(txt)
        print(txt)
        names = [s[0] for s in test_s]
        out.append("Held-out images: " + ", ".join(names) + "\n")
        save_confusion(cm, os.path.join(config.RESULTS_DIR, "confusion.png"))
    X, y, _ = build(samples)
    clf = fit(X, y)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "info": {"boards": len(samples), "trained": time.strftime("%Y-%m-%d %H:%M"),
                                        "classes": sorted(set(y))}}, config.CLASSIFIER_FILE)
    counts = {lab: int((y == lab).sum() // 2) for lab in LABELS}
    out.append("Training samples per label (all boards): " +
               ", ".join(f"{k}={v}" for k, v in counts.items()) + "\n")
    with open(os.path.join(config.RESULTS_DIR, "accuracy.md"), "w") as f:
        f.write("\n".join(out))
    print(f"saved {config.CLASSIFIER_FILE} ({time.time() - t0:.0f}s)")
    print(out[-1])


def save_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    ax.set_title("Piece recognition, held-out boards")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()

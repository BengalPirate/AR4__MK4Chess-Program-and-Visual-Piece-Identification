"""Accuracy of the current model on the held-out webcam boards (data_webcam_test/).

  python tests/webcam_report.py      writes results/accuracy_webcam.md and
                                     results/confusion_webcam_all.png
Groups: start position (either colour at the bottom) and every other position.
One frame per saved board; boards saved in bursts of 3 frames count separately.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from chess_logic import LABELS  # noqa: E402
from pieces import PieceClassifier, load_samples  # noqa: E402
from train import report, save_confusion  # noqa: E402

START = "RNBQKBNRPPPPPPPP" + "." * 32 + "pppppppprnbqkbnr"


def main():
    clf = PieceClassifier()
    L = np.array(LABELS)
    groups = {"all": [], "start": [], "other": []}
    per_board = []
    for stamp, img, geo, labels in load_samples("data_webcam_test"):
        pred = [LABELS[i] for i in np.asarray(clf.logp([geo.crops(img)])).argmax(-1)]
        s = "".join(labels)
        key = "start" if s in (START, START.swapcase()) else "other"   # either colour at rank 1
        for g in ("all", key):
            groups[g].append((labels, pred))
        per_board.append((stamp, key, sum(p == t for p, t in zip(pred, labels))))
    out = [f"# Webcam recognition accuracy (held-out boards)\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} "
           f"with models/piece_cnn.pt (trained {clf.info.get('trained')}). Boards in data_webcam_test/ "
           "were never used for training. One frame per saved board.\n"]
    for g, title in (("all", "All held-out webcam boards"), ("start", "Start positions"),
                     ("other", "Other positions")):
        t = [x for b in groups[g] for x in b[0]]
        p = [x for b in groups[g] for x in b[1]]
        txt, cm = report(L[[LABELS.index(x) for x in t]], L[[LABELS.index(x) for x in p]],
                         f"{title} ({len(groups[g])} boards)")
        out.append(txt)
        print(txt.split("\n|")[0])
        if g == "all":
            save_confusion(cm, os.path.join(config.RESULTS_DIR, "confusion_webcam_all.png"))
    out.append("### Per board\n\n| Board | Group | Exact squares |\n|---|---|---|\n" +
               "\n".join(f"| {s} | {k} | {n}/64 |" for s, k, n in per_board))
    with open(os.path.join(config.RESULTS_DIR, "accuracy_webcam.md"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

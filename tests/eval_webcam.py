"""Evaluate the current classifier on the held-out webcam boards (data_webcam_test/).

  python tests/eval_webcam.py [name-prefix]

Per board: exact / occupancy accuracy and the predicted grid (! wrong, ? below
MIN_SQUARE_CONF). For start-position boards also the move-inference test: the
game is set one knight move away from the start position, so the correct answer
is that knight moving back; each case is counted as correct+accepted, paused
(margin/surprise thresholds -> operator correction) or wrong+accepted (unsafe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from chess_logic import LABELS, Game  # noqa: E402
from pieces import PieceClassifier, load_samples  # noqa: E402
from train_photos import WEBCAM_TEST_DIR  # noqa: E402

CASES = [("g1f3", "f3g1"), ("g1h3", "h3g1"), ("b1c3", "c3b1"), ("b1a3", "a3b1"),
         ("g8f6", "f6g8"), ("g8h6", "h6g8"), ("b8c6", "c6b8"), ("b8a6", "a6b8")]
START = "".join(LABELS[0] if p is None else p.symbol() for p in
                (chess.Board().piece_at(s) for s in range(64)))


def main(prefix=""):
    clf = PieceClassifier()
    L = np.array(LABELS)
    tot = {"ok": 0, "pause": 0, "wrong": 0}
    accs = []
    for name, img, geo, lab in load_samples(WEBCAM_TEST_DIR):
        if not name.startswith(prefix):
            continue
        lp = clf.logp([geo.crops(img)])
        pred, conf = L[lp.argmax(1)], np.exp(lp.max(1))
        lab = np.array(lab)
        accs.append(np.mean(pred == lab))
        print(f"{name}: exact {np.mean(pred == lab):.1%}  occupancy {np.mean((pred == '.') == (lab == '.')):.1%}")
        for r in range(7, -1, -1):
            print(f"  {r + 1} " + " ".join(pred[r * 8 + f] + ("!" if pred[r * 8 + f] != lab[r * 8 + f] else " ")
                                         + ("?" if conf[r * 8 + f] < config.MIN_SQUARE_CONF else " ")
                                         for f in range(8)))
        if "".join(lab) != START:
            continue
        for fwd, back in CASES:
            b = chess.Board()
            side = chess.WHITE if fwd[1] == "1" else chess.BLACK
            b.turn = side
            b.push_uci(fwd)
            b.turn = side
            r = Game(b.fen()).infer_move(lp)
            safe = r["margin"] >= config.MOVE_MIN_MARGIN and r["surprise"] <= config.MOVE_MAX_SQUARE_SURPRISE
            right = r["move"] is not None and r["move"].uci() == back
            k = "ok" if right and safe else "pause" if not safe else "wrong"
            tot[k] += 1
            if k != "ok":
                print(f"    {back}: got {r['move']} {k} margin {r['margin']:.1f} "
                      f"surprise {r['surprise']:.1f} at {chess.square_name(r['worst_square'])}")
    print(f"mean exact {np.mean(accs):.1%} over {len(accs)} boards")
    print(f"move tests: {tot['ok']} correct+accepted, {tot['pause']} paused, {tot['wrong']} wrong+accepted")


if __name__ == "__main__":
    main(*sys.argv[1:])

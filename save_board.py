"""Save a labelled webcam board (frame + calibrated corners + 64 labels from a FEN).

  python save_board.py --fen "<FEN or board part>" [--test] [--image PATH]

--test   store in data_webcam_test/ (held out, used only to measure accuracy);
         default data/ (used for training by train.py / train_photos.py).
--image  use a saved frame (e.g. live_view.py --snapshot) instead of opening the camera.
Writes a crops contact sheet next to the sample so the labels can be checked.
"""
import argparse
import os

import chess
import cv2
import numpy as np

import config
from board_vision import BoardGeometry, open_camera
from pieces import save_sample
from train_photos import WEBCAM_TEST_DIR


def labels_from_fen(fen):
    board = chess.Board(fen.split()[0] + " w - - 0 1")
    return [(board.piece_at(sq).symbol() if board.piece_at(sq) else ".") for sq in range(64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--image")
    a = ap.parse_args()
    geo = BoardGeometry.load()
    if a.image:
        frame = cv2.imread(a.image)
    else:
        cap, _ = open_camera()
        # autofocus settles in ~1 s; keep the sharpest of the next frames
        best = -1.0
        for i in range(45):
            ok, f = cap.read()
            if not ok or i < 15:
                continue
            sharp = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp > best:
                best, frame = sharp, f
        print(f"sharpness {best:.0f}")
    labels = labels_from_fen(a.fen)
    out_dir = WEBCAM_TEST_DIR if a.test else config.DATA_DIR
    stamp = save_sample(frame, geo, labels, "save_board.py " + a.fen, out_dir)
    crops = geo.crops(frame)
    rows = []
    for r in range(7, -1, -1):
        row = []
        for f in range(8):
            c = crops[r * 8 + f].copy()
            cv2.putText(c, labels[r * 8 + f], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            row.append(c)
        rows.append(np.hstack(row))
    cv2.imwrite(os.path.join(out_dir, stamp + "_crops.png"), np.vstack(rows))
    print(f"saved {out_dir}/{stamp} ({sum(l != '.' for l in labels)} pieces)")


if __name__ == "__main__":
    main()

"""Save a labelled board from every camera (frame + calibrated corners + 64 labels from a FEN).

  python save_board.py --fen "<FEN or board part>" [--test] [--image PATH --camera KEY]

--test   store in data_webcam_test/ (held out, used only to measure accuracy);
         default data/ (used for training by train.py / train_photos.py).
--image  use a saved frame of one camera (--camera, default front) instead of the cameras.
All cameras' frames of the position share one stamp (<stamp>.jpg = front,
<stamp>_side.jpg = side). Writes a crops contact sheet per camera so the labels
can be checked.
"""
import argparse
import os

import chess
import cv2
import numpy as np

import config
from board_vision import BoardGeometry
from cameras import open_views
from pieces import save_sample
from train_photos import WEBCAM_TEST_DIR


def labels_from_fen(fen):
    board = chess.Board(fen.split()[0] + " w - - 0 1")
    return [(board.piece_at(sq).symbol() if board.piece_at(sq) else ".") for sq in range(64)]


def contact_sheet(geo, frame, labels):
    crops = geo.crops(frame)
    rows = []
    for r in range(7, -1, -1):
        row = []
        for f in range(8):
            c = crops[r * 8 + f].copy()
            cv2.putText(c, labels[r * 8 + f], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            row.append(c)
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--camera", default="front", choices=list(config.CAMERAS))
    a = ap.parse_args()
    labels = labels_from_fen(a.fen)
    out_dir = WEBCAM_TEST_DIR if a.test else config.DATA_DIR
    if a.image:
        shots = [(a.camera, BoardGeometry.load(config.CAMERAS[a.camera]["file"]), cv2.imread(a.image))]
    else:
        shots = []
        for v in open_views():
            if v.geo is None:
                print(f"[{v.key}] not calibrated - skipped")
                continue
            shots.append((v.key, v.geo, v.settled_frame(45)))
    stamp = None
    for key, geo, frame in shots:
        stamp = save_sample(frame, geo, labels, "save_board.py " + a.fen, out_dir, camera=key, stamp=stamp)
        name = stamp if key == "front" else f"{stamp}_{key}"
        cv2.imwrite(os.path.join(out_dir, name + "_crops.png"), contact_sheet(geo, frame, labels))
        print(f"saved {out_dir}/{name} ({sum(l != '.' for l in labels)} pieces)")


if __name__ == "__main__":
    main()

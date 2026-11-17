"""Build piece-classifier training crops from the team's phone photos.

Source: ~/Desktop/Chess_Classification/wimages/<colour>_<piece>/*.HEIC
(iPhone Live Photos: one piece per photo, our board, many angles; the matching
.MOV clips in wvideos/ are not used.)

Steps (each cached, re-run safely):
  decode   HEIC -> data_photos/img/<class>/<name>.jpg, long side 1024 px
  detect   zero-shot box of the piece: OWL-ViT (google/owlvit-base-patch32,
           Apache-2.0, Minderer et al. 2022 via Hugging Face transformers)
           -> data_photos/boxes.json
  sheet    contact sheet of detections for checking -> results/photo_boxes.jpg

Crops are generated on the fly in train_photos.py from the image + box so the
crop geometry can match the webcam square crops (board_vision.BoardGeometry).

  python photo_dataset.py decode detect sheet
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import config

SRC = os.path.expanduser(os.environ.get("CHESS_PHOTOS", "~/Desktop/Chess_Classification/wimages"))
OUT = os.path.join(config.HERE, "data_photos")
IMG = os.path.join(OUT, "img")
BOXES = os.path.join(OUT, "boxes.json")
LONG_SIDE = 1024
# folder name -> FEN letter used in chess_logic.LABELS
CLASS_LETTER = {"white_pawn": "P", "white_knight": "N", "white_bishop": "B", "white_rook": "R",
                "white_queen": "Q", "white_king": "K", "black_pawn": "p", "black_knight": "n",
                "black_bishop": "b", "black_rook": "r", "black_queen": "q", "black_king": "k"}


def _decode_one(src):
    import pillow_heif
    from PIL import Image, ImageOps
    pillow_heif.register_heif_opener()
    cls = os.path.basename(os.path.dirname(src))
    dst = os.path.join(IMG, cls, os.path.splitext(os.path.basename(src))[0] + ".jpg")
    if os.path.exists(dst):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
    im.save(dst, quality=92)
    return 1


def decode():
    srcs = sorted(glob.glob(os.path.join(SRC, "*", "*.HEIC")) + glob.glob(os.path.join(SRC, "*", "*.heic")))
    srcs = [s for s in srcs if os.path.basename(os.path.dirname(s)) in CLASS_LETTER]
    with ProcessPoolExecutor(4) as ex:
        n = sum(ex.map(_decode_one, srcs, chunksize=8))
    print(f"decode: {len(srcs)} photos, {n} new")


def images():
    return sorted(glob.glob(os.path.join(IMG, "*", "*.jpg")))


def detect():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import OwlViTForObjectDetection, OwlViTProcessor
    torch.set_num_threads(os.cpu_count() or 1)
    boxes = json.load(open(BOXES)) if os.path.exists(BOXES) else {}
    todo = [p for p in images() if os.path.relpath(p, IMG) not in boxes]
    if not todo:
        print("detect: nothing to do")
        return
    name = "google/owlvit-base-patch32"
    proc = OwlViTProcessor.from_pretrained(name)
    model = OwlViTForObjectDetection.from_pretrained(name).eval()
    queries = [["a photo of a chess piece", "a chess pawn", "a chess king", "a chess knight"]]
    for i, p in enumerate(todo):
        im = Image.open(p).convert("RGB")
        with torch.no_grad():
            inp = proc(text=queries, images=im, return_tensors="pt")
            out = model(**inp)
        logits = out.logits[0].max(-1).values.sigmoid().numpy()      # best query per patch box
        cxcywh = out.pred_boxes[0].numpy()
        # OwlViTProcessor stretches the photo to a square: boxes are relative to W and H
        W, H = im.size
        bx = np.stack([(cxcywh[:, 0] - cxcywh[:, 2] / 2) * W, (cxcywh[:, 1] - cxcywh[:, 3] / 2) * H,
                       (cxcywh[:, 0] + cxcywh[:, 2] / 2) * W, (cxcywh[:, 1] + cxcywh[:, 3] / 2) * H], 1)
        order = np.argsort(-logits)[:5]
        boxes[os.path.relpath(p, IMG)] = {"size": list(im.size), "fmt": 2,
                                          "cands": [[float(logits[j])] + [round(float(v), 1) for v in bx[j]]
                                                    for j in order]}
        if i % 50 == 0 or i == len(todo) - 1:
            json.dump(boxes, open(BOXES, "w"))
            print(f"detect: {i + 1}/{len(todo)}", flush=True)
    json.dump(boxes, open(BOXES, "w"))


def fix_old_boxes():
    """Entries written before the scaling fix used max(W, H) for both axes."""
    boxes = json.load(open(BOXES))
    n = 0
    for e in boxes.values():
        if e.get("fmt") != 2:
            W, H = e["size"]
            s = max(W, H)
            e["cands"] = [[c[0], c[1] * W / s, c[2] * H / s, c[3] * W / s, c[4] * H / s] for c in e["cands"]]
            e["fmt"] = 2
            n += 1
    json.dump(boxes, open(BOXES, "w"))
    print(f"fixed {n} old entries")


def best_box(entry):
    """Top-scoring candidate that is not a whole-board box, clipped to the image:
    (score, x0, y0, x1, y1). Score 0 if every candidate is too large."""
    w, h = entry["size"]
    ok = [c for c in entry["cands"] if (c[3] - c[1]) * (c[4] - c[2]) < 0.3 * w * h]
    if not ok:
        return 0.0, 0, 0, w, h
    sc, x0, y0, x1, y1 = ok[0]
    return sc, max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def sheet(n_per_class=6, path=None):
    import random
    import cv2
    import numpy as np
    boxes = json.load(open(BOXES))
    random.seed(0)
    rows = []
    for cls in sorted(CLASS_LETTER):
        keys = [k for k in boxes if k.startswith(cls + "/")]
        tiles = []
        for k in random.sample(keys, min(n_per_class, len(keys))):
            im = cv2.imread(os.path.join(IMG, k))
            sc, x0, y0, x1, y1 = best_box(boxes[k])
            cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 6)
            cv2.putText(im, f"{cls} {sc:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            tiles.append(cv2.resize(im, (180, 240)))
        while len(tiles) < n_per_class:
            tiles.append(np.zeros((240, 180, 3), np.uint8))
        rows.append(np.hstack(tiles))
    out = path or os.path.join(config.RESULTS_DIR, "photo_boxes.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.vstack(rows))
    print("sheet:", out)


if __name__ == "__main__":
    for step in sys.argv[1:] or ["decode", "detect", "sheet"]:
        {"decode": decode, "detect": detect, "sheet": sheet, "fix": fix_old_boxes}[step]()

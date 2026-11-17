"""Fine-tune ResNet-18 (all layers) on the phone-photo crops -> models/piece_cnn.pt

Same split as train_photos.py (groups of 10 consecutive shots; train / validation /
test = 70 / 15 / 15 % per class, fixed seed). Fresh random crops + webcam-style
augmentation every epoch; the epoch with the best validation accuracy is kept;
the test split is evaluated once at the end, plus webcam boards in
data_webcam_test/ if any. pieces.PieceClassifier uses this model when present.

  python finetune.py [--epochs 10]

Webcam boards in data/ (L key in chess_demo.py, or save_board.py) are added to
training; boards in data_webcam_test/ are only used for the final test.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torchvision

import config
from chess_logic import LABELS
from photo_dataset import BOXES, CLASS_LETTER, IMG, best_box
from pieces import load_samples
from train_photos import SEED, WEBCAM_TEST_DIR, empty_crop, piece_crop, split, to_input

CNN_FILE = os.path.join(config.MODEL_DIR, "piece_cnn.pt")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_net():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(512, len(LABELS))
    return m


def to_tensor(crops):
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255
    x = (x - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


class Source:
    """Per photo: a reduced region around the piece (for re-cropping each epoch)
    and a few fixed empty-square crops."""

    def __init__(self, keys, boxes, rng):
        self.items = []
        for k in keys:
            img = cv2.imread(os.path.join(IMG, k))
            sc, x0, y0, x1, y1 = best_box(boxes[k])
            H, W = img.shape[:2]
            bw, bh = x1 - x0, y1 - y0
            rx0, rx1 = int(max(0, x0 - bw)), int(min(W, x1 + bw))
            ry0, ry1 = int(max(0, y0 - 0.8 * bh)), int(min(H, y1 + 0.8 * bw))
            reg = img[ry0:ry1, rx0:rx1]
            f = min(1.0, 360 / max(reg.shape[:2]))
            reg = cv2.resize(reg, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            box = (sc, (x0 - rx0) * f, (y0 - ry0) * f, (x1 - rx0) * f, (y1 - ry0) * f)
            _, rect = piece_crop(img, best_box(boxes[k]), rng, jitter=False)
            empties = [e for e in (empty_crop(img, best_box(boxes[k]), rect, rng) for _ in range(3))
                       if e is not None and e.size]
            empties = [cv2.resize(e, (112, 224), interpolation=cv2.INTER_AREA) for e in empties]
            self.items.append((reg, box, CLASS_LETTER[k.split("/")[0]], empties))

    def sample(self, rng, augment, empties_per_photo):
        crops, labels = [], []
        for reg, box, letter, empties in self.items:
            c, _ = piece_crop(reg, box, rng, jitter=augment)
            if c.size:
                crops.append(to_input(c, rng, augment))
                labels.append(letter)
            for e in empties[:empties_per_photo]:
                crops.append(to_input(e, rng, augment))
                labels.append(".")
        return crops, np.array([LABELS.index(l) for l in labels])


class WebcamSource:
    """Labelled webcam boards from data/ (chess_demo L key, save_board.py): the real
    square crops. Repeated `repeat` times per epoch with mild augmentation (they are
    already at webcam resolution), so they outweigh the phone photos."""

    def __init__(self, data_dir=config.DATA_DIR, repeat=3):
        self.crops, self.labels, self.repeat = [], [], repeat
        self.boards = 0
        for _, img, geo, labels in load_samples(data_dir):
            self.crops += geo.crops(img)
            self.labels += labels
            self.boards += 1

    def sample(self, rng):
        crops, labels = [], []
        for _ in range(self.repeat if self.crops else 0):
            for c, l in zip(self.crops, self.labels):
                f = c.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
                c = np.clip(f, 0, 255).astype(np.uint8)
                if l in ".":                       # empty squares: mirror freely
                    c = cv2.flip(c, 1) if rng.random() < 0.5 else c
                crops.append(c)
                labels.append(LABELS.index(l))
        return crops, np.array(labels, dtype=np.int64)


def evaluate(net, crops, y):
    net.eval()
    with torch.no_grad():
        pred = np.concatenate([net(to_tensor(crops[i:i + 128])).argmax(1).numpy()
                               for i in range(0, len(crops), 128)])
    return pred


def main():
    from train import report, save_confusion
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--init", help="start from this saved model (e.g. models/piece_cnn.pt) instead of ImageNet")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (use ~3e-4 with --init)")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    boxes = json.load(open(BOXES))
    sets = split(boxes)
    src = {k: Source(v, boxes, rng) for k, v in sets.items()}
    va_c, va_y = src["val"].sample(rng, False, 1)
    te_c, te_y = src["test"].sample(rng, False, 1)
    web_train = WebcamSource()
    print(f"loaded in {time.time() - t0:.0f}s; val {len(va_y)} test {len(te_y)} crops; "
          f"{web_train.boards} webcam training boards", flush=True)
    net = make_net()
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu")["state_dict"])
        print("initialised from", a.init, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * 40, pct_start=0.2)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    best = (-1, None)
    for ep in range(a.epochs):
        tr_c, tr_y = src["train"].sample(rng, True, 2)
        wc, wy = web_train.sample(rng)
        tr_c, tr_y = tr_c + wc, np.concatenate([tr_y, wy]).astype(np.int64)
        order = rng.permutation(len(tr_y))
        bs = int(np.ceil(len(order) / 40))
        net.train()
        tot = 0.0
        for i in range(40):
            idx = order[i * bs:(i + 1) * bs]
            if len(idx) == 0:
                continue
            x, y = to_tensor([tr_c[j] for j in idx]), torch.from_numpy(tr_y[idx])
            loss = lossf(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        acc = float(np.mean(evaluate(net, va_c, va_y) == va_y))
        print(f"epoch {ep + 1}/{a.epochs}: loss {tot / len(order):.3f}  val {acc:.1%}  ({time.time() - t0:.0f}s)",
              flush=True)
        if acc >= best[0]:
            best = (acc, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save({"state_dict": best[1], "labels": LABELS, "crop_size": config.CROP_SIZE,
                "trained": time.strftime("%Y-%m-%d %H:%M"), "val_acc": best[0],
                "photos": len(sets["train"]), "boards": web_train.boards}, CNN_FILE)
    L = np.array(LABELS)
    out = [f"# Piece recognition accuracy (fine-tuned ResNet-18)\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')}; "
           f"best validation accuracy {best[0]:.1%}; {a.epochs} epochs.\n"]
    txt, cm = report(L[te_y], L[evaluate(net, te_c, te_y)], "Held-out phone photos (test split)")
    out.append(txt)
    print(txt)
    save_confusion(cm, os.path.join(config.RESULTS_DIR, "confusion_cnn_photos.png"))
    web = list(load_samples(WEBCAM_TEST_DIR))
    if web:
        crops, y = [], []
        for _, img, geo, labels in web:
            crops += geo.crops(img)
            y += labels
        yi = np.array([LABELS.index(l) for l in y])
        txt, cm = report(L[yi], L[evaluate(net, crops, yi)], f"Webcam boards ({len(web)}, never trained on)")
        out.append(txt)
        print(txt)
        save_confusion(cm, os.path.join(config.RESULTS_DIR, "confusion_cnn_webcam.png"))
    with open(os.path.join(config.RESULTS_DIR, "accuracy_cnn.md"), "w") as f:
        f.write("\n".join(out))
    print(f"saved {CNN_FILE} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""Figures for the README and the Checkpoint 2 report.

Run from the repository root:  python docs/make_figures.py
Writes docs/system_diagram.png, docs/recognition_examples.png, docs/board_robot_map.png.
"""
import glob
import json
import os
import sys

import chess
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import config  # noqa: E402
from chess_logic import LABELS  # noqa: E402

DOCS = os.path.join(ROOT, "docs")


def system_diagram():
    boxes = [
        ("cameras.py: two webcams, 1280×720 @ 15 fps\nfront C930e (behind White) + side C920 PRO (a-file side)", "#dfe9f5"),
        ("board_vision.py\nper-camera model fitted to the tiles → 64 square crops; bump check", "#dfe9f5"),
        ("pieces.py / offset_model.py\nfine-tuned ResNet-18 per view, fused: 13 probabilities per square;\n"
         "piece offset from the square centre (mm)", "#dfe9f5"),
        ("chess_logic.py  (python-chess 1.11)\ngame state: turn, castling / en passant rights, tray, stand-ins\n"
         "legal-move inference, 2 agreeing scans → accept, else PAUSE", "#e6f2e1"),
        ("Move selection\nStockfish 16 over UCI (robot's side) / player's move seen by the cameras", "#e6f2e1"),
        ("planner.py\nmove → pick / place on squares and tray slots\n(captures, en passant, castling, promotion, set-down stops)", "#fbefd9"),
        ("motion.py + kin.py\npreflight, vertical approach / retreat, rank-8 tilt, backlash approach,\n"
         "pick check (cameras + gripper current), placement check + fix, interlock", "#fbefd9"),
        ("ar4link.py → AR4 MK4 (Teensy, firmware 6.7.1)\ngripper.py → servo gripper (Nano, current sensor)", "#f6dede"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(boxes) * 1.25)
    ax.axis("off")
    for i, (text, col) in enumerate(boxes):
        y = (len(boxes) - 1 - i) * 1.25 + 0.15
        ax.add_patch(FancyBboxPatch((0.6, y), 8.8, 0.95, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=col, ec="#555", lw=1))
        ax.text(5, y + 0.475, text, ha="center", va="center", fontsize=9)
        if i:
            ax.annotate("", xy=(5, y + 0.97), xytext=(5, y + 1.23),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))
    ax.text(9.75, 8.2, "perception", rotation=90, va="center", color="#2b5d99", fontsize=9)
    ax.text(9.75, 5.0, "chess logic", rotation=90, va="center", color="#3d7a2e", fontsize=9)
    ax.text(9.75, 1.9, "motion", rotation=90, va="center", color="#99661a", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "system_diagram.png"), dpi=170)
    plt.close(fig)


def recognition_examples():
    """Held-out mid-game boards with the classifier's labels: green correct,
    orange correct but below the confidence threshold, red wrong."""
    from pieces import PieceClassifier, load_samples
    clf = PieceClassifier()
    boards = []
    for stamp, img, geo, labels in load_samples("data_webcam_test"):
        start = "".join(labels[:16]) == "RNBQKBNRPPPPPPPP"
        boards.append((stamp, img, geo, labels, start))
    show = ("20261116-141910-144", "20261116-231455-889", "20261116-233436-740", "20261116-233907-252")
    sel = [b for k in show for b in boards if b[0] == k]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for ax, (stamp, img, geo, labels, start) in zip(axes.flat, sel):
        lp = np.asarray(clf.logp([geo.crops(img)]))
        pred, conf = lp.argmax(-1), np.exp(lp.max(-1))
        vis = img.copy()
        ok = 0
        for sq in range(64):
            p, t = LABELS[pred[sq]], labels[sq]
            ok += p == t
            if p == "." and t == ".":
                continue
            col = (0, 170, 0) if p == t else (0, 0, 230)
            if p == t and conf[sq] < config.MIN_SQUARE_CONF:
                col = (0, 140, 255)
            cx, cy = (int(v) for v in geo.square_center(sq))
            txt = p if p != "." else "_"
            cv2.putText(vis, txt, (cx - 9, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
            cv2.putText(vis, txt, (cx - 9, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        x0, y0 = np.min(geo.project(np.array([[-1, -1, 0], [9, 9, 0]], float)), axis=0)
        crop = vis[max(0, int(y0) - 120):, :]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{'start position' if start else 'other position'}  {stamp}:  {ok}/64 exact", fontsize=9)
        ax.axis("off")
    fig.suptitle("Recognition on held-out webcam boards (label = prediction; green correct, "
                 "red wrong, orange low confidence; '_' = predicted empty)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "recognition_examples.png"), dpi=130)
    plt.close(fig)


def board_robot_map():
    from planner import BoardRobot
    br = BoardRobot()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for sq in chess.SQUARES:
        x, y, _ = br.square_xyz(sq)
        tilt = br.cal and chess.square_rank(sq) == 7
        ax.scatter(y, x, s=150, c="#e8a33a" if tilt else "#4c8bd6", edgecolors="k", lw=0.4)
        ax.text(y, x, chess.square_name(sq), ha="center", va="center", fontsize=5.5, color="white")
    for k, v in br.cal["squares"].items():
        ax.scatter(v["y"], v["x"], s=320, facecolors="none", edgecolors="red", lw=1.4)
    if br.park:
        ax.scatter(br.park["y"], br.park["x"], marker="s", s=60, c="k")
        ax.text(br.park["y"] + 12, br.park["x"], "PARK", fontsize=8)
    ax.scatter(0, 0, marker="^", s=120, c="k")
    ax.text(10, 5, "robot base", fontsize=8)
    ax.set_xlabel("robot Y (mm)")
    ax.set_ylabel("robot X (mm)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_title("Board in robot coordinates. Blue: gripper straight down; orange: rank 8,\n"
                 "gripper tilted back 30° on J5 (J3 limit). Red rings: taught corners.", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "board_robot_map.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    system_diagram()
    recognition_examples()
    board_robot_map()
    print("figures written to", DOCS)

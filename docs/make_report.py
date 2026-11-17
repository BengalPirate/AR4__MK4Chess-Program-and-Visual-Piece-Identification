"""Build the Checkpoint 2 team report PDF.

Run from the repository root:  python docs/make_report.py
(after python docs/make_figures.py). Writes docs/Team2_Checkpoint2_Report.pdf.
Screenshots are taken from results/screens/ (chess_demo.py, key S).
"""
import glob
import os
import re
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/BengalPirate/AR4__MK4Chess-Program-and-Visual-Piece-Identification"
OUT = os.path.join(ROOT, "docs", "Team2_Checkpoint2_Report.pdf")

ss = getSampleStyleSheet()
H1, H2, BODY = ss["Title"], ss["Heading2"], ss["BodyText"]
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=10.5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5)


def P(t, st=BODY):
    return Paragraph(t, st)


def table(rows, widths, header=True):
    data = [[P(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")))
    t.setStyle(TableStyle(style))
    return t


def img(path, width, max_h=None):
    from reportlab.lib.utils import ImageReader
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    w, h = ImageReader(path).getSize()
    if max_h and width * h / w > max_h:
        width = max_h * w / h
    return Image(path, width=width, height=width * h / w)


def md_section(path, title):
    """The bullet lines of one '### title' section of a results markdown file."""
    txt = open(os.path.join(ROOT, path)).read()
    m = re.search(r"### " + re.escape(title) + r".*?\n\n(.*?)\n\n", txt, re.S)
    return dict(re.findall(r"- (.*?): \*?\*?([\d.]+%)", m.group(1))) if m else {}


def contributions():
    rows = []
    for line in open(os.path.join(ROOT, "CONTRIBUTIONS.md")):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] not in ("Team member", "---") and not cells[0].startswith("-"):
            rows.append(cells)
    return rows


def main():
    web = md_section("results/accuracy_webcam.md", "All held-out webcam boards")
    mid = md_section("results/accuracy_webcam.md", "Other positions")
    start = md_section("results/accuracy_webcam.md", "Start positions")
    photo = md_section("results/accuracy_cnn.md", "Held-out phone photos")
    screens = sorted(glob.glob(os.path.join(ROOT, "results", "screens", "*.png")))

    doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            title="Team 2 - Vision-Guided AR4 Chess, Checkpoint 2")
    W = doc.width
    s = []
    s += [P("Vision-Guided AR4 Chess – Checkpoint 2:<br/>Chess Program and Visual Piece Identification", H1),
          P("<b>Team 2:</b> Brandon Newton, Roshan Padhan, Andrew Overton<br/>"
            f"<b>Repository:</b> <link href='{REPO_URL}'>{REPO_URL}</link><br/>"
            "<b>Robot:</b> Annin AR4 MK4, Teensy 4.1, Annin firmware 6.7.1, servo gripper on an "
            "Arduino Nano &nbsp; <b>Camera:</b> Logitech C930e (1280×720)<br/>"
            f"<b>Report generated:</b> {time.strftime('%B %d, %Y')}"),
          Spacer(1, 8),
          P("<b>Summary.</b> A fixed webcam sees the whole board. A calibrated camera model cuts "
            "out each of the 64 squares, and a fine-tuned ResNet-18 labels each one as empty or "
            "one of the twelve pieces (type and colour). The recognised board is never trusted "
            "on its own: a python-chess game keeps the position, the turn and the castling and "
            "en passant rights, and a change on the board is accepted only when one legal move "
            "explains it clearly; otherwise the program pauses for the operator. A selected "
            "legal move is turned into pick-and-place operations and checked with Annin's "
            "kinematics before the arm moves. On 38 webcam frames never used for training (20 start "
            "positions, 18 frames of 8 other positions) the "
            f"labels were {web.get('exact label (empty / type+colour)', '?')} exact "
            f"(colour {web.get('colour, on occupied squares detected as occupied', '?')}); "
            "in 88 move-inference tests no wrong move was accepted; and the real arm moved all "
            "32 pieces of a start position two ranks forward.")]

    s += [P("1. Architecture and interfaces", H2),
          img("docs/system_diagram.png", W * 0.6, max_h=3.3 * inch),
          table([["Interface", "Data passed", "Contract"],
                 ["Camera → perception", "BGR frame 1280×720",
                  "board_vision.BoardGeometry (from config/board_image.json) gives 64 crops, "
                  "112×224 px, square + space above it"],
                 ["Perception → chess logic", "64×13 log-probabilities (5 frames averaged)",
                  "labels in order . P N B R Q K p n b r q k; square 0 = a1"],
                 ["Chess logic → planner", "chess.Move + Game (board, tray contents)",
                  "only legal moves; illegal input rejected before planning"],
                 ["Planner → motion", "list of Op(kind pick/place/pause, where e2 / tray:3 / reserve, "
                  "piece, xyz)", "xyz from config/board_robot.json; pause ops stop execution"],
                 ["Motion → arm / gripper", "joint targets (RJ at % speed), gripper angle",
                  "every point preflighted; interlock held; E-stop flag stops the sequence"]],
                [W * 0.2, W * 0.35, W * 0.45])]

    s += [PageBreak(), P("2. Chess program and legal-move handling", H2),
          P("The position and rules come from <b>python-chess</b> (GPL-3.0): board "
            "representation, turn, castling rights, en passant square, legal-move generation "
            "and check / checkmate. Moves can be selected by an engine (Stockfish through "
            "python-chess's UCI interface when installed, otherwise a small built-in search) "
            "or by the operator, who clicks a piece and one of its highlighted legal "
            "destinations. Typed or detected moves that are not legal are rejected with a "
            "message; nothing is planned for them. Our integration (chess_logic.Game) adds the "
            "tray of captured pieces, undo, and the inference of the opponent's move from the "
            "camera."),
          P("<b>Move inference from the camera.</b> For every legal move, the position after it "
            "is scored against the 64 square probabilities. The best move is accepted only if "
            "it beats the second best by a margin (log-likelihood 2.0) and no square strongly "
            "contradicts it (−log p ≤ 4.0). A misread square therefore cannot create a move: it "
            "either does not change the decision or the program pauses and asks the operator "
            "to correct the square (click it, type the piece)."),
          table([["Special move", "Game state", "Physical handling (planner.py)"],
                 ["Capture", "captured piece added to the tray list",
                  "captured piece to the next tray slot first, then the moving piece"],
                 ["En passant", "en passant square kept by the game, not the image",
                  "pawn taken from beside the destination square"],
                 ["Castling", "rights kept through the game (a rook moved and back has lost them)",
                  "king two squares, then the rook over it"],
                 ["Promotion", "defaults to queen; choice typed as e7e8n etc.",
                  "pawn to the tray; new piece from the tray, else the spare queen at the reserve "
                  "spot; an unavailable under-promotion pauses for the operator"]],
                [W * 0.15, W * 0.35, W * 0.5]),
          Spacer(1, 4),
          P("Still to complete physically before the final: teaching the tray and reserve "
            "positions and running captures and a promotion on the arm (planned and tested in "
            "simulation; unit tests cover all four special moves).", SMALL)]

    s += [PageBreak(), P("3. Visual identification of square, piece type and colour", H2),
          P("<b>Board and squares.</b> With the board empty, OpenCV finds the 49 inner tile "
            "corners and a camera model (focal length, radial distortion, pose) is fitted to "
            "them; fit error below 1 px. From it each square's crop is an upright box one "
            "square wide and 2.2 squares tall, so it contains the piece standing on the square "
            "seen in perspective. The live view overlays the grid so camera movement is "
            "visible; recalibration takes one minute."),
          P("<b>Classifier.</b> ResNet-18 with ImageNet weights, all layers fine-tuned, 13 "
            "outputs. Training data: 1,592 iPhone photos of our own set from many angles (12 "
            "classes; HEIC decoded, piece found with the OWL-ViT open-vocabulary detector, then "
            "cropped and degraded to look like webcam crops) plus 30 labelled webcam boards. "
            "Photos are split train / validation / test 70 / 15 / 15 % by runs of 10 "
            "consecutive shots, so near-duplicates never cross the split. Webcam test boards "
            "are kept in a separate folder that training never reads."),
          table([["Test set", "Squares", "Exact label", "Occupancy", "Colour", "Type"],
                 ["Phone photos, test split", "483", photo.get("exact label (empty / type+colour)", "?"),
                  photo.get("occupancy", "?"), photo.get("colour, on occupied squares detected as occupied", "?"),
                  photo.get("piece type, on occupied squares", "?")],
                 ["Webcam start positions (20 frames)", "1280",
                  start.get("exact label (empty / type+colour)", "?"), start.get("occupancy", "?"),
                  start.get("colour, on occupied squares detected as occupied", "?"),
                  start.get("piece type, on occupied squares", "?")],
                 ["Webcam other positions (18 frames, 8 positions)", "1152",
                  mid.get("exact label (empty / type+colour)", "?"), mid.get("occupancy", "?"),
                  mid.get("colour, on occupied squares detected as occupied", "?"),
                  mid.get("piece type, on occupied squares", "?")],
                 ["All webcam frames (38)", "2432", web.get("exact label (empty / type+colour)", "?"),
                  web.get("occupancy", "?"), web.get("colour, on occupied squares detected as occupied", "?"),
                  web.get("piece type, on occupied squares", "?")]],
                [W * 0.34, W * 0.1, W * 0.14, W * 0.14, W * 0.14, W * 0.14]),
          P("Colour and type are measured on occupied squares that were detected as occupied. "
            "4 of the 20 start-position frames come from the first, lower camera angle (89–94 %); "
            "at the final angle start positions score 98.4–100 %. The other positions include three "
            "mid-game positions (one with both sides castled), colours swapped, two scrambled "
            "arrangements and a sparse endgame.", SMALL),
          img("docs/recognition_examples.png", W),
          P("<b>Errors and ambiguous cases.</b> All 9 errors on the three mid-game boards are pieces "
            "partly hidden behind a taller piece in the camera's line of sight (h2 pawn behind "
            "the g1 king, f8 rook behind the g8 king, e8 king and e7 queen in line, c7 / f7 "
            "pawns behind back-rank pieces); 7 of the 9 were below the confidence threshold. "
            "On the five other arrangements every error (1–2 squares per frame) was below the "
            "threshold. "
            "Other ambiguous cases: queen vs king and bishop vs pawn when only the top of the "
            "piece is visible, and black pieces on dark squares. At the first camera angle the "
            "e2 pawn was hidden behind the e1 king; every move test there paused (safe but "
            "unusable), and raising the camera fixed it.", SMALL),
          P("<b>Move tests.</b> On the start-position boards, 88 moves were simulated by "
            "swapping squares (knight out and back): 56 correct and accepted, 32 paused (all "
            "from the lower camera angle), <b>0 wrong moves accepted</b>. Confusion matrix: results/confusion_webcam_all.png.", SMALL)]

    s += [PageBreak(), P("4. Board-state integration and robot interface", H2)]
    if screens:
        s += [P("Screenshots of chess_demo.py: live recognition (labels on every square), the "
                "recognised position reconciled with the game, and an engine-selected legal "
                "move with its source and destination squares and the robot plan.")]
        for path in screens[-3:]:
            s += [img(path, W, max_h=3.4 * inch), Spacer(1, 4)]
    else:
        s += [P("<i>Screenshots: results/screens/ (chess_demo.py, key S).</i>")]
    s += [P("<b>From move to commands.</b> planner.plan_move turns a legal move into pick / "
            "place operations; BoardRobot maps a square to robot X, Y and board height by "
            "bilinear interpolation between four taught corners (closed jaw tips touching the "
            "centres of a1, h1, h8, a8) and a plane fit; small per-rank corrections measured on "
            "the arm are added. motion.Executor then, for each operation: rises to travel "
            "height, opens the gripper, moves above the square, descends vertically in 15 mm "
            "steps to the grip height, closes, and retreats vertically. Before the first "
            "motion, every point of the plan is checked reachable within the joint limits."),
          table([["Piece", "Grip height (mm)", "Gripper"],
                 ["Pawn", "23", "close to 20° (foam pads touching)"],
                 ["Knight", "35", "close to 16° (narrow body)"],
                 ["Bishop", "45", "close to 21°"],
                 ["Rook, queen", "35, 57", "open 47°, close until 20° or motor stall"],
                 ["King", "68", "open 47°, close until 18° or stall (cross may face either way)"]],
                [W * 0.2, W * 0.2, W * 0.6]),
          P("Travel height clears the tallest piece (95 mm) by 15 mm plus the part of the "
            "carried piece below the jaws. Rank 8 is too close to the base to reach with the "
            "gripper straight down (J3 would need 55–63°, its stop is +52°; no joint combination "
            "avoids this), so rank 8 is worked with the gripper tilted back 30° on J5, and a "
            "piece picked tilted is placed at the same tilt.", SMALL),
          img("docs/board_robot_map.png", W * 0.62),
          P("<b>Arm test.</b> At 10 % speed the arm moved every piece of a start position two "
            "ranks forward (32 pick-and-place operations, including all of rank 8 tilted). "
            "Problems found and fixed on the way are listed in results/test_log.md: pawn grip "
            "height raised 3 mm, rank-8 position corrections, keeping the tilt when placing, "
            "king grip for either cross orientation, and a gripper that cannot be fooled by "
            "its own motor current.", SMALL)]

    s += [PageBreak(), P("5. Safety and integration", H2),
          table([["Layer", "Behaviour"],
                 ["Emergency stop", "Hardware E-stop; the firmware's E-stop flag is detected and "
                  "the sequence stops with the arm left in place."],
                 ["Joint / workspace limits", "Firmware limits plus our measured J5 range "
                  "(−99.6…+106.7°, used 2° inside) so the gripper cannot fold into the forearm; "
                  "every move simulated with Annin's kinematics; each joint move stays within "
                  "3 mm of its straight line."],
                 ["Approach and retreat", "Vertical only, from a travel height that clears all "
                  "pieces; the whole plan is preflighted before the first motion."],
                 ["Reduced speed", "10 % for all testing; the serial library refuses more than 25 %."],
                 ["Shared board", "motion.BoardLock: an exclusive lock file (can be on a shared "
                  "filesystem); only the arm holding it moves over the board."],
                 ["Uncertain vision / inconsistent board / illegal move",
                  "Move inference pauses below the margin or on a contradicting square; a board "
                  "that does not match the game state, a camera that moved, or an illegal move "
                  "blocks execution until the operator corrects it."],
                 ["Gripper", "Stall stop eases the servo off; a jam check refuses to continue "
                  "if the jaws do not open freely."]],
                [W * 0.27, W * 0.73])]

    s += [P("6. Testing and validation", H2),
          P("<b>Unit tests</b> (pytest tests, 25 pass): turn tracking and illegal moves, pinned "
            "piece, castling rights beyond the image, en passant window, promotion with tray, "
            "undo, engine legality, move inference (simple, none, one misread square, castling "
            "and capture, inconsistent board pauses), square and tray mapping, plans for quiet "
            "moves / captures / en passant / castling / promotion / illegal moves, gripper-down "
            "jaw direction and J5 window, vertical descent, unreachable plan refused before "
            "moving, interlock. <b>Vision</b>: tests/eval_webcam.py on held-out boards (tables "
            "above). <b>Arm</b>: hover, grip-and-lift for each piece type, full two-rank shift "
            "(results/test_log.md).")]

    s += [P("7. Known limitations", H2),
          P("• Pieces hidden behind taller pieces are the main recognition error; a higher or "
            "second camera would reduce them.<br/>"
            "• The camera mount moved several times during testing; the grid overlay shows it, "
            "and recalibration needs an empty board.<br/>"
            "• Tray and reserve not yet taught; captures and promotion not yet run on the arm.<br/>"
            "• With the foam pads the gripper current cannot tell an empty grip from a held "
            "piece, so a missed pick is detected by the next camera scan, not the gripper.<br/>"
            "• Rank-8 positions needed small hand-measured corrections (2–5 mm).")]

    s += [P("8. Plan for the final (November 19)", H2),
          P("1. Fix the camera mount and recalibrate; add webcam boards from mid-game positions "
            "to training. 2. Teach the tray and reserve spots; test capture, castling, en "
            "passant and promotion on the arm. 3. After every robot move, scan and compare with "
            "the expected position before the opponent moves (detects a missed or dropped "
            "piece). 4. Full games at 10 %, then at the final speed. 5. Agree the interlock "
            "procedure for the shared board.")]

    s += [PageBreak(), P("9. Individual contributions", H2)]
    rows = contributions()
    s += [table([["Team member", "Contribution"]] + rows, [W * 0.22, W * 0.78]) if rows
          else P("<i>See CONTRIBUTIONS.md.</i>"),
          P("Each member explains their part and the interfaces between perception, chess "
            "logic and motion during the live demonstration.", SMALL)]

    s += [P("10. Sources and AI assistance", H2),
          P("python-chess (GPL-3.0) – board, rules, legal moves, UCI engine interface; "
            "Stockfish (GPL-3.0), optional engine; PyTorch / torchvision – ResNet-18 and ImageNet "
            "weights; OWL-ViT (google/owlvit-base-patch32, Hugging Face transformers) – piece "
            "boxes for building the photo dataset only; OpenCV – camera, corner detection, image "
            "processing; scikit-learn – baseline classifier; Annin Robotics AR4 HMI 6.7 "
            "kinematics module and Teensy firmware 6.7.1 (unmodified)."),
          P("<b>AI assistance.</b> Claude (Anthropic), used through Claude Code, wrote and "
            "debugged much of the code, ran training and tests, and drafted the documentation. "
            "The team built and wired the hardware, calibrated and operated the arm, set up and "
            "labelled the test positions, judged every physical test, chose the changes, and "
            "reviewed the results."),
          P("<b>Reproducing.</b> Setup, calibration order and commands are in README.md; "
            "calibration and configuration files are in config/; the trained model is "
            "models/piece_cnn.pt; this report is rebuilt with python docs/make_figures.py and "
            "python docs/make_report.py.")]

    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

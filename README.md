# AR4 MK4 – Team 2 Chess Program and Visual Piece Identification

Checkpoint 2 of the vision-guided AR4 chess assignment. A webcam watches a
physical chess board; every square is classified as empty or one of the twelve
pieces (type **and** colour); the recognised position is reconciled with a
chess program that keeps the full game state (turn, castling and en passant
rights, captured pieces); a legal move is selected; and the move is turned into
pick-and-place commands for an Annin Robotics AR4 MK4 arm with a servo gripper.
Anything uncertain pauses for the operator instead of moving the arm.

Report: [`docs/Team2_Checkpoint2_Report.pdf`](docs/Team2_Checkpoint2_Report.pdf)

## Results at a glance

| Test | Result |
|---|---|
| Phone photos, held-out test split (483 squares) | **97.1 %** exact label (empty / type + colour) |
| Webcam boards never trained on (14 boards: 11 start positions, 3 mid-game) | **96.0 %** exact, colour 100 % on detected pieces |
| Mid-game positions (incl. both sides castled) | 95.3 % each; every error is a piece hidden behind a taller one |
| Move inference on webcam boards (88 simulated moves) | 56 correct and accepted, 32 paused for the operator, **0 wrong moves accepted** |
| Unit tests (`pytest tests`) | 25 / 25 pass: legal / illegal moves, castling rights, en passant, promotion, undo, move inference, special-move plans, J5 window, interlock |
| Real arm, 10 % speed | all 32 pieces moved two ranks forward (pawns, back ranks, tilted grips on rank 8) |

Details: `results/accuracy_cnn.md` (photos + webcam), `results/test_log.md`
(arm and vision tests with the problems found and fixed).

## How it works

```
webcam ─► board_vision.py ─► pieces.py ─────────► chess_logic.py ─► planner.py ─► motion.py ─► AR4 + gripper
          (calibrated grid,  (ResNet-18 per        (python-chess     (move → pick/  (checked joint   (ar4link.py,
           64 square crops)   square: 13 labels,    game state, legal  place ops,      moves, tilt,     gripper.py)
                              probabilities)        move inference,    tray, reserve)  interlock)
                                                    engine)
```

1. **Board → squares** (`board_vision.py`, `calibrate_camera.py`). With the
   board empty, OpenCV finds the 49 inner tile corners; a camera model (focal
   length, lens distortion, pose) is fitted to them (fit error < 1 px). Each
   square is cut out as an upright crop one square wide and 2.2 squares tall,
   so the piece standing on it is included (112 × 224 px).
2. **Squares → pieces** (`pieces.py`, `finetune.py`). A ResNet-18 (ImageNet
   weights, all layers fine-tuned) outputs 13 probabilities per square:
   empty, P N B R Q K, p n b r q k. Training data: 1,592 iPhone photos of our
   own pieces (`photo_dataset.py` decodes the HEIC files, OWL-ViT finds the
   piece box, crops are made to look like webcam crops) plus 30 labelled webcam
   boards. Photos are split by runs of 10 consecutive shots so near-identical
   pictures never sit on both sides of the split.
3. **Pieces → game** (`chess_logic.py`). The game state lives in
   python-chess, not in the image. After a move, 5 frames are averaged and every
   *legal* move is scored by how well the resulting position explains the
   probabilities. The best move is accepted only if it beats the runner-up by a
   clear margin and no square disagrees strongly; otherwise the program pauses
   and the operator corrects the squares (click a square, type the piece).
4. **Move → robot** (`planner.py`, `motion.py`). A move becomes pick / place
   operations on squares, tray slots or the reserve spot (special moves below).
   Each operation is simulated with Annin's own kinematics before anything
   moves: approach at travel height, vertical descent in 15 mm steps, grip,
   vertical retreat. The whole plan is checked for reach and joint limits first.

### Special moves

| Move | Physical handling |
|---|---|
| Capture | captured piece to the next tray slot first, then the move |
| En passant | the captured pawn is taken from beside the destination, not from it |
| Castling | king two squares first, then the rook over it |
| Promotion | pawn to the tray; new piece from the tray if one of that kind was captured, else the spare queen from the reserve spot; under-promotion with no piece available pauses |

Castling rights, the en passant square and the turn are kept by the game
state; a position read from the camera alone is never trusted for them.

## Safety

- Hardware E-stop (the firmware's E-stop flag is detected and stops the sequence).
- Joint limits from the firmware plus our measured J5 range (−99.6…+106.7°,
  used with a 2° margin) so the gripper can never fold into the forearm.
- Every point of a plan is checked reachable before the first move; every joint
  move stays within 3 mm of the straight line it stands for.
- Vertical approach and retreat only, at a travel height that clears the
  tallest piece plus the part of the carried piece hanging below the jaws.
- 10 % speed for all testing.
- One-arm interlock (`motion.BoardLock`, a lock file that can sit on a shared
  filesystem): only the arm holding it may move over the board.
- Uncertain recognition, a board that does not match the game state, or an
  illegal move pauses; the arm is not moved.
- Gripper: a stall stop backs the servo off when it squeezes too hard; a jam
  check refuses to continue if the jaws do not open freely.

## Setup

1. Python 3.12 virtual environment: `pip install -r requirements.txt`.
2. Annin's kinematics module: from the AR4 HMI 6.7 source, build
   `ARrobots/robot_kinematics` for your Python (cmake + pybind11) and set
   `HMI_DIR` in `kin.py` if the HMI is not at `~/AR4/AR4 HMI Interface 6.7 source`.
3. Firmware: Teensy 4.1 with Annin `AR4_teensy41_sketch_v6.7.1` (unmodified,
   in `firmware/`), gripper Nano with `AR4_nano_sketch_v1.5`.
4. Optional: `stockfish` on the PATH; without it a small built-in engine is used.
5. Phone-photo dataset (only for retraining): put the class folders
   (`white_pawn/…`) in `~/Desktop/Chess_Classification/wimages`, then
   `python photo_dataset.py decode` and `python photo_dataset.py detect`.

## Calibration (in this order)

| Step | Tool | Output |
|---|---|---|
| Joint calibration J1–J4, J6 (limit switches; J5 excluded) | Annin HMI | controller angles |
| J5 zero by eye (gripper in line with the forearm) and its safe range | `manual_calib.py` | `config/joint_limits.json` |
| Camera: empty board in view | `calibrate_camera.py` | `config/board_image.json` |
| Board in robot coordinates: closed jaw tips touching a1, h1, h8, a8; tray; PARK | `robot_setup.py` | `config/board_robot.json` |
| Check reach / heights on real pieces | `reach_check.py`, `hover_test.py` | – |

Measured grip settings (`config.py`): grip heights above the board P 23,
N 35, B 45, R 35, Q 57, K 68 mm; gripper tool length 64 mm past the J6 flange;
rank 8 is too close to the base to reach straight down (J3 would pass its
+52° stop), so it is worked with the gripper tilted back 30° on J5, and a piece
picked tilted is placed at the same tilt.

## Running

```
python chess_demo.py            # camera + chess; robot plan shown, not executed
python chess_demo.py --robot    # also connects the AR4 and gripper; X executes
```

Keys are listed at the top of `chess_demo.py` (SPACE scan, P engine move,
M manual / AI mode, X execute, Y live labels, V set game from the board, U undo …).

Other tools: `live_view.py` (aim the camera), `save_board.py` (save a labelled
board for training or testing), `tests/eval_webcam.py` (accuracy and
move-inference test on saved boards), `shift_test.py` (arm test: move pieces
two ranks), `gripper_test.py`, `jog_joint.py`.

## Files

| File | Purpose |
|---|---|
| `chess_demo.py` | Main program: live view, recognition, game, move selection, robot plan / execution |
| `chess_logic.py` | Game state (python-chess), legal-move inference from recognition, engine |
| `board_vision.py` | Camera discovery, board geometry, square crops, empty-board reference |
| `pieces.py` | Piece classifier (fine-tuned ResNet-18; logistic-regression fallback) |
| `photo_dataset.py`, `train_photos.py`, `finetune.py`, `train.py` | Dataset building and training |
| `planner.py` | Moves → pick / place ops; square, tray and reserve positions |
| `motion.py` | Executes ops as checked joint moves; interlock; preflight |
| `kin.py` | Annin kinematics wrapper; gripper-down and tilted inverse kinematics; limits |
| `ar4link.py`, `gripper.py` | Serial links to the Teensy (firmware 6.7.1) and the gripper Nano |
| `calibrate_camera.py`, `robot_setup.py`, `manual_calib.py` | Calibration tools |
| `config.py`, `config/` | All settings and our calibration files |
| `models/` | Trained classifier (`piece_cnn.pt`) |
| `data/`, `data_webcam_test/` | Labelled webcam boards for training / held-out testing |
| `data_photos/boxes.json` | Piece boxes for the phone photos |
| `results/` | Accuracy reports, confusion matrices, test log, screenshots |
| `tests/` | Unit tests and the webcam evaluation |
| `firmware/` | Annin firmware (unmodified, with Annin's license) |

## Sources and AI assistance

- [python-chess](https://github.com/niklasf/python-chess) (GPL-3.0): board
  representation, legal moves, UCI engine interface.
- [Stockfish](https://stockfishchess.org) (GPL-3.0), optional engine.
- [PyTorch / torchvision](https://pytorch.org): ResNet-18 with ImageNet weights.
- [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) via Hugging Face
  transformers: piece boxes in the phone photos (dataset building only).
- [OpenCV](https://opencv.org): camera, chessboard corners, image processing.
- Annin Robotics AR4 HMI 6.7 (kinematics module) and Teensy firmware 6.7.1.
- AI assistance: Claude (Anthropic) was used through Claude Code to write and
  debug much of the code, run the training and tests, and draft this
  documentation. The team set up and operated the hardware, chose the
  approach, labelled positions, judged every physical test and reviewed the
  results.

## Known limitations

- Pieces hidden behind taller pieces from the camera's viewpoint are the main
  error source (all mid-game errors). A higher or second camera would help.
- The camera mount has moved several times; recalibrate (empty board, 1 min)
  whenever the grid overlay drifts.
- Captures, promotion and the tray are planned and tested in simulation; the
  tray position still has to be taught and a capture run on the arm.
- The gripper's current sensor cannot tell an empty grip from a held piece with
  the foam pads, so grips are position-based; a missed pick is caught by the
  next camera scan, not by the gripper.

Team members and contributions: [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

# AR4 MK4 – Team 2 Chess Program and Visual Piece Identification

Vision-guided AR4 chess (final, Milestone 6; started as Checkpoint 2). Two webcams watch a
physical chess board; every square is classified as empty or one of the twelve
pieces (type **and** colour); the recognised position is reconciled with a
chess program that keeps the full game state (turn, castling and en passant
rights, captured pieces); a legal move is selected; and the move is turned into
pick-and-place commands for an Annin Robotics AR4 MK4 arm with a servo gripper.
Anything uncertain pauses for the operator instead of moving the arm.

Final report: [`docs/Team2_Final_Report.pdf`](docs/Team2_Final_Report.pdf) – rebuild with
`python docs/game_stats.py && python docs/make_final_report.py`.
Checkpoint 2 report: [`docs/Team2_Checkpoint2_Report.pdf`](docs/Team2_Checkpoint2_Report.pdf)

## Results at a glance

| Test | Result |
|---|---|
| Phone photos, held-out test split (483 squares) | **97.1 %** exact label (empty / type + colour) |
| Webcam frames never trained on (38: 20 start positions, 18 frames of 8 other positions) | **98.1 %** exact, colour 100 % on detected pieces |
| Other positions (mid-games incl. both sides castled, colours swapped, scrambled, endgame) | 98.4 % exact; mid-game errors are pieces hidden behind taller ones |
| Move inference on webcam boards (88 simulated moves) | 56 correct and accepted, 32 paused for the operator, **0 wrong moves accepted** |
| Unit tests (`pytest tests`) | 38 / 38 pass: legal / illegal moves, castling rights, en passant, promotion, undo, move inference, special-move plans, J5 window, interlock, pick checks and retries, diagonal camera calibration |
| Real arm, 10 % speed | all 32 pieces moved two ranks forward (pawns, back ranks, tilted grips on rank 8) |
| Two webcams, 40 held-out boards | **99.4 %** exact, colour 100 % (current model, `results/accuracy_cnn_diag.md`) |
| Live games 2026-11-20 and 11-22 (person White, robot Black, Stockfish 16) | 28 robot moves incl. 6 captures to the tray; 60 / 68 picks first try; every move verified by the cameras (`results/games/summary.md`) |
| Live game 2026-11-22 | full game to **checkmate by the robot**: 1.h4 e5 2.Rh3 d5 3.Re3 Nc6 4.a4 Nf6 5.Ra3 Ng4 6.Rac3 d4 7.Rxc6 dxe3 8.fxe3 Qxh4+ 9.g3 Qxg3# (`results/games/`) |
| Scope | one AR4 against a person – confirmed as sufficient by the course advisor, who watched it play live (see Known limitations) |

Details: `results/accuracy_cnn.md` (photos), `results/accuracy_webcam.md` (webcam,
`python tests/webcam_report.py`), `results/test_log.md`
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

## Cameras

Two webcams watch the board (`config.CAMERAS`):

| View | Camera | Position | Calibration file |
|---|---|---|---|
| `front` | Logitech C930e | diagonally across the h1 corner (since 2026-11-20) | `config/board_image.json` |
| `side` | Logitech C920 PRO | beside the a-file (Black on the left of its image) | `config/board_image_side.json` |

`cameras.py` opens every attached camera by name, preferring the direct USB
device over the Parallels-virtualised copy (which gives no frames), so the
`/dev/videoN` numbers can change freely. Every scan classifies all 64 squares in
each view and fuses the two per-square probability tables (weighted geometric
mean, `pieces.fuse`), so a square hidden from one camera can still be read from
the other, and a square the cameras disagree on becomes uncertain and pauses
instead of being guessed. Each camera has its own empty-board reference and bump
check; a bumped camera pauses the game. `live_view.py` shows both views for aiming.

**USB bandwidth fix (resolution/frame rate).** Both webcams run at 1280 x 720
(the resolution the classifier crops were made at), but with both open at the
default 30 fps the second camera delivered no frames at all: the Parallels USB
passthrough cannot reserve the isochronous bandwidth for two 720p/30 fps MJPG
streams. Measured over 4 s with both cameras open:

| Setting (both cameras, MJPG) | front frames | side frames |
|---|---|---|
| 1280 x 720 @ 30 fps | 83 | **0** |
| 1280 x 720 @ 15 fps | 42 | 43 |
| 1280 x 720 @ 10 fps | 25 | 26 |
| 1024 x 576 @ 30 fps | 89 | 90 |
| 640 x 480 @ 30 fps | 89 | 90 |

We keep 1280 x 720 and cap the frame rate at 15 fps (`config.CAMERA_FPS`, applied
in `board_vision.open_camera`); a scan needs only 5 frames, and full resolution
keeps the crops sharp. Lowering the resolution would also work but would change
the crops the classifier was trained on. The decoder occasionally reports a
"Corrupt JPEG data" frame; averaging over 5 frames per scan absorbs it.

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
| Cameras: empty board in view of both | `calibrate_camera.py` (or K in the app) | `config/board_image.json`, `config/board_image_side.json`, `results/calibration_<camera>.png` |
| Board in robot coordinates: closed jaw tips touching a1, h1, h8, a8; tray; PARK | `robot_setup.py` | `config/board_robot.json` |
| Check reach / heights on real pieces | `reach_check.py`, `hover_test.py` | – |
| Camera nudged with pieces on the board | `camera_realign.py front` (`--dry-run` to only measure) | updated `config/board_image*.json`, old file in `config/old/` |

Measured grip settings (`config.py`): grip heights above the board P 23,
N 35, B 45, R 35, Q 57, K 68 mm; gripper tool length 64 mm past the J6 flange;
rank 8 is too close to the base to reach straight down (J3 would pass its
+52° stop), so it is worked with the gripper tilted back 30° on J5, and a piece
picked tilted is placed at the same tilt.

## Running

```
python chess_demo.py            # camera + chess; robot plan shown, not executed
python chess_demo.py --robot    # also connects the AR4 and gripper; X executes
# live game, hands-free: start from the board as the cameras see it, robot plays Black
python chess_demo.py --robot --auto --from-board --robot-color black
#   --to-move robot     non-standard start position with the robot to move
#   --skill 0-20        Stockfish strength (default 20)
#   --fix f1=B,f7=p     square corrections applied until the game starts
#   --tray nRpP         restart: tray slots already holding captured pieces (. = empty)
```

In auto play the game scans every 1.5 s; the player's move is accepted when two scans agree,
the robot replies at once and the board is verified before the next turn. TAB pauses/resumes;
any stop pauses until the operator fixes the board and presses TAB. Do not press other keys in the
game window during a stop (L / I / T save training frames or start retraining). Keep the host computer
awake (macOS: `caffeinate -is`): if it sleeps, the VM's USB devices drop.

Keys are listed at the top of `chess_demo.py` (SPACE scan, P engine move,
M manual / AI mode, X execute, Y live labels, V set game from the board, U undo …).

Other tools: `live_view.py` (both cameras side by side, to aim them), `save_board.py` (save a labelled
board for training or testing), `tests/eval_webcam.py` (accuracy and
move-inference test on saved boards), `shift_test.py` (arm test: move pieces
two ranks), `gripper_test.py`, `jog_joint.py`, `grip_measure.py` (where the jaws meet each
piece type, after changing the gripper pads), `camera_realign.py`.

## Files

| File | Purpose |
|---|---|
| `chess_demo.py` | Main program: live view, recognition, game, move selection, robot plan / execution |
| `chess_logic.py` | Game state (python-chess), legal-move inference from recognition, engine |
| `board_vision.py` | Camera discovery, board geometry, square crops, empty-board reference |
| `cameras.py` | All board cameras: capture threads, per-camera calibration and bump check |
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

- **One arm against a person.** The assignment text describes two AR4s alternating turns; the
  course advisor confirmed that the robot playing a person is sufficient, and a person plays
  White. For two arms, each would run its own program with its own taught board and the shared
  `motion.BoardLock` (set `AR4_BOARD_LOCK` to a shared path); the interlock is unit-tested only.
- The side camera reads h4 poorly in the current setup (one false stop on 2026-11-22).
- The gripper's foam pads wear; worn or too-thick pads raise the servo's resting current and
  cause false "jammed" stops – replace them (`grip_measure.py` to check).
- Castling, en passant and promotion are planned and unit-tested but not yet run on the arm.
- Rank 8 is at the edge of reach (worked with the gripper tilted 30°); rank-8 picks are the
  least reliable and the camera offset reading there is unreliable. Third rank-8 try aims
  15 mm toward the robot (`config.RANK8_THIRD_TRY_BACK_MM`).
- Pieces hidden behind taller pieces are the main recognition error; the side camera helps.
- Cameras get bumped: the tile-alignment check pauses the game; `camera_realign.py` (pieces on)
  or recalibrate with an empty board.
- A USB re-attach can restart the Teensy, which then reads all joints as 0: do not move the
  arm until the angles are restored or the arm is recalibrated.

Team members and contributions: [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

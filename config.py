"""Settings for the Team 2 AR4 chess system. Distances in mm, speeds in percent."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "config")
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
RESULTS_DIR = os.path.join(HERE, "results")

BOARD_IMAGE_FILE = os.path.join(CONFIG_DIR, "board_image.json")   # camera calibration
BOARD_ROBOT_FILE = os.path.join(CONFIG_DIR, "board_robot.json")   # robot calibration
CLASSIFIER_FILE = os.path.join(MODEL_DIR, "piece_classifier.joblib")
JOINT_LIMITS_FILE = os.path.join(CONFIG_DIR, "joint_limits.json")    # manual_calib.py
LIMIT_MARGIN_DEG = 2.0              # moves stay this far inside the manually recorded joint ranges
LOCK_FILE = os.environ.get("AR4_BOARD_LOCK", "/tmp/ar4_board.lock")

# ---------------------------------------------------------------- camera
CAMERA_NAME = "Logitech Webcam C930e"
# Board cameras. "near" = the board side closest to the camera (orientation for
# auto calibration): rank1 / rank8 / fileA / fileH. "file" = its calibration.
# Scans classify every square in every attached view and fuse the results.
CAMERAS = {
    "front": {"name": "Logitech Webcam C930e", "near": "rank1",
              "file": os.path.join(CONFIG_DIR, "board_image.json")},
    "side": {"name": "C920", "near": "fileA",
             "file": os.path.join(CONFIG_DIR, "board_image_side.json")},
}
# Weight of each view when fusing (geometric mean of the probabilities)
VIEW_WEIGHT = {"front": 1.0, "side": 1.0}
SQUARE_MM = 47.0                    # board square size (folding board, 47 mm)
FRAME_SIZE = (1280, 720)
CAMERA_FPS = 15                     # two 720p cameras only fit the VM's USB link together at <= 15 fps
MARKER_IDS = (4, 5, 6, 7)           # ArUco DICT_4X4_50 markers on the mat (bump check)
CAMERA_MOVED_PX = 8.0               # ArUco marker shift (if markers are visible) that invalidates calibration
MIN_TILE_ALIGNMENT = 0.45           # tile-edge correlation with the empty-board reference (tune on the real board)

# ---------------------------------------------------------------- recognition
# A square's crop covers the square and the space above it up to this height,
# in square widths (the king is the tallest piece, about 2 squares tall).
PIECE_BOX_HEIGHT = 2.2
CROP_SIZE = (112, 224)              # (w, h) fed to the feature network
SCAN_FRAMES = 5                     # frames averaged per scan
MIN_SQUARE_CONF = 0.60              # raw recognition: lower -> square marked uncertain
MOVE_MIN_MARGIN = 2.0               # log-likelihood margin best vs 2nd-best move hypothesis
MOVE_MAX_SQUARE_SURPRISE = 4.0      # -log p of any square under the chosen hypothesis

# ---------------------------------------------------------------- chess engine
STOCKFISH_PATHS = ("/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish")
ENGINE_TIME_S = 0.5
FALLBACK_DEPTH = 3

# ---------------------------------------------------------------- robot
# J5 (wrist pitch) must stay where the wide servo gripper cannot fold into the
# forearm: J5 is never auto-calibrated (its limit switch is in the collision
# zone; it is zeroed by eye with j5_zero.py) and every move keeps J5 in this
# window. With the gripper pointing down J5 is ~20-70 deg over the board.
J5_WINDOW = (-97.6, 104.7)        # deg; 0 = gripper in line with the forearm. Measured by eye 2026-11-16: -99.6 .. +106.7, minus 2 deg margin
# J6 may turn: the jaws are held at one direction relative to the board. 45 deg
# to the files opens them toward the diagonal neighbours (1.4 squares away).
GRIPPER_YAW_DEG = 45.0              # jaw direction offset (set by eye in board_teach.py)
SPEED_PCT = 10                      # reduced speed for all testing (ar4link caps at 25)
TOOL_LENGTH_MM = 64.0               # J6 flange -> closed jaw tips: user measured 105 from the J5 hinge; model J5 axis -> flange = 41
MAX_TILT_DEG = 60.0                 # squares unreachable straight down (b8-g8: J3 would pass +52) are
                                    # approached with the gripper tilted in the arm's vertical plane
RANK8_TILT_DEG = 30.0              # rank 8 is always worked tilted back this much with J5 (tips toward the robot); needs >= 25 with the 64 mm tool
# A piece picked tilted on rank 8 is carried at that tilt; it is set down on a free
# square midway and picked up again straight down before going to ranks 1-2, or
# wherever the tilted pose would bring J5 within this margin of its window.
REGRIP_RANKS = (1, 2)
TILT_J5_MARGIN_DEG = 10.0
# Missed grip (cameras still see the piece on its square after the lift): retry
# shifted this far along the a-h direction, first in GRIP_RETRY_FIRST (+1 = toward
# h, -1 = toward a; the camera's hint wins when it has one), then the other way,
# at most GRIP_RETRIES times. 2026-11-17: tilted miss at h5 had the jaws on the a side.
# Before each pick the cameras read how far off centre the piece stands
# (offset_model.py, trained on data_offsets/). Operator's rule: more than 10 mm off in
# x or y -> the pick is shifted 15 mm that way (offset_model.SHIFT_MM). False = off.
OFFSET_CORRECTION = True
# After every set-down on a square: step aside, read the piece's offset with the
# cameras (offset_model); if it is more than PLACE_TOL_MM off in x or y, pick it up
# where it really stands and set it down shifted the other way by that much, up to
# PLACE_FIX_TRIES times (operator's request 2026-11-19).
PLACE_CHECK = True
PLACE_TOL_MM = 4.0
PLACE_FIX_TRIES = 2
PLACE_ASIDE_SQ = 1.5          # step this many squares aside (toward the robot; ranks 7-8: toward rank 1) to look
GRIP_RETRY_MM = 5.0
GRIP_RETRIES = 2
GRIP_RETRY_FIRST = +1
TRAVEL_CLEARANCE = 15               # gap kept above the tallest piece during transit
APPROACH_Z = 40                     # above the piece top before the slow descent
VERTICAL_STEP_MM = 15               # (no longer used for splitting: _vertical is one move, split by PATH_TOL_MM)
# J1 has ~2 deg of slack (2026-11-19, rod + cameras: e5 reached from a5 read 15.7 mm
# toward a, from h5 1.6 mm). Every final approach therefore ends with the base turning
# toward the a-file: first to a point this far toward h at travel height, then onto
# the target, so the slack is always taken up the same way. 0 = off.
BACKLASH_APPROACH_MM = 20.0
# Measured aim error per square (rod_calib.py: blue rod in the jaws, both cameras,
# backlash approach on) -> config/aim_correction.json; planner subtracts it from every
# square's aim. Rank 8: a-h part only. False = off (back to the taught corners alone).
AIM_CORRECTION = True
# Operator's call 2026-11-19 after a real h8 pick (gripper too far toward h with the full
# map): use HALF the measured correction; then 1/3 after the next pick ("a bit too close to missing").
AIM_CORRECTION_SCALE = 1.0 / 3.0
AIM_CORRECTION_FILE = os.path.join(CONFIG_DIR, "aim_correction.json")
PATH_TOL_MM = 3.0
MAX_JOINT_FLIP_DEG = 15.0           # a step turning a joint more than this ...
MAX_DEG_PER_MM = 1.5                # ... and more than this per mm of tip travel is a configuration flip: refused                   # max deviation of a joint move from the straight line

# Piece heights above the board (mm) and grasp height (fingertip height above the
# board surface when gripping). MEASURE THESE on the real set before the final.
PIECE_HEIGHT = {"P": 48, "N": 62, "B": 70, "R": 52, "Q": 84, "K": 95}
# Measured on the real set 2026-11-16 (fingertip height above the board when gripping and releasing)
GRASP_Z = {"P": 23.0, "N": 35.0, "R": 35.0, "B": 45.0, "Q": 57.0, "K": 68.0}   # P checked on the arm; others -5 mm after the first arm test
PLACE_EXTRA_Z = 2.0
# Promotion: a captured piece of the new kind in the tray is swapped in; otherwise
# the pawn stays on the board as a stand-in (both players treat it as the new
# piece; the software tracks it). A spare piece on a taught reserve spot could be
# used instead by setting its FEN letter here (e.g. "q"); not used.
RESERVE_PIECE = None                 # release this far above the grasp height


# Small per-rank position corrections (mm, robot frame: +X away from the robot,
# +Y to the robot's left), from watching the arm 2026-11-16.
RANK_OFFSET_MM = {8: (5.0, 0.0), 7: (0.0, 2.0), 5: (2.0, 0.0)}
SQUARE_OFFSET_MM = {s: (2.0, 0.0) for s in ("c8", "d8", "e8", "f8")}   # overrides the rank offset (middle of rank 8 needed less)
# Used instead of the two above for rank 8 once a8/h8 are taught TILTED (robot_setup step 5)
TILTED_SQUARE_OFFSET_MM = {}
# Whole rank 8 (tilted picks/places), robot frame mm. 2026-11-19: with the re-taught
# tilted a8/h8 the jaws gripped the rook at its far edge (away from the arm) on h8;
# the operator's call: centre 10 mm toward the arm (-X), then 7 mm after the next test. Only rank 8 - the tilt
# correction for ranks 3-7 still uses the taught a8/h8 points unchanged.
RANK8_TILTED_OFFSET_MM = (-7.0, 0.0)
# Tilted SET-DOWNS on ranks 3-7 (a piece carried from rank 8), board mm toward rank 8.
# 2026-11-19: with the backlash approach h5 set-downs landed 5.7 / 7.8 / 8.9 mm toward
# rank 1 -> operator approved +7.5.
TILTED_PLACE_TOWARD_RANK8_MM = 7.5


def travel_z(carrying=None):
    """Fingertip height above the board for transit: clear the tallest piece, plus
    the part of the carried piece that hangs below the fingertips."""
    z = max(PIECE_HEIGHT.values()) + TRAVEL_CLEARANCE
    if carrying:
        z += GRASP_Z[carrying.upper()]
    return z

# Gripper (servo 0 on the Nano): fitted travel 15 (closed) .. 45 (open)
GRIP_OPEN_DEG = 45
GRIP_YAW_PIECE = {p: 0.0 for p in "PNBRQK"}   # extra jaw rotation (deg) on top of BoardRobot.jaw_yaw (jaws parallel to the square lines): all 0
GRIP_OPEN_PIECE = {"P": 45, "N": 45, "B": 45, "R": 47, "Q": 47, "K": 47}   # open before descending; servo end stop ~52
GRIP_SQUEEZE_DEG = {p: 0 for p in "PNBRQK"}   # extra closing after first contact (tightness) - tune with hover_test.py --grip
GRIP_HOLD_DEG = {"P": 20, "N": 16, "B": 21, "R": 20, "Q": 20, "K": 18}   # position grip, close until this angle OR a stall (K: cross may be thin or wide -> 18 + stall stop)
GRIP_CLOSE_DEG = {"P": 20, "N": 16, "B": 20, "R": 20, "Q": 20, "K": 20}   # N narrow: may compress the pads past 20   # foam pads fitted 2026-11-16: 20 = pads touching (servo power-on position); contact load stops it earlier


def joint_limits():
    """Manually recorded joint ranges (manual_calib.py), shrunk by LIMIT_MARGIN_DEG:
    {joint index 0-5: (min, max)} for the joints that have both ends recorded."""
    import json
    if not os.path.exists(JOINT_LIMITS_FILE):
        return {}
    with open(JOINT_LIMITS_FILE) as f:
        lim = json.load(f).get("limits", {})
    out = {}
    for j, (lo, hi) in lim.items():
        if lo is not None and hi is not None:
            out[int(j[1]) - 1] = (lo + LIMIT_MARGIN_DEG, hi - LIMIT_MARGIN_DEG)
    return out

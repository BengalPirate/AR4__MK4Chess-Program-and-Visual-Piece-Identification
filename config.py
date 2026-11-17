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
FRAME_SIZE = (1280, 720)
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
TRAVEL_CLEARANCE = 15               # gap kept above the tallest piece during transit
APPROACH_Z = 40                     # above the piece top before the slow descent
VERTICAL_STEP_MM = 15               # waypoint spacing on vertical approach/retreat
PATH_TOL_MM = 3.0                   # max deviation of a joint move from the straight line

# Piece heights above the board (mm) and grasp height (fingertip height above the
# board surface when gripping). MEASURE THESE on the real set before the final.
PIECE_HEIGHT = {"P": 48, "N": 62, "B": 70, "R": 52, "Q": 84, "K": 95}
# Measured on the real set 2026-11-16 (fingertip height above the board when gripping and releasing)
GRASP_Z = {"P": 23.0, "N": 35.0, "R": 35.0, "B": 45.0, "Q": 57.0, "K": 68.0}   # P checked on the arm; others -5 mm after the first arm test
PLACE_EXTRA_Z = 2.0                 # release this far above the grasp height


# Small per-rank position corrections (mm, robot frame: +X away from the robot,
# +Y to the robot's left), from watching the arm 2026-11-16.
RANK_OFFSET_MM = {8: (5.0, 0.0), 7: (0.0, 2.0), 5: (2.0, 0.0)}
SQUARE_OFFSET_MM = {s: (2.0, 0.0) for s in ("c8", "d8", "e8", "f8")}   # overrides the rank offset (middle of rank 8 needed less)


def travel_z(carrying=None):
    """Fingertip height above the board for transit: clear the tallest piece, plus
    the part of the carried piece that hangs below the fingertips."""
    z = max(PIECE_HEIGHT.values()) + TRAVEL_CLEARANCE
    if carrying:
        z += GRASP_Z[carrying.upper()]
    return z

# Gripper (servo 0 on the Nano): fitted travel 15 (closed) .. 45 (open)
GRIP_OPEN_DEG = 45
GRIP_YAW_PIECE = {p: 0.0 for p in "PNBRQK"}   # jaw rotation (deg): all straight - the gripper clears neighbours without the 45 slant
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

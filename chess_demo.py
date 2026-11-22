"""Vision-guided AR4 chess: live recognition, game state, move selection, robot plan.

Run:  python chess_demo.py            camera + chess, robot plan shown but not executed
      python chess_demo.py --robot    also connect the AR4 and gripper (X executes)
      python chess_demo.py --robot --auto --from-board --robot-color black
                                      real game, hands-free: the game starts from the board as
                                      the cameras see it; your move is accepted once two scans
                                      agree and the robot replies at once (Stockfish, --skill
                                      0-20); an illegal move is put back. TAB pauses/resumes.

Keys (camera window). Moves are typed in UCI (e2e4, e7e8q) and sent with ENTER,
so letters a-h, 1-8 and q r b n are kept for typing:
  SPACE   scan the board: classify all 64 squares (5 frames averaged), then find
          the legal move that explains the change (or confirm nothing changed)
  ENTER   accept the detected move / submit the typed move
  ESC     clear typing, cancel a pending move or a square selection
  M       mode: AI (engine chooses the robot's move) / MANUAL (operator chooses)
  P       AI mode: engine selects the move for the side to move; shows it and the robot plan
          MANUAL mode: click a piece on the diagram, then one of its highlighted legal
          destinations - that becomes the robot's next move (X executes it)
  X       execute the plan on the robot (--robot) or simulate it (dry run)
  Y       live recognition on/off (re-scans continuously, labels on every square)
  V       set the game from the recognised position (any arrangement) - ENTER confirms
  L       learn: save this frame labelled with the current game position (training)
  T       train the classifier on all saved boards (background)
  I       training setups: cycle through positions to arrange for data collection
  K       calibrate the board: click the outer corners of a1, h1, h8, a8
  E       store the empty-board reference (board must be empty)
  U       undo last move         Z  new game         S  screenshot        J  quit
  click   a square in the camera view (or the diagram in AI mode), then type its
          content to correct the scan: P N B R Q K (white, shift) / p n b r q k (black) / . or 0 = empty

Safety: the robot plan can only be executed when the last scan confirmed that the
physical board matches the game state, the classifier was confident, the board
calibration is valid (tile alignment) and the move is legal. Anything else
PAUSES and asks the operator to correct the board or the recognition.
"""
import argparse
import csv
import os
import threading
import time

import chess
import cv2
import numpy as np

import config
from board_vision import BoardGeometry, detect_markers, refine_corners_empty_board
from cameras import open_views
from chess_logic import LABELS, Engine, Game, board_from_labels
from planner import BoardRobot, plan_move, plan_restore, plan_robot_move, robot_sq

VW, VH = 720, 405                       # displayed size of each camera view
H_CAM = VH
CANVAS_H = 720
PANEL_W = 520
GREEN, RED, AMBER, WHITE, CYAN, GREY = (0, 200, 0), (0, 0, 255), (0, 170, 255), \
    (255, 255, 255), (255, 255, 0), (150, 150, 150)

SETUPS = [   # positions to arrange physically for training data (I key)
    ("start position", chess.STARTING_FEN),
    ("colours swapped (white on 7-8, black on 1-2)",
     "RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w - - 0 1"),
    ("pieces mixed 1", "r1bk3r/P1pp1Qpp/1n3n2/2b1p3/1PB1P3/2N2N2/p1PP1PPP/R1B1K2R w - - 0 1"),
    ("pieces mixed 2", "1RnqKb1r/pP1N1ppP/2b5/3pP3/Q2p1B2/2n1k3/PPp2pRp/r3N2B w - - 0 1"),
    ("all 32 shuffled A", "RBP2pNP/p1pr1q2/N3pp1n/1bPK1PPb/rp4P1/Q6P/Pnk1p1p1/R3B3 w - - 0 1"),
    ("all 32 shuffled B", "1pRqP1rP/3nP3/1P4pp/R1NpnBkr/1Ppp4/7B/1P3Qbp/2KPPpNb w - - 0 1"),
    ("sparse: one of each", "8/2k5/4q3/1r6/5B2/2N5/3P1n2/6K1 w - - 0 1"),
]


class App:
    def __init__(self, args):
        self.args = args
        if args.rank_near_camera:
            config.CAMERAS["front"]["near"] = f"rank{args.rank_near_camera}"
        self.views = open_views()           # every attached board camera (cameras.py)
        self.calib_view = None              # view being calibrated by clicks
        self.game = Game()
        self.robot_color = chess.WHITE if args.robot_color == "white" else chess.BLACK
        if getattr(args, "skill", None) is not None:
            config.STOCKFISH_SKILL = args.skill
        self.engine = Engine()
        self.clf = self._load_classifier()
        self.msg, self.msg_col = "Ready. SPACE = scan, P = engine move, K = calibrate", WHITE
        self.typed = ""
        self.scan = None            # dict: labels, conf, logp, frame, time, inference
        self.pending = None         # move detected by vision, awaiting ENTER
        self.pending_board = None   # board from V (set position), awaiting ENTER
        self.plan = None            # (move, ops) chosen by the engine
        self.verified = False       # last scan matched the game state
        self.selected = None        # square chosen for manual correction
        self.overrides = {}         # square -> label set by the operator
        for item in filter(None, (getattr(args, "fix", None) or "").split(",")):
            sq, lab = item.split("=")       # --fix f1=B,f7=p (start-position corrections)
            self.overrides[chess.parse_square(sq.strip())] = lab.strip()
        self.live = False
        self.busy = False
        # --auto: the game runs itself - scans while waiting, accepts the player's move once
        # two scans in a row agree, and the robot replies at once. TAB pauses / resumes.
        self.auto = bool(getattr(args, "auto", False))
        self.auto_paused = False
        self.started = not getattr(args, "from_board", False)   # --from-board: game set from the first scan
        self._seen = None           # last scan's reading, for the two-scans-agree rule
        self.calib_clicks = None
        self.setup_idx = -1
        self.highlight = {}         # square -> colour (errors, move squares)
        self.manual = False         # MANUAL mode: operator picks the robot's move
        self.manual_from = None     # square picked as the move source
        self.br = self._load_board_robot()
        self.arm = self.gripper = self.executor = None
        if args.robot:
            self._connect_robot()
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        self.scan_log = os.path.join(config.RESULTS_DIR, "scans.csv")

    # ------------------------------------------------------------ setup helpers
    def _load_classifier(self):
        try:
            from pieces import PieceClassifier
            c = PieceClassifier()
            print(f"Classifier: {c.info}")
            return c
        except Exception as e:      # noqa: BLE001
            print(f"No classifier yet ({e})")
            return None

    def _load_board_robot(self):
        try:
            return BoardRobot()
        except Exception as e:      # noqa: BLE001
            print(f"Robot board mapping not available: {e}")
            return None

    def _connect_robot(self):
        from ar4link import AR4
        from gripper import Gripper
        from motion import BoardLock, Executor
        if self.br is None:
            raise SystemExit("--robot needs config/board_robot.json (run board_teach.py)")
        self.arm = AR4(safe=True, j5_window=config.J5_WINDOW)
        self.gripper = Gripper()
        check = None
        if self.clf is not None:
            from grip_check import PickChecker
            check = PickChecker(self.views, self.clf)   # cameras confirm each pick; misses are retried
        reader = None
        if config.OFFSET_CORRECTION and self.views:
            from offset_model import OffsetReader
            reader = OffsetReader(self.views)          # piece off centre -> pick shifted 15 mm
        self.executor = Executor(self.arm, self.gripper, self.br, BoardLock(), log=print, pick_check=check,
                                 offset_reader=reader)
        print(f"Robot connected: {self.arm.pose()}")

    def say(self, text, col=WHITE):
        self.msg, self.msg_col = text, col
        print(text)

    # ------------------------------------------------------------ recognition
    @property
    def geo(self):
        """Geometry of the first calibrated view (for code that needs any one)."""
        return next((v.geo for v in self.views if v.geo is not None), None)

    def grab(self, k):
        """k frames from every view: [(view, frames)]."""
        return [(v, v.frames(k)) for v in self.views]

    def calibration_ok(self, grabbed):
        """Every view calibrated and not moved; a bumped camera pauses the game."""
        for v, frames in grabbed:
            if not frames:
                return False, f"{v.key} camera gives no frames"
            ok, why = v.check(frames[-1])
            if not ok:
                return False, why
        return True, ""

    def run_scan(self, infer=True):
        """Classify the board; if infer, explain it as a legal move. Runs on a thread."""
        if self.busy:
            return
        if self.clf is None:
            self.say("No classifier trained yet: arrange a known position, L to save it, T to train", AMBER)
            return
        self.busy = True

        def work():
            try:
                from pieces import scan_views
                grabbed = self.grab(config.SCAN_FRAMES)
                ok, why = self.calibration_ok(grabbed)
                if not ok:
                    self.verified = False
                    self.say("PAUSED: " + why, RED)
                    return
                logp, per_view = scan_views(self.clf, [(v.key, v.geo, fr) for v, fr in grabbed])
                # cameras index robot squares; the game uses chess squares (files mirrored)
                idx = [robot_sq(sq) for sq in range(64)]
                logp = np.asarray(logp)[idx].copy()
                per_view = {k: np.asarray(lp)[idx] for k, lp in per_view.items()}
                for sq, lab in self.overrides.items():          # operator corrections
                    logp[sq][:] = np.log(1e-6)
                    logp[sq][LABELS.index(lab)] = 0.0
                p = np.exp(logp)
                labels = [LABELS[i] for i in p.argmax(axis=1)]
                conf = p.max(axis=1)
                view_labels = {k: [LABELS[i] for i in lp.argmax(axis=1)] for k, lp in per_view.items()}
                self.scan = {"labels": labels, "conf": conf, "logp": logp, "view_labels": view_labels,
                             "frames": {v.key: fr[-1] for v, fr in grabbed}, "time": time.time()}
                if infer:
                    self._interpret()
                else:
                    unsure = int((conf < config.MIN_SQUARE_CONF).sum())
                    dis = self._disagree()
                    self.say(f"Live: {sum(c != '.' for c in labels)} pieces, {unsure} uncertain squares"
                             + (f", cameras disagree on {len(dis)}" if dis else ""),
                             AMBER if unsure else GREEN)
            except Exception as e:      # noqa: BLE001
                self.verified = False
                self.say(f"PAUSED: scan failed: {e}", RED)
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def _disagree(self):
        """Squares where the camera views give different labels."""
        vl = list(self.scan.get("view_labels", {}).values()) if self.scan else []
        if len(vl) < 2:
            return []
        return [sq for sq in range(64) if len({v[sq] for v in vl}) > 1]

    def _interpret(self):
        if not self.started:
            self._start_from_board()
            return
        s, g = self.scan, self.game
        inf = g.infer_move(s["logp"])
        s["inference"] = inf
        truth_now = g.physical_labels()
        self.highlight = {}
        if inf["move"] is None and g.confident(inf):
            self.verified = True
            self._seen = None
            errs = [sq for sq in range(64) if s["labels"][sq] != truth_now[sq]]
            for sq in errs:
                self.highlight[sq] = AMBER
            self._log_scan(truth_now, "verify")
            whose = ""
            if self.auto and not g.board.is_game_over():
                whose = " - robot's move" if g.board.turn == self.robot_color else " - YOUR move"
            self.say(f"Board matches the game ({len(errs)} square(s) misread but explained "
                     f"by the game state). {g.status()}{whose}", GREEN)
            self._auto_robot_move()
            return
        self.verified = False
        if inf["move"] is not None and g.confident(inf) and self.auto \
                and g.board.turn != self.robot_color:
            m = inf["move"]
            if not self._agrees(("move", m.uci())):
                self.say(f"Seen {g.board.san(m)} ({m.uci()}) - checking once more (hands clear)", CYAN)
                return
            self.accept_move(m, "vision")        # board verified by this scan
            self._auto_robot_move()
            return
        if inf["move"] is not None and g.confident(inf):
            self.pending = inf["move"]
            self.highlight = {inf["move"].from_square: CYAN, inf["move"].to_square: CYAN}
            self.say(f"Detected move {g.board.san(inf['move'])} ({inf['move'].uci()}), margin "
                     f"{inf['margin']:.1f}. ENTER = accept, ESC = reject", CYAN)
            return
        if self._illegal_move():
            return
        # Not confident: explain why and wait for the operator
        cands = ", ".join((m.uci() if m else "no move") for m, _ in inf["ranked"][:3])
        worst = chess.square_name(inf["worst_square"])
        self.highlight[inf["worst_square"]] = RED
        for sq in range(64):
            if s["conf"][sq] < config.MIN_SQUARE_CONF:
                self.highlight.setdefault(sq, AMBER)
        self.say(f"PAUSED - unsure (best: {cands}; {worst} disagrees). Type the move + ENTER, "
                 "or click squares to correct and SPACE again", RED)
        self._log_scan(None, "uncertain")

    def _illegal_move(self):
        """The player moved one piece to a square it may not go to (operator's rule
        2026-11-20): the arm takes it back to its square and the game waits for a legal
        move. Only a piece moved to an EMPTY square is taken back automatically - if an
        illegal move also took or knocked another piece, the arm cannot know where that
        piece belongs, so the game pauses for the operator. Returns True if handled."""
        s, g = self.scan, self.game
        truth, labels, conf = g.physical_labels(), s["labels"], s["conf"]
        diff = [sq for sq in range(64) if labels[sq] != truth[sq]]
        vacated = [sq for sq in diff if truth[sq] != "." and labels[sq] == "."]
        appeared = [sq for sq in diff if truth[sq] == "." and labels[sq] != "."]
        name = chess.square_name
        if len(diff) == 2 and len(vacated) == 1 and len(diff) - len(vacated) == 1 and not appeared:
            frm, to = vacated[0], next(sq for sq in diff if sq != vacated[0])
            self.verified = False
            self.highlight = {frm: RED, to: RED}
            self.say(f"ILLEGAL: the {truth[frm]} from {name(frm)} seems to have taken the {truth[to]} on "
                     f"{name(to)}. Put both pieces back by hand, then SPACE", RED)
            return True
        if not (len(diff) == 2 and len(vacated) == 1 and len(appeared) == 1):
            return False
        frm, to = vacated[0], appeared[0]
        if min(conf[frm], conf[to]) < config.MIN_SQUARE_CONF:
            return False                 # not sure enough of the reading to move anything
        if any(m.from_square == frm and m.to_square == to for m in g.board.legal_moves):
            return False                 # a legal move read uncertainly: the normal pause handles it
        piece = truth[frm]               # the game knows what stood there (the reading may misname it)
        if self.auto and not self._agrees(("illegal", frm, to)):
            self.say(f"Looks like an illegal move {name(frm)}{name(to)} - checking once more", AMBER)
            return True
        why = g.why_illegal(chess.Move(frm, to))
        self.verified = False
        self.highlight = {frm: RED, to: RED}
        if self.br is None:
            self.say(f"ILLEGAL move {name(frm)}{name(to)} ({why}). Put the {piece} back on {name(frm)}, "
                     "then make a legal move and SPACE", RED)
            return True
        ops = plan_restore(g.board, frm, to, piece, self.br)
        print(f"\nIllegal move {name(frm)}{name(to)} ({why}) - taking it back:")
        for o in ops:
            print("   ", o)
        if self.executor is None:
            self.say(f"ILLEGAL move {name(frm)}{name(to)} ({why}). Dry run: put the {piece} back on "
                     f"{name(frm)} by hand, then make a legal move and SPACE", RED)
            return True
        # runs on the scan thread, which keeps self.busy set until the arm is done
        try:
            self.say(f"ILLEGAL move {name(frm)}{name(to)} ({why}) - the arm puts the {piece} back "
                     f"on {name(frm)}. Hands off the board (E-stop to stop)", RED)
            self.executor.run(ops)
            self.say(f"The {piece} is back on {name(frm)}. Make a LEGAL move, then SPACE "
                     f"({why})", AMBER)
        except Exception as e:      # noqa: BLE001
            self.say(f"ROBOT STOPPED while taking the illegal move back: {e}", RED)
        return True

    def _log_scan(self, truth, kind):
        s = self.scan
        new = not os.path.exists(self.scan_log)
        with open(self.scan_log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "kind", "fen_game", "recognised", "truth", "errors",
                            "occupied", "min_conf", "move", "margin"])
            inf = s.get("inference", {})
            errs = sum(a != b for a, b in zip(s["labels"], truth)) if truth else ""
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), kind, self.game.board.fen(),
                        "".join(s["labels"]), "".join(truth) if truth else "", errs,
                        sum(c != "." for c in (truth or s["labels"])), f"{s['conf'].min():.2f}",
                        inf.get("move").uci() if inf.get("move") else "",
                        f"{inf.get('margin', 0):.2f}"])

    # ------------------------------------------------------------ game actions
    def accept_move(self, move, how):
        g = self.game
        try:
            san = g.board.san(move)
            steps = g.tray_after(move)[2] if move in g.board.legal_moves else []
            g.push(move)
        except ValueError as e:
            self.say(f"REJECTED: {e}", RED)
            return
        self.pending, self.plan, self.overrides = None, None, {}
        self.highlight = {move.from_square: CYAN, move.to_square: CYAN}
        if self.scan is not None and how == "vision":
            self._log_scan(g.physical_labels(), "move")
            if self.args.auto_learn:
                self._save_views(self.scan["frames"], g.physical_labels(), "game")
        self.verified = how == "vision"
        tray_note = ""
        if steps and how not in ("robot", "simulated"):
            # a move made by hand: tell the operator where the tray pieces go, so the
            # physical tray keeps matching the game's record of it
            parts = []
            for kind, slot, pc in steps:
                if kind == "in":
                    parts.append(f"put the {pc} in tray slot {slot}")
                elif slot == "reserve":
                    parts.append(f"use the spare {pc} from the reserve spot")
                else:
                    parts.append(f"take the {pc} from tray slot {slot}")
            tray_note = " TRAY: " + "; ".join(parts) + "."
        self.say(f"{san} played ({how}). {g.status()}{tray_note}", AMBER if tray_note else GREEN)
        if self.args.play and not self.auto and how in ("vision", "typed") \
                and g.board.turn == self.robot_color and not g.board.is_game_over() and not tray_note:
            self.engine_move()          # --play: the robot's reply is planned at once; X executes

    def engine_move(self):
        g = self.game
        if g.board.is_game_over():
            self.say(f"Game over: {g.status()}", AMBER)
            return
        m = self.engine.choose(g.board)
        self.set_plan(m, f"Engine ({self.engine.name})")

    def set_plan(self, m, who):
        g = self.game
        ops = plan_robot_move(g, m, self.br) if self.br else plan_move(g, m)
        self.plan = (m, ops)
        self.highlight = {m.from_square: CYAN, m.to_square: CYAN}
        print(f"\n{who} selects {g.board.san(m)} = {m.uci()}")
        for o in ops:
            print("   ", o)
        self.say(f"{who.split(' (')[0]}: {g.board.san(m)} ({chess.square_name(m.from_square)} -> "
                 f"{chess.square_name(m.to_square)}). X = execute", CYAN)

    def execute_plan(self):
        if not self.plan:
            self.say("No plan - press P first", AMBER)
            return
        m, ops = self.plan
        # safety gates: verified board, legal move, valid calibration
        if not self.verified:
            self.say("REFUSED: board state not verified - SPACE to scan first", RED)
            return
        if m not in self.game.board.legal_moves:
            self.say("REFUSED: planned move is no longer legal", RED)
            return
        if self.executor is None:
            self._simulate(m, ops)
            return
        if self.busy:
            return
        self.busy = True

        def work():
            try:
                self._execute_now(m, ops)
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def _execute_now(self, m, ops):
        """Run the robot's move on the calling (worker) thread. True if it was played."""
        try:
            self.say(f"Robot executing {self.game.board.san(m)} ({m.uci()}) ... (SPACE-bar does not "
                     "stop the arm: use the E-stop)", AMBER)
            self.executor.run(ops)
            self.accept_move(m, "robot")
            self.verified = False
            self.say(f"Robot played {m.uci()}. " + ("Checking the board..." if self.auto
                                                    else "SPACE to verify the board"), GREEN)
            return True
        except Exception as e:      # noqa: BLE001
            if self.auto:
                self.auto_paused = True
            self.say(f"ROBOT STOPPED: {e}" + (" - AUTO PAUSED: fix the board, then TAB" if self.auto else ""),
                     RED)
            return False

    def _agrees(self, key):
        """--auto: act only when two scans in a row read the same thing (the player's
        hand is out of the way and the reading is stable)."""
        same = self._seen == key
        self._seen = None if same else key
        return same

    def _auto_robot_move(self):
        """--auto: the board is verified and it is the robot's turn - choose with the
        engine and play it at once (on the scan thread, which holds self.busy)."""
        g = self.game
        if not self.auto or self.auto_paused or not self.verified or g.board.is_game_over() \
                or g.board.turn != self.robot_color:
            if self.auto and g.board.is_game_over():
                self.say(f"GAME OVER: {g.status()}", CYAN)
            return
        self.engine_move()
        if not self.plan:
            return
        m, ops = self.plan
        if any(o.kind == "pause" for o in ops):
            self.auto_paused = True
            self.say(f"AUTO PAUSED: the plan for {m.uci()} needs the operator (see the panel). "
                     "X executes it, TAB resumes auto", AMBER)
            return
        if self.executor is None:
            self._simulate(m, ops)
            return
        self._execute_now(m, ops)

    def _start_from_board(self):
        """--from-board: the game starts from whatever the cameras see (no fixed
        start position). The standard start is White to move; any other position is
        taken as the player's move (--to-move robot changes that)."""
        s = self.scan
        unsure = [chess.square_name(sq) for sq in range(64) if s["conf"][sq] < config.MIN_SQUARE_CONF]
        if unsure:
            for sq in range(64):
                if s["conf"][sq] < config.MIN_SQUARE_CONF:
                    self.highlight[sq] = AMBER
            self.say(f"Reading the start position: unsure of {', '.join(unsure[:6])} - hands clear? "
                     "(click a square + type its piece to correct)", AMBER)
            return
        b = board_from_labels(s["labels"])
        if b.board_fen() == chess.STARTING_BOARD_FEN:
            b.turn = chess.WHITE
        else:
            robot = getattr(self.args, "to_move", "player") == "robot"
            b.turn = self.robot_color if robot else (not self.robot_color)
        if not b.is_valid():
            self.say(f"The board as read is not a legal chess position ({b.status()!r}) - "
                     "correct it (click a square + type its piece)", RED)
            return
        if not self._agrees(("start", b.fen())):
            self.say("Start position read - checking once more", CYAN)
            return
        self.game, self.plan, self.pending, self.overrides = Game(b.fen()), None, None, {}
        tray = getattr(self.args, "tray", None)
        if tray:
            # 2026-11-22: a restarted game must know the tray slots already in use, or the
            # robot sets a captured piece down on top of one (h4 pawn onto slot 0's knight)
            self.game.tray = [None if c in ".-" else c for c in tray]
        self.started, self.verified = True, True
        side = "White" if b.turn == chess.WHITE else "Black"
        self.say(f"Game set from the board: {b.board_fen()} - {side} to move "
                 f"({'robot' if b.turn == self.robot_color else 'YOU'})", GREEN)
        self._auto_robot_move()

    def _simulate(self, m, ops):
        """Dry run: compute every joint move of the plan without a robot."""
        if self.br is None:
            print("Dry run (no board_robot.json): plan steps only")
            for o in ops:
                print("   ", o)
            self.accept_move(m, "simulated")
            return
        import kin
        from motion import BoardLock, DryArm, DryGripper, Executor
        park = self.br.park["joints"] if self.br.park else [0, 30, 0, 0, 60, 0]
        arm = DryArm(park)
        ex = Executor(arm, DryGripper(), self.br, BoardLock(config.LOCK_FILE + ".dry"), log=print)
        try:
            ex.run(ops, park=False)
        except Exception as e:      # noqa: BLE001
            self.say(f"SIMULATION STOPPED: {e}", RED)
            return
        print(f"Dry run: {len(arm.log)} joint moves, gripper down, jaws at yaw {ex.yaw:.0f}, "
              f"J5 within {config.J5_WINDOW}")
        for kind, js in arm.log[:6]:
            print(f"    {kind} {js}  -> XYZ {np.round(kin.fk(js)[:3], 1)}")
        self.accept_move(m, "simulated")

    # ------------------------------------------------------------ input
    def _colour_like_cameras(self, sq, ch):
        """The OpenCV window often loses Shift/Caps Lock (2026-11-20: 'B' arrived as 'b'),
        so a lowercase piece letter takes the colour the cameras see on that square
        (colour is read reliably; the piece type is what gets corrected). Uppercase is
        always white; a square the cameras see as empty keeps the letter as typed."""
        if ch.isupper() or not self.scan:
            return ch
        seen = [v[sq] for v in self.scan.get("view_labels", {}).values() if v[sq] != "."]
        if seen and sum(c.isupper() for c in seen) * 2 > len(seen):
            return ch.upper()
        return ch

    def on_key(self, k):
        if k < 0:
            return True
        ch = chr(k) if k < 128 else ""
        if self.selected is not None and ch in "PNBRQKpnbrqk.0" and ch:
            lab = "." if ch in ".0" else self._colour_like_cameras(self.selected, ch)
            self.overrides[self.selected] = lab
            self.say(f"{chess.square_name(self.selected)} set to '{lab}'. SPACE to re-scan with "
                     f"{len(self.overrides)} correction(s)", CYAN)
            self.selected = None
            return True
        if k == 27:                         # ESC
            n = len(self.overrides)
            self.typed, self.pending, self.pending_board, self.selected = "", None, None, None
            self.calib_clicks, self.overrides = None, {}
            self.say(f"Cleared (and {n} square correction(s))" if n else "Cleared", WHITE)
        elif k in (13, 10):                 # ENTER
            if self.typed:
                t, self.typed = self.typed, ""
                try:
                    self.accept_move(self.game.parse_move(t), "typed")
                except ValueError as e:
                    self.say(f"REJECTED: {e}", RED)
            elif self.pending is not None:
                self.accept_move(self.pending, "vision")
            elif self.pending_board is not None:
                self.game = Game(self.pending_board.fen())
                self.pending_board, self.plan = None, None
                self.verified = True
                self.say(f"Game set from vision: {self.game.board.fen()}", GREEN)
        elif k == 9:                        # TAB: pause / resume auto play
            if self.auto:
                self.auto_paused = not self.auto_paused
                self._seen = None
                self.say("AUTO PAUSED (TAB resumes)" if self.auto_paused else
                         "AUTO resumed - scanning", AMBER if self.auto_paused else GREEN)
        elif k == 8 or k == 127:            # backspace
            self.typed = self.typed[:-1]
        elif ch == " ":
            self.run_scan(infer=True)
        elif ch and ch in "abcdefgh12345678qrbn" and (self.typed or ch in "abcdefgh"):
            self.typed += ch
        elif ch == "p":
            if self.manual:
                self.say("MANUAL mode: click a piece on the diagram, then its destination", CYAN)
            else:
                self.engine_move()
        elif ch == "m":
            self.manual = not self.manual
            self.manual_from, self.plan = None, None
            self.highlight = {}
            self.say("MANUAL mode: click a piece on the board diagram, then a highlighted square"
                     if self.manual else "AI mode: P = engine chooses the robot's move", CYAN)
        elif ch == "x":
            self.execute_plan()
        elif ch == "y":
            self.live = not self.live
            self.say(f"Live recognition {'on' if self.live else 'off'}", WHITE)
        elif ch == "v":
            self.set_from_vision()
        elif ch == "l":
            self.learn()
        elif ch == "t":
            self.train()
        elif ch == "i":
            self.setup_idx = (self.setup_idx + 1) % (len(SETUPS) + 1)
            if self.setup_idx == len(SETUPS):
                self.setup_idx = -1
                self.game = Game()
                self.say("Training setups off - new game", WHITE)
            else:
                name, fen = SETUPS[self.setup_idx]
                self.game = Game(fen)
                self.say(f"Setup {self.setup_idx + 1}/{len(SETUPS)}: {name}. Arrange it, then L", CYAN)
        elif ch == "k":
            self.calibrate_all()
        elif ch == "e":
            self.store_empty()
        elif ch == "u":
            m = self.game.undo()
            self.plan = None
            self.say(f"Undid {m.uci() if m else 'nothing'}. {self.game.status()}", WHITE)
        elif ch == "z":
            self.game, self.plan, self.pending = Game(), None, None
            self.verified = False
            self.say("New game - set up the start position and press SPACE", WHITE)
        elif ch == "s":
            self.screenshot()
        elif ch == "j":
            return False
        return True

    def on_mouse(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < VW * len(self.views) and y < VH:
            v = self.views[x // VW]
            fx, fy = (x % VW) * config.FRAME_SIZE[0] / VW, y * config.FRAME_SIZE[1] / VH
            if self.calib_clicks is not None:
                if v is not self.calib_view:
                    self.say(f"Click in the {self.calib_view.key} view", AMBER)
                    return
                self.calib_clicks.append((fx, fy))
                names = ["a1", "h1", "h8", "a8"]
                if len(self.calib_clicks) < 4:
                    self.say(f"Now click the outer corner of {names[len(self.calib_clicks)]}", CYAN)
                else:
                    self.finish_calibration()
                return
            if v.geo is None:
                return
            # pick the square whose column (z = 0.5 square) is nearest the click
            d = [np.hypot(*(v.geo.square_center(sq, 0.5) - (fx, fy))) for sq in range(64)]
            self.selected = robot_sq(int(np.argmin(d)))
        elif x >= VW * len(self.views):
            sq = self.panel_square(x - VW * len(self.views), y)
            if sq is None:
                return
            if self.manual:
                self.manual_click(sq)
                return
            self.selected = sq
        if self.selected is None:
            return
        self.say(f"{chess.square_name(self.selected)} selected: type P N B R Q K / "
                 "p n b r q k / . (empty), ESC cancels", CYAN)

    def manual_click(self, sq):
        """MANUAL mode: first click picks a piece of the side to move, second click a
        legal destination. Only legal moves can be chosen."""
        b = self.game.board
        name = chess.square_name(sq)
        if self.manual_from is not None:
            cands = [m for m in b.legal_moves
                     if m.from_square == self.manual_from and m.to_square == sq]
            if cands:
                m = next((c for c in cands if c.promotion in (None, chess.QUEEN)), cands[0])
                self.manual_from = None
                self.set_plan(m, "Operator")
                return
            if sq == self.manual_from:
                self.manual_from, self.highlight = None, {}
                self.say("Selection cleared", WHITE)
                return
        pc = b.piece_at(sq)
        if pc is None or pc.color != b.turn:
            side = "White" if b.turn else "Black"
            self.say(f"REJECTED: {name} has no {side} piece ({side} to move)", RED)
            return
        dests = [m.to_square for m in b.legal_moves if m.from_square == sq]
        if not dests:
            self.say(f"REJECTED: the {pc.symbol()} on {name} has no legal move", RED)
            return
        self.manual_from = sq
        self.plan = None
        self.highlight = {sq: (255, 0, 255), **{d: GREEN for d in dests}}
        self.say(f"{pc.symbol()} on {name}: {len(dests)} legal destination(s) shown in green - "
                 "click one", CYAN)

    # ------------------------------------------------------------ calibration / data
    def calibrate_all(self):
        """K: auto-calibrate every camera on the empty board. A view whose tiles are
        not found falls back to clicking its four corners."""
        if self.busy:
            return
        self.busy = True

        def work():
            try:
                done, failed = [], []
                for v in self.views:
                    geo = v.calibrate()
                    if geo is None:
                        failed.append(v)
                    else:
                        done.append(f"{v.key} {geo.reproj_px:.1f} px")
                if failed:
                    self.calib_view, self.calib_clicks = failed[0], []
                    self.say(f"Calibrated: {', '.join(done) or 'none'}. {failed[0].key}: tiles not found "
                             f"(pieces on the board?) - click the OUTER corner of a1, then h1, h8, a8 "
                             f"in the {failed[0].key} view", CYAN)
                else:
                    self.say(f"All cameras calibrated (fit: {', '.join(done)}); empty references stored",
                             GREEN)
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def finish_calibration(self):
        v = self.calib_view
        frame, _ = v.get()
        corners = self.calib_clicks
        self.calib_clicks, self.calib_view = None, None
        refined = refine_corners_empty_board(frame, corners)
        if refined is not None:
            corners = refined
            note = "refined from the 49 inner tile corners"
        else:
            note = "from clicks (board not empty or pattern not found: no refinement)"
        v.geo = BoardGeometry(corners, config.FRAME_SIZE, detect_markers(frame))
        v.geo.save(v.cfg["file"])
        v.geo.set_reference(frame) if refined is not None else None
        self.say(f"{v.key} camera calibrated ({note}). "
                 + ("Empty reference stored." if refined is not None else "Press E with the board empty."),
                 GREEN)

    def store_empty(self):
        if any(v.geo is None for v in self.views):
            self.say("Calibrate first (K)", AMBER)
            return
        for v in self.views:
            v.geo.set_reference(v.get()[0])
        self.say("Empty-board references stored for every camera", GREEN)

    def _save_views(self, frames_by_view, labels, source):
        """One labelled sample per camera, sharing a stamp."""
        from pieces import save_sample
        stamp = None
        labels = [labels[robot_sq(sq)] for sq in range(64)]     # chess -> robot squares
        for v in self.views:
            f = frames_by_view.get(v.key)
            if f is not None and v.geo is not None:
                stamp = save_sample(f, v.geo, labels, source, camera=v.key, stamp=stamp)

    def learn(self):
        if any(v.geo is None for v in self.views):
            self.say("Calibrate first (K)", AMBER)
            return
        labels = self.game.physical_labels()
        grabbed = self.grab(3)
        for i in range(3):
            self._save_views({v.key: fr[i] for v, fr in grabbed if len(fr) > i}, labels,
                             "setup" if self.setup_idx >= 0 else "manual")
        n = len([f for f in os.listdir(config.DATA_DIR) if f.endswith(".json")])
        self.say(f"Saved 3 labelled frames of this position from {len(self.views)} camera(s) "
                 f"({n} samples total). Move pieces to another position, then L again; T trains", GREEN)

    def train(self):
        if self.busy:
            return
        self.busy = True

        def work():
            import subprocess
            import sys
            try:
                # fine-tune on phone photos + saved webcam boards (finetune.py, ~20-30 min on the VM)
                self.say("Training classifier (fine-tune, ~25 min, app stays usable) ...", AMBER)
                r = subprocess.run([sys.executable, "finetune.py", "--epochs", "8"],
                                   cwd=config.HERE, capture_output=True, text=True)
                print(r.stdout[-3000:], r.stderr[-2000:])
                self.clf = self._load_classifier()
                acc = [ln for ln in r.stdout.splitlines() if "exact label" in ln][-1:]
                self.say("Trained. " + (acc[0].strip("- ").replace("*", "") if acc else ""),
                         GREEN if r.returncode == 0 else RED)
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def set_from_vision(self):
        if self.scan is None:
            self.say("Scan first (SPACE or Y)", AMBER)
            return
        b = board_from_labels(self.scan["labels"], self.game.board.turn)
        if not b.is_valid():
            self.say(f"PAUSED: recognised position is not a valid chess position "
                     f"({b.status().name if hasattr(b.status(), 'name') else b.status()}) - "
                     "correct squares by clicking", RED)
            return
        self.pending_board = b
        self.say(f"Recognised position {b.board_fen()} - ENTER to use it as the game", CYAN)

    def screenshot(self):
        path = os.path.join(config.RESULTS_DIR, "screens",
                            time.strftime("%Y%m%d-%H%M%S") + ".png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, self.canvas)
        self.say(f"Screenshot saved: {path}", WHITE)

    # ------------------------------------------------------------ drawing
    SQ = 54
    BX, BY = 30, 40

    def panel_square(self, x, y):
        f, r = (x - self.BX) // self.SQ, 7 - (y - self.BY) // self.SQ
        return chess.square(f, r) if 0 <= f < 8 and 0 <= r < 8 else None

    def draw_panel(self):
        p = np.full((CANVAS_H, PANEL_W, 3), 30, np.uint8)
        g = self.game.board
        S, X0, Y0 = self.SQ, self.BX, self.BY
        cv2.putText(p, f"Game state - {'MANUAL' if self.manual else 'AI'} mode", (X0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255) if self.manual else WHITE, 1)
        for sq in range(64):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x, y = X0 + f * S, Y0 + (7 - r) * S
            light = (f + r) % 2 == 1
            cv2.rectangle(p, (x, y), (x + S, y + S), (181, 217, 240) if light else (99, 136, 181), -1)
            if sq in self.highlight:
                cv2.rectangle(p, (x + 2, y + 2), (x + S - 2, y + S - 2), self.highlight[sq], 3)
            if sq == self.selected:
                cv2.rectangle(p, (x + 1, y + 1), (x + S - 1, y + S - 1), (255, 0, 255), 3)
            pc = g.piece_at(sq)
            if pc:
                fill, edge = ((245, 245, 245), (0, 0, 0)) if pc.color else ((20, 20, 20), (230, 230, 230))
                cv2.circle(p, (x + S // 2, y + S // 2), S // 2 - 7, fill, -1, cv2.LINE_AA)
                cv2.circle(p, (x + S // 2, y + S // 2), S // 2 - 7, edge, 1, cv2.LINE_AA)
                cv2.putText(p, pc.symbol().upper(), (x + S // 2 - 9, y + S // 2 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, edge, 2, cv2.LINE_AA)
                if sq in self.game.standins:     # a pawn standing in for a promoted piece
                    cv2.circle(p, (x + S // 2, y + S // 2), S // 2 - 3, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(p, "p", (x + S - 13, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 0, 255), 1, cv2.LINE_AA)
        for i in range(8):
            cv2.putText(p, "abcdefgh"[i], (X0 + i * S + S // 2 - 5, Y0 + 8 * S + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
            cv2.putText(p, str(8 - i), (X0 - 18, Y0 + i * S + S // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
        arrow = self.plan[0] if self.plan else self.pending
        if arrow:
            def ctr(sq):
                return (X0 + chess.square_file(sq) * S + S // 2, Y0 + (7 - chess.square_rank(sq)) * S + S // 2)
            cv2.arrowedLine(p, ctr(arrow.from_square), ctr(arrow.to_square), (0, 140, 255), 4,
                            cv2.LINE_AA, tipLength=0.25)
        y = Y0 + 8 * S + 40
        lines = [(self.game.status(), WHITE), (self.game.rights_text(), GREY),
                 (f"stand-ins: {self.game.standin_text()}", GREY),
                 (f"tray (slot:piece): {self.game.tray_text()}" + ("  spare used" if self.game.reserve_used else ""), GREY),
                 (f"engine: {self.engine.name}", GREY),
                 (f"classifier: {(self.clf.info.get('kind') or 'trained on ' + str(self.clf.info['boards']) + ' boards') if self.clf else 'NOT TRAINED'}",
                  GREY if self.clf else AMBER),
                 (f"board state verified: {'YES' if self.verified else 'no'}", GREEN if self.verified else AMBER),
                 (f"robot: {'connected' if self.executor else 'dry run'}", GREY)]
        if self.game.board.move_stack:
            lines.append(("moves: " + " ".join(m.uci() for m in self.game.board.move_stack[-8:]), GREY))
        if self.plan:
            lines.append((f"robot plan for {self.plan[0].uci()}:", CYAN))
            for o in self.plan[1]:
                lines.append(("  " + str(o).split("  (")[0], CYAN))
        for txt, col in lines:
            cv2.putText(p, txt[:62], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)
            y += 20
        return p

    def draw_view(self, v, frame):
        img = frame.copy()
        geo = v.geo
        dis = set(self._disagree())
        if geo is not None:
            labels, colors = {}, [(170, 170, 170)] * 64
            if self.scan is not None:
                for sq in range(64):
                    lab, c = self.scan["labels"][sq], self.scan["conf"][sq]
                    if lab != ".":
                        labels[sq] = (lab, GREEN if c >= config.MIN_SQUARE_CONF else AMBER)
                    elif c < config.MIN_SQUARE_CONF:
                        labels[sq] = ("?", AMBER)
                vl = self.scan.get("view_labels", {}).get(v.key)
                for sq in dis:               # this camera's own reading where the views differ
                    if vl:
                        labels[sq] = (vl[sq], (255, 0, 255))
                colors = list(colors)
                for sq, col in self.highlight.items():
                    colors[sq] = col
            labels = {robot_sq(sq): lab for sq, lab in labels.items()}
            colors = [colors[robot_sq(sq)] for sq in range(64)]
            geo.draw(img, labels, colors)
            if self.selected is not None:
                cv2.polylines(img, [geo.square_polygon(robot_sq(self.selected)).astype(np.int32)], True,
                              (255, 0, 255), 3)
            arrow = self.plan[0] if self.plan else self.pending
            if arrow:
                a = tuple(int(c) for c in geo.square_center(robot_sq(arrow.from_square)))
                b = tuple(int(c) for c in geo.square_center(robot_sq(arrow.to_square)))
                cv2.arrowedLine(img, a, b, (0, 140, 255), 5, cv2.LINE_AA, tipLength=0.2)
        if self.calib_clicks is not None and v is self.calib_view:
            for pt in self.calib_clicks:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 8, CYAN, 2)
        img = cv2.resize(img, (VW, VH))
        title = f"{v.key} {v.device}" + ("" if geo else "  NOT CALIBRATED")
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    CYAN if geo else RED, 2, cv2.LINE_AA)
        return img

    def draw(self, frames):
        n = len(self.views)
        canvas = np.full((CANVAS_H, VW * n + PANEL_W, 3), 20, np.uint8)
        for i, (v, f) in enumerate(zip(self.views, frames)):
            canvas[:VH, i * VW:(i + 1) * VW] = self.draw_view(v, f) if f is not None else 0
        canvas[:, VW * n:] = self.draw_panel()
        y = H_CAM + 28
        cv2.putText(canvas, self.msg[:120], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.msg_col, 1, cv2.LINE_AA)
        if len(self.msg) > 120:
            cv2.putText(canvas, self.msg[120:240], (12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        self.msg_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"typed move: {self.typed}_" + ("   [working...]" if self.busy else ""),
                    (12, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        help_txt = ("SPACE scan  ENTER accept/submit  M mode  P engine  X execute  Y live  V set-from-vision  "
                    "L learn  T train  I setups  K calib  E empty-ref  U undo  Z new  S shot  J quit")
        cv2.putText(canvas, help_txt, (12, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
        cv2.putText(canvas, "click a square + type a piece letter to correct the scan; "
                    "magenta = this camera's reading where the cameras disagree",
                    (12, y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
        self.canvas = canvas
        return canvas

    # ------------------------------------------------------------ main loop
    def loop(self):
        cv2.namedWindow("AR4 chess", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("AR4 chess", self.on_mouse)
        last_live = 0
        while True:
            frames = [v.get()[0] for v in self.views]
            if all(f is None for f in frames):
                time.sleep(0.05)
                continue
            if self.auto and not self.auto_paused and not self.busy and not self.game.board.is_game_over() \
                    and self.calib_clicks is None and self.selected is None and not self.typed \
                    and time.time() - last_live > config.AUTO_SCAN_S:
                last_live = time.time()
                self.run_scan(infer=True)
            elif self.live and not self.busy and time.time() - last_live > 1.0:
                last_live = time.time()
                self.run_scan(infer=False)
            cv2.imshow("AR4 chess", self.draw(frames))
            if self.args.snapshot and time.time() - getattr(self, "_snap_t", 0) > 2:
                self._snap_t = time.time()
                cv2.imwrite(self.args.snapshot, self.canvas)
            if not self.on_key(cv2.waitKey(30)):
                break
        self.engine.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", action="store_true", help="connect the AR4 and gripper")
    ap.add_argument("--snapshot", help="also write the window image to this file every 2 s")
    ap.add_argument("--rank-near-camera", type=int, choices=(1, 8),
                    help="override the front camera's nearest rank (config.CAMERAS)")
    ap.add_argument("--no-auto-learn", dest="auto_learn", action="store_false",
                    help="do not save confirmed game positions as training data")
    ap.add_argument("--play", action="store_true",
                    help="real game: after the player's move is accepted, plan the robot's reply at once")
    ap.add_argument("--robot-color", choices=("white", "black"), default="white",
                    help="the side the robot plays (--play)")
    ap.add_argument("--auto", action="store_true",
                    help="auto play: scan continuously, accept the player's move when two scans agree, "
                         "robot replies at once (TAB pauses)")
    ap.add_argument("--from-board", action="store_true",
                    help="start the game from the position the cameras see")
    ap.add_argument("--to-move", choices=("player", "robot"), default="player",
                    help="--from-board, non-start position: who moves first")
    ap.add_argument("--tray", help="--from-board: tray slots already holding pieces, one FEN letter per "
                                   "slot from slot 0, '.' = empty (e.g. nRpP)")
    ap.add_argument("--fix", help="square corrections applied to the scans until the game starts, "
                                  "e.g. f1=B,f7=p (uppercase white, lowercase black, . empty)")
    ap.add_argument("--skill", type=int, choices=range(21), metavar="0-20",
                    help="Stockfish skill level (default config.STOCKFISH_SKILL)")
    App(ap.parse_args()).loop()


if __name__ == "__main__":
    main()

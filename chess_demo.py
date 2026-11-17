"""Vision-guided AR4 chess: live recognition, game state, move selection, robot plan.

Run:  python chess_demo.py            camera + chess, robot plan shown but not executed
      python chess_demo.py --robot    also connect the AR4 and gripper (X executes)

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
from board_vision import BoardGeometry, open_camera, refine_corners_empty_board
from chess_logic import LABELS, Engine, Game, board_from_labels, square_labels
from planner import BoardRobot, plan_move

W_CAM, H_CAM = 960, 540                 # displayed camera size
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


class Camera:
    """Grabs frames on a thread so the newest frame is always available."""

    def __init__(self):
        self.cap, self.device = open_camera()
        self.frame, self.n = None, 0
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.n = f, self.n + 1

    def get(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.n

    def frames(self, k):
        out, last = [], -1
        t_end = time.time() + 5
        while len(out) < k and time.time() < t_end:
            f, n = self.get()
            if f is not None and n != last:
                out.append(f)
                last = n
            time.sleep(0.03)
        return out


class App:
    def __init__(self, args):
        self.args = args
        self.cam = Camera()
        self.geo = BoardGeometry.load()
        if self.geo:
            self.geo.load_reference()
        self.game = Game()
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
        self.live = False
        self.busy = False
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
        self.executor = Executor(self.arm, self.gripper, self.br, BoardLock(), log=print)
        print(f"Robot connected: {self.arm.pose()}")

    def say(self, text, col=WHITE):
        self.msg, self.msg_col = text, col
        print(text)

    # ------------------------------------------------------------ recognition
    def calibration_ok(self, frame):
        if self.geo is None:
            return False, "Board not calibrated - press K"
        al = self.geo.tile_alignment(frame)
        if al is not None and al < config.MIN_TILE_ALIGNMENT:
            return False, f"Board/camera moved (tile alignment {al:.2f}) - recalibrate (K)"
        mv = self.geo.camera_moved(frame)
        if mv is not None and mv > config.CAMERA_MOVED_PX:
            return False, f"Camera moved ({mv:.0f} px at the markers) - recalibrate (K)"
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
                frames = self.cam.frames(config.SCAN_FRAMES)
                ok, why = self.calibration_ok(frames[-1])
                if not ok:
                    self.verified = False
                    self.say("PAUSED: " + why, RED)
                    return
                logp = self.clf.logp([self.geo.crops(f) for f in frames])
                for sq, lab in self.overrides.items():          # operator corrections
                    logp[sq][:] = np.log(1e-6)
                    logp[sq][LABELS.index(lab)] = 0.0
                p = np.exp(logp)
                labels = [LABELS[i] for i in p.argmax(axis=1)]
                conf = p.max(axis=1)
                self.scan = {"labels": labels, "conf": conf, "logp": logp,
                             "frame": frames[-1], "time": time.time()}
                if infer:
                    self._interpret()
                else:
                    unsure = int((conf < config.MIN_SQUARE_CONF).sum())
                    self.say(f"Live: {sum(c != '.' for c in labels)} pieces, "
                             f"{unsure} uncertain squares", AMBER if unsure else GREEN)
            except Exception as e:      # noqa: BLE001
                self.verified = False
                self.say(f"PAUSED: scan failed: {e}", RED)
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def _interpret(self):
        s, g = self.scan, self.game
        inf = g.infer_move(s["logp"])
        s["inference"] = inf
        truth_now = square_labels(g.board)
        self.highlight = {}
        if inf["move"] is None and g.confident(inf):
            self.verified = True
            errs = [sq for sq in range(64) if s["labels"][sq] != truth_now[sq]]
            for sq in errs:
                self.highlight[sq] = AMBER
            self._log_scan(truth_now, "verify")
            self.say(f"Board matches the game ({len(errs)} square(s) misread but explained "
                     f"by the game state). {g.status()}", GREEN)
            return
        self.verified = False
        if inf["move"] is not None and g.confident(inf):
            self.pending = inf["move"]
            self.highlight = {inf["move"].from_square: CYAN, inf["move"].to_square: CYAN}
            self.say(f"Detected move {g.board.san(inf['move'])} ({inf['move'].uci()}), margin "
                     f"{inf['margin']:.1f}. ENTER = accept, ESC = reject", CYAN)
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
            g.push(move)
        except ValueError as e:
            self.say(f"REJECTED: {e}", RED)
            return
        self.pending, self.plan, self.overrides = None, None, {}
        self.highlight = {move.from_square: CYAN, move.to_square: CYAN}
        if self.scan is not None and how == "vision":
            self._log_scan(square_labels(g.board), "move")
            if self.args.auto_learn:
                from pieces import save_sample
                save_sample(self.scan["frame"], self.geo, square_labels(g.board), "game")
        self.verified = how == "vision"
        self.say(f"{san} played ({how}). {g.status()}", GREEN)

    def engine_move(self):
        g = self.game
        if g.board.is_game_over():
            self.say(f"Game over: {g.status()}", AMBER)
            return
        m = self.engine.choose(g.board)
        self.set_plan(m, f"Engine ({self.engine.name})")

    def set_plan(self, m, who):
        g = self.game
        ops = plan_move(g, m)
        if self.br:
            ops = [self.br.locate(o) for o in ops]
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
                self.say(f"Robot executing {m.uci()} ... (SPACE-bar does not stop the arm: "
                         "use the E-stop)", AMBER)
                self.executor.run(ops)
                self.accept_move(m, "robot")
                self.verified = False
                self.say(f"Robot played {m.uci()}. SPACE to verify the board", GREEN)
            except Exception as e:      # noqa: BLE001
                self.say(f"ROBOT STOPPED: {e}", RED)
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

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
    def on_key(self, k):
        if k < 0:
            return True
        ch = chr(k) if k < 128 else ""
        if self.selected is not None and ch in "PNBRQKpnbrqk.0" and ch:
            lab = "." if ch in ".0" else ch
            self.overrides[self.selected] = lab
            self.say(f"{chess.square_name(self.selected)} set to '{lab}'. SPACE to re-scan with "
                     f"{len(self.overrides)} correction(s)", CYAN)
            self.selected = None
            return True
        if k == 27:                         # ESC
            self.typed, self.pending, self.pending_board, self.selected = "", None, None, None
            self.calib_clicks = None
            self.say("Cleared", WHITE)
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
            frame, _ = self.cam.get()
            from board_vision import auto_calibrate, detect_markers
            found = auto_calibrate(frame, self.args.rank_near_camera)
            if found is not None:
                self.geo = BoardGeometry(found[0], config.FRAME_SIZE, detect_markers(frame), found[1])
                self.geo.save()
                self.geo.set_reference(frame)
                self.say(f"Board found automatically (fit {self.geo.reproj_px:.1f} px); empty "
                         "reference stored", GREEN)
            else:
                self.calib_clicks = []
                self.say("Tile pattern not found (pieces on the board?). Click the OUTER corner "
                         "of a1, then h1, h8, a8", CYAN)
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
        if x < W_CAM:
            fx, fy = x * config.FRAME_SIZE[0] / W_CAM, y * config.FRAME_SIZE[1] / H_CAM
            if self.calib_clicks is not None:
                self.calib_clicks.append((fx, fy))
                names = ["a1", "h1", "h8", "a8"]
                if len(self.calib_clicks) < 4:
                    self.say(f"Now click the outer corner of {names[len(self.calib_clicks)]}", CYAN)
                else:
                    self.finish_calibration()
                return
            if self.geo is None:
                return
            # pick the square whose column (z = 0.5 square) is nearest the click
            d = [np.hypot(*(self.geo.square_center(sq, 0.5) - (fx, fy))) for sq in range(64)]
            self.selected = int(np.argmin(d))
        else:
            sq = self.panel_square(x - W_CAM, y)
            if sq is None:
                return
            if self.manual:
                self.manual_click(sq)
                return
            self.selected = sq
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
    def finish_calibration(self):
        frame, _ = self.cam.get()
        from board_vision import detect_markers
        corners = self.calib_clicks
        self.calib_clicks = None
        refined = refine_corners_empty_board(frame, corners)
        if refined is not None:
            corners = refined
            note = "refined from the 49 inner tile corners"
        else:
            note = "from clicks (board not empty or pattern not found: no refinement)"
        self.geo = BoardGeometry(corners, config.FRAME_SIZE, detect_markers(frame))
        self.geo.save()
        self.geo.set_reference(frame) if refined is not None else None
        self.say(f"Board calibrated ({note}). "
                 + ("Empty reference stored." if refined is not None else "Press E with the board empty."),
                 GREEN)

    def store_empty(self):
        if self.geo is None:
            self.say("Calibrate first (K)", AMBER)
            return
        frame, _ = self.cam.get()
        self.geo.set_reference(frame)
        self.say("Empty-board reference stored (tile alignment and change detection)", GREEN)

    def learn(self):
        if self.geo is None:
            self.say("Calibrate first (K)", AMBER)
            return
        from pieces import save_sample
        frames = self.cam.frames(3)
        labels = square_labels(self.game.board)
        for f in frames:
            save_sample(f, self.geo, labels, "setup" if self.setup_idx >= 0 else "manual")
        n = len([f for f in os.listdir(config.DATA_DIR) if f.endswith(".json")])
        self.say(f"Saved {len(frames)} labelled frames of this position ({n} total). "
                 "Move pieces to another position, then L again; T trains", GREEN)

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
        p = np.full((H_CAM + 180, PANEL_W, 3), 30, np.uint8)
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
                 (f"captured tray: {' '.join(self.game.tray) or '-'}", GREY),
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

    def draw(self, frame):
        img = frame.copy()
        if self.geo is not None:
            labels, colors = {}, [(170, 170, 170)] * 64
            if self.scan is not None:
                for sq in range(64):
                    lab, c = self.scan["labels"][sq], self.scan["conf"][sq]
                    if lab != ".":
                        labels[sq] = (lab, GREEN if c >= config.MIN_SQUARE_CONF else AMBER)
                    elif c < config.MIN_SQUARE_CONF:
                        labels[sq] = ("?", AMBER)
                for sq, col in self.highlight.items():
                    colors = list(colors)
                    colors[sq] = col
            self.geo.draw(img, labels, colors)
            if self.selected is not None:
                cv2.polylines(img, [self.geo.square_polygon(self.selected).astype(np.int32)], True,
                              (255, 0, 255), 3)
            arrow = self.plan[0] if self.plan else self.pending
            if arrow:
                a = tuple(int(v) for v in self.geo.square_center(arrow.from_square))
                b = tuple(int(v) for v in self.geo.square_center(arrow.to_square))
                cv2.arrowedLine(img, a, b, (0, 140, 255), 5, cv2.LINE_AA, tipLength=0.2)
        if self.calib_clicks is not None:
            for pt in self.calib_clicks:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 8, CYAN, 2)
        cam = cv2.resize(img, (W_CAM, H_CAM))
        canvas = np.full((H_CAM + 180, W_CAM + PANEL_W, 3), 20, np.uint8)
        canvas[:H_CAM, :W_CAM] = cam
        canvas[:, W_CAM:] = self.draw_panel()
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
        cv2.putText(canvas, "click a square + type a piece letter to correct the scan",
                    (12, y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
        self.canvas = canvas
        return canvas

    # ------------------------------------------------------------ main loop
    def loop(self):
        cv2.namedWindow("AR4 chess", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("AR4 chess", self.on_mouse)
        last_live = 0
        while True:
            frame, _ = self.cam.get()
            if frame is None:
                time.sleep(0.05)
                continue
            if self.live and not self.busy and time.time() - last_live > 1.0:
                last_live = time.time()
                self.run_scan(infer=False)
            cv2.imshow("AR4 chess", self.draw(frame))
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
    ap.add_argument("--rank-near-camera", type=int, default=1, choices=(1, 8),
                    help="printed rank nearest the camera (orientation for auto calibration)")
    ap.add_argument("--no-auto-learn", dest="auto_learn", action="store_false",
                    help="do not save confirmed game positions as training data")
    App(ap.parse_args()).loop()


if __name__ == "__main__":
    main()

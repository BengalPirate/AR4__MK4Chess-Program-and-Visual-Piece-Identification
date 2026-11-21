"""Build the final (Milestone 6) team report PDF.

Run from the repository root:
    python docs/make_figures.py          (figures; system diagram)
    python docs/game_stats.py            (results/games/summary.md from the game logs)
    python docs/make_final_report.py     -> docs/Team2_Final_Report.pdf

Numbers come from results/ (accuracy files, game logs, grips.csv, scans.csv) and from
config.py, so the report follows the code and the logs rather than hand-typed values.
"""
import os
import sys
import time

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_report import (H1, H2, REPO_URL, ROOT, SMALL, P, contributions, img,  # noqa: E402
                         md_section, table)
import game_stats  # noqa: E402
import config  # noqa: E402

OUT = os.path.join(ROOT, "docs", "Team2_Final_Report.pdf")


def bullets(items, st=None):
    return P("<br/>".join("• " + i for i in items), st) if st else P("<br/>".join("• " + i for i in items))


def main():
    games, tot, gs, sc = game_stats.main()
    cam2 = md_section("results/accuracy_cnn_2cam.md", "Webcam boards (40, never trained on)")
    web = md_section("results/accuracy_webcam.md", "All held-out webcam boards")
    photo = md_section("results/accuracy_cnn_2cam.md", "Held-out phone photos (test split)")
    played = [g for g in games if g["moves"]]
    robot_moves = tot["robot_moves"]
    window = os.path.join(ROOT, "results", "games", "2026-11-20_game6_window.png")

    doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            title="Team 2 - Vision-Guided AR4 Chess, Final Report")
    W = doc.width
    s = []

    # ------------------------------------------------------------------ title
    s += [P("Vision-Guided AR4 Chess – Final Report<br/>(Project Milestone 6) – DRAFT", H1),
          P("<b>Team 2:</b> Brandon Newton, Roshan Padhan, Andrew Overton<br/>"
            f"<b>Repository:</b> <link href='{REPO_URL}'>{REPO_URL}</link><br/>"
            "<b>Hardware:</b> Annin AR4 MK4 (Teensy 4.1, Annin firmware 6.7.1), DS3225 servo gripper "
            "on an Arduino Nano with a current sensor, two webcams (Logitech C930e front, C920 PRO side), "
            "47 mm-square folding board<br/>"
            f"<b>Report generated:</b> {time.strftime('%B %d, %Y')}"),
          Spacer(1, 6),
          P("<b>Summary.</b> Two webcams watch the board; every square is classified as empty or one of "
            "the twelve pieces and the two views are fused. A python-chess game keeps the position and "
            "accepts the opponent's move only when two scans agree on one legal move that explains the "
            "change. Stockfish 16 chooses the robot's reply, which is planned as pick-and-place "
            "operations, checked for reach, and executed by the AR4 with camera checks that the piece "
            "left its square and landed centred on its destination. The board is verified against the "
            "game before the next turn. In live games on November 20 the robot made "
            f"<b>{robot_moves} moves for Black chosen by the running engine</b>, including "
            f"<b>{tot['captures_robot']} captures</b> with the taken piece moved to the tray; "
            f"{gs['first_ok']} of {gs['first_tries']} picks succeeded on the first try. "
            "<b>Disclosure:</b> the system was demonstrated with <b>one</b> physical AR4 playing Black "
            "against a human playing White; the two-arm configuration required by the assignment was "
            "not run (section 10)."),
          ]

    # ------------------------------------------------------------------ coverage
    s += [P("1. Requirements and evidence", H2),
          table([["Requirement", "Status", "Evidence (section)"],
                 ["Calibrate board and robot; establish start position", "Done",
                  "camera auto-calibration from tiles, joint + corner teaching, aim map; game starts from the "
                  "position the cameras read (3)"],
                 ["Identify piece type, colour and square from live video", "Done",
                  f"{cam2.get('exact label (empty / type+colour)', '?')} exact on 40 held-out webcam boards, "
                  f"colour {cam2.get('colour, on occupied squares detected as occupied', '?')} (4)"],
                 ["Full loop: observe, reconcile, select legal move, robot coordinates, grasp, place, "
                  "retreat, verify", "Done", f"{robot_moves} robot moves in live games, each verified (5, 7)"],
                 ["Both AR4 arms alternate turns", "<b>Not done</b>",
                  "one arm vs a human; interlock implemented and unit-tested only (8, 10)"],
                 ["Moves selected live by the chess program", "Done", "Stockfish 16 over UCI, skill 20 (5)"],
                 ["Captures with removal to a designated location", "Done",
                  f"{tot['captures_robot']} robot captures, {tot['tray_places']} pieces placed in tray slots (7)"],
                 ["Special moves (castling, promotion, en passant)", "Implemented; not run on the arm",
                  "planner + unit tests; limitations disclosed (5, 10)"],
                 ["Post-move verification; stop and recover on failures", "Done",
                  f"{tot['place_fix']} placement fixes, {tot['retries']} pick retries, "
                  f"{tot['stops']} safe stops with recovery in the Nov 20 games (9)"],
                 ["Safety: E-stop, limits, reduced speed, approach heights, interlock", "Done (interlock: one arm)",
                  "(8)"],
                 ["Report, repository, README, versions, AI acknowledgement, contributions", "Done",
                  "(11–13); README.md; CONTRIBUTIONS.md"]],
                [W * 0.37, W * 0.17, W * 0.46])]

    # ------------------------------------------------------------------ architecture
    s += [PageBreak(), P("2. Software architecture", H2),
          img("docs/system_diagram.png", W * 0.55, max_h=4.6 * inch),
          table([["Module", "Role"],
                 ["chess_demo.py", "Operator app and game loop: live views, scans, game state panel; "
                  "--auto plays hands-free (scan every 1.5 s, accept the player's move after two agreeing "
                  "scans, robot replies at once; TAB pauses)"],
                 ["cameras.py, board_vision.py", "Open both webcams by name; per-camera board model, square "
                  "crops, tile-alignment bump check"],
                 ["pieces.py, offset_model.py", "Square classifier (13 classes) per view, fused by weighted "
                  "geometric mean; regression of a piece's offset from its square centre"],
                 ["chess_logic.py", "Game (python-chess board, tray, stand-ins), move inference, Stockfish"],
                 ["planner.py", "Legal move → pick/place operations in robot coordinates (BoardRobot from "
                  "taught corners + measured aim corrections)"],
                 ["motion.py, kin.py", "Executor: preflight, approach/retreat, grip, pick and placement "
                  "checks, recovery; inverse kinematics incl. the tilted rank-8 wrist"],
                 ["ar4link.py, gripper.py", "Serial drivers for the Teensy (percent speed, joint and J5 limits) "
                  "and the gripper Nano (stall stop, current)"],
                 ["robot_setup.py, manual_calib.py, calibrate_camera.py, rod_calib.py",
                  "Calibration tools (section 3)"]],
                [W * 0.3, W * 0.7])]

    # ------------------------------------------------------------------ calibration
    s += [PageBreak(), P("3. Calibration procedure", H2),
          table([["Step", "Tool", "What it does / result"],
                 ["1. Joints J1–J4, J6", "Annin HMI auto-calibration (limit switches)",
                  "J5 excluded: its auto-calibration drives the wide gripper into the forearm"],
                 ["2. J5 zero and joint ranges", "manual_calib.py",
                  "J5 zeroed by eye; safe ranges recorded to config/joint_limits.json; "
                  f"J5 window {config.J5_WINDOW[0]}…{config.J5_WINDOW[1]}° (2° inside the measured range)"],
                 ["3. Cameras", "calibrate_camera.py (K in the app)",
                  "empty board: inner tile corners found, camera model fitted (fit ≈ 1 px); empty-board "
                  "reference stored for the bump check; one file per camera"],
                 ["4. Board in robot coordinates", "robot_setup.py",
                  "closed jaw tips touched on a1, h1 (straight) and a8, h8 (tilted 30°); tray slot 0 and park "
                  "taught; config/board_robot.json"],
                 ["5. Aim map", "rod_calib.py",
                  "a rod held in the jaws is triangulated by both cameras over measured squares; residuals stored in "
                  "config/aim_correction.json; verified within 1 mm on e5, h3, c6, h7 and f8/h8"],
                 ["6. Backlash", "measurement over e5",
                  f"J1 backlash ≈ 2°; every final approach ends turning toward the a-file "
                  f"({config.BACKLASH_APPROACH_MM:.0f} mm pre-point)"],
                 ["Check before play", "chess_demo.py start-up",
                  f"tile alignment ≥ {config.MIN_TILE_ALIGNMENT} per camera, else the game pauses; the arm's "
                  "reported pose is compared with the taught park pose"]],
                [W * 0.2, W * 0.24, W * 0.56]),
          Spacer(1, 4),
          img("results/calibration_front.png", W * 0.49, max_h=2.4 * inch),
          img("results/calibration_side.png", W * 0.49, max_h=2.4 * inch),
          P("Camera calibration overlays (front, side).", SMALL)]

    # ------------------------------------------------------------------ perception
    s += [PageBreak(), P("4. Perception and post-move verification", H2),
          P("Each camera's model gives an upright crop per square (one square wide, 2.2 tall, so the "
            "piece is included in perspective). A ResNet-18 (ImageNet weights, fully fine-tuned, 13 "
            "outputs) was trained on 1,079 phone photos of our own set plus 78 labelled webcam boards from "
            "both cameras. Webcam test boards are kept in a folder training never reads."),
          table([["Test set", "Exact label", "Occupancy", "Colour", "Type"],
                 ["Phone photos, test split (483 squares)",
                  photo.get("exact label (empty / type+colour)", "?"), photo.get("occupancy", "?"),
                  photo.get("colour, on occupied squares detected as occupied", "?"),
                  photo.get("piece type, on occupied squares", "?")],
                 ["Webcam boards, both cameras (40 boards, 2,560 squares)",
                  cam2.get("exact label (empty / type+colour)", "?"), cam2.get("occupancy", "?"),
                  cam2.get("colour, on occupied squares detected as occupied", "?"),
                  cam2.get("piece type, on occupied squares", "?")],
                 ["Earlier single-camera model, 38 held-out boards",
                  web.get("exact label (empty / type+colour)", "?"), web.get("occupancy", "?"),
                  web.get("colour, on occupied squares detected as occupied", "?"),
                  web.get("piece type, on occupied squares", "?")]],
                [W * 0.44, W * 0.14, W * 0.14, W * 0.14, W * 0.14]),
          P("Details and confusion matrices: results/accuracy_cnn_2cam.md, results/accuracy_webcam.md.", SMALL),
          P("<b>Reconciling with the game.</b> For every legal move the position after it is scored against "
            "the fused probabilities. A move is accepted only if it beats the runner-up by a log-likelihood "
            f"margin of {config.MOVE_MIN_MARGIN} and no square contradicts it by more than "
            f"{config.MOVE_MAX_SQUARE_SURPRISE} (−log p); in auto play two consecutive scans must agree, "
            "so a hand over the board cannot be read as a move. In the 88 simulated move tests no wrong move "
            "was accepted. An illegal physical move is detected and put back by the arm."),
          P("<b>Verification.</b> After each robot move the board is scanned and must match the game "
            f"before the next turn: {tot['verified']} \"board matches the game\" results in the Nov 20 "
            f"logs, {sc.get('verify', 0)} verified scans logged in results/scans.csv overall, "
            f"{sc.get('uncertain', 0)} uncertain scans that paused instead of guessing. Within a move, "
            "a pick counts only when the source square is seen empty, and every set-down is measured by "
            f"the offset model; more than {config.PLACE_TOL_MM:.0f} mm off → the piece is re-picked and "
            f"re-placed (up to {config.PLACE_FIX_TRIES} times)."),
          P("<b>Known perception weaknesses.</b> Pieces hidden behind taller pieces (mitigated by the side "
            "camera); the front camera barely sees a8/b8; the offset reading is unreliable on rank 8 while "
            "the arm is tilted over it (a centred c8 bishop read 11 mm off on Nov 20).", SMALL)]

    # ------------------------------------------------------------------ chess / special moves
    s += [PageBreak(), P("5. Chess program, move selection and special moves", H2),
          P("Rules, legal moves and game state come from python-chess 1.11.2. The robot's moves are chosen "
            f"by Stockfish 16 through python-chess's UCI interface (skill {config.STOCKFISH_SKILL}); "
            "nothing is replayed from a script. The game can start from any position the cameras read "
            "(--from-board); the standard start means White to move."),
          table([["Special move", "Game state", "Physical handling", "Tested"],
                 ["Capture", "taken piece added to the tray list",
                  "taken piece to the next tray slot first, then the moving piece", "on the arm (live games)"],
                 ["Castling", "rights kept by the game, not the image", "king two squares, then the rook",
                  "unit tests only"],
                 ["En passant", "en passant square kept by the game", "pawn removed from beside the destination",
                  "unit tests only"],
                 ["Promotion", "queen by default; e7e8n etc. typed",
                  "pawn to the tray, new piece from the tray; if that piece is not in the tray the pawn stays "
                  "as a stand-in that the game treats as the promoted piece", "unit tests only"]],
                [W * 0.14, W * 0.26, W * 0.4, W * 0.2])]

    # ------------------------------------------------------------------ motion / gripper
    s += [P("6. Motion planning and gripper control", H2),
          P("A square maps to robot X, Y and board height by bilinear interpolation between the taught "
            "corners plus the measured aim map. For each operation the Executor rises to travel height "
            f"({config.travel_z():.0f} mm above the board, more when carrying a tall piece), approaches so "
            "that the base finishes turning toward the a-file, descends vertically, grips, rises vertically "
            "and checks. Every point is solved with Annin's kinematics before the first motion; an "
            "unreachable point refuses the whole move. Rank 8 cannot be reached straight down "
            f"(J3 would pass its +52° stop), so it is worked with the gripper tilted {config.RANK8_TILT_DEG:.0f}° "
            "back; pieces moving between rank 8 and ranks 1–2 are set down midway and re-picked."),
          table([["Piece", "Grip height (mm)", "Open / hold (servo °)"]] +
                [[n, f"{config.GRASP_Z[k]:.0f}", f"{config.GRIP_OPEN_PIECE[k]} / {config.GRIP_HOLD_DEG[k]}"]
                 for k, n in zip("PNBRQK", ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"])],
                [W * 0.25, W * 0.3, W * 0.45]),
          P("<b>Gripper.</b> The servo closes in small steps while the supply current is read; contact is "
            "felt as a current rise, a stall backs the servo off. With the foam pads narrow pieces raise "
            "the current little, so a pick is confirmed by the cameras (source square empty) and, after "
            "the set-down, the destination must be seen occupied. Misses: retry shifted "
            f"{config.GRIP_RETRY_MM:.0f} mm (up to {config.GRIP_RETRIES} retries), aimed where the cameras now "
            f"see the piece; on rank 8 the third try aims {config.RANK8_THIRD_TRY_BACK_MM:.0f} mm toward the "
            "robot (operator's rule, Nov 20 – it recovered the c8 bishop on the third try).", SMALL),
          P(f"<b>Speeds.</b> All development and testing at 10 % (travel 25 % from Nov 20 morning). For "
            f"the final games: {config.SPEED_PCT} % for the vertical moves at the piece and "
            f"{config.TRAVEL_SPEED_PCT} % for travel at height; ar4link refuses anything above 75 %.", SMALL)]

    # ------------------------------------------------------------------ live results
    s += [PageBreak(), P("7. Live game results (November 20)", H2),
          P("Human plays White; the robot plays Black with moves from Stockfish. Moves marked * were "
            "made by the robot. Logs: results/games/ (raw program output); summary from docs/game_stats.py."),
          table([["Log", "Moves", "Robot captures", "Stopped by"]] +
                [[g["file"].replace("2026-11-20_", ""),
                  " ".join(f"{m}{'*' if w == 'robot' else ''}" for m, w in g["moves"]),
                  ", ".join(g["captures_robot"]) or "–", "; ".join(g["stops"]) or "–"] for g in played],
                [W * 0.13, W * 0.47, W * 0.15, W * 0.25]),
          Spacer(1, 4),
          table([["Measure (Nov 20)", "Value"],
                 ["Robot moves executed, chosen by Stockfish", str(robot_moves)],
                 ["Player moves recognised by the cameras", str(tot["player_moves"])],
                 ["Robot captures / pieces placed in the tray", f"{tot['captures_robot']} / {tot['tray_places']}"],
                 ["Picks confirmed on the first try", f"{gs['first_ok']} / {gs['first_tries']}"],
                 ["  of which rank 8 (tilted)", f"{gs['rank8_first_ok']} / {gs['rank8_first']}"],
                 ["Set-down checks passed / automatic fixes needed", f"{tot['place_ok']} / {tot['place_fix']}"],
                 ["Pick retries", str(tot["retries"])],
                 ["Safe stops (operator recovery)", str(tot["stops"])],
                 ["Scans that paused as uncertain", str(tot["paused_unsure"])]],
                [W * 0.6, W * 0.4]),
          P("An earlier game the same afternoon (1.e4 Nf6 2.Bb5 Nxe4 3.Nh3 a6 4.Ba4 Nc5) also included a robot "
            "capture to the tray; its raw log was lost, the moves are recovered in "
            "results/games/2026-11-20_afternoon_recovered.log. Earlier arm tests (Nov 16–19): all 32 pieces "
            "moved two ranks forward at 10 %; results/test_log.md.", SMALL)]
    if os.path.exists(window):
        s += [img(window, W, max_h=3.0 * inch),
              P("chess_demo.py during a live game: both camera views with recognised labels, game panel "
                "(position, side to move, tray, engine, board verified).", SMALL)]

    # ------------------------------------------------------------------ safety
    s += [PageBreak(), P("8. Safety and interlock", H2),
          table([["Layer", "Behaviour", "How it was tested"],
                 ["Emergency stop", "hardware E-stop; the firmware's E-stop flag stops the sequence and "
                  "leaves the arm in place", "used during the J1 cable incident (Nov 16) and recovery"],
                 ["Joint / workspace limits", "recorded joint ranges and the J5 window are enforced in the serial "
                  "driver; every plan preflighted", "unit tests (J5 outside window, unreachable plan refused)"],
                 ["Approach / retreat", "vertical only, from travel height above the tallest piece; lifts go "
                  "straight up before any re-orientation", "real games; simulated path heights checked before "
                  "manual parks"],
                 ["Reduced speed", "10–25 % for all testing; hard cap in ar4link", "–"],
                 ["Uncertain detection / inconsistent state", "the game pauses, the arm is not moved",
                  f"{tot['paused_unsure']} pauses in the Nov 20 games; 0 wrong moves in 88 move tests"],
                 ["Shared-board interlock", "motion.BoardLock: exclusive lock file "
                  f"({config.LOCK_FILE}, or a shared path via AR4_BOARD_LOCK); an arm must hold it to move over "
                  "the board and releases it only after retreat; held through a stop",
                  "unit test: a second arm is refused while the lock is held. <b>Not tested with two "
                  "physical arms.</b>"],
                 ["Gripper", "stall stop; jam check before each grip", "bench and game use"]],
                [W * 0.2, W * 0.45, W * 0.35])]

    # ------------------------------------------------------------------ recovery
    s += [P("9. Failures and recovery (human intervention identified)", H2),
          table([["Event", "What the system did", "Human intervention"],
                 ["Set-down off-centre", "re-picked and re-placed automatically (8 times on Nov 20)", "none"],
                 ["Missed pick (piece still on its square)", "opened in the air, re-aimed, retried; c8 bishop "
                  "recovered on the third try with the rank-8 rule", "none"],
                 ["Missed c8 three times (before the rank-8 rule)", "stopped, arm left in place, game paused",
                  "operator re-centred the bishop; arm parked; game restarted from the board"],
                 ["Gripper Nano lost (host computer slept, Nov 20)", "stopped mid-move, arm in place",
                  "USB re-attached; arm controller restarted, so the last read joint angles were restored "
                  "(arm confirmed unmoved), park path simulated, arm parked at 10 %"],
                 ["Recognition unsure at start (f1, f7)", "start-up waited", "operator corrected the squares"],
                 ["Knight tipped during a set-down fix (Nov 19)", "stopped: destination seen empty",
                  "operator stood the knight up; recover_setdown.py for a piece left in the jaws"]],
                [W * 0.27, W * 0.41, W * 0.32])]

    # ------------------------------------------------------------------ limitations
    s += [PageBreak(), P("10. Limitations", H2),
          bullets(["<b>Only one physical arm was used.</b> The assignment asks for two AR4s alternating "
                   "turns. Our second and third arms were not ready (one had J1, J3 and J6 not moving at "
                   "build-up), so White was played by a human. The software is written for it – each arm "
                   "runs its own Executor with its own taught board, both must hold the shared BoardLock "
                   "to move over the board – but this was never run with two arms.",
                   "Castling, en passant and promotion were not executed on the arm (unit-tested plans only).",
                   "Rank 8 is at the edge of the arm's reach and needs the tilted wrist; rank-8 picks are the "
                   "least reliable and the offset reading there is not trustworthy.",
                   "If the host computer sleeps, the VM's USB devices drop; the arm controller can lose its "
                   "joint angles when re-attached.",
                   "Speed was raised to 30 / 50 % only for the final games.",
                   "The game assumes it is the player's move when started from a non-standard position "
                   "unless told otherwise (--to-move robot)."])]

    # ------------------------------------------------------------------ reproduce / versions
    s += [P("11. Reproducing and operating", H2),
          P("Setup, calibration order and operation are in README.md. A game: "
            "<font face='Courier'>python chess_demo.py --robot --auto --from-board --robot-color black</font>. "
            "Configuration and calibration files: config/ (board_image*.json, board_robot.json, "
            "joint_limits.json, aim_correction.json) and config.py; model models/piece_cnn.pt."),
          table([["Component", "Version"],
                 ["Python", "3.12.3 (Ubuntu 24.04 arm64, Parallels VM)"],
                 ["python-chess", "1.11.2 (GPL-3.0)"],
                 ["Stockfish", "16 (GPL-3.0), Ubuntu package unpacked locally"],
                 ["PyTorch / torchvision", "2.15.0 / 0.29.2 (CPU)"],
                 ["OpenCV", "5.0.0"],
                 ["NumPy, SciPy, scikit-learn", "2.5.4, 1.18.1, 1.9.1"],
                 ["pyserial, reportlab", "3.5, 5.1.0"],
                 ["Annin AR4 HMI kinematics / Teensy firmware", "6.7 / 6.7.1 (unmodified)"],
                 ["Gripper Nano sketch", "Annin AR4_nano_sketch v1.5"]],
                [W * 0.4, W * 0.6])]

    s += [P("12. Sources and AI assistance", H2),
          P("python-chess (board, rules, UCI); Stockfish (engine); PyTorch/torchvision (ResNet-18, ImageNet "
            "weights); OWL-ViT via Hugging Face transformers (photo dataset boxes only); OpenCV (cameras, "
            "corner detection); scikit-learn (baseline); Annin Robotics AR4 HMI and firmware."),
          P("<b>AI assistance.</b> Claude (Anthropic), used through Claude Code, wrote and debugged much of the "
            "code, ran training, tests and dry runs, and drafted this report. <b>Validation:</b> every arm "
            "motion was run first as a dry run or a simulated height check; new motion was approved by the "
            "operator, who watched it with a hand on the E-stop; recognition accuracy was measured only on "
            "boards never used for training; 34 unit tests cover rules, move inference, planning, limits "
            "and the interlock; the numbers in this report are generated from the logs in results/."),
          ]

    s += [P("13. Individual contributions", H2)]
    rows = contributions()
    s += [table([["Team member", "Contribution"]] + rows, [W * 0.22, W * 0.78]) if rows
          else P("<i>See CONTRIBUTIONS.md.</i>"),
          P("Each member explains their own part and its interfaces during the demonstration.", SMALL)]

    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

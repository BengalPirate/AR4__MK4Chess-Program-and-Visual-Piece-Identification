# Hand-off: arm parked and calibrated, ready to play (2026-11-23, about 06:10 CST)

This file records the exact state the system was left in, so that someone who
clones this repository on another computer (or another Claude session) can start
a game after redoing only the joint calibration (Annin HMI + J5 by eye):
board positions, cameras and everything else are used as they are. Tag of this state: see the commit
"Hand-off snapshot 2026-11-23" (git tag `handoff-20261123`).

## State when it was handed over

| Part | State |
|---|---|
| Arm (AR4 MK4, Teensy 4.1 serial `18042250`, firmware 6.7.1) | **Standing at PARK**, verified by the controller: J1 -4.421, J2 6.705, J3 50.211, J4 0.000, J5 33.097, J6 40.590 deg (= `config/board_robot.json` `park.joints`) |
| Joint calibration | Annin HMI auto-cal J1-J4 + J6 on 2026-11-23 05:48 (J5 excluded); J5 zeroed by eye in `manual_calib.py` 05:49; J5 safe range -99.63 .. +106.68 deg (used with 2 deg margin, `config/joint_limits.json`) |
| Board in robot coordinates | `config/board_robot.json`, corners re-taught 2026-11-23 05:25-05:36 (tag `calibration-20261123-0537`); hover check after the joint re-calibration: a1, h1, h8, a8 all centred (06:0x) |
| Cameras | front = Logitech C930e (diagonal across the h1 corner), side = Logitech C920 (beside the a-file). Calibrated on the empty board 2026-11-23 05:2x; with the start position set: alignment front 0.62, side 0.87 (game needs >= 0.45); start position read **64/64**. Picture: `results/handoff_cameras_20261123.jpg` |
| Gripper | DS3225 servo on the Arduino Nano (CH340, `/dev/ttyUSB*`). Opens/closes; resting current open 0.15-0.54 A (limit raised to 0.6 A, `gripper.IDLE_MAX_AMPS`) |
| Board | **standard starting position set up**, tray empty. Human plays White, robot plays Black |
| Classifier | `models/piece_cnn.pt` (ResNet-18, trained 2026-11-20 22:19) - in the repository |

## On the new computer: only the joint calibration is redone

Everything else (board positions, camera calibrations, classifier, settings) is in
this repository and is used as it is. Unplugging the arm restarts its controller
(it is powered over USB) and it then reads every joint as 0; the joint calibration
below takes care of that.

1. **Software** (once): Linux with Python 3.12.
   ```
   git clone https://github.com/BengalPirate/AR4__MK4Chess-Program-and-Visual-Piece-Identification.git
   cd AR4__MK4Chess-Program-and-Visual-Piece-Identification
   python3 -m venv venv && venv/bin/pip install -r requirements.txt
   sudo apt install stockfish            # or put the binary on PATH (else a small built-in engine plays)
   sudo usermod -aG dialout $USER        # serial ports; log out and in again
   ```
   Annin's **AR4 HMI 6.7 source** is needed twice: its app `AR4.py` (step 3) and its
   kinematics module, which `kin.py` uses: build `ARrobots/robot_kinematics` for this
   Python (cmake + pybind11, see README "Setup") and
   `export AR4_HMI_DIR=/path/to/"AR4 HMI Interface 6.7 source"`.
2. **Plug in** the arm (Teensy), the gripper Nano and both webcams. Ports and cameras
   are found automatically (Teensy by serial number, Nano by its CH340 chip, cameras by
   name). The cameras stay where they are on the table.
3. **Joint calibration in the Annin HMI** (`AR4.py` in the HMI source; in its
   `ARconfig.json` set `"comPort"` to the Teensy's `/dev/ttyACMx`): Calibration tab,
   auto-calibrate **J1, J2, J3, J4, J6 only - never J5** (its limit switch is where the
   gripper folds into the forearm). Hand on the E-stop. Close the HMI.
4. **J5 by eye**: `venv/bin/python manual_calib.py` - jog J5 until the gripper is in line
   with the forearm, press **J5 "= 0"** (not "all joints"), Save (keeps the J5 range).
   Close it.
5. **Park**: `venv/bin/python go_park.py` - checks the arm reads the calibrated zero pose,
   simulates the path, moves to PARK at 10 %.
6. **Check everything** (nothing moves): `venv/bin/python preflight_check.py` - must end
   with `READY TO PLAY`. It says what to do for any item marked FIX (e.g. a nudged camera:
   `camera_realign.py front`, works with the pieces on the board).
7. Optional, recommended: one hover at 10 % - `venv/bin/python hover_test.py e4 --above 40`;
   the jaws stop centred about 6 cm above e4.
8. **Play** (human White, robot Black, start position on the board):
   ```
   venv/bin/python -u chess_demo.py --robot --auto --from-board --robot-color black
   ```
   Add `--fix d1=Q,g1=N` if a start square is read "unsure" (d1/g1/d8/h8 often are).
   Speeds: 30 % at the piece, 50 % travel (`config.py`). During a stop do not press keys
   in the game window except TAB (L / I / T save training frames or retrain).

Shortcut instead of steps 3-4 (only if the arm has certainly not moved since it was
parked here): `venv/bin/python restore_park.py` tells the restarted controller it is at
PARK (nothing moves), then continue at step 6.

The joint calibration in steps 3-4 was redone this way on 2026-11-23 05:48 and the
taught corners were centred afterwards (a1, h1, h8, a8 hover check), so the board
positions do not depend on the computer or on re-calibrating the joints.

**If the board or the table was moved**: re-teach the corners in `robot_setup.py`
(press "20 mm above X", then **jog down until the closed jaw tips touch the board**,
then save; heights must come out about 97 mm), and if the cameras no longer see the
board in place, empty it and run `calibrate_camera.py`.

## Rules learned the hard way (read before driving the arm)

- **Before any motion, check the controller's pose is what you expect** (park).
  A replug of any USB device near it can restart the Teensy; it then reads all 0s and
  any move goes from a false origin. On 2026-11-23 this folded J5 into the forearm.
- The `20 mm above` buttons in `robot_setup.py` stop 20 mm up; saving there puts the
  board 20 mm too high (happened twice on 2026-11-22).
- The front camera gets nudged when pieces are set: the game pauses on the
  alignment check; `camera_realign.py front` fixes it with the pieces on.
- Gripper foam can hold a piece a moment too long; the placement check now looks
  again and re-opens wider before stopping (`motion._recheck_set_down`).
- Rank 8 is worked with the gripper tilted 30 deg; rank-8 misses retry 15 mm toward the robot.
- After a stop, restart from the board: `--from-board --to-move robot` if it is the
  robot's turn, `--tray pP...` for the pieces already in tray slots 0, 1, ...
- Directions given by the operator are from the robot's view (right = toward the a-file).

Full documentation: `README.md`, report `docs/Team2_Final_Report.pdf`, logs `results/games/`.

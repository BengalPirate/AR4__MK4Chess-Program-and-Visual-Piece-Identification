# Hand-off: arm parked and calibrated, ready to play (2026-11-23, about 06:10 CST)

This file records the exact state the system was left in, so that someone who
clones this repository on another computer (or another Claude session) can start
a game **without recalibrating anything**. Tag of this state: see the commit
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

## Moving the arm to another computer - IMPORTANT

The Teensy is powered over USB. **Unplugging it restarts it, and on the new
computer it reports every joint as 0.** The arm does not move by itself, so the
calibration is still physically valid **as long as nobody moves the arm** (do not
push it, do not jog it, do not run the HMI calibration or any motion script).
`restore_park.py` tells the controller it is at PARK again; nothing moves.

Motor power can stay on during the swap (the drivers hold the joints). If motor
power is cut, J2/J3 may sag under gravity: then recalibrate (below) instead.

## On the new computer, in this order

1. **Software** (once): Ubuntu/Linux with Python 3.12.
   ```
   git clone https://github.com/BengalPirate/AR4__MK4Chess-Program-and-Visual-Piece-Identification.git
   cd AR4__MK4Chess-Program-and-Visual-Piece-Identification
   python3 -m venv venv && venv/bin/pip install -r requirements.txt
   sudo apt install stockfish            # or put the binary on PATH (else a small built-in engine plays)
   sudo usermod -aG dialout $USER        # serial ports; log out and in again
   ```
   Annin's kinematics module is required (`kin.py`): get the **AR4 HMI 6.7 source**
   from Annin Robotics, build `ARrobots/robot_kinematics` for this Python (cmake +
   pybind11, see README "Setup"), and set `export AR4_HMI_DIR=/path/to/"AR4 HMI Interface 6.7 source"`.
2. **Plug in** the Teensy (arm), the gripper Nano and both webcams. Ports are found
   automatically (Teensy by its serial number via `/dev/serial/by-id`, Nano by its
   CH340 chip, cameras by name). Do not start the Annin HMI.
3. **Restore the park angles** (nothing moves):
   ```
   venv/bin/python restore_park.py --check     # expect: "Controller was reset (all joints 0)"
   venv/bin/python restore_park.py             # expect: "Restored: ... J1=-4.42 J2=6.71 J3=50.21 J4=0.00 J5=33.10 J6=40.59"
   ```
   It refuses if the controller reads anything other than all-0 or PARK.
4. **Check with one hover** at 10 % (hand on the E-stop): the jaws must stop
   centred about 6 cm above e4. If they are off, stop and recalibrate.
   ```
   venv/bin/python hover_test.py e4 --above 40
   ```
5. **Cameras**: if they were moved with the computer, check the alignment:
   `venv/bin/python camera_realign.py front --dry-run` (and `side`). Small nudge:
   `camera_realign.py front` fixes it with the pieces on. Big move: empty the
   board and run `calibrate_camera.py`.
6. **Play** (human White, robot Black, start position on the board):
   ```
   venv/bin/python -u chess_demo.py --robot --auto --from-board --robot-color black
   ```
   Add `--fix d1=Q,g1=N` if a start square is read "unsure" (d1/g1/d8/h8 often
   are). Speeds: 30 % at the piece, 50 % travel (`config.py`). During a stop do not
   press keys in the game window except TAB (L / I / T save training frames or retrain).

## If the arm was moved: full recalibration (about 5 minutes)

1. Annin HMI (`AR4.py` in the HMI source, set `comPort` in `ARconfig.json` to the
   Teensy's `/dev/ttyACMx`): Calibration tab, auto-calibrate **J1, J2, J3, J4, J6
   only - never J5** (its switch is where the gripper folds into the forearm). Close the HMI.
2. `python manual_calib.py`: jog J5 until the gripper is in line with the forearm,
   press **J5 "= 0"** (not "all joints"), Save (keeps the J5 range).
3. The board positions in `config/board_robot.json` stay valid. Verify with hovers
   (`hover_test.py a1 h1 h8 a8 --above 40`); if off, re-teach in `robot_setup.py`
   (press "20 mm above X", then **jog down until the closed jaw tips touch the board**, then save).

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

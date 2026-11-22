"""Robot execution of a pick-and-place plan.

Every move is a joint move (RJ) computed by kin.ik_down: gripper pointing straight
down, jaws at a fixed direction over the board (config.GRIPPER_YAW_DEG), and J5
inside its gripper-safe window. Cartesian (ML) moves are disabled in ar4link with
the servo gripper fitted, so the firmware never chooses wrist angles itself.

Joint moves do not follow straight lines, so each move is checked in simulation
(forward kinematics at 20 points) and split into waypoints until no point
deviates more than PATH_TOL_MM from the straight line - vertical descents stay
vertical and transit stays at travel height.

One pick:   open gripper at travel height -> above square -> descend in
            VERTICAL_STEP_MM steps to the grasp height -> close (gripper current
            shows contact; reaching the closed angle with no load = missed piece,
            STOP) -> retreat vertically -> travel height.
One place:  above target -> descend to grasp height + PLACE_EXTRA_Z -> open ->
            retreat -> travel height.

Safety: reduced speed (SPEED_PCT, capped at 25% in ar4link); E-stop flag checked
after every move (ar4link raises EStopError); any refused move, unreachable
point, gripper fault or software stop raises and leaves the arm where it is;
the board-workspace interlock (BoardLock) must be held for any motion over the
board and is released only after returning to PARK.
"""
import fcntl
import math
import os
import time

import chess
import numpy as np

import config
import kin


class MotionError(RuntimeError):
    pass


class StopRequested(MotionError):
    pass


class BoardLock:
    """Interlock for a shared board: only the holder may move over it.

    A non-blocking exclusive lock on a file that every arm controller uses (set
    AR4_BOARD_LOCK to a path on a shared filesystem for arms on several machines)."""

    def __init__(self, path=config.LOCK_FILE):
        self.path, self.f = path, None

    def acquire(self):
        if self.f:
            return True
        f = open(self.path, "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            return False
        self.f = f
        return True

    def release(self):
        if self.f:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.f = None


class DryArm:
    """Simulated arm: accepts joint moves, reports them. Used without --robot."""

    def __init__(self, joints):
        self.joints = list(joints)
        self.log = []

    def move_joints(self, js, speed_pct=10):
        self.joints = list(js)
        self.log.append(("RJ", [round(a, 2) for a in js]))

    def current_joints(self):
        return list(self.joints)


class DryGripper:
    def __init__(self):
        self.log = []

    def open(self, deg=None):
        self.log.append(("open", deg or config.GRIP_OPEN_DEG))

    trace = []

    def grip(self, deg, squeeze=0):
        self.log.append(("grip", deg))
        return True

    def hold(self, deg):
        self.log.append(("hold", deg))
        return True


class Executor:
    def __init__(self, arm, gripper, board_robot, lock, log=print, pick_check=None, offset_hint=None,
                 offset_reader=None):
        self.arm, self.grip, self.br, self.lock = arm, gripper, board_robot, lock
        # pick_check(square) -> "empty" | "occupied" | None: cameras after the lift
        # offset_hint(square) -> mm offset of the piece along a->h, or None
        self.pick_check, self.offset_hint = pick_check, offset_hint
        # offset_reader(square) -> {"dx_mm", "dy_mm", "shift_x_mm", "shift_y_mm"} or None:
        # the cameras' reading of where the piece stands, BEFORE the arm comes over it
        self.offset_reader = offset_reader
        self.last_offset = None
        self.log = log
        self.stop = False
        self.min_tilt = 0.0           # >0 while working a rank-8 square (config.RANK8_TILT_DEG)
        self.carry_tilt = 0.0         # tilt the held piece was picked at: placed at the same tilt
        self.unconfirmed = None       # source square of a pick the cameras could not yet confirm
        # jaws parallel to the board's square lines (from the taught corners)
        self.base_yaw = board_robot.jaw_yaw
        self.yaw = self.base_yaw
        lo, hi = config.J5_WINDOW
        j5 = self._joints()[4]
        if not lo <= j5 <= hi:
            raise MotionError(f"J5 is at {j5:.1f} deg, outside the gripper-safe window "
                              f"{lo:.0f}..{hi:.0f}. Is J5 zeroed (j5_zero.py)?")

    def _joints(self):
        if hasattr(self.arm, "current_joints"):
            return self.arm.current_joints()
        return list(self.arm.pose().joints)

    def _check_stop(self):
        if self.stop:
            raise StopRequested("Stopped by operator")

    # ---- primitive: straight-line flange move, as checked joint moves
    def _goto(self, xyz, speed=None):
        self._check_stop()
        start = self._joints()
        start_xyz = kin.vfk(start)
        end = kin.ik_down(xyz, self.yaw, start, self.min_tilt)
        if end is None:
            raise MotionError(f"Unreachable with gripper down: X {xyz[0]:.0f} Y {xyz[1]:.0f} "
                              f"Z {xyz[2]:.0f}")
        path = self._continuous(start, self._waypoints(start, start_xyz, end, xyz))
        for js in path:
            self._check_stop()
            self.arm.move_joints(js, speed or config.SPEED_PCT)

    def _continuous(self, start, path):
        """Re-solve every waypoint from the previous one, so the arm keeps one
        configuration through the move (solving the end from far away can land on
        the other wrist branch - a sudden flip at the end). Then refuse any step that
        turns a joint far more than the tip moves (a flip that got through anyway)."""
        out, prev = [], list(start)
        for js in path:
            again = kin.ik_down(kin.vfk(js), self.yaw, prev, self.min_tilt)
            js = again if again is not None else js
            turn = max(abs(a - b) for a, b in zip(js, prev))
            moved = float(np.linalg.norm(np.subtract(kin.vfk(js), kin.vfk(prev))))
            retilt = abs(kin.tilt_deg(js) - kin.tilt_deg(prev)) > 5.0   # intended straight <-> tilted change
            if not retilt and turn > config.MAX_JOINT_FLIP_DEG and turn > config.MAX_DEG_PER_MM * moved:
                raise MotionError(f"Refused before moving: a joint would turn {turn:.0f} deg while the "
                                  f"gripper moves {moved:.0f} mm (arm configuration flip) near "
                                  f"{np.round(kin.vfk(js))}")
            out.append(js)
            prev = js
        return out

    def _waypoints(self, start, start_xyz, end, end_xyz, depth=0):
        """Split a joint move until its simulated path stays within PATH_TOL_MM of
        the straight line between its ends."""
        a, b = np.array(start_xyz), np.array(end_xyz)
        worst = 0.0
        for t in np.linspace(0, 1, 21)[1:-1]:
            p = np.array(kin.vfk([s + (e - s) * t for s, e in zip(start, end)]))
            ab = b - a
            u = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            worst = max(worst, float(np.linalg.norm(p - (a + u * ab))))
        if worst <= config.PATH_TOL_MM or depth >= 6:
            return [end]
        mid_xyz = tuple((a + b) / 2)
        mid = kin.ik_down(mid_xyz, self.yaw, start, self.min_tilt)
        if mid is None:
            raise MotionError(f"Path leaves the reachable space near {np.round(mid_xyz)}")
        return self._waypoints(start, start_xyz, mid, mid_xyz, depth + 1) + \
            self._waypoints(mid, mid_xyz, end, end_xyz, depth + 1)

    def _vertical(self, x, y, z_from, z_to):
        """Straight up/down as ONE move (2026-11-19: 15 mm steps made the arm stop and
        start every step - jerky, and each stop lets the joints settle in their slack);
        _goto still splits it where the path would leave the line by > PATH_TOL_MM."""
        self._goto((x, y, z_to))

    def _approach(self, xyz):
        """Go to xyz (travel height above a target) so that the base ENDS turning toward
        the a-file: via a point config.BACKLASH_APPROACH_MM toward h. J1's slack is then
        always taken up the same way (it does not move on the way straight down)."""
        d = float(getattr(config, "BACKLASH_APPROACH_MM", 0.0) or 0.0)
        if d > 0:
            u = self.br.file_step[:2] / np.linalg.norm(self.br.file_step[:2])
            pre = (xyz[0] + d * u[0], xyz[1] + d * u[1], xyz[2])
            if kin.ik_down(pre, self.yaw, self._joints(), self.min_tilt) is not None:
                self._goto(pre, config.TRAVEL_SPEED_PCT)
            else:
                self.log("    (backlash pre-approach point out of reach - direct approach)")
        self._goto(xyz, config.TRAVEL_SPEED_PCT)

    # ---- pick / place
    def yaw_for(self, piece):
        """Jaw rotation for a piece: config.GRIP_YAW_PIECE (king, queen: straight)
        or the taught one."""
        return self.base_yaw + config.GRIP_YAW_PIECE.get(piece.upper(), 0.0)

    @staticmethod
    def tilt_for(where):
        from planner import tilt_for
        return tilt_for(where)

    def travel(self, carrying=None):
        x, y, z = kin.vfk(self._joints())
        ztr = self.br.board_z(x, y) + config.travel_z(carrying)
        if z < ztr - 1:
            self._vertical(x, y, z, ztr)

    def _target(self, op, tilt):
        """op.xyz, corrected for the arm model's tilted-pose error where the gripper is
        tilted away from rank 8 (rank-8 squares already use tilted-taught corners)."""
        x, y, zb = op.xyz
        if tilt > 0 and self.tilt_for(op.where) == 0:
            # tilted off rank 8: the tilted-taught a4/h4/a8/h8 correction applies, NOT the
            # straight-down measured aim map (2026-11-19: both together put h5 ~5 mm off)
            if len(op.where) == 2 and hasattr(self.br, "_aim_fix"):
                cx, cy = self.br._aim_fix(chess.parse_square(op.where))
                x, y = x - cx, y - cy
            dx, dy, dz = self.br.tilt_correction(x, y)
            x, y, zb = x + dx, y + dy, zb + dz
            if op.kind == "place":
                d = float(getattr(config, "TILTED_PLACE_TOWARD_RANK8_MM", 0.0))
                u8 = self.br.rank_step[:2] / np.linalg.norm(self.br.rank_step[:2])
                x, y = x + d * u8[0], y + d * u8[1]
        return x, y, zb

    def pick(self, op):
        piece = op.piece.upper()
        t_empty, t_full = config.travel_z(), config.travel_z(piece)
        self.log(f"  pick {op.piece} at {op.where}")
        self.travel()
        self.min_tilt = self.tilt_for(op.where) if op.tilt is None else op.tilt
        x0, y0, zb = self._target(op, self.min_tilt)
        bx, by = x0, y0                                  # the square's own aim, before any shift
        self.yaw = self.yaw_for(piece)
        is_square = len(op.where) == 2
        along = self.br.file_step[:2] / np.linalg.norm(self.br.file_step[:2])
        up = self.br.rank_step[:2] / np.linalg.norm(self.br.rank_step[:2])
        x0, y0, hint = self._offset_shift(op, piece, x0, y0, zb, along, up)
        # fallback retry offsets along a->h (used only when the cameras cannot say where
        # the piece is now): first the camera's hint, else GRIP_RETRY_FIRST, then the other way
        if hint is None:
            hint = self.offset_hint(op.where) if self.offset_hint else None
        first = float(config.GRIP_RETRY_FIRST) if hint is None else (1.0 if hint > 0 else -1.0)
        fallback = [first * config.GRIP_RETRY_MM, -first * config.GRIP_RETRY_MM]
        tries = 1 + (config.GRIP_RETRIES if (self.pick_check is not None and is_square) else 0)
        x, y = x0, y0
        for attempt in range(tries):
            if attempt:
                self.log(f"    RETRY {attempt}")
            back = float(getattr(config, "RANK8_THIRD_TRY_BACK_MM", 0.0) or 0.0)
            if attempt == 2 and is_square and op.where[1] == "8" and back > 0:
                # operator's rule: two misses on rank 8 -> aim 15 mm toward the robot
                nx, ny = x0 + back * up[0], y0 + back * up[1]
                seed, ok = self._joints(), True
                for z in (zb + t_full, zb + config.GRASP_Z[piece]):
                    seed = kin.ik_down((nx, ny, z), self.yaw, seed, self.min_tilt)
                    if seed is None:
                        ok = False
                        break
                if ok:
                    x, y = nx, ny
                    self.log(f"    rank 8, two misses: aiming {back:.0f} mm toward the robot")
                else:
                    self.log(f"    rank 8: {back:.0f} mm toward the robot is out of reach - normal retry")
            self.grip.open(config.GRIP_OPEN_PIECE[piece])
            # approach at the LIFTED height: the cameras photograph the empty jaws here,
            # and the arm comes back to exactly this pose after gripping
            self._approach((x, y, zb + t_full))
            before = None
            if self.pick_check is not None and hasattr(self.pick_check, "snapshot"):
                before = self.pick_check.snapshot()
            self._vertical(x, y, zb + t_full, zb + config.GRASP_Z[piece])
            felt = self.grip.hold(config.GRIP_HOLD_DEG[piece])
            self.log(f"    grip at Z {zb + config.GRASP_Z[piece]:.0f} (board {zb:.0f}): "
                     f"{'felt' if felt else 'NOT felt'} {getattr(self.grip, 'trace', '')}")
            if not felt:
                self.grip.open(config.GRIP_OPEN_PIECE[piece])
                self._vertical(x, y, zb + config.GRASP_Z[piece], zb + t_empty)
                if attempt + 1 < tries:
                    x, y = self._next_aim(op, bx, by, x0, y0, fallback[min(attempt, 1)], along, up)
                    continue
                raise MotionError(f"No piece felt at {op.where} (gripper closed with no load) - "
                                  "stopped for recovery")
            self.carry_tilt = self.min_tilt   # the piece now sits at this angle in the jaws
            self._vertical(x, y, zb + config.GRASP_Z[piece], zb + t_full)   # straight UP first
            if self.pick_check is None or not is_square:
                return
            held, diffs = None, {}
            if before:
                js = self._joints()
                tip = kin.tip(js)
                axis = tip - np.array(kin.fk(js)[:3])
                held, diffs = self.pick_check.held(before, self.br, tip, axis / np.linalg.norm(axis),
                                                   config.GRASP_Z[piece])
                from grip_check import grip_rise
                rise = grip_rise(getattr(self.grip, "trace", None), piece)
                if rise:
                    # the jaws closed on something: that beats a low side-camera reading
                    held = True
                diffs = {**diffs, "current_rise": rise}
            seen = self.pick_check(chess.parse_square(op.where))
            self._log_grip(op, attempt, diffs, held, seen)
            self.log(f"    camera check: jaws {'HOLD a piece' if held else 'EMPTY' if held is False else 'undecided'}"
                     f" {({k: round(d, 1) for k, d in diffs.items()})}; {op.where} looks {seen or 'undecided'}")
            aside = False
            if seen is None:
                # the arm itself often hides the square (tilted over rank 8). Operator's rule:
                # never travel on before the source is seen EMPTY -> step aside (still
                # holding) and look again from there
                self.log(f"    {op.where} undecided with the arm over it - stepping aside to look again")
                self._step_aside(op.where, piece)
                aside = True
                seen = self.pick_check(chess.parse_square(op.where))
                self.log(f"    {op.where} seen from aside: {seen or 'undecided'}")
                if seen is None:
                    # operator 2026-11-22 (capture of the h4 pawn stopped here): do not stop -
                    # the piece may or may not be in the jaws, so set it back down on its square
                    # (harmless if the jaws are empty) and retry, aimed where the cameras see it
                    if attempt + 1 >= tries:
                        raise MotionError(f"Cannot confirm the pick from {op.where} (still undecided from "
                                          f"aside after {tries} tries) - arm stopped, gripper closed: "
                                          "check the jaws and the square")
                    if diffs.get("current_rise"):
                        self.log("    still undecided, contact was felt - setting the piece back down, then retrying")
                        held = True
                    else:
                        # no contact felt: most likely a miss, the piece still standing - open
                        # just above where its top would be (a held piece drops only a little,
                        # onto its own square), then retry
                        self.log("    still undecided, no contact felt - opening just above the piece's "
                                 "top over its square, then retrying")
                        self._approach((x, y, zb + t_full))
                        z_top = zb + config.PIECE_HEIGHT[piece] + 5.0
                        self._vertical(x, y, zb + t_full, z_top)
                        self.grip.open(config.GRIP_OPEN_PIECE[piece])
                        self._vertical(x, y, z_top, zb + t_full)
                        self.carry_tilt = 0.0
                        x, y = self._next_aim(op, bx, by, x0, y0, fallback[min(attempt, 1)], along, up)
                        continue
            # ONLY the square seen empty confirms a pick
            if seen == "empty" and held is not False:
                return
            if seen == "empty" and held is False:
                if felt:
                    # narrow pieces are held by the foam pads: little current rise, and the
                    # side camera often reads low (2026-11-19: pawn c7 twice, knight b8 - all
                    # really in the jaws). Square empty + contact felt = picked; the set-down
                    # check then requires the destination to be seen occupied, else it stops.
                    self.log("    square empty and contact felt - taken as picked "
                             "(the destination will be checked)")
                    return
                raise MotionError(f"The jaws are empty but {op.where} is empty too - the piece was "
                                  "knocked or dragged away. Put it back on its square, then continue")
            # missed: the piece is still on its square (nudged or untouched).
            if held:
                # the jaws DO show something while the square still looks occupied (the
                # readings contradict): go back down and set it back, never drop it
                self.log("    square occupied but the jaws show a piece - setting it back down first")
                if aside:
                    self._approach((x, y, zb + t_full))
                z_rel = zb + config.GRASP_Z[piece] + config.PLACE_EXTRA_Z
                self._vertical(x, y, zb + t_full, z_rel)
                self.grip.open(config.GRIP_OPEN_PIECE[piece])
                self._vertical(x, y, z_rel, zb + t_full)
            else:
                # operator's rule 2026-11-20: the piece is on the board right under the jaws -
                # do NOT go back down onto it; open in the air where the arm is
                self.log("    missed: the piece is still on the square - opening in the air (no descent)")
                self.grip.open(config.GRIP_OPEN_PIECE[piece])
            self.carry_tilt = 0.0
            if attempt + 1 < tries:
                x, y = self._next_aim(op, bx, by, x0, y0, fallback[min(attempt, 1)], along, up)
        raise MotionError(f"Grip missed {op.where} {tries} times (camera still sees the piece "
                          f"there) - re-centre it by hand, then continue")

    def _log_grip(self, op, attempt, diffs, held, seen):
        """One row per grip in results/grips.csv (to set the holding thresholds per piece)."""
        import csv
        trace = getattr(self.grip, "trace", None) or []
        if not trace:                       # simulated gripper (tests, dry runs): nothing to record
            return
        path = os.path.join(config.RESULTS_DIR, "grips.csv")
        new = not os.path.exists(path)
        try:
            with open(path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["time", "piece", "square", "attempt", "tilt", "peak_amps", "final_amps", "final_deg",
                                "diff_front", "diff_side", "jaws", "square_seen", "trace"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), op.piece, op.where, attempt,
                            f"{self.min_tilt:g}", f"{max((a for _, a in trace), default=0):.2f}",
                            f"{trace[-1][1]:.2f}" if trace else "", trace[-1][0] if trace else "",
                            f"{diffs.get('front', ''):}", f"{diffs.get('side', ''):}",
                            {True: "held", False: "empty", None: "undecided"}[held], seen or "undecided",
                            " ".join(f"{d}:{a:.2f}" for d, a in trace)])
        except OSError:
            pass

    def _step_aside(self, square, carrying=None):
        """Up to travel height, then PLACE_ASIDE_SQ squares aside (toward the robot;
        ranks 7-8: toward rank 1) so the cameras see the square without the arm over it."""
        uh, u8 = self._board_dirs()
        r = chess.square_rank(chess.parse_square(square))
        d = config.PLACE_ASIDE_SQ * self.br.pitch * (-u8 if r >= 6 else u8)
        self.travel(carrying)
        x, y, z = kin.vfk(self._joints())
        aside = (x + d[0], y + d[1], z)
        if kin.ik_down(aside, self.yaw, self._joints(), self.min_tilt) is not None:
            self._goto(aside, config.TRAVEL_SPEED_PCT)
        time.sleep(0.8)                     # let the arm and the camera image settle

    def _next_aim(self, op, bx, by, x0, y0, fallback_mm, along, up):
        """Where to aim the next try after a miss: the cameras read where the piece
        stands NOW (it may have been nudged) from aside; else the old fixed shift."""
        if self.offset_reader is not None:
            self._step_aside(op.where)
            try:
                r = self.offset_reader(op.where)
            except Exception as e:      # noqa: BLE001
                r = None
                self.log(f"    offset reading failed ({e})")
            if r is not None:
                dx, dy = r["dx_mm"], r["dy_mm"]
                self.log(f"    cameras: the piece is now {dx:+.1f} mm toward h, {dy:+.1f} mm toward rank 8 "
                         "- aiming there")
                return bx + dx * along[0] + dy * up[0], by + dx * along[1] + dy * up[1]
        self.log(f"    no camera reading - shifting {fallback_mm:+.0f} mm along a->h")
        return x0 + fallback_mm * along[0], y0 + fallback_mm * along[1]

    def _offset_shift(self, op, piece, x0, y0, zb, along, up):
        """Operator's rule: the cameras read the piece's offset before the arm comes
        over it; more than 10 mm off in x or y -> aim 15 mm that way. Returns the
        (possibly shifted) aim and the a->h reading for the retry direction."""
        self.last_offset = None
        if getattr(self, "_correcting", False):          # placement fix: exact shift already in the op
            return x0, y0, None
        if not (config.OFFSET_CORRECTION and self.offset_reader is not None and len(op.where) == 2):
            return x0, y0, None
        try:
            r = self.offset_reader(op.where)
        except Exception as e:                       # never let the reading stop a pick
            self.log(f"    offset reading failed ({e}) - picking at the square centre")
            return x0, y0, None
        if r is None:
            self.log("    offset reading: no camera frame - picking at the square centre")
            return x0, y0, None
        self.last_offset = r
        sx, sy = r["shift_x_mm"], r["shift_y_mm"]
        # 2026-11-22: below the 15-mm rule's threshold, aim at the measured offset itself
        # once it is at least OFFSET_AIM_MIN_MM (h4 pawn read 6.8 mm off -> picked at the
        # centre and missed)
        m = float(getattr(config, "OFFSET_AIM_MIN_MM", 0.0) or 0.0)
        if m > 0:
            if not sx and abs(r["dx_mm"]) >= m:
                sx = r["dx_mm"]
            if not sy and abs(r["dy_mm"]) >= m:
                sy = r["dy_mm"]
        self.log(f"    cameras: piece {r['dx_mm']:+.1f} mm toward h, {r['dy_mm']:+.1f} mm toward rank 8"
                 + ("" if not (sx or sy) else f" -> pick shifted {sx:+.0f} mm toward h, {sy:+.0f} mm toward rank 8"))
        if not (sx or sy):
            return x0, y0, r["dx_mm"]
        x, y = x0 + sx * along[0] + sy * up[0], y0 + sx * along[1] + sy * up[1]
        seed = self._joints()
        for z in (zb + config.travel_z(piece), zb + config.GRASP_Z[piece]):
            js = kin.ik_down((x, y, z), self.yaw, seed, self.min_tilt)
            if js is None:
                self.log("    shifted pick is out of reach - picking at the square centre")
                return x0, y0, r["dx_mm"]
            seed = js
        return x, y, r["dx_mm"]

    def _confirm_pick(self, piece, x, y, zb, t_full):
        """The pick's square read undecided with the arm over it. Now the arm is over
        the destination: the source square must look EMPTY, else nothing is set down."""
        src, self.unconfirmed = self.unconfirmed, None
        seen = self.pick_check(chess.parse_square(src)) if self.pick_check is not None else None
        self.log(f"    {src} seen from the destination: {seen or 'undecided'}")
        if seen == "empty":
            return
        # not confirmed: release where it was picked? No - the jaws may hold the piece.
        # Stop here, hovering, gripper closed: the operator looks and decides.
        raise MotionError(
            f"Pick from {src} NOT confirmed ({src} looks {seen or 'undecided'} from here) - "
            + ("the piece is probably still on its square: the jaws are likely EMPTY. "
               if seen == "occupied" else "")
            + "Arm stopped above the destination with the gripper closed - check the jaws, "
              "then set the piece back by hand")

    def place(self, op):
        self._place_once(op)
        if (getattr(config, "PLACE_CHECK", False) and self.offset_reader is not None
                and len(op.where) == 2 and not getattr(self, "_correcting", False)):
            self._check_placement(op)

    def _board_dirs(self):
        uh = self.br.file_step[:2] / np.linalg.norm(self.br.file_step[:2])
        u8 = self.br.rank_step[:2] / np.linalg.norm(self.br.rank_step[:2])
        return uh, u8

    def _shifted(self, op, kind, sh, s8):
        """Copy of op as `kind`, aim moved sh mm toward h and s8 mm toward rank 8."""
        from planner import Op
        uh, u8 = self._board_dirs()
        x, y, z = op.xyz
        o = Op(kind, op.where, op.piece, "placement fix")
        o.xyz = (x + sh * uh[0] + s8 * u8[0], y + sh * uh[1] + s8 * u8[1], z)
        return o

    def _look(self, square):
        """Step aside at travel height and read the piece's offset (or None)."""
        self._step_aside(square)
        try:
            return self.offset_reader(square)
        except Exception as e:              # never let the check itself stop a game
            self.log(f"    placement check failed ({e})")
            return None

    def _check_placement(self, op):
        tol = config.PLACE_TOL_MM
        fix_h = fix_8 = 0.0                 # total counter-shift of the set-down so far
        for attempt in range(config.PLACE_FIX_TRIES + 1):
            r = self._look(op.where)
            if self.pick_check is not None:
                there = self.pick_check(chess.parse_square(op.where))
                if there == "empty":
                    raise MotionError(f"The piece is NOT on {op.where} after the set-down (the jaws were "
                                      "probably empty) - check the gripper and the board")
            if r is None:
                self.log(f"    placement check: no reading on {op.where} - left as placed")
                return
            dx, dy = r["dx_mm"], r["dy_mm"]
            ok = abs(dx) <= tol and abs(dy) <= tol
            self.log(f"    placement check {op.where}: {dx:+.1f} mm toward h, {dy:+.1f} mm toward rank 8"
                     + (" - OK" if ok else f" - more than {tol:g} mm off"))
            if ok:
                return
            if attempt == config.PLACE_FIX_TRIES:
                self.log(f"    placement still off after {attempt} fixes - left for the operator")
                return
            # pick it where it really stands, put it down shifted the other way
            fix_h -= dx
            fix_8 -= dy
            self.log(f"    FIX {attempt + 1}: re-pick at the piece, set down {fix_h:+.1f} / {fix_8:+.1f} mm")
            self._correcting = True                                   # exact shifts, no 15 mm rule
            try:
                p = self._shifted(op, "pick", dx, dy)
                p.tilt = self.last_place_tilt or None     # same tilt as the set-down being fixed
                self.pick(p)
                self._place_once(self._shifted(op, "place", fix_h, fix_8))
            finally:
                self._correcting = False

    def _place_once(self, op):
        piece = op.piece.upper()
        t_empty, t_full = config.travel_z(), config.travel_z(piece)
        self.log(f"  place {op.piece} at {op.where}")
        self.travel(piece)
        # keep the tilt it was picked with (straightening would tip the piece)
        self.min_tilt = max(self.tilt_for(op.where), self.carry_tilt)
        self.last_place_tilt = self.min_tilt
        x, y, zb = self._target(op, self.min_tilt)
        self.yaw = self.yaw_for(piece)
        self._approach((x, y, zb + t_full))
        if self.unconfirmed:
            self._confirm_pick(piece, x, y, zb, t_full)
        z_rel = zb + config.GRASP_Z[piece] + config.PLACE_EXTRA_Z
        self._vertical(x, y, zb + t_full, z_rel)
        self.grip.open(config.GRIP_OPEN_PIECE[piece])
        self.carry_tilt = 0.0
        self._vertical(x, y, z_rel, zb + t_empty)

    def preflight(self, ops):
        """Check every point of the plan is reachable before moving at all."""
        seed = self._joints()
        carry = self.carry_tilt
        for op in ops:
            if op.kind == "pause":
                continue
            if op.kind == "pick":
                tilt = self.tilt_for(op.where) if op.tilt is None else op.tilt
            else:
                tilt = max(self.tilt_for(op.where), carry)
            carry = tilt if op.kind == "pick" else 0.0
            x, y, zb = self._target(op, tilt)
            p = op.piece.upper()
            for z in (zb + config.travel_z(p), zb + config.travel_z(), zb + config.GRASP_Z[p]):
                js = kin.ik_down((x, y, z), self.yaw_for(p), seed, tilt)
                if js is None:
                    raise MotionError(f"{op.where} is out of reach at Z {z:.0f} "
                                      "(gripper down, J5 in its safe window) - plan refused before moving")
                seed = js

    def run(self, ops, park=True):
        """Execute a whole plan. Raises (and stops) on any problem."""
        if not self.lock.acquire():
            raise MotionError("Board workspace is in use by another arm (interlock held)")
        self.stop = False
        try:
            self.preflight(ops)
            for op in ops:
                if op.kind == "pause":
                    raise MotionError(f"Paused: {op.note}")
                (self.pick if op.kind == "pick" else self.place)(op)
            self.travel()
            if park:
                self.go_park()
        except Exception:
            self.log("  motion stopped - arm left in place, interlock still held")
            raise
        else:
            self.lock.release()

    def go_park(self):
        p = self.br.park
        if not p:
            return
        self.travel()
        self.arm.move_joints(list(p["joints"]), config.TRAVEL_SPEED_PCT)
        self.lock.release()

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
    def __init__(self, arm, gripper, board_robot, lock, log=print):
        self.arm, self.grip, self.br, self.lock = arm, gripper, board_robot, lock
        self.log = log
        self.stop = False
        self.min_tilt = 0.0           # >0 while working a rank-8 square (config.RANK8_TILT_DEG)
        self.carry_tilt = 0.0         # tilt the held piece was picked at: placed at the same tilt
        self.yaw = board_robot.yaw if board_robot.yaw is not None else config.GRIPPER_YAW_DEG
        self.base_yaw = self.yaw
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
        for js in self._waypoints(start, start_xyz, end, xyz):
            self._check_stop()
            self.arm.move_joints(js, speed or config.SPEED_PCT)

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
        n = max(1, int(math.ceil(abs(z_to - z_from) / config.VERTICAL_STEP_MM)))
        for i in range(1, n + 1):
            self._goto((x, y, z_from + (z_to - z_from) * i / n))

    # ---- pick / place
    def yaw_for(self, piece):
        """Jaw rotation for a piece: config.GRIP_YAW_PIECE (king, queen: straight)
        or the taught one."""
        return config.GRIP_YAW_PIECE.get(piece.upper(), self.base_yaw)

    @staticmethod
    def tilt_for(where):
        """Minimum gripper tilt for a target: rank 8 is too close to the base to be
        worked straight down (J3 limit), so it always uses config.RANK8_TILT_DEG."""
        return config.RANK8_TILT_DEG if len(where) == 2 and where[1] == "8" else 0.0

    def travel(self, carrying=None):
        x, y, z = kin.vfk(self._joints())
        ztr = self.br.board_z(x, y) + config.travel_z(carrying)
        if z < ztr - 1:
            self._vertical(x, y, z, ztr)

    def pick(self, op):
        x, y, zb = op.xyz
        piece = op.piece.upper()
        t_empty, t_full = config.travel_z(), config.travel_z(piece)
        self.log(f"  pick {op.piece} at {op.where}")
        self.travel()
        self.min_tilt = self.tilt_for(op.where)
        self.yaw = self.yaw_for(piece)
        self.grip.open(config.GRIP_OPEN_PIECE[piece])
        self._goto((x, y, zb + t_empty))
        self._vertical(x, y, zb + t_empty, zb + config.GRASP_Z[piece])
        felt = self.grip.hold(config.GRIP_HOLD_DEG[piece])
        self.log(f"    grip at Z {zb + config.GRASP_Z[piece]:.0f} (board {zb:.0f}): "
                 f"{'felt' if felt else 'NOT felt'} {getattr(self.grip, 'trace', '')}")
        if not felt:
            self.grip.open(config.GRIP_OPEN_PIECE[piece])
            self._vertical(x, y, zb + config.GRASP_Z[piece], zb + t_empty)
            raise MotionError(f"No piece felt at {op.where} (gripper closed with no load) - "
                              "stopped for recovery")
        self.carry_tilt = self.min_tilt   # the piece now sits at this angle in the jaws
        self._vertical(x, y, zb + config.GRASP_Z[piece], zb + t_full)

    def place(self, op):
        x, y, zb = op.xyz
        piece = op.piece.upper()
        t_empty, t_full = config.travel_z(), config.travel_z(piece)
        self.log(f"  place {op.piece} at {op.where}")
        self.travel(piece)
        # keep the tilt it was picked with (straightening would tip the piece)
        self.min_tilt = max(self.tilt_for(op.where), self.carry_tilt)
        self.yaw = self.yaw_for(piece)
        self._goto((x, y, zb + t_full))
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
            tilt = self.tilt_for(op.where) if op.kind == "pick" else max(self.tilt_for(op.where), carry)
            carry = tilt if op.kind == "pick" else 0.0
            x, y, zb = op.xyz
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
        self.arm.move_joints(list(p["joints"]), config.SPEED_PCT)
        self.lock.release()

"""Robot setup for chess: calibrate (without J5), zero J5 by eye, teach the board.

Run:  python robot_setup.py      (close the AR4 HMI first - one program per port)

1. CALIBRATE J1-J4 + J6: Annin auto-calibration for every joint except J5, whose
   limit switch is where the wide servo gripper would fold into the forearm.
2. ZERO J5 BY EYE: jog J5 in small steps until the gripper points straight along
   the forearm, then press "This is J5 = 0". Nothing moves on that press; the
   firmware is told J5's angle (SP command).
3. GRIPPER DOWN: turns the wrist so the gripper points straight down above where
   it is, jaws at the chosen direction over the board. All later jogs keep that.
4. TEACH: jog until the closed jaw tips just touch the board at the centre of
   a1, h1, h8, a8, saving each. More exact: hold a pointed tool in the closed
   jaws, enter how far its point sticks out below the jaw tips, and touch the
   point to the square centres (X/Y are exact straight down; the height is corrected). Then the tray slot 0 (first captured-piece spot,
   jaw tips on the mat), the reserve spot (spare queen for promotion), and PARK.
   Results go to config/board_robot.json.
5. TILTED: rank 8 is worked tilted back, a different arm pose from the
   straight-down teaching, and the arm's geometry model is 8-10 mm off between
   the two. So a8 and h8 are also taught tilted (jaw tips on the centre; rank-8
   squares are interpolated between them), and a4 and h4 too: their difference
   from the straight-down positions corrects every tilted set-down on ranks 3-7.
Every jog is a checked joint move: gripper down, J5 inside config.J5_WINDOW.
Speed is capped at 25 %; keep a hand on the E-stop.
"""
import json
import os
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np

import config
import kin
from ar4link import AR4, AR4Error

STEPS_MM = (0.5, 1, 2, 5, 10, 25)
TEACH = ("a1", "h1", "h8", "a8", "tray", "reserve")
# points taught in the TILTED working pose: a8/h8 (rank 8 is worked tilted) and
# a4/h4 (pieces carried tilted from rank 8 are set down tilted on ranks 3-7)
TILTED_KEYS = ("a8", "h8", "a4", "h4")


class SetupApp:
    def __init__(self, root):
        self.root = root
        self.arm = AR4(safe=True, j5_window=config.J5_WINDOW)
        self.gripper = None
        root.title("Team 2 chess - robot setup")
        self.data = {"squares": {}}
        if os.path.exists(config.BOARD_ROBOT_FILE):
            with open(config.BOARD_ROBOT_FILE) as f:
                self.data = json.load(f)
            self.data.setdefault("squares", {})
        # jaws parallel to the board's square lines, from the taught corners if any
        try:
            from planner import BoardRobot
            jy = BoardRobot().jaw_yaw
        except Exception:      # noqa: BLE001
            jy = 0.0
        self.yaw = tk.DoubleVar(value=jy)
        # pointer tool held in the closed jaws: mm its point sticks out below the jaw tips
        self.pointer = tk.DoubleVar(value=0.0)
        # rank-8 corners are taught in the pose they are worked in: tilted back
        self.tilted = tk.BooleanVar(value=False)
        self.step = tk.DoubleVar(value=5)
        self.speed = tk.IntVar(value=10)
        self.pose_var, self.saved_var = tk.StringVar(), tk.StringVar()
        self.status = tk.StringVar(value=f"Connected: {self.arm.port}")
        self.buttons = []

        tk.Label(root, textvariable=self.pose_var, font=("monospace", 11), justify="left")\
            .grid(row=0, column=0, sticky="w", padx=8, pady=6)

        f1 = tk.LabelFrame(root, text="1-2. Calibrate (J5 excluded) and zero J5 by eye")
        f1.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self._btn(f1, "Calibrate J1-J4 + J6", self.calibrate, 0, 0, bg="#b71c1c")
        for i, d in enumerate((-2, -0.5, 0.5, 2)):
            self._btn(f1, f"J5 {d:+g}", lambda d=d: self.jog_j5(d), 0, i + 1)
        self._btn(f1, "This is J5 = 0 (gripper in line with forearm)", self.zero_j5, 1, 0,
                  span=5, bg="#2e7d32")

        f2 = tk.LabelFrame(root, text="3. Gripper pointing down - jog (fingertip moves straight)")
        f2.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._btn(f2, "Point gripper DOWN here", self.point_down, 0, 0, span=2, bg="#1565c0")
        tk.Label(f2, text="Jaw direction (deg):").grid(row=0, column=2)
        tk.Spinbox(f2, from_=-90, to=90, increment=5, width=5, textvariable=self.yaw,
                   command=self.point_down).grid(row=0, column=3)
        tk.Label(f2, text="Pointer below jaw tips (mm, 0 = none):").grid(row=0, column=4)
        tk.Entry(f2, width=6, textvariable=self.pointer).grid(row=0, column=5)
        st = tk.Frame(f2)
        st.grid(row=1, column=0, columnspan=6, sticky="w")
        tk.Label(st, text="Step mm:").pack(side="left")
        for s in STEPS_MM:
            tk.Radiobutton(st, text=str(s), variable=self.step, value=s).pack(side="left")
        tk.Label(st, text="  Speed %:").pack(side="left")
        tk.Spinbox(st, from_=2, to=25, width=4, textvariable=self.speed).pack(side="left")
        for col, (ax, neg, pos) in enumerate((("X", "X- toward base", "X+ away"),
                                              ("Y", "Y- robot's right", "Y+ robot's left"),
                                              ("Z", "Z- down", "Z+ up"))):
            self._btn(f2, pos, lambda a=col: self.jog(a, +1), 2, col * 2, span=2)
            self._btn(f2, neg, lambda a=col: self.jog(a, -1), 3, col * 2, span=2)

        f3 = tk.LabelFrame(root, text="4. Teach (closed jaw tips touching the board / mat)")
        f3.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        for i, k in enumerate(TEACH):
            self._btn(f3, f"Save {k}", lambda k=k: self.save_point(k), 0, i, bg="#2e7d32")
            self._btn(f3, f"20 mm above {k}", lambda k=k: self.goto_above(k), 1, i)
        self._btn(f3, "Save PARK (clear of board & camera)", self.save_park, 2, 0, span=3, bg="#1565c0")
        self._btn(f3, "Lift 30 mm", lambda: self.jog(2, +1, 30), 2, 3)
        self._btn(f3, "Gripper open", lambda: self.grip("open"), 2, 4)
        self._btn(f3, "Gripper close", lambda: self.grip("close"), 2, 5)
        self._btn(f3, "Go to PARK", self.go_park, 3, 0, span=3, bg="#1565c0")

        f4 = tk.LabelFrame(root, text=f"5. TILTED ({config.RANK8_TILT_DEG:g} deg back, as when gripping "
                                      "on rank 8 / carrying from it): jaw tips touching the centre")
        f4.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        tk.Checkbutton(f4, text="Jog tilted", variable=self.tilted).grid(row=0, column=0)
        self._btn(f4, f"Tilt back {config.RANK8_TILT_DEG:g} deg here", self.tilt_here, 0, 1, bg="#1565c0")
        for i, k in enumerate(TILTED_KEYS):
            self._btn(f4, f"Save {k} TILTED", lambda k=k: self.save_tilted(k), 1 + i // 2, 2 * (i % 2),
                      bg="#2e7d32")
            self._btn(f4, f"20 mm above {k} tilted", lambda k=k: self.goto_above_tilted(k), 1 + i // 2,
                      2 * (i % 2) + 1)

        tk.Label(root, textvariable=self.saved_var, font=("monospace", 10), justify="left")\
            .grid(row=5, column=0, sticky="w", padx=8)
        tk.Label(root, textvariable=self.status, fg="#444", wraplength=720, justify="left")\
            .grid(row=6, column=0, sticky="w", padx=8, pady=6)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        p = self.refresh()
        if p and kin.tilt_deg(p.joints) > config.RANK8_TILT_DEG - 5:
            self.tilted.set(True)        # already tilted (rank 8): keep jogging tilted
        if p and all(abs(a) < 1e-6 for a in p.joints):
            self.say("All joints read 0: the controller restarted. Start with step 1.")
        self.show_saved()

    # ---------------------------------------------------------------- helpers
    def _btn(self, parent, text, cmd, r, c, span=1, bg=None):
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg="white" if bg else None)
        b.grid(row=r, column=c, columnspan=span, sticky="ew", padx=3, pady=2)
        self.buttons.append(b)
        return b

    def say(self, t):
        self.status.set(t)
        print(time.strftime("%H:%M:%S"), t, flush=True)

    def busy(self, on):
        for b in self.buttons:
            b.config(state="disabled" if on else "normal")
        self.root.update()

    def refresh(self):
        try:
            p = self.arm.pose()
        except AR4Error as e:
            self.pose_var.set(f"Position unavailable: {e}")
            return None
        down = kin.tool_axis_down(p.joints, 2.0)
        self.pose_var.set(
            f"X {p.x:8.2f}  Y {p.y:8.2f}  Z {p.z:8.2f}   (flange, mm)   gripper "
            f"{'DOWN' if down else 'not down'}\n" +
            "  ".join(f"J{i + 1} {a:7.2f}" for i, a in enumerate(p.joints)))
        return p

    def show_saved(self):
        lines = []
        for k in TEACH:
            d = self.data["squares"].get(k) if k not in ("tray", "reserve") else self.data.get(k)
            lines.append(f"{k:8s}: " + (f"X {d['x']:7.1f} Y {d['y']:7.1f} Z {d['z']:7.1f}" if d else "-"))
        lines.append(f"park    : {'saved' if self.data.get('park') else '-'}    jaw direction "
                     f"{self.data.get('yaw', '-')}")
        sq = self.data["squares"]
        if all(k in sq for k in ("a1", "h1", "h8", "a8")):
            p = np.array([[sq[k]["x"], sq[k]["y"], sq[k]["z"]] for k in ("a1", "h1", "h8", "a8")])
            A = np.c_[p[:, 0], p[:, 1], np.ones(4)]
            coef, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
            res = p[:, 2] - A @ coef
            side = [np.linalg.norm(p[i, :2] - p[(i + 1) % 4, :2]) / 7 for i in range(4)]
            lines.append(f"board check: square pitch {min(side):.1f}-{max(side):.1f} mm "
                         f"(expect 47), plane residual {np.abs(res).max():.1f} mm")
        self.saved_var.set("\n".join(lines))

    def move(self, js, allow_j5_outside=False):
        self.busy(True)
        try:
            self.arm.move_joints(js, int(self.speed.get()), allow_j5_outside=allow_j5_outside)
        except AR4Error as e:
            self.say(f"Refused / failed: {e}")
        finally:
            self.busy(False)
            self.refresh()

    # ------------------------------------------------------- safe moves (2026-11-19)
    # "go above" once moved in ONE joint move from a pose pressed against a8 to over
    # h1: the joints swung through the board. Every go-to / re-orientation now lifts
    # STRAIGHT UP first, changes the wrist only at a safe height, travels across at
    # that height and comes STRAIGHT DOWN. All waypoints are solved before moving;
    # if any fails, nothing moves.
    SAFE_ABOVE_MM = 80.0         # jaw tips this far above the board while travelling / re-orienting
    STEP_MM = 10.0               # vertical waypoint spacing
    MAX_STEP_DEG = 12.0          # a waypoint turning any joint more than this -> refuse (wrist flip)

    def _board_z(self):
        zs = [d["z"] for d in self.data.get("squares", {}).values() if isinstance(d, dict) and "z" in d]
        return max(zs) if zs else 100.0

    @staticmethod
    def _rise_step(seed, dz):
        """Joints that move the whole tool straight up by dz mm, wrist orientation
        unchanged, solved by small least-squares steps FROM the current joints (stays on
        the same wrist solution; works at J5 ~ 0 where the analytic solver flips)."""
        from scipy.optimize import least_squares
        seed = np.asarray(seed, float)
        refs = [(0.0, 0.0, 0.0), (0.0, 0.0, 100.0), (50.0, 0.0, 0.0)]     # flange, along the tool, across
        goal = [kin._axis_point(seed, r) + np.array([0.0, 0.0, dz]) for r in refs]

        def res(q):
            return np.concatenate([kin._axis_point(q, r) - g for r, g in zip(refs, goal)] + [0.02 * (q - seed)])
        sol = least_squares(res, seed, diff_step=1e-4, xtol=1e-6)
        q = sol.x
        err = max(np.linalg.norm(kin._axis_point(q, r) - g) for r, g in zip(refs, goal))
        return (list(q), err) if err < 1.0 else (None, err)

    def _lift(self, dz):
        """Straight up by dz mm with the wrist held as it is, in STEP_MM steps
        -> (joint list, "") or (None, why)."""
        js = list(self.arm.pose().joints)
        z0 = kin.fk(js)[2]
        out, seed = [], js
        lo, hi = config.J5_WINDOW
        n = int(np.ceil(max(dz, 0.0) / self.STEP_MM))
        for i in range(1, n + 1):
            q, err = self._rise_step(seed, dz / n)
            z = z0 + dz * i / n
            why = (f"out of reach ({err:.0f} mm short)" if q is None else
                   "a joint would pass its limit " + str([f"J{k + 1} {a:.0f}" for k, (a, hi_, lo_) in
                                                          enumerate(zip(q, kin.POS_LIM, kin.NEG_LIM))
                                                          if not -lo_ <= a <= hi_])
                   if not kin.within_limits(q) else
                   f"J5 would reach {q[4]:.0f} deg, outside its window" if not lo <= q[4] <= hi else
                   "a joint would jump (wrist flip)" if max(abs(u - v) for u, v in zip(q, seed)) > self.MAX_STEP_DEG
                   else "")
            if why:
                print("lift refused from joints", [round(a, 1) for a in js], flush=True)
                return None, f"cannot rise straight up past {z:.0f} mm: {why}"
            out.append(q)
            seed = q
        return out, ""

    OVERRIDE_PRESSES = 3         # the same refused button this many times within OVERRIDE_SECS ...
    OVERRIDE_SECS = 6.0          # ... -> warning dialog -> the move is made anyway (operator's call)

    def _refuse(self, msg, xyz, tilt, action=None):
        """A safe move was refused. Pressing the same button OVERRIDE_PRESSES times
        quickly asks for confirmation and then moves anyway (lift by joint move to the
        safe height, over, down - no checks beyond the controller's own limits)."""
        key = (round(xyz[0]), round(xyz[1]), round(xyz[2]), round(tilt))
        now = time.time()
        hist = [t for t in getattr(self, "_refusals", {}).get(key, []) if now - t < self.OVERRIDE_SECS] + [now]
        self._refusals = {key: hist}
        left = self.OVERRIDE_PRESSES - len(hist)
        if left > 0:
            self.say(f"{msg}  (press {left} more time{'s' * (left > 1)} quickly to override)")
            return False
        self._refusals = {}
        if not messagebox.askyesno(
                "Override safety check",
                f"The software thinks this move is unsafe:\n\n{msg}\n\nMove anyway? The arm goes "
                "UP to the safe height at its current spot, then across, then down - each as one "
                "joint move with NO path checks. Make sure the jaws are clear of the board and "
                "keep a hand on the E-stop."):
            self.say("Override cancelled - nothing moved")
            return False
        return action() if action else self._direct_goto(xyz, tilt)

    def _direct_goto(self, xyz, tilt):
        """Operator override: always moves if any joint solution exists. Up (wrist held)
        if possible, else from where it is; over the target at the safe height; down.
        J5's window is NOT enforced here (the operator decided; E-stop at hand)."""
        p = self.arm.pose()
        cur = kin.vfk(p.joints)
        z_safe = max(cur[2], xyz[2], self._board_z() + self.SAFE_ABOVE_MM)
        seed = list(p.joints)
        steps = []
        lift, _ = self._lift(z_safe - cur[2])
        if lift:
            steps += lift
            seed = lift[-1]
        else:
            q, _ = self._rise_step(seed, z_safe - cur[2])
            if q is not None:
                steps.append(q)
                seed = q
            else:
                self.say("Override: cannot rise with the wrist as it is - going from the current height")
        for q in ((xyz[0], xyz[1], z_safe), tuple(xyz)):
            js = kin.ik_down(q, float(self.yaw.get()), seed, tilt)
            if js is None:
                js = kin.ik_down(q, float(self.yaw.get()), seed, tilt, )
            if js is None:
                self.say(f"Override: the arm has no joint solution at {np.round(q, 0)} - cannot move there")
                return False
            steps.append(js)
            seed = js
        self.say("OVERRIDE: moving without path checks (J5 window not enforced)")
        self.busy(True)
        try:
            for js in steps:
                self.arm.move_joints(js, int(self.speed.get()), allow_j5_outside=True)
        except AR4Error as e:
            self.say(f"Controller refused: {e}")
            return False
        finally:
            self.busy(False)
            self.refresh()
        return True

    def safe_goto(self, xyz, tilt=0.0):
        """Lift straight up -> re-orient at safe height -> across -> straight down to xyz."""
        p = self.arm.pose()
        cur = np.array(kin.vfk(p.joints), float)
        cur_tilt = kin.tilt_deg(p.joints)
        cur_tilt = 0.0 if cur_tilt < 2 else cur_tilt
        z_safe = max(cur[2], xyz[2], self._board_z() + self.SAFE_ABOVE_MM)
        up = (cur[0], cur[1], z_safe)
        over = (xyz[0], xyz[1], z_safe)
        plan, why = self._lift(z_safe - cur[2])
        if plan is None and cur[2] >= self._board_z() + 40.0:
            self.say(f"(could not rise further: {why} - continuing from the current height)")
            plan = []
            z_safe = cur[2]
            up = (cur[0], cur[1], z_safe)
            over = (xyz[0], xyz[1], z_safe)
        if plan is None:
            return self._refuse(f"Refused (lifting straight up): {why}. Jog the arm up by hand first.", xyz, tilt)
        # re-orient in place at the safe height (allowed to turn joints more), then across and down
        seed_js = plan[-1] if plan else list(p.joints)
        js_re = kin.ik_down(up, float(self.yaw.get()), seed_js, tilt)
        if js_re is None:
            return self._refuse(f"Refused: no {'tilted' if tilt else 'gripper-down'} pose above the current point", xyz, tilt)
        rest_seed = js_re
        seed = rest_seed
        rest = [js_re]
        for a, b, step in ((up, over, self.STEP_MM), (over, tuple(xyz), self.STEP_MM)):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            for i in range(1, n + 1):
                q = a + (b - a) * i / n
                js = kin.ik_down(tuple(q), float(self.yaw.get()), seed, tilt)
                if js is None:
                    return self._refuse(f"Refused: no solution at {np.round(q, 0)}", xyz, tilt)
                if max(abs(u - v) for u, v in zip(js, seed)) > self.MAX_STEP_DEG:
                    return self._refuse(f"Refused: a joint would jump near {np.round(q, 0)} (wrist flip)", xyz, tilt)
                rest.append(js)
                seed = js
        if max(abs(u - v) for u, v in zip(js_re, seed_js)) > 60 and not messagebox.askyesno(
                "Large move", "Re-orienting at the safe height turns some joints by more than 60 deg. Continue?"):
            return False
        self.busy(True)
        try:
            for js in plan + rest:
                self.arm.move_joints(js, int(self.speed.get()))
        except AR4Error as e:
            self.say(f"Refused / failed: {e}")
            return False
        finally:
            self.busy(False)
            self.refresh()
        return True

    # ---------------------------------------------------------------- steps
    def calibrate(self):
        if not messagebox.askokcancel(
                "Calibrate", "J1, J2, J3 then J4 and J6 drive to their limit switches and back "
                "off quickly (firmware speed). J5 does NOT move.\n\nArea around the arm clear? "
                "Hand on the E-stop?"):
            return
        self.busy(True)
        try:
            p = self.arm.calibrate_without_j5()
            self.say(f"Calibrated (J5 untouched): {p}. Now zero J5 by eye.")
        except AR4Error as e:
            self.say(f"Calibration failed: {e}")
        finally:
            self.busy(False)
            self.refresh()

    def jog_j5(self, d):
        js = list(self.arm.pose().joints)
        js[4] += d
        self.move(js, allow_j5_outside=True)

    def zero_j5(self):
        if not messagebox.askyesno("Zero J5", "Is the gripper pointing exactly along the "
                                   "forearm (straight out, not bent up or down)?"):
            return
        p = self.arm.set_joint(5, 0.0)
        self.data["j5_zeroed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.say(f"J5 set to 0 here. {p}")
        self.refresh()

    def point_down(self):
        """Gripper straight down over the current point: lifts to the safe height
        first, turns the wrist up there, comes back down (never turns at the board)."""
        p = self.arm.pose()
        x, y, z = kin.vfk(p.joints)
        if self.safe_goto((x, y, max(z, self._board_z() + 40.0)), 0.0):
            self.tilted.set(False)

    def jog(self, axis, sign, dist=None):
        p = self.arm.pose()
        target = list(kin.vfk(p.joints))       # = flange when straight down; tips + tool length when tilted
        target[axis] += sign * (dist or float(self.step.get()))
        tilt = config.RANK8_TILT_DEG if self.tilted.get() else 0.0
        js = kin.ik_down(tuple(target), float(self.yaw.get()), p.joints, tilt)
        if js is None:
            self.say(f"Refused: no {'tilted' if tilt else 'gripper-down'} solution at {np.round(target, 1)}")
            return
        if kin.tilt_deg(p.joints) > config.MAX_TILT_DEG + 1:
            self.say("Press 'Point gripper DOWN here' first")
            return
        self.move(js)

    def save_point(self, k):
        p = self.arm.pose()
        if k in TILTED_KEYS and kin.tilt_deg(p.joints) > config.RANK8_TILT_DEG - 5:
            # 2026-11-19: straight a8/h8 saves silently went to the TILTED points -> ask
            if messagebox.askyesno(
                    "Gripper is tilted",
                    f"The gripper is tilted {kin.tilt_deg(p.joints):.0f} deg, so this would save the "
                    f"TILTED {k} (rank-8 pick point), not the straight-down corner.\n\n"
                    f"Yes = save as TILTED {k}\nNo = cancel (press 'Point gripper DOWN here' first "
                    f"to teach the straight corner)"):
                self.save_tilted(k)
            else:
                self.say(f"Not saved - for the straight {k}: 'Point gripper DOWN here', centre, then Save {k}")
            return
        if not kin.tool_axis_down(p.joints, 1.0):
            self.say("Not saved: the gripper must point straight down")
            return
        # Straight down, flange, jaw tips and a pointer's point share X and Y; only
        # the height differs: with the point touching, the jaw tips (what the stored
        # point stands for) would be `pointer` mm lower.
        ptr = float(self.pointer.get() or 0.0)
        d = {"x": p.x, "y": p.y, "z": p.z - ptr, "joints": p.joints}
        if ptr:
            d["pointer_mm"] = ptr
        if k in ("tray", "reserve"):
            self.data[k] = d
        else:
            self.data["squares"][k] = d
        self.data["yaw"] = float(self.yaw.get())
        self._write()
        self.say(f"Saved {k}" + (f" (pointer {ptr:g} mm: height corrected)" if ptr else ""))

    def tilt_here(self):
        """Tilt the gripper straight back along X (J4 rolls as needed, J6 = J1 + yaw as
        everywhere), jaw tips staying put."""
        p = self.arm.pose()
        x, y, z = kin.vfk(p.joints)
        if self.safe_goto((x, y, max(z, self._board_z() + 40.0)), config.RANK8_TILT_DEG):
            self.tilted.set(True)

    def save_tilted(self, k):
        """Rank-8 corner taught in the tilted working pose. Stored in planning
        coordinates (kin.vfk: jaw tips + tool length straight up), like every target."""
        p = self.arm.pose()
        t = kin.tilt_deg(p.joints)
        # the tilt solver uses RANK8_TILT_DEG or the next 5-deg step it can reach
        # (games use the same solver, so any of those is the working pose)
        if not config.RANK8_TILT_DEG - 3 <= t <= config.MAX_TILT_DEG + 1:
            self.say(f"Not saved: the gripper is tilted {t:.0f} deg, expected {config.RANK8_TILT_DEG:g} "
                     f"or more (tick 'Jog tilted' / press 'Tilt back' first)")
            return
        x, y, z = kin.vfk(p.joints)
        self.data.setdefault("squares_tilted", {})[k] = {
            "x": float(x), "y": float(y), "z": float(z), "joints": p.joints, "tilt": round(t, 1)}
        self._write()
        print(f"saved {k} TILTED tilt {t:.1f} at {x:.1f} {y:.1f} {z:.1f}", flush=True)
        missing = [m for m in TILTED_KEYS if m not in self.data["squares_tilted"]]
        self.say(f"Saved {k} TILTED (tips at X {x:.1f} Y {y:.1f})"
                 + (f" - still to teach tilted: {', '.join(missing)}" if missing else " - all tilted points taught"))

    def goto_above_tilted(self, k):
        d = self.data.get("squares_tilted", {}).get(k) or self.data["squares"].get(k)
        if not d:
            self.say(f"{k} not taught yet")
            return
        if self.safe_goto((d["x"], d["y"], d["z"] + 20), config.RANK8_TILT_DEG):
            self.tilted.set(True)

    def go_park(self):
        """Straight up to the safe height (wrist held), then the joints move evenly into
        the saved park pose. Every point of that path is checked beforehand to keep the
        jaw tips at least 30 mm over the board; if not -> refused (triple-press overrides)."""
        park = self.data.get("park")
        if not park:
            self.say("No PARK saved yet")
            return
        p = self.arm.pose()
        cur = kin.vfk(p.joints)
        z_safe = max(cur[2], self._board_z() + self.SAFE_ABOVE_MM)
        plan, why = self._lift(z_safe - cur[2])
        if plan is None:
            if cur[2] < self._board_z() + 40.0:
                self._refuse_park(f"Refused (lifting straight up): {why}. Jog the arm up first.")
                return
            plan = []
        start = plan[-1] if plan else list(p.joints)
        goal = list(park["joints"])
        n = max(1, int(np.ceil(max(abs(u - v) for u, v in zip(start, goal)) / 5.0)))
        path = [list(np.array(start) + (np.array(goal) - np.array(start)) * i / n) for i in range(1, n + 1)]
        board_tip = self._board_z() - config.TOOL_LENGTH_MM
        low = min(kin.tip(js)[2] for js in path) - board_tip
        if low < 30.0:
            self._refuse_park(f"Refused: on the way to PARK the jaws would come within {low:.0f} mm of the board")
            return
        self.busy(True)
        try:
            for js in plan + path:
                self.arm.move_joints(js, int(self.speed.get()))
        except AR4Error as e:
            self.say(f"Refused / failed: {e}")
            return
        finally:
            self.busy(False)
            self.refresh()
        self.tilted.set(False)
        self.say("At PARK")

    def _refuse_park(self, msg):
        x, y, z = kin.vfk(self.data["park"]["joints"])
        self._refuse(msg, (x, y, z), -1.0, action=self._park_direct)

    def _park_direct(self):
        """Override: up with the wrist held (if possible), then one joint move to PARK."""
        p = self.arm.pose()
        cur = kin.vfk(p.joints)
        lift, _ = self._lift(max(cur[2], self._board_z() + self.SAFE_ABOVE_MM) - cur[2])
        self.say("OVERRIDE: to PARK without path checks")
        self.busy(True)
        try:
            for js in (lift or []) + [list(self.data["park"]["joints"])]:
                self.arm.move_joints(js, int(self.speed.get()), allow_j5_outside=True)
        except AR4Error as e:
            self.say(f"Controller refused: {e}")
            return False
        finally:
            self.busy(False)
            self.refresh()
        self.tilted.set(False)
        return True

    def save_park(self):
        p = self.arm.pose()
        lo, hi = config.J5_WINDOW
        if not lo <= p.joints[4] <= hi:
            self.say("Not saved: J5 outside its safe window")
            return
        self.data["park"] = {"joints": p.joints, "x": p.x, "y": p.y, "z": p.z}
        self._write()
        self.say("Saved PARK")

    def goto_above(self, k):
        d = self.data["squares"].get(k) if k not in ("tray", "reserve") else self.data.get(k)
        if not d:
            self.say(f"{k} not taught yet")
            return
        ptr = float(self.pointer.get() or 0.0)              # a pointer sticks out below the tips
        if self.safe_goto((d["x"], d["y"], d["z"] + 20 + ptr), 0.0):
            self.tilted.set(False)

    def grip(self, what):
        try:
            if self.gripper is None:
                from gripper import Gripper
                self.say("Connecting to the gripper Nano ...")
                self.root.update()
                self.gripper = Gripper()
            self.gripper.open() if what == "open" else self.gripper.close()
            self.say(f"Gripper {what}")
        except Exception as e:      # noqa: BLE001
            self.say(f"Gripper: {e}")

    def _write(self):
        self.data["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data["port"] = self.arm.port
        os.makedirs(os.path.dirname(config.BOARD_ROBOT_FILE), exist_ok=True)
        with open(config.BOARD_ROBOT_FILE, "w") as f:
            json.dump(self.data, f, indent=2)
        self.show_saved()

    def quit(self):
        try:
            self.arm.sync_app_config()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    SetupApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

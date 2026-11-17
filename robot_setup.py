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
   a1, h1, h8, a8, saving each. Then the tray slot 0 (first captured-piece spot,
   jaw tips on the mat), the reserve spot (spare queen for promotion), and PARK.
   Results go to config/board_robot.json.
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
        self.yaw = tk.DoubleVar(value=self.data.get("yaw", config.GRIPPER_YAW_DEG))
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

        tk.Label(root, textvariable=self.saved_var, font=("monospace", 10), justify="left")\
            .grid(row=4, column=0, sticky="w", padx=8)
        tk.Label(root, textvariable=self.status, fg="#444", wraplength=720, justify="left")\
            .grid(row=5, column=0, sticky="w", padx=8, pady=6)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        p = self.refresh()
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
        print(t)

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
        p = self.arm.pose()
        js = kin.ik_down(p.xyz, float(self.yaw.get()), p.joints)
        if js is None:
            self.say("No gripper-down solution here (out of reach or J5 outside its window) - "
                     "move the arm over the board first")
            return
        if max(abs(a - b) for a, b in zip(js, p.joints)) > 60 and not messagebox.askyesno(
                "Large move", "That turns some joints by more than 60 deg. Continue?"):
            return
        self.move(js)

    def jog(self, axis, sign, dist=None):
        p = self.arm.pose()
        target = list(kin.vfk(p.joints))       # = flange when straight down; tips + tool length when tilted
        target[axis] += sign * (dist or float(self.step.get()))
        js = kin.ik_down(tuple(target), float(self.yaw.get()), p.joints)
        if js is None:
            self.say(f"Refused: no gripper-down solution at {np.round(target, 1)}")
            return
        if kin.tilt_deg(p.joints) > config.MAX_TILT_DEG + 1:
            self.say("Press 'Point gripper DOWN here' first")
            return
        self.move(js)

    def save_point(self, k):
        p = self.arm.pose()
        if not kin.tool_axis_down(p.joints, 1.0):
            self.say("Not saved: the gripper must point straight down")
            return
        d = {"x": p.x, "y": p.y, "z": p.z, "joints": p.joints}
        if k in ("tray", "reserve"):
            self.data[k] = d
        else:
            self.data["squares"][k] = d
        self.data["yaw"] = float(self.yaw.get())
        self._write()
        self.say(f"Saved {k}")

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
        p = self.arm.pose()
        for z in (max(p.z, d["z"] + 60), d["z"] + 20):      # over at a safe height, then down
            js = kin.ik_down((d["x"], d["y"], z), float(self.yaw.get()), self.arm.pose().joints)
            if js is None:
                self.say("Refused: unreachable")
                return
            self.move(js)

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

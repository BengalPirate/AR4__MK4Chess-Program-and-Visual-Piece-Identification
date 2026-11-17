"""Manual calibration: zero the joints by eye and record each joint's safe range.

Run:  python manual_calib.py      (close the AR4 HMI and robot_setup.py first)

Replaces limit-switch auto-calibration (which ran J1 into its cables on
2026-11-16). Nothing here drives a joint to a switch.

1. REFERENCE POSE -> ZERO. Jog the arm into the pose the controller calls all
   zeros, then press "Readings = 0 (all joints)". Nothing moves; the controller is
   told the angles (SP command). The zero pose of this arm:
     J1 base pointing straight forward (along the robot's +X, toward the board)
     J2 upper arm vertical                      (phone level on the upper arm)
     J3 forearm horizontal, pointing forward    (phone level on the forearm)
     J4 wrist not rolled (J5's axis horizontal)
     J5 gripper in line with the forearm
     J6 your chosen jaw reference (e.g. jaws opening left/right)
   A single joint can also be re-zeroed with its "= 0" button.
   If a jog is refused before zeroing (the controller still believes old angles
   that are outside its limits), press "Readings = 0" once first, jog to the
   reference pose, then zero again there.
2. LIMITS. For each joint, jog to the end of its SAFE travel (cables slack, nothing
   hits anything, gripper clear of the forearm) and press "min here" / "max here".
3. SAVE -> config/joint_limits.json. From then on every move from our software
   (ar4link.move_joints, kin.within_limits) stays LIMIT_MARGIN_DEG inside these
   ranges. The angles are also written to the HMI config so its startup doesn't
   overwrite them.

Jogs are single-joint joint moves at the chosen speed (default 5 %). Watch the
arm, keep a hand on the E-stop. The FIRST jog of each joint should be a small
one, to check which way + and - turn.
"""
import json
import os
import time
import tkinter as tk
from tkinter import messagebox

import config
from ar4link import AR4, AR4Error

STEPS = (0.5, 1, 2, 5, 10)
LOCKED = {int(j) for j in os.environ.get("AR4_LOCKED_JOINTS", "").split(",") if j}   # joint numbers
J1_MAX_STEP = float(os.environ.get("AR4_J1_MAX_STEP", "10"))
JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
HINT = {"J1": "base: + / - rotation", "J2": "shoulder", "J3": "elbow", "J4": "wrist roll",
        "J5": "wrist pitch (gripper vs forearm!)", "J6": "gripper rotation"}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Team 2 - manual calibration (zero by eye + joint limits)")
        self.arm = AR4(safe=True, j5_window=config.J5_WINDOW)
        self.data = {"limits": {}}
        if os.path.exists(config.JOINT_LIMITS_FILE):
            with open(config.JOINT_LIMITS_FILE) as f:
                self.data = json.load(f)
            self.data.setdefault("limits", {})
        self.step = tk.DoubleVar(value=1)
        self.speed = tk.IntVar(value=5)
        self.status = tk.StringVar(value=f"Connected: {self.arm.port}")
        self.angle_vars = {j: tk.StringVar(value="?") for j in JOINTS}
        self.lim_vars = {j: tk.StringVar() for j in JOINTS}
        self.buttons = []

        top = tk.Frame(root)
        top.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        tk.Label(top, text="Step deg:").pack(side="left")
        for s in STEPS:
            tk.Radiobutton(top, text=str(s), variable=self.step, value=s).pack(side="left")
        tk.Label(top, text="   Speed %:").pack(side="left")
        tk.Spinbox(top, from_=2, to=15, width=4, textvariable=self.speed).pack(side="left")

        g = tk.Frame(root)
        g.grid(row=1, column=0, padx=8)
        for c, h in enumerate(("joint", "", "reading", "jog -", "jog +", "zero", "min", "max", "saved range")):
            tk.Label(g, text=h, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=c, padx=3)
        for r, j in enumerate(JOINTS, 1):
            i = r - 1
            tk.Label(g, text=j, font=("TkDefaultFont", 11, "bold")).grid(row=r, column=0)
            tk.Label(g, text=HINT[j], fg="#555").grid(row=r, column=1, sticky="w")
            tk.Label(g, textvariable=self.angle_vars[j], width=9, font=("monospace", 11)).grid(row=r, column=2)
            self._btn(g, " - ", lambda i=i: self.jog(i, -1), r, 3)
            self._btn(g, " + ", lambda i=i: self.jog(i, +1), r, 4)
            self._btn(g, f"{j} = 0", lambda i=i: self.zero(i), r, 5)
            self._btn(g, "min here", lambda j=j: self.mark(j, 0), r, 6, bg="#455a64")
            self._btn(g, "max here", lambda j=j: self.mark(j, 1), r, 7, bg="#455a64")
            tk.Label(g, textvariable=self.lim_vars[j], width=18, font=("monospace", 10)).grid(row=r, column=8)

        b = tk.Frame(root)
        b.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        self._btn(b, "Readings = 0 (all joints) - arm is in the reference pose", self.zero_all, 0, 0,
                  bg="#2e7d32")
        self._btn(b, "SAVE limits", self.save, 0, 1, bg="#1565c0")
        self._btn(b, "Refresh", self.refresh, 0, 2)
        tk.Label(root, textvariable=self.status, fg="#333", wraplength=900, justify="left")\
            .grid(row=3, column=0, sticky="w", padx=8, pady=6)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.refresh()
        self.show_limits()

    def _btn(self, parent, text, cmd, r, c, bg=None):
        w = tk.Button(parent, text=text, command=cmd, bg=bg, fg="white" if bg else None)
        w.grid(row=r, column=c, sticky="ew", padx=2, pady=2)
        self.buttons.append(w)

    def say(self, t):
        self.status.set(t)
        print(time.strftime("%T"), t, flush=True)

    def busy(self, on):
        for w in self.buttons:
            w.config(state="disabled" if on else "normal")
        self.root.update()

    def refresh(self):
        try:
            p = self.arm.pose()
        except AR4Error as e:
            self.say(f"Position unavailable: {e}")
            return None
        for j, a in zip(JOINTS, p.joints):
            self.angle_vars[j].set(f"{a:8.2f}")
        return p

    def show_limits(self):
        for j in JOINTS:
            lo, hi = self.data["limits"].get(j, [None, None])
            self.lim_vars[j].set(f"{'-' if lo is None else f'{lo:.1f}'} .. {'-' if hi is None else f'{hi:.1f}'}")

    def jog(self, i, sign):
        if i + 1 in LOCKED:
            self.say(f"J{i + 1} jogging is locked (AR4_LOCKED_JOINTS).")
            return
        if i == 0 and float(self.step.get()) > J1_MAX_STEP:
            self.say(f"J1 steps are capped at {J1_MAX_STEP:g} deg until its direction is confirmed")
            return
        p = self.refresh()
        if p is None:
            return
        js = list(p.joints)
        js[i] += sign * float(self.step.get())
        self.busy(True)
        try:
            # enforce_limits=False: we are discovering the limits here
            q = self.arm.move_joints(js, int(self.speed.get()), allow_j5_outside=True, enforce_limits=False)
            self.say(f"J{i + 1} {sign * float(self.step.get()):+g} deg -> {q.joints[i]:.2f}")
        except AR4Error as e:
            self.say(f"Jog refused / stopped: {e}")
        finally:
            self.busy(False)
            self.refresh()

    def zero(self, i):
        if not messagebox.askyesno("Zero", f"Is J{i + 1} exactly at its reference position "
                                   "(see the list in the window title help)? Nothing moves."):
            return
        p = self.arm.set_joint(i + 1, 0.0)
        self.data[f"J{i + 1}_zeroed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.say(f"J{i + 1} reading set to 0. {p}")
        self.refresh()

    def zero_all(self):
        if not messagebox.askyesno(
                "Zero all", "Base forward, upper arm vertical, forearm horizontal, wrist straight, "
                "gripper in line with the forearm? All six readings become 0. Nothing moves."):
            return
        self.arm._cmd("SPA0.000B0.000C0.000D0.000E0.000F0.000G0.000H0.000I0.000")
        self.data["zeroed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.say("All joint readings set to 0 at the reference pose. Now record min / max per joint.")
        self.refresh()

    def mark(self, j, k):
        p = self.refresh()
        if p is None:
            return
        a = p.joints[JOINTS.index(j)]
        lim = self.data["limits"].setdefault(j, [None, None])
        lim[k] = round(a, 2)
        self.show_limits()
        self.say(f"{j} {'min' if k == 0 else 'max'} = {a:.2f} (not saved yet - press SAVE)")

    def save(self):
        bad = [j for j, (lo, hi) in self.data["limits"].items()
               if lo is not None and hi is not None and lo >= hi]
        if bad:
            self.say(f"Not saved: min >= max for {', '.join(bad)}")
            return
        self.data["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data["margin_deg"] = config.LIMIT_MARGIN_DEG
        with open(config.JOINT_LIMITS_FILE, "w") as f:
            json.dump(self.data, f, indent=2)
        try:
            self.arm.sync_app_config()
        except Exception as e:      # noqa: BLE001
            self.say(f"Saved limits; HMI config not updated ({e})")
            return
        self.say(f"Saved {config.JOINT_LIMITS_FILE} and synced the HMI's saved angles.")

    def quit(self):
        try:
            self.arm.sync_app_config()
        except Exception:           # noqa: BLE001
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()

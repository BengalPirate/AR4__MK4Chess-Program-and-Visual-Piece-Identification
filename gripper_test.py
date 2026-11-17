"""Servo gripper test. The arm does not move; only the gripper servo is commanded.

  python gripper_test.py            interactive: type an angle, o / c, a (current), q
  python gripper_test.py --cycle 3  open/close cycles with current readings
  python gripper_test.py --port /dev/ttyUSB0

15 = closed, 45 = fully open (fitted travel; up to 50 allowed for R/Q/K, end stop ~52).
With the foam pads, 20 = pads touching.
Start with small steps, and keep fingers out of the gripper.
"""
import argparse
import time

from gripper import Gripper


def show(g, label):
    time.sleep(0.3)
    print(f"{label:>12}: angle {g.angle:3d} deg   current {g.current():.3f} A")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--open", type=int, default=45)
    ap.add_argument("--closed", type=int, default=15)
    ap.add_argument("--cycle", type=int, default=0)
    a = ap.parse_args()

    g = Gripper(a.port, open_deg=a.open, closed_deg=a.closed)
    print(f"Nano on {g.port}")
    show(g, "power-on")
    try:
        if a.cycle:
            for i in range(a.cycle):
                g.open(); show(g, f"open {i + 1}")
                g.close(); show(g, f"closed {i + 1}")
            g.open(); show(g, "open")
            return
        print("Commands: <angle 15-45>, o = open, c = close, a = read current, q = quit")
        while True:
            s = input("> ").strip().lower()
            if s in ("q", "quit", ""):
                break
            if s == "o":
                g.open(); show(g, "open")
            elif s == "c":
                g.close(); show(g, "closed")
            elif s == "a":
                show(g, "current")
            elif s.isdigit():
                g.set_angle(int(s)); show(g, "moved")
            else:
                print("?")
    finally:
        g.close_port()


if __name__ == "__main__":
    main()

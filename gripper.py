"""Servo gripper driver: Annin AR4 Nano IO board running AR4_nano_sketch_v1.5.

The DS3225 servo gripper's signal wire is on Nano pin A0 (servo 0) and its supply
current is measured by an ACS712 on A7 (AR4 MK4 manual, chapter 5). Commands:
  SV0P<deg>\n  -> "Servo Done"   move servo 0 (15 = closed, 45 = fully open on our gripper)
  TG\n         -> "<amps>"       gripper supply current
The sketch also backs the servo off (opens it by 0.25 deg steps) while the
current is above its limit, so a firm grip on a piece cannot stall the servo.
Opening the port resets the Nano (DTR); on boot it drives servo 0 to 20 deg.

Setup notes (learned 2026-11-14):
- The servo arm must be fitted with the servo at 15 deg and the jaws closed by
  hand (manual ch. 5). Mis-fitted, every position stalls at ~2.2 A.
- A stall's current drop resets the Nano / hangs its USB link (replug needed),
  so set_angle() watches the current and backs off above STALL_AMPS.
- Plug the gripper's 5 V adapter in before the Nano's USB.
- The first connection after opening the port sometimes gets no reply; the
  constructor retries.
"""
import glob
import os
import time

import serial
from serial.tools import list_ports

BAUD = 9600
OPEN_DEG, CLOSED_DEG = 45, 15   # measured: full travel, no stall
MIN_DEG, MAX_DEG = 15, 50       # never command outside the fitted travel (2026-11-16: 51 loads, 53 stalls)
STALL_AMPS = 0.8
OPEN_STALL_AMPS = 1.5            # opening only (see open())
OPEN_GRACE_S = 0.5
STALL_GRACE_S = 0.3              # ignore this long after each step: the servo's start burst is not a stall
# an open, free gripper reads 0.1-0.3 A; a bound one 0.4-0.6. 2026-11-23: after the J5
# fold the gripper read 0.15-0.54 A open and the operator found nothing wrong ->
# 0.6 for now (env AR4_GRIP_IDLE_MAX_AMPS overrides; 0.45 is the original value)
IDLE_MAX_AMPS = float(os.environ.get("AR4_GRIP_IDLE_MAX_AMPS", 0.6))
# USB IDs of common Nano boards: CH340 clone, FTDI, genuine Arduino
NANO_IDS = {(0x1A86, 0x7523), (0x0403, 0x6001), (0x2341, 0x0043), (0x2341, 0x0058),
            (0x2341, 0x0001)}
TEENSY_VID = 0x16C0


def find_port():
    """Return the Nano's serial port (never the Teensy)."""
    cands = [p.device for p in list_ports.comports()
             if (p.vid, p.pid) in NANO_IDS]
    if not cands:   # unknown USB-serial chip: any ttyUSB, or a non-Teensy ttyACM
        cands = sorted(glob.glob("/dev/ttyUSB*")) + [
            p.device for p in list_ports.comports()
            if p.device.startswith("/dev/ttyACM") and p.vid != TEENSY_VID]
    if not cands:
        raise RuntimeError(
            "No Nano found. Plug the USB jack on the robot base into the computer "
            "and pass the device through to the VM (Parallels: Devices > USB). "
            "On Ubuntu, if a CH340 Nano appears in lsusb but no /dev/ttyUSB0 shows "
            "up, brltty has claimed it: sudo apt remove brltty")
    if len(cands) > 1:
        raise RuntimeError(f"Several candidate ports, pass one explicitly: {cands}")
    return cands[0]


class GripperError(RuntimeError):
    pass


class Gripper:
    def __init__(self, port=None, open_deg=OPEN_DEG, closed_deg=CLOSED_DEG, tries=3):
        self.port = port or find_port()
        self.open_deg, self.closed_deg = open_deg, closed_deg
        for attempt in range(tries):
            self.ser = serial.Serial(self.port, BAUD, timeout=0.1)
            time.sleep(6.0)             # Nano reboots when the port opens
            self.ser.reset_input_buffer()
            try:
                self.current()
                break
            except GripperError:
                self.ser.close()
                if attempt == tries - 1:
                    raise GripperError("Nano not answering. Unplug the 5 V, replug the "
                                       "Nano USB, then plug the 5 V back in.")
                time.sleep(2)
        self.angle = 20                 # sketch's power-on position

    def close_port(self):
        self.ser.close()

    def _cmd(self, cmd, expect_line=True, timeout=2.0):
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\n").encode())
        buf = b""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            buf += self.ser.read(self.ser.in_waiting or 1)
            text = buf.decode(errors="replace")
            if (expect_line and text.endswith("\n")) or (not expect_line and "Done" in text):
                break
        reply = buf.decode(errors="replace").strip()
        if not reply:
            raise GripperError(f"No reply from the Nano to {cmd!r} - wrong port, sketch "
                               "not loaded, or board unpowered")
        return reply

    def set_angle(self, deg, step=5, settle=0.5, stall_amps=None, grace_s=None):
        """Move servo 0 in `step`-degree increments, watching the current.

        A short spike while moving is normal; a current above STALL_AMPS on three
        readings in a row (~0.5 s) is a stall: the servo backs off to the last good
        angle before its current drop can reset the Nano."""
        deg = int(round(deg))
        if not MIN_DEG <= deg <= MAX_DEG:
            raise ValueError(f"gripper angle must be {MIN_DEG}-{MAX_DEG} deg")
        while self.angle != deg:
            prev = self.angle
            nxt = prev + max(-step, min(step, deg - prev))
            reply = self._cmd(f"SV0P{nxt}", expect_line=False)
            if "Servo Done" not in reply:
                raise GripperError(f"Unexpected reply to servo command: {reply!r}")
            self.angle = nxt
            high = 0
            start = time.monotonic()
            end = start + settle              # no position feedback: wait, watching current
            while time.monotonic() < end or high:
                amps = self.current()
                if time.monotonic() - start < (STALL_GRACE_S if grace_s is None else grace_s):
                    continue                  # start-of-move burst (up to ~1.6 A, 2026-11-21)
                high = high + 1 if amps > (STALL_AMPS if stall_amps is None else stall_amps) else 0
                if high >= 3:
                    self._cmd(f"SV0P{prev}", expect_line=False)
                    self.angle = prev
                    raise GripperError(f"Stall at {nxt} deg (> {STALL_AMPS} A); backed off "
                                       f"to {prev} deg. Check the jaws for an obstruction.")
        return "Servo Done"

    def grip(self, target_deg, contact_amps=0.35, step=2, squeeze=0):
        """Close toward target_deg in small steps until the jaws meet a piece.

        Contact shows as a current rise above contact_amps (a free servo draws
        ~0.1 A). On contact the jaws stay at that angle (holding the piece) and
        True is returned. Reaching target_deg with no load means nothing is
        between the jaws: returns False (caller stops for recovery).
        After contact the jaws close `squeeze` more degrees, 1 deg at a time (grip
        tightness; foam pads compress), still easing off on a stall.
        self.trace lists (angle, amps) for every step."""
        target_deg = max(MIN_DEG, int(round(target_deg)))
        self.trace = []
        base = self.idle_amps()
        if base > IDLE_MAX_AMPS:
            raise GripperError(f"Gripper already loaded before closing ({base:.2f} A) - not open/jammed?")

        def go(nxt):
            self._cmd(f"SV0P{nxt}", expect_line=False)
            self.angle = nxt
            time.sleep(0.25)
            amps = max(self.current() for _ in range(2))
            self.trace.append((nxt, round(amps, 2)))
            return amps

        while self.angle > target_deg:
            nxt = max(target_deg, self.angle - step)
            amps = go(nxt)
            if amps > STALL_AMPS:          # squeezing hard: ease off one step, hold there
                go(nxt + step)
                return True
            if amps > contact_amps:
                for _ in range(int(squeeze)):
                    if self.angle - 1 < MIN_DEG:
                        break
                    prev = self.angle
                    if go(self.angle - 1) > STALL_AMPS:
                        go(prev)
                        break
                return True
        return False

    def hold(self, target_deg, step=2):
        """Position grip: close to target_deg (a per-piece angle a few degrees past
        contact, so the foam pads squeeze the piece). The current is only a jam
        alarm: on a stall, ease off one step and hold there. Used instead of
        grip(): with the foam pads the current cannot tell 'holding a piece' from
        'empty' reliably (idle 0.1-0.4 A). self.trace lists (angle, amps)."""
        target_deg = max(MIN_DEG, int(round(target_deg)))
        self.trace = []
        while self.angle > target_deg:
            nxt = max(target_deg, self.angle - step)
            self._cmd(f"SV0P{nxt}", expect_line=False)
            self.angle = nxt
            time.sleep(0.25)
            amps = max(self.current() for _ in range(2))
            self.trace.append((nxt, round(amps, 2)))
            if amps > STALL_AMPS:
                self._cmd(f"SV0P{nxt + step}", expect_line=False)
                self.angle = nxt + step
                break
        return True

    def idle_amps(self, n=6):
        return sorted(self.current() for _ in range(n))[n // 2]      # median

    def open(self, deg=None, tries=4):
        """Open, then check the servo is idle (jaws free). After a Nano restart the
        servo snaps to 20 deg and binds on the foam pads; a bound gripper must never
        look 'open' (2026-11-16: jaws stayed shut through a whole run). Small steps
        out of the squeezed range, retries, then a hard error."""
        deg = self.open_deg if deg is None else deg
        last = None
        for attempt in range(tries):
            try:
                if self.angle < 30:
                    # squeezed on the pads (e.g. Nano power-on at 20): the servo draws
                    # a lot until released - jump straight out, no stall check needed
                    # when opening
                    self._cmd(f"SV0P{min(30, deg)}", expect_line=False)
                    self.angle = min(30, deg)
                    time.sleep(0.8)
                # 2026-11-22: opening from pads pressed together draws long bursts that
                # tripped the 0.8 A stall check. Opening cannot crush anything, and a real
                # jam is caught by the idle check below -> looser limits while opening.
                lim = dict(stall_amps=OPEN_STALL_AMPS, grace_s=OPEN_GRACE_S)
                if self.angle > deg:
                    # coming back from a wider opening (49 for R/Q/K) the jaws stay
                    # loaded at deg (0.46 A); approached from below they are free
                    self.set_angle(max(MIN_DEG, deg - 6), **lim)
                # 2026-11-21: above ~42 deg the jaws meet light resistance (0.1-0.24 A);
                # 5-deg jumps into it spiked past STALL_AMPS -> false "jammed". Open in
                # 2-deg steps over the last part.
                if deg > 40 and self.angle < 40:
                    self.set_angle(40, **lim)
                self.set_angle(deg, step=2 if deg > 40 else 5, **lim)
            except GripperError as e:
                last = e
                time.sleep(1.0)
                continue
            for _ in range(5):              # let it settle (up to ~2 s); a jam stays high
                time.sleep(0.4)
                amps = self.idle_amps()
                if amps <= IDLE_MAX_AMPS:
                    return "Servo Done"
            last = GripperError(f"Gripper not idle after opening to {deg} deg ({amps:.2f} A)")
            self._cmd(f"SV0P{max(MIN_DEG, deg - 6)}", expect_line=False)   # wiggle and retry
            self.angle = max(MIN_DEG, deg - 6)
            time.sleep(0.5)
        raise GripperError(f"{last} - jaws may be jammed; check the gripper before continuing")

    def close(self, deg=None):
        return self.set_angle(self.closed_deg if deg is None else deg)

    def current(self):
        """Gripper supply current in amps (ACS712 on A7)."""
        reply = self._cmd("TG")
        try:
            return float(reply)
        except ValueError:
            raise GripperError(f"Bad current reply: {reply!r}")

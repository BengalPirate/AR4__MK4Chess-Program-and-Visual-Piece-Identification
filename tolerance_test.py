"""Grip tolerance test: how far off centre can a piece stand and still be picked?

  python tolerance_test.py h5 R --offset "a 15"      piece on h5 placed 15 mm toward a
  python tolerance_test.py e4 P --offset "robot 20"  ... 20 mm toward the robot
  python tolerance_test.py --note 7 "missed, pushed 5 mm toward a"   add what the operator saw

Directions: a / h (along the ranks, Y) and robot / away (along the files, X).
The arm does a REAL pick exactly as in a game (tilted on rank 8 or when given
--tilt, both camera checks, NO retries): lift to travel height, then
  picked  -> set back down on the square, centred
  missed  -> released at the piece's height (never dropped), arm lifts away
Every test is a row in results/grip_tolerance.csv: piece, square, offset, grip
current, the gripper-in-air camera differences and the square check, plus the
operator's note. Ends at PARK (or, with --stay, hovering above the square).
"""
import argparse
import csv
import os
import time


import config
from motion import BoardLock, Executor, MotionError
from planner import BoardRobot, Op

LOG = os.path.join(config.RESULTS_DIR, "grip_tolerance.csv")
FIELDS = ["n", "time", "image", "square", "piece", "tilt", "offset_dir", "offset_mm", "max_amps", "close_deg",
          "jaws", "diff_front", "diff_side", "square_seen", "result", "operator_note"]


class Recorder:
    """Wraps the PickChecker: same answers, but remembers them for the log."""

    def __init__(self, checker):
        self.c, self.last = checker, {}

    def snapshot(self):
        return self.c.snapshot()

    def held(self, *a, **k):
        h, d = self.c.held(*a, **k)
        self.last.update(jaws=h, diffs=d)
        return h, d

    def __call__(self, sq):
        s = self.c(sq)
        self.last["seen"] = s
        return s


def rows():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return list(csv.DictReader(f))


def write(all_rows):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("square", nargs="?")
    ap.add_argument("piece", nargs="?", choices=list("PNBRQK"))
    ap.add_argument("--offset", default="none 0", help='"a 15", "h 20", "robot 25", "away 15", "none 0"')
    ap.add_argument("--tilt", type=float, help="force a tilted pick (deg) off rank 8")
    ap.add_argument("--stay", action="store_true",
                    help="stay at travel height above the square afterwards (next test starts there) instead of PARK")
    ap.add_argument("--note", nargs=2, metavar=("N", "TEXT"), help="add the operator's observation to test N")
    a = ap.parse_args()
    if a.note:
        rs = rows()
        for r in rs:
            if r["n"] == a.note[0]:
                r["operator_note"] = a.note[1]
        write(rs)
        print(f"note added to test {a.note[0]}")
        return
    d, mm = a.offset.split()
    assert d in ("a", "h", "robot", "away", "none"), d
    from ar4link import AR4
    from grip_check import open_checker
    from gripper import Gripper
    rec = Recorder(open_checker())
    # labelled training image of the piece at this known offset, before the arm touches it
    import offset_capture
    shot = offset_capture.save({v.key: v.settled_frame(10) for v in rec.c.views}, a.square, a.piece, d, mm,
                               "tolerance_test.py (arm may be hovering above)")
    br = BoardRobot()
    saved = config.GRIP_RETRIES
    config.GRIP_RETRIES = 0                          # measure the first grip only
    ex = Executor(AR4(safe=True, j5_window=config.J5_WINDOW), Gripper(), br, BoardLock(), pick_check=rec)
    pick = br.locate(Op("pick", a.square, a.piece))
    if a.tilt is not None:
        pick.tilt = a.tilt
    place = br.locate(Op("place", a.square, a.piece))
    result = "picked"
    try:
        if not ex.lock.acquire():
            raise SystemExit("board interlock held")
        ex.preflight([pick, place])
        try:
            ex.pick(pick)
        except MotionError as e:
            result = "missed" if "missed" in str(e) else "stopped: " + str(e)[:80]
            print("  ", e)
        if result == "picked":
            ex.place(place)
        ex.travel()
        if not a.stay:
            ex.go_park()
    finally:
        config.GRIP_RETRIES = saved
        ex.lock.release()
    trace = getattr(ex.grip, "trace", []) or []
    rs = rows()
    row = {"n": str(len(rs) + 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "image": shot,
           "square": a.square,
           "piece": a.piece, "tilt": f"{pick.tilt if pick.tilt is not None else ex.tilt_for(a.square):g}",
           "offset_dir": d, "offset_mm": mm,
           "max_amps": f"{max((t[1] for t in trace), default=0):.2f}",
           "close_deg": f"{trace[-1][0] if trace else ''}",
           "jaws": {True: "held", False: "empty", None: "undecided"}[rec.last.get("jaws")],
           "diff_front": f"{rec.last.get('diffs', {}).get('front', ''):}",
           "diff_side": f"{rec.last.get('diffs', {}).get('side', ''):}",
           "square_seen": rec.last.get("seen") or "undecided", "result": result, "operator_note": ""}
    rs.append(row)
    write(rs)
    print(f"test {row['n']}: {a.piece} on {a.square} {mm} mm toward {d}: {result}; jaws {row['jaws']} "
          f"(front {row['diff_front']}, side {row['diff_side']}); square {row['square_seen']}; "
          f"grip current up to {row['max_amps']} A")
    os._exit(0)          # camera threads block in read(); leave without tearing them down


if __name__ == "__main__":
    main()

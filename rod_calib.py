"""Where does the arm REALLY point? The cameras watch a blue rod in the gripper.

A blue rod is clamped in the closed jaws, sticking ROD_MM below the jaw tips. The
arm hovers so the rod tip should be HOVER_MM above a square's centre (planning
coordinates, exactly as a game aims there). Both calibrated cameras find the rod
tip (the blue pixel farthest along the rod, near where the tip should appear) and
the two viewing rays give its real 3D position over the board. Real minus square
centre = the arm's aim error at that square, in the same approach direction a
game uses (joint slack included).

  python rod_calib.py --detect                 no motion: find the tip with the arm where it is
  python rod_calib.py e4 --hover 40            hover over e4 (rod tip 40 mm up), measure, park
  python rod_calib.py a8 b8 ... h8 --hover 15  one rank; results appended to results/rod_calib.csv

Heights: the rod's length is ADDED to every height (11-17 lesson: the tool
length alone is not enough); the rod tip never goes below MIN_HOVER_MM. Every
run is simulated first and refused unless the simulated rod tip stays at or
above the hover height (--simulate: only that).
"""
import argparse
import csv
import json
import os
import sys
import time

import chess
import cv2
import numpy as np

import config
import kin

ROD_MM = 80.0
MIN_HOVER_MM = 15.0
MAX_RAY_MM = 2.0          # the two viewing rays must pass this close, else the reading is rejected
LOG = os.path.join(config.RESULTS_DIR, "rod_calib.csv")
OUT_IMG = os.path.join(config.RESULTS_DIR, "rod_calib_last.jpg")
LEAN_FILE = os.path.join(config.RESULTS_DIR, "rod_lean.json")   # rod tip offset from the jaw axis, per tilt
# blue in OpenCV HSV (H 0-180)
H_LO, H_HI, S_MIN, V_MIN = 95, 135, 70, 30


def robot_to_board(br, p):
    """Robot mm (x, y, z) of a point -> board coords (squares: x along files from the
    a-file edge, y along ranks from the rank-1 edge, z up from the board)."""
    u, w = br.board_uv(p[0], p[1])
    h = (p[2] - (br.board_z(p[0], p[1]) - config.TOOL_LENGTH_MM)) / config.SQUARE_MM
    return np.array([u + 0.5, w + 0.5, h])


def _tool_rot(js):
    """Rotation (robot frame) of the gripper: columns = tool x, y, z axes."""
    f0 = np.array(kin.fk(js)[:3])
    return np.stack([kin._axis_point(js, e) - f0 for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], 1)


def _board_axes(br):
    """Robot-frame unit vectors of board +x (toward h), +y (toward rank 8), +z (up)."""
    ex = br.file_step[:2] / np.linalg.norm(br.file_step[:2])
    ey = br.rank_step[:2] / np.linalg.norm(br.rank_step[:2])
    return np.array([[ex[0], ex[1], 0.0], [ey[0], ey[1], 0.0], [0.0, 0.0, 1.0]]).T


def board_to_tool(br, js, v_board):
    return _tool_rot(js).T @ (_board_axes(br) @ np.asarray(v_board, float))


def tool_to_board(br, js, v_tool):
    return np.linalg.solve(_board_axes(br), _tool_rot(js) @ np.asarray(v_tool, float))


def find_tip(frame, geo, tip_b, up_b):
    """Image position of the rod tip near its predicted place, or (None, why).
    tip_b / up_b: board coords of the predicted tip and of a point 60 mm up the rod."""
    p0, p1 = geo.project([tip_b, up_b])
    d = p0 - p1
    L = np.linalg.norm(d)
    if L < 3:
        return None, "rod too short in this view"
    d = d / L
    n = np.array([-d[1], d[0]])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (H_LO, S_MIN, V_MIN), (H_HI, 255, 255))
    # search window: a band along the rod around the predicted tip (blue floor-mat
    # edges elsewhere are ignored)
    h, w = mask.shape
    ys, xs = np.mgrid[0:h, 0:w]
    rel = np.stack([xs - p0[0], ys - p0[1]], -1)
    along = rel @ d                    # + beyond the predicted tip, - up the rod
    across = np.abs(rel @ n)
    win = (along > -max(L, 40) - 40) & (along < 60) & (across < 45)
    m = (mask > 0) & win
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if k < 2:
        return None, "no blue near the predicted tip"
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < 15:
        return None, "blue area too small"
    yy, xx = np.nonzero(lab == big)
    a = (np.stack([xx - p0[0], yy - p0[1]], -1) @ d)
    i = np.argsort(a)[-5:]             # the 5 pixels farthest down the rod
    return np.array([xx[i].mean(), yy[i].mean()]), ""


def damaged(f):
    """Frames broken on the shared USB link: a garbled band (sudden row-to-row brightness
    jump) or a strong colour cast over part of the image (blue bands fool the tip finder)."""
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)
    if np.abs(np.diff(g.mean(axis=1))).max() >= 15.0:
        return True
    b, gch, r = (f[..., i].astype(float).mean(axis=1) for i in range(3))
    cast = b - (gch + r) / 2                       # per row: how much bluer than grey
    return float(np.percentile(cast, 90)) > 40.0


def clean_frame(v, tries=5):
    f = v.settled_frame(8)
    for _ in range(tries):
        if not damaged(f):
            return f
        print(f"   {v.key}: damaged frame, taking another", flush=True)
        f = v.settled_frame(4)
    return f


def ray(geo, px):
    """Viewing ray (origin, unit direction) in board coords for an image point."""
    R, _ = cv2.Rodrigues(geo.rvec)
    t = np.asarray(geo.tvec, float).reshape(3)
    nrm = cv2.undistortPoints(np.float32([[px]]), geo.K, geo.dist).reshape(2)
    dcam = np.array([nrm[0], nrm[1], 1.0])
    o = -R.T @ t
    dw = R.T @ dcam
    s = np.array([1.0, 1.0, geo.z_sign])
    o, dw = o * s, dw * s
    return o, dw / np.linalg.norm(dw)


def triangulate(rays):
    """Closest point to all rays (least squares) and the rms ray distance (squares)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    x = np.linalg.solve(A, b)
    dist = [np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - o)) for o, d in rays]
    return x, float(np.sqrt(np.mean(np.square(dist))))


def measure(views, br, js, rod_mm=ROD_MM, img_path=OUT_IMG):
    """Rod tip: predicted (from the joints) and measured (cameras), board coords."""
    tip_r = kin.tip(js, config.TOOL_LENGTH_MM + rod_mm)
    up_r = kin.tip(js, config.TOOL_LENGTH_MM + rod_mm - 60.0)
    tip_b, up_b = robot_to_board(br, tip_r), robot_to_board(br, up_r)
    rays, found, tiles = [], {}, []
    for v in views:
        f = clean_frame(v)
        if img_path:
            cv2.imwrite(img_path.replace(".jpg", f"_raw_{v.key}.jpg"), f)
        px, why = find_tip(f, v.geo, tip_b, up_b)
        img = f.copy()
        pp = v.geo.project([tip_b])[0].astype(int)
        cv2.drawMarker(img, tuple(pp), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        if px is None:
            found[v.key] = why
        else:
            found[v.key] = "ok"
            rays.append(ray(v.geo, px))
            cv2.circle(img, tuple(px.astype(int)), 7, (0, 0, 255), 2)
        x0, y0 = max(0, pp[0] - 150), max(0, pp[1] - 200)
        crop = img[y0:y0 + 300, x0:x0 + 300]
        crop = cv2.resize(crop, (450, 450))
        cv2.putText(crop, f"{v.key}: yellow=predicted, red=found", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        tiles.append(crop)
    if tiles and img_path:
        cv2.imwrite(img_path, np.hstack(tiles))
    res = {"pred": tip_b, "found": found}
    if len(rays) >= 2:
        x, rms = triangulate(rays)
        if rms * config.SQUARE_MM <= MAX_RAY_MM:
            res.update(meas=x, rms_mm=rms * config.SQUARE_MM)
        else:
            res["found"]["rays"] = f"miss each other by {rms * config.SQUARE_MM:.1f} mm (a camera found the wrong spot)"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("squares", nargs="*")
    ap.add_argument("--detect", action="store_true", help="no motion: measure with the arm where it is")
    ap.add_argument("--hover", type=float, default=40.0, help="rod tip height above the board (mm)")
    ap.add_argument("--rod", type=float, default=ROD_MM, help="rod length below the jaw tips (mm)")
    ap.add_argument("--pause", type=float, default=1.5, help="settle time at each square (s)")
    ap.add_argument("--simulate", action="store_true", help="simulate only (no motion)")
    ap.add_argument("--close", action="store_true",
                    help="operator check: allow the rod tip down to 3 mm (give the MEASURED --rod length)")
    ap.add_argument("--lean", action="store_true",
                    help="at each square also turn J6 +90/-90 deg: measures how far the rod tip sits off the "
                         "jaw axis (saved, then subtracted from every later reading at that tilt)")
    a = ap.parse_args()
    from cameras import open_views
    from planner import BoardRobot
    from ar4link import AR4
    br = BoardRobot()
    arm = AR4(safe=True, j5_window=config.J5_WINDOW)
    views = open_views()
    for v in views:
        v.settled_frame(10)
    mm = config.SQUARE_MM
    if a.detect or not a.squares:
        js = list(arm.pose().joints)
        r = measure(views, br, js, a.rod)
        print(f"cameras: {r['found']}")
        print(f"predicted rod tip (board): {np.round(r['pred'], 3)} squares")
        if "meas" in r:
            d = (r["meas"] - r["pred"]) * mm
            print(f"measured rod tip (board):  {np.round(r['meas'], 3)} squares; rays meet within {r['rms_mm']:.1f} mm")
            print(f"measured - predicted: {d[0]:+.1f} mm toward h, {d[1]:+.1f} mm toward rank 8, {d[2]:+.1f} mm up")
        print(f"picture: {OUT_IMG}", flush=True)
        sys.stdout.flush()
        os._exit(0)
    lowest = 3.0 if a.close else MIN_HOVER_MM
    if a.hover < lowest:
        raise SystemExit(f"--hover below {lowest:g} mm refused")
    # ---- motion: same code as a game, rod length added to EVERY height
    from motion import BoardLock, DryArm, DryGripper, Executor
    tool0 = config.TOOL_LENGTH_MM
    config.TOOL_LENGTH_MM += a.rod
    config.TRAVEL_CLEARANCE += a.rod
    # the board must be EMPTY for this: travel only has to clear the board itself (rod
    # tip ~55 mm up), else travel height would exceed PARK and flip the wrist near the base
    for k in config.PIECE_HEIGHT:
        config.PIECE_HEIGHT[k] = 40
    # 1. SIMULATE the whole run; refuse unless the rod tip never goes below the hover height
    sim = Executor(DryArm(list(arm.pose().joints)), DryGripper(), br, BoardLock(config.LOCK_FILE + ".dry"),
                   log=lambda *x: None)
    low = []
    for name in a.squares:
        sq = chess.parse_square(name)
        x, y, zb = br.square_xyz(sq)
        sim.travel()
        sim.min_tilt = sim.tilt_for(name)
        sim.yaw = sim.yaw_for("P")
        sim._approach((x, y, zb + config.travel_z()))
        sim._vertical(x, y, zb + config.travel_z(), zb + a.rod + a.hover)
        sim._vertical(x, y, zb + a.rod + a.hover, zb + config.travel_z())
    sim.go_park()
    prev = None
    for _, js in sim.arm.log:              # endpoints AND the joint-space path between them
        for s_ in ([1.0] if prev is None else np.linspace(0.1, 1.0, 10)):
            q = js if prev is None else list(np.array(prev) + (np.array(js) - np.array(prev)) * s_)
            t = kin.tip(q, tool0 + a.rod)
            low.append(t[2] - (br.board_z(t[0], t[1]) - tool0))
        prev = js
    print(f"simulated: rod tip never lower than {min(low):.1f} mm above the board "
          f"(asked {a.hover:g}); {len(sim.arm.log)} joint moves", flush=True)
    if min(low) < a.hover - 3 or min(low) < lowest - 2:
        raise SystemExit("REFUSED: the simulated rod tip would go below the hover height - nothing moved")
    if "--simulate" in sys.argv:
        sys.stdout.flush()
        os._exit(0)
    ex = Executor(arm, DryGripper(), br, BoardLock())
    if not ex.lock.acquire():
        raise SystemExit("board interlock held")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    new = not os.path.exists(LOG)
    rows = []
    try:
        for name in a.squares:
            sq = chess.parse_square(name)
            x, y, zb = br.square_xyz(sq)
            ex.travel()
            ex.min_tilt = ex.tilt_for(name)
            ex.yaw = ex.yaw_for("P")
            if ex.min_tilt > 0 and ex.tilt_for(name) == 0:
                dx, dy, dz = br.tilt_correction(x, y)
                x, y, zb = x + dx, y + dy, zb + dz
            t_empty = config.travel_z()
            # planning z is the FLANGE reference: zb = JAW TIPS on the board. The rod
            # sticks a.rod below the tips, so the tip is at a.hover only at
            # zb + a.rod + a.hover (11-17: forgetting the a.rod drove a pointer into the board)
            ex._approach((x, y, zb + t_empty))
            ex._vertical(x, y, zb + t_empty, zb + a.rod + a.hover)
            time.sleep(a.pause)
            config.TOOL_LENGTH_MM -= a.rod             # measure() adds the rod itself
            js0 = ex._joints()
            r = measure(views, br, js0, a.rod, os.path.join(config.RESULTS_DIR, f"rod_calib_{name}.jpg"))
            if a.lean and "meas" in r:
                turned = []
                for dj in (90.0, -90.0):
                    q = list(js0)
                    q[5] += dj
                    if not kin.within_limits(q):
                        print(f"   J6 {q[5]:.0f} outside its limits - lean not measured", flush=True)
                        break
                    arm.move_joints(q, config.SPEED_PCT)
                    time.sleep(a.pause)
                    rr = measure(views, br, q, a.rod, None)
                    if "meas" not in rr:
                        print(f"   J6 {dj:+.0f}: tip not found {rr['found']}", flush=True)
                        break
                    turned.append(rr["meas"])
                arm.move_joints(list(js0), config.SPEED_PCT)
                if len(turned) == 2:
                    axis = (turned[0] + turned[1]) / 2
                    lean_b = (r["meas"] - axis) * mm                  # board frame, mm
                    lean_t = board_to_tool(br, js0, lean_b)           # gripper frame: valid at any pose
                    json.dump({"tool_frame_mm": [float(v) for v in lean_t], "square": name,
                               "board_frame_mm_there": [float(v) for v in lean_b],
                               "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(LEAN_FILE, "w"), indent=1)
                    print(f"   rod lean: tip sits {lean_b[0]:+.1f} mm toward h, {lean_b[1]:+.1f} mm toward "
                          f"rank 8 of the jaw axis here (saved in the gripper frame)", flush=True)
            if "meas" in r and os.path.exists(LEAN_FILE):   # where the JAWS point, not the leaning rod tip
                lean_t = np.array(json.load(open(LEAN_FILE))["tool_frame_mm"])
                r["meas"] = r["meas"] - tool_to_board(br, js0, lean_t) / mm
                r["lean_corrected"] = True
            config.TOOL_LENGTH_MM += a.rod
            centre = np.array([chess.square_file(sq) + 0.5, chess.square_rank(sq) + 0.5])
            if "meas" in r:
                e = (r["meas"][:2] - centre) * mm
                ep = (r["pred"][:2] - centre) * mm
                print(f"{name}: tilt {kin.tilt_deg(ex._joints()):.0f}; rod tip is {e[0]:+.1f} mm toward h, "
                      f"{e[1]:+.1f} mm toward rank 8 of the centre, {r['meas'][2] * mm:.0f} mm up "
                      f"(model said {ep[0]:+.1f}, {ep[1]:+.1f}); rays within {r['rms_mm']:.1f} mm"
                      + ("" if r.get("lean_corrected") else "  [rod lean NOT yet measured at this tilt]"), flush=True)
                rows.append([time.strftime("%Y-%m-%d %H:%M:%S"), name, f"{kin.tilt_deg(ex._joints()):.0f}",
                             f"{a.hover:g}", f"{e[0]:.1f}", f"{e[1]:.1f}", f"{r['meas'][2] * mm:.1f}",
                             f"{r['rms_mm']:.1f}", ""])
            else:
                print(f"{name}: NOT MEASURED {r['found']}", flush=True)
                rows.append([time.strftime("%Y-%m-%d %H:%M:%S"), name, "", f"{a.hover:g}", "", "", "", "",
                             str(r["found"])])
            ex._vertical(x, y, zb + a.rod + a.hover, zb + t_empty)
        ex.go_park()
    finally:
        ex.lock.release()
        with open(LOG, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "square", "tilt", "hover_mm", "err_toward_h_mm", "err_toward_rank8_mm",
                            "tip_height_mm", "ray_rms_mm", "note"])
            w.writerows(rows)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

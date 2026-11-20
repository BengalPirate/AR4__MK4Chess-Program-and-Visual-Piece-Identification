"""Recovery after a stop with a piece still in the jaws, arm hovering over its square.

  python recover_setdown.py c7 P      lower straight down onto c7 (keeping the tilt), THEN connect
                                      the gripper (the Nano reboots and the jaws can go limp), open,
                                      lift, park. Refuses if the arm is more than 15 mm from the hold point.
"""
import os, sys, numpy as np
sys.path.insert(0, os.getcwd())
import config, kin
from ar4link import AR4
from motion import BoardLock, Executor, DryGripper
from planner import BoardRobot, Op
SQ, PC = sys.argv[1], sys.argv[2].upper()
br = BoardRobot()
arm = AR4(safe=True, j5_window=config.J5_WINDOW)
ex = Executor(arm, DryGripper(), br, BoardLock())          # gripper NOT connected yet (Nano reboot)
if not ex.lock.acquire():
    raise SystemExit("interlock held by another program")
js = ex._joints()
ex.carry_tilt = kin.tilt_deg(js)
op = br.locate(Op("place", SQ, PC))
ex.min_tilt = max(ex.tilt_for(SQ), ex.carry_tilt)
x, y, zb = ex._target(op, ex.min_tilt)
ex.yaw = ex.yaw_for(PC)
t_full, t_empty = config.travel_z(PC), config.travel_z()
here = np.array(kin.vfk(js))
target = np.array([x, y, zb + t_full])
d = np.linalg.norm(here - target)
print(f"arm at {np.round(here,1)}, tilt {ex.carry_tilt:.0f}; hold point {np.round(target,1)}; {d:.1f} mm apart", flush=True)
if d > 15:
    raise SystemExit("NOT above the square as expected - nothing moved")
ex._goto(tuple(target))
z_rel = zb + config.GRASP_Z[PC] + config.PLACE_EXTRA_Z
ex._vertical(x, y, zb + t_full, z_rel)
print("piece set down (gripper still closed); connecting to the gripper ...", flush=True)
from gripper import Gripper
g = Gripper()
g.open(config.GRIP_OPEN_PIECE[PC])
print("gripper open; lifting and parking", flush=True)
ex.grip = g
ex._vertical(x, y, z_rel, zb + t_empty)
ex.go_park()
ex.lock.release()
print("done - at PARK", flush=True)
sys.stdout.flush(); os._exit(0)

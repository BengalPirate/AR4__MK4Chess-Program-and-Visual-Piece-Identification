"""All board cameras (config.CAMERAS): open, grab frames, calibrate, show.

Each attached camera is a View: a capture thread that always holds the newest
frame, plus that camera's BoardGeometry (its own calibration file and empty-board
reference). Programs use open_views() and treat every view the same way;
a camera that is not plugged in is skipped with a warning.
"""
import threading
import time

import cv2
import numpy as np

import config
from board_vision import BoardGeometry, auto_calibrate, detect_markers, find_camera, open_camera


class View:
    def __init__(self, key, device=None, start=True):
        self.key = key
        self.cfg = config.CAMERAS[key]
        self.cap, self.device = open_camera(device or find_camera(self.cfg["name"]))
        self.geo = BoardGeometry.load(self.cfg["file"])
        if self.geo is not None:
            self.geo.load_reference()
        self.frame, self.n = None, 0
        self.lock = threading.Lock()
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.n = f, self.n + 1
            else:
                time.sleep(0.01)

    def get(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.n

    def frames(self, k, timeout=5.0):
        """k distinct new frames."""
        out, last = [], self.n
        t_end = time.time() + timeout
        while len(out) < k and time.time() < t_end:
            f, n = self.get()
            if f is not None and n != last:
                out.append(f)
                last = n
            time.sleep(0.02)
        return out

    def settled_frame(self, n=30, timeout=6.0):
        """Sharpest of the next n frames (lets exposure / autofocus settle)."""
        best, frame = -1.0, None
        for f in self.frames(n, timeout):
            s = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if s > best:
                best, frame = s, f
        return frame

    def calibrate(self, frame=None):
        """Auto-calibrate this camera on the EMPTY board; saves its file and the
        empty reference. Returns the new geometry or None if the tiles were not found."""
        frame = self.settled_frame() if frame is None else frame
        found = auto_calibrate(frame, self.cfg["near"])
        if found is None:
            return None
        geo = BoardGeometry(found[0], (frame.shape[1], frame.shape[0]), detect_markers(frame), found[1])
        geo.save(self.cfg["file"])
        geo.set_reference(frame)
        self.geo = geo
        return geo

    def check(self, frame):
        """(ok, reason): calibrated and not moved since calibration."""
        if self.geo is None:
            return False, f"{self.key} camera not calibrated - empty the board and press K"
        al = self.geo.tile_alignment(frame)
        if al is not None and al < config.MIN_TILE_ALIGNMENT:
            return False, f"{self.key} camera or board moved (tile alignment {al:.2f}) - recalibrate (K)"
        mv = self.geo.camera_moved(frame)
        if mv is not None and mv > config.CAMERA_MOVED_PX:
            return False, f"{self.key} camera moved ({mv:.0f} px at the markers) - recalibrate (K)"
        return True, ""

    def release(self):
        self.cap.release()


def open_views(keys=None, start=True):
    """Every configured camera that is attached, in config order."""
    views = []
    for key in keys or config.CAMERAS:
        try:
            views.append(View(key, start=start))
            print(f"camera {key}: {views[-1].device}"
                  + ("" if views[-1].geo else "  (not calibrated)"))
        except Exception as e:      # noqa: BLE001
            print(f"camera {key}: NOT AVAILABLE ({e})")
    if not views:
        raise RuntimeError("No board camera found")
    return views


def tile(images, labels=None, width=None):
    """Side by side, same height; each optionally captioned."""
    h = min(i.shape[0] for i in images)
    out = []
    for k, img in enumerate(images):
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        if labels:
            cv2.putText(img, labels[k], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, labels[k], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        out.append(img)
    img = np.hstack(out)
    if width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    return img

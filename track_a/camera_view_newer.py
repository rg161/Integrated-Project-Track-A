"""
Onboard camera view: what the robot's camera sees, with the sensing result drawn on top.

Why it is here (41014 Sensors and Control):
  The brief asks for a camera-based pose measurement that feeds the EKF. This
  module shows that measurement chain on screen and in the demo video:
    rendered RGB -> ArUco corners -> PnP pose -> EKF correction accepted/rejected.

Rule: this file only uses what the robot itself would know (the image, the
ArUco result, the EKF estimate, the controller mode). It never reads simulator
ground truth, so it does not break the "controller must not use ground truth"
constraint.

Outputs:
  - camera_view.mp4 in the results folder (when recording is on)
  - an optional live window (cv2.imshow)
"""

from pathlib import Path
import numpy as np
import cv2

GREEN = (60, 220, 60)    # BGR colours, because OpenCV draws in BGR
RED = (60, 60, 240)
YELLOW = (0, 210, 255)
WHITE = (255, 255, 255)


class CameraView:
    """
    Draws the robot camera frame with an ArUco/EKF overlay and shows or records it
    """

    def __init__(self, output=None, record=True, window=False, fps=4.0):
        """
        output: folder for camera_view.mp4 (None = do not record)
        record: write the mp4
        window: also show a live cv2 window (needs a GUI-enabled OpenCV)
        fps: playback rate. The camera runs at 1/camera_period Hz, so pass that
        """
        self.output = Path(output) if output is not None else None
        self.record = bool(record and output is not None)
        self.window = bool(window)
        self.fps = float(fps)
        self._writer = None
        self._window_ok = self.window
        self.frames = 0
        self.video_path = None

    def draw(self, rgb, measurement, reason, t, mode, camera_age, ekf_pose, accepted=None, dropped=False,
             corners=None):
        """
        Return a BGR frame with the overlay drawn.

        rgb: the rendered camera image (RGB, uint8)
        measurement: ArucoMeasurement, or None if nothing usable was found
        reason: camera.last_reason ('ok', 'not_found', 'wrong_id', ...)
        t, mode, camera_age: sim time, controller mode, seconds since the last accepted fix
        ekf_pose: the EKF estimate [x, y, theta]
        accepted: whether the EKF gate accepted the measurement (None = not tested)
        dropped: True if this frame was lost (dropout or blackout trial)
        corners: the 4 marker corner pixels to draw (see marker_corners); None = draw none
        """
        img = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8)[..., :3], cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        if dropped:
            # Simulated camera loss: the EKF gets no update, so show it plainly
            img = (img * 0.25).astype(np.uint8)
            cv2.putText(img, "NO CAMERA FRAME", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)
        elif measurement is not None and corners is not None:
            corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
            colour = GREEN if accepted in (None, True) else YELLOW
            cv2.polylines(img, [corners.astype(np.int32)], True, colour, 2, cv2.LINE_AA)
            for i, c in enumerate(corners):
                cv2.circle(img, tuple(int(v) for v in c), 4, RED if i == 0 else colour, -1)
                cv2.putText(img, str(i), (int(c[0]) + 5, int(c[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
            centre = corners.mean(axis=0)
            cv2.drawMarker(img, (int(centre[0]), int(centre[1])), colour, cv2.MARKER_CROSS, 14, 1)

        # Image centre cross, so the bearing to the marker is easy to judge by eye
        cv2.drawMarker(img, (w // 2, h // 2), WHITE, cv2.MARKER_CROSS, 10, 1)

        # Text panel (top-left), dark strip behind it for readability
        lines = [f"t={t:5.1f}s  mode={mode}"]
        if dropped:
            status, colour = "camera dropped", RED
        elif measurement is None:
            status, colour = f"ArUco: {reason}", YELLOW
        else:
            gate = "" if accepted is None else (" | EKF accepted" if accepted else " | EKF REJECTED (NIS gate)")
            status, colour = f"ArUco: marker found{gate}", (GREEN if accepted in (None, True) else YELLOW)
        lines.append(status)
        if measurement is not None and not dropped:
            p = measurement.pose
            reproj = getattr(measurement, "reprojection_px", None)
            extra = f"  reproj {reproj:.2f} px" if reproj is not None else ""
            lines.append(f"range {measurement.range_m:.2f} m{extra}")
            lines.append(f"camera pose  x={p[0]:+.2f} y={p[1]:+.2f} th={np.degrees(p[2]):+.0f}deg")
        lines.append(f"EKF pose     x={ekf_pose[0]:+.2f} y={ekf_pose[1]:+.2f} th={np.degrees(ekf_pose[2]):+.0f}deg")
        lines.append(f"time since last fix {camera_age:4.1f}s" if np.isfinite(camera_age) else "no fix yet")

        strip_h = 8 + 18 * len(lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, strip_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, text in enumerate(lines):
            col = colour if i == 1 else WHITE
            cv2.putText(img, text, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def marker_corners(camera, rgb):
        """
        Pixel corners of the charging-station marker in this image, or None.

        Re-runs the same ArUco detector the sensing code uses, only to draw the
        outline. Nothing here feeds the EKF or the controller.
        """
        try:
            gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8)[..., :3], cv2.COLOR_RGB2GRAY)
            corners, ids, _ = camera._detect(gray)
            if ids is None:
                return None
            hits = np.flatnonzero(np.asarray(ids).reshape(-1) == camera.c.marker_id)
            return None if hits.size == 0 else np.asarray(corners[int(hits[0])]).reshape(4, 2)
        except Exception:
            return None

    def show(self, frame_bgr):
        """
        Send one finished frame to the window and/or the video file
        """
        if self.record:
            if self._writer is None:
                self._open_writer(frame_bgr.shape[1], frame_bgr.shape[0])
            if self._writer is not None:
                self._writer.write(frame_bgr)
        if self._window_ok:
            try:
                cv2.imshow("Robot camera", frame_bgr)
                cv2.waitKey(1)
            except cv2.error:
                # Headless OpenCV build or no display: stop trying, keep recording
                self._window_ok = False
                print("Camera window unavailable (OpenCV has no GUI support); recording only.")
        self.frames += 1

    def _open_writer(self, w, h):
        """
        Open the mp4 writer on first use (so we know the frame size)
        """
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "camera_view.mp4"
        # H.264 ('avc1') plays everywhere. Plain 'mp4v' can show as a green screen in
        # QuickTime/browsers, so try H.264 first and only fall back if OpenCV lacks it.
        for code in ("avc1", "H264", "mp4v"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), self.fps, (w, h))
            if writer.isOpened():
                self._writer = writer
                self.video_path = path
                if code == "mp4v":
                    print("H.264 not available in this OpenCV; saved as mp4v "
                          "(if it plays green, open it in VLC or convert with ffmpeg -pix_fmt yuv420p).")
                return
        self.record = False
        print("Could not open a video writer; camera video not saved.")

    def close(self):
        """
        Finish the video file and close the window
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._window_ok:
            try:
                cv2.destroyWindow("Robot camera")
            except cv2.error:
                pass

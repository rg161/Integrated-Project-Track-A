"""
Onboard camera view: updated perspective of what the robot sees

records view to results folder (mp4)

each frame shows the detected ArUco marker, the result and whether the EKF is accepted or not.

off: --no-camera-video
open window: --camera-window
"""

from pathlib import Path
import numpy as np
import cv2

#BGR Colours for OpenCV (r and b swapped)
GREEN = (60, 220, 60)
RED = (60, 60, 240)
YELLOW = (0, 21, 255)
WHITE = (255, 255, 255)

class CameraView:
    """
    Draws the camera frame w/ an ArUco/EKF overlay
    """
    def __init__(self, output=None, record=True, window=False, fps=4.0):
        """
        output - folder for camera_view.mp4
        record - write mp4
        window - allow live window to open
        """
        self.output = Path(output) if output is not None else None
        self.record = bool(record and output is not None)
        self.window = bool(window)
        self.fps = float(fps)
        self._writer = None
        self._window_ok = self.window
        self.frames = 0
        self.video_path = None

    def draw(self, rgb, measurement, reason, t, mode, camera_age, ekf_pose, accepted=None, dropped=False):
        """
        return bgr frame w/ drawn overlay

        rgb - rendered image
        measurement - ArUco measurement
        reason - error display
        mode - controller mode
        ekf_pose - ekf estimate
        accepted - if ekf gate allowed measurement
        dropped - true if frame was lost
        """
        img = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8)[..., :3], cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        if dropped:
            #simulated camera loss: the EKF gets no update, display error
            img = (img * 0.25).astype(np.uint8)
            cv2.putText(img, "NO CAMERA FRAME", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)
        elif measurement is not None:
            corners = np.asarray(measurement.corners, dtype=np.float32).reshape(4, 2)
            colour = GREEN if accepted in (None, True) else YELLOW
            cv2.polylines(img, [corners.astype(np.int32)], True, colour, 2, cv2.LINE_AA)
            for i, c in enumerate(corners):
                cv2.circle(img, tuple(int(v) for v in c), 4, RED if i == 0 else colour, -1)
                cv2.putText(img, str(i), (int(c[0]) + 5, int(c[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
            centre = corners.mean(axis=0)
            cv2.drawMarker(img, (int(centre[0]), int(centre[1])), colour, cv2.MARKER_CROSS, 14, 1)

        #image centre cross (clearer visability of bearing)
        cv2.drawMarker(img, (w // 2, h // 2), WHITE, cv2.MARKER_CROSS, 10, 1)

        #updating text for clarity
        #add block behind
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
            lines.append(f"range {measurement.range_m:.2f} m  reproj {measurement.reprojection_px:.2f} px")
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

    def show(self, frame_bgr):
        """
        send complete frame to window and video file
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
        open mp4 writer for first use
        """
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "camera_view.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if writer.isOpened():
            self._writer = writer
            self.video_path = path
        else:
            self.record = False
            print("Could not open a video writer; camera video not saved.")

    def close(self):
        """
        finish video and close file
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._window_ok:
            try:
                cv2.destroyWindow("Robot camera")
            except cv2.error:
                pass




import unittest
import numpy as np
from frames import(FrameTree, Transform, build_static_frames, pose2d_to_transform, quat_wxyz_to_R, transform_to_pose2d, wrap_angle, R_MJ_TO_CV)

R_B_CMJ_FORWARD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)

def R_to_quat_wxyz(R):
    #Test-only helper (valid away from 180 deg rotations)
    w = np.sqrt(max(0, 1 + np.trace(R))) / 2
    return [w, 
            (R[2, 1] - R[1, 2]) / (4 * w), 
            (R[0, 2] - R[2, 0]) / (4 * w), 
            (R[1, 0] - R[0, 1]) / (4 * w)]

def make_tree(base = (1, 0, 0)):
    tree = FrameTree()
    build_static_frames(tree, 
                        cam_pos_B = [0.07, 0, 0.1], 
                        cam_quat_B_wxyz = R_to_quat_wxyz(R_B_CMJ_FORWARD), 
                        lidar_pos_B = [-0.03, 0, 0.17], 
                        marker_pos_W = [2, 0, 0.1])
    tree.set_base_pose(*base)
    return tree

def predicted_range_bearing(pose, landmark_xy):
    #Test only, real is in EKF module
    dx, dy = landmark_xy[0] - pose[0], landmark_xy[1] - pose[1]
    return np.array([np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - pose[2])])

class TestFrames(unittest.TestCase):
    def test_quat_identity_and_normalisation(self):
        assert np.allclose(quat_wxyz_to_R([1, 0, 0, 0]), np.eye(3))
        assert np.allclose(quat_wxyz_to_R([2, 0, 0, 0]), np.eye(3))

    def test_inverse_and_pose_roundtrip(self):
        T = pose2d_to_transform("W", "B", 1, 2, 0.7)
        assert np.allclose((T @ T.inv()).matrix(), np.eye(4))
        assert np.allclose(transform_to_pose2d(T), (1, 2, 0.7))

    def test_compose_mismatched_frames_raises(self):
        T = pose2d_to_transform("W", "B", 0, 0, 0)
        with self.assertRaises(ValueError):
            _ = T @ T

    def test_missing_frame_raises(self):
        with self.assertRaises(KeyError):
            FrameTree().transform([0, 0, 0], "C", "B")

    def test_mujoco_to_opencv_camera_axes(self):
        R_B_C = make_tree().get("B", "C").R
        assert np.allclose(R_B_C @ [0, 0, 1], [1, 0, 0])  #optical forward -> body +x
        assert np.allclose(R_B_C @ [1, 0, 0], [0, -1, 0]) #optical right -> body -y
        assert np.allclose(R_B_C @ [0, 1, 0], [0, 0, -1]) #optical down -> body -z
        assert np.isclose(np.linalg.det(R_B_C), 1)        #proper rotation, no mirror
        assert np.isclose(np.linalg.det(R_MJ_TO_CV), 1)

    def test_marker_straight_ahead(self):
        #A point 1m along the optical axis is 1m in front o the camera in B
        p_B = make_tree().transform([0, 0, 1], "C", "B")
        assert np.allclose(p_B, [0.07 + 1, 0, 0.1])

    def test_tree_matches_analytic_range_bearing(self):
        for pose in [(1, 0, 0), (0.5, -0.3, 0.9), (1.5, 0.4, -2.0)]:
            tree = make_tree(pose)
            p = tree.transform([0, 0, 0], "M", "B")
            rb_tree = np.array([np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])])
            assert np.allclose(rb_tree, predicted_range_bearing(pose, [2, 0]))

    def test_bearing_sign_marker_on_right(self):
        assert predicted_range_bearing((1, 0, np.pi / 4), [2, 0])[1] < 0

    def test_wrap_angle(self):
        assert np.isclose(wrap_angle(3 * np.pi / 2 + 0.1), -np.pi / 2 + 0.1)
        assert np.isclose(wrap_angle(-3 * np.pi / 2 - 0.1), np.pi / 2 - 0.1)

if __name__ == "__main__":
    unittest.main()
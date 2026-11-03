import unittest
import numpy as np
from frames import rot_y
from noise import (ArucoGate, ArucoNoiseModel, aruco_check, clean_scan, cluster_scan,
                   incidence_cos, mean_side_px, odom_cov, scan_to_points,
                   wheel_increments_to_motion)
 
N = 360
INC = np.deg2rad(1.0)
R_FACING = np.diag([1.0, -1.0, -1.0])        # marker facing the camera head-on
SQUARE_40 = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], float)
 
 
class TestLidar(unittest.TestCase):
    def test_invalid_returns_become_nan(self):
        r = np.full(N, 2.0)
        r[[3, 7, 11, 15, 19, 23]] = [-1, np.nan, np.inf, 0.0, 0.05, 4.0]
        out = clean_scan(r, window=1)
        self.assertTrue(np.all(np.isnan(out[[3, 7, 11, 15, 19, 23]])))
        self.assertEqual(np.sum(np.isfinite(out)), N - 6)
 
    def test_spike_removed_including_across_the_seam(self):
        r = np.full(N, 2.0)
        r[[10, 0]] = 3.0                      # index 0 needs the circular wrap
        out = clean_scan(r, window=3)
        self.assertAlmostEqual(out[10], 2.0)
        self.assertAlmostEqual(out[0], 2.0)
 
    def test_median_reduces_noise(self):
        rng = np.random.default_rng(0)
        r = 2.0 + rng.normal(0, 0.01, N)
        rms = lambda x: np.sqrt(np.mean((x - 2.0) ** 2))
        self.assertLess(rms(clean_scan(r, window=5)), rms(r))
 
    def test_median_never_invents_data(self):
        r = np.full(N, 2.0)
        r[50] = np.nan
        self.assertTrue(np.isnan(clean_scan(r, window=5)[50]))
 
    def test_even_window_rejected(self):
        with self.assertRaises(ValueError):
            clean_scan(np.ones(N), window=4)
 
    def test_scan_to_points_axes(self):
        p = scan_to_points([1.0, 1.0], angle_min=0.0, angle_inc=np.pi / 2)
        self.assertTrue(np.allclose(p, [[1, 0], [0, 1]]))       # CCW positive, y left
 
    def test_clusters_split_on_gap_and_on_depth_jump(self):
        r = np.full(N, np.nan)
        r[10:31] = 1.0
        r[50:71] = 2.0                         # 20 deg gap -> separate
        self.assertEqual(len(cluster_scan(r, INC)), 2)
        r2 = np.full(N, np.nan)
        r2[10:31] = 1.0
        r2[31:51] = 2.0                        # adjacent beams, 1 m depth jump -> separate
        self.assertEqual(len(cluster_scan(r2, INC)), 2)
 
    def test_continuous_surface_is_one_cluster_and_min_points(self):
        r = np.full(N, np.nan)
        r[10:60] = 1.5
        self.assertEqual(len(cluster_scan(r, INC)), 1)
        r[200:202] = 1.0
        self.assertEqual(len(cluster_scan(r, INC, min_points=3)), 1)   # 2-point blob dropped
 
 
class TestAruco(unittest.TestCase):
    T = np.array([0.0, 0.0, 1.5])
 
    def test_incidence_head_on_grazing_and_behind(self):
        self.assertAlmostEqual(incidence_cos(self.T, R_FACING), 1.0)
        R80 = rot_y(np.deg2rad(80)) @ R_FACING
        self.assertAlmostEqual(incidence_cos(self.T, R80), np.cos(np.deg2rad(80)))
        self.assertLess(incidence_cos(self.T, np.eye(3)), 0)
 
    def test_mean_side(self):
        self.assertAlmostEqual(mean_side_px(SQUARE_40), 40.0)
 
    def test_good_detection_accepted(self):
        self.assertEqual(aruco_check(self.T, R_FACING, SQUARE_40, 0.4), (True, "ok"))
 
    def test_each_rejection_rule(self):
        g = ArucoGate()
        self.assertEqual(aruco_check(self.T, R_FACING, SQUARE_40, 3.0)[1], "reproj")
        self.assertEqual(aruco_check(self.T, R_FACING, SQUARE_40 * 0.3, 0.4)[1], "small")
        self.assertEqual(aruco_check([0, 0, g.max_range + 0.5], R_FACING, SQUARE_40, 0.4)[1], "range")
        R80 = rot_y(np.deg2rad(80)) @ R_FACING
        self.assertEqual(aruco_check(self.T, R80, SQUARE_40, 0.4)[1], "grazing")
        self.assertEqual(aruco_check(self.T, np.eye(3), SQUARE_40, 0.4)[1], "grazing")
        self.assertEqual(aruco_check([np.nan, 0, 1], R_FACING, SQUARE_40, 0.4)[1], "invalid")
 
 
class TestArucoCovariance(unittest.TestCase):
    M = ArucoNoiseModel(f_px=526.6, marker_side=0.15)
 
    def test_range_variance_grows_with_distance(self):
        sig = [np.sqrt(self.M.cov(r)[0, 0]) for r in (0.5, 1.0, 2.0, 3.0)]
        self.assertTrue(all(a < b for a, b in zip(sig, sig[1:])))
 
    def test_range_sigma_scales_as_r_squared_without_floor(self):
        m = ArucoNoiseModel(f_px=526.6, marker_side=0.15, sigma_r_floor=0.0)
        self.assertAlmostEqual(np.sqrt(m.cov(2.0)[0, 0] / m.cov(1.0)[0, 0]), 4.0)
 
    def test_sigma_at_2m_is_a_few_cm(self):
        self.assertTrue(0.01 < np.sqrt(self.M.cov(2.0)[0, 0]) < 0.05)
 
    def test_grazing_view_inflates_and_matrix_is_valid(self):
        head, graz = self.M.cov(2.0, 1.0), self.M.cov(2.0, 0.3)
        self.assertTrue(np.all(np.diag(graz) > np.diag(head)))
        self.assertTrue(np.allclose(head, head.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(head) > 0))
 
    def test_floor_active_at_close_range(self):
        self.assertGreaterEqual(np.sqrt(self.M.cov(0.1)[0, 0]), self.M.sigma_r_floor)
 
 
class TestOdometry(unittest.TestCase):
    r, b = 0.033, 0.288
 
    def test_straight_and_spin(self):
        ds, dth = wheel_increments_to_motion(1.0, 1.0, self.r, self.b)
        self.assertAlmostEqual(ds, self.r)
        self.assertAlmostEqual(dth, 0.0)
        ds, dth = wheel_increments_to_motion(-1.0, 1.0, self.r, self.b)
        self.assertAlmostEqual(ds, 0.0)
        self.assertAlmostEqual(dth, 2 * self.r / self.b)        # right faster -> CCW (+)
 
    def test_cov_linear_in_distance_symmetric_and_zero_at_rest(self):
        c1 = odom_cov(1.0, 1.0, self.r, self.b)
        c2 = odom_cov(2.0, 2.0, self.r, self.b)
        self.assertTrue(np.allclose(c2, 2 * c1))
        self.assertTrue(np.allclose(c1, c1.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(c1) >= -1e-15))
        self.assertTrue(np.allclose(odom_cov(0, 0, self.r, self.b), 0))
 
 
if __name__ == "__main__":
    unittest.main()
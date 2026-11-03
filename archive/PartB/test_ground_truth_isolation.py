import ast
import pathlib
import re
import unittest
from types import SimpleNamespace
import numpy as np
from frames import rot_z
from ground_truth import GroundTruth

ALLOWED_STEMS = {"ground_truth", "logger", "evaluator", "calibration"}
RAW_STATE = re.compile(r"\.(qpos|qvel|xpos|xmat|xquat|cvel)\b")

def _fake_sim(x, y, th):
    model = SimpleNamespace(body=lambda name: SimpleNamespace(id=1))
    R = rot_z(th)
    data = SimpleNamespace(time = 1.5, 
                           xpos = np.array([[0, 0, 0], [x, y, 0]]), 
                           xmat = np.array([np.eye(3).ravel(), R.ravel()]))
    return model, data

def _project_files():
    here = pathlib.Path(__file__).parent
    for p in here.glob("*.py"):
        if p.stem not in ALLOWED_STEMS and not p.stem.startswith("test_"):
            yield p

class TestGroundTruth(unittest.TestCase):
    def test_pose_and_time(self):
        model, data = _fake_sim(1, -2, 0.6)
        gt = GroundTruth(model, data)
        self.assertTrue(np.allclose(gt.pose(), [1, -2, 0.6]))
        self.assertEqual(gt.time, 1.5)

    def test_pose_error_wraps_angle(self):
        model, data = _fake_sim(0, 0, 3)
        e = GroundTruth(model, data).pose_error([0, 0, -3])
        self.assertTrue(abs(e[2]) < np.pi)
        self.assertAlmostEqual(e[2], -6 + 2 * np.pi)

class TestIsolation(unittest.TestCase):
    def test_no_forbidden_imports(self):
        for p in _project_files():
            tree = ast.parse(p.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                self.assertNotIn("ground_truth", names,
                                 f"{p.name} imports ground_truth")

    def test_no_raw_simulator_state_reads(self):
        for p in _project_files():
            hits = RAW_STATE.findall(p.read_text())
            self.assertFalse(hits, f"{p.name} reads raw simulator state: {set(hits)}")

if __name__ == "__main__":
    unittest.main()

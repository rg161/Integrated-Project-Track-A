# Part D — integrated controller

The original ROS 1 controller has been replaced by the runtime controller in `track_a/controller.py`.
It receives the Part C EKF estimate directly from `track_a/runner.py`; no ROS topics or subprocess launcher are required.

Run the complete project with:

```bash
python main.py --headless --scenario nominal --duration 65
```

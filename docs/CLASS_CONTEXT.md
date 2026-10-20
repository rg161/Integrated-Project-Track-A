# Prior Industrial Robotics context

The request specifically asked to use the Industrial Robotics chat. Context
retrieval was attempted twice, including a targeted search for exact mobile
sensing/control class code.

The relevant retrievable context concerned SE3/frame conventions, separating
scene/model/motion modules, checking Jacobians and collisions, camera versus
world coordinates, and case/path/dependency problems in earlier Swift projects.
It also recovered earlier discussion of the Track A EKF state `[x, y, theta]`,
encoder prediction and camera pose/range-bearing corrections. These reinforce
the interface checks, transform tests and shared model used here.

The full chat transcript and all original lab files were not available through
retrieval. Exact ArUco/calibration/mobile-controller class snippets were not
recovered. Accordingly, the supplied archives and PDF were the authoritative
implementation inputs; missing connections are identified as new integration
work, not attributed to an unseen class solution.

The separate KUKA KR25/KR35 / UR3e tyre-removal work is a different platform and
task. Its unverified joint pivots and calibration-gated IK were not transferred
to the TurtleBot. Swift, `ir-support` and Robotics Toolbox dependencies were not
added to this MuJoCo project merely because they were used in that class.

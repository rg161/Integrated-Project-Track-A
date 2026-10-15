# Model, sensing, estimation and control

## Part A — motion model

State is \(\mathbf{x}=[x,y,\theta]^T\), input is
\(\mathbf{u}=[v,\omega]^T\). Left/right encoder increments are
\(\Delta\phi_L,\Delta\phi_R\).

\[
\Delta s=\frac{r}{2}(\Delta\phi_L+\Delta\phi_R),\qquad
\Delta\theta=\frac{r}{b}(\Delta\phi_R-\Delta\phi_L).
\]

The supplied Part A midpoint discretisation is used consistently:

\[
\begin{aligned}
\bar\theta &= \theta_k+\tfrac12\omega_k\Delta t,\\
x_{k+1}&=x_k+v_k\Delta t\cos\bar\theta,\\
y_{k+1}&=y_k+v_k\Delta t\sin\bar\theta,\\
\theta_{k+1}&=\operatorname{wrap}(\theta_k+\omega_k\Delta t).
\end{aligned}
\]

It assumes locally planar motion and modest change over a 0.05 s step. It is an
approximation to the continuous unicycle model, not an exact rigid-body dynamics
model. MuJoCo separately integrates the free body, wheel actuators and contacts
at 0.002 s. Encoder samples measure wheel motion; commanded wheel speed is not
substituted for measured speed in the EKF.

`F_jac` and `G_jac` are derivatives of this same discrete function. The standard
linearised controllability matrix `[G, FG, F²G]` has rank 2 at rest and rank 3
at the tested nonzero forward speed. The rank-2 rest result does not contradict
nonlinear manoeuvrability using a sequence of turns and forward moves.

For a point landmark \(m=(m_x,m_y)\), the retained alternative measurement is

\[
h(\mathbf{x})=\begin{bmatrix}
\sqrt{(m_x-x)^2+(m_y-y)^2}\\
\operatorname{wrap}(\operatorname{atan2}(m_y-y,m_x-x)-\theta)
\end{bmatrix}.
\]

A single point's range/bearing leaves a rotation-about-landmark gauge. The
stacked local observability check therefore reports rank 2. Two distinct mapped
points can give rank 3, with poor conditioning if they are close together.

The default runtime instead uses **full planar pose from an oriented marker**
whose world transform is known. Marker orientation supplies information absent
from a point-only range/bearing measurement. Its EKF measurement Jacobian is
`I_3` when a valid pose has been recovered. It is not valid to use this rank-3
statement while the marker is out of view.

## Part B — sensing and frames

### Encoders and LiDAR

MuJoCo named `jointpos` sensors provide cumulative left/right wheel angles.
Independent Gaussian reading noise is added, then consecutive readings are
differenced. LiDAR uses 180 horizontal `mj_ray` intersections from the fixed
LiDAR site, rotates rays with the simulated sensor, and returns at most 3.5 m.
Robot geometry is excluded from scans. Noise is added only to finite hits.
No-return beams remain infinity; invalid readings become NaN.

Part B's median filtering and invalid-sample handling are retained. Scan points
are placed in world coordinates using the **estimated** base pose plus the
known LiDAR extrinsic. Ray traversal decreases occupancy log odds and endpoints
increase them. Occupied cells are inflated by the robot radius plus margin.

### Camera model and ArUco

The ideal simulated camera has zero distortion. Intrinsics are derived from the
specified vertical field of view:

\[
f_x=f_y=\frac{H}{2\tan(\mathrm{fovy}/2)},\qquad
c_x=(W-1)/2,\quad c_y=(H-1)/2.
\]

At 480 × 360 and 58°, the focal length is about 324.729 pixels. This is analytic
simulation calibration, independently checked with rendered marker views. It
does not replace a real camera's calibration images and fitted distortion.

The scene contains an actual 4×4 ArUco code plus its black border, generated
from ID 7 in `DICT_4X4_50`. OpenCV detects its four corners in the rendered image;
`solvePnP` estimates `T_C_M`. No true camera or robot pose is used by PnP.

Transforms use column-vector semantics:

\[
p_{parent}=R_{parent,child}p_{child}+t_{parent,child},\qquad
T_{W,B}=T_{W,M}T_{C,M}^{-1}T_{B,C}^{-1}.
\]

The optical frame's +z is base +x; optical +x is base −y; optical +y is base −z.
MuJoCo's camera looks along its local −z with +y up. Part B's 180° x-axis camera
conversion and the explicit mounting transform keep those conventions distinct.

The camera rejects nonfinite/behind-camera solutions, too-small markers,
excessive range, incidence above 70° and reprojection RMS above 1.5 pixels.
PnP covariance is approximated by finite-differencing the pixel-to-planar-pose
mapping while retaining the local PnP solution branch:

\[
R=J_{pixels}\,\sigma_{pixels}^2 I\,J_{pixels}^T+
\operatorname{diag}(0.012^2,0.012^2,(1.5^\circ)^2).
\]

This retains position/heading cross-covariance from the camera lever arm.
Pixel sigma is at least 0.5 pixels, accounting for rasterisation and detection
uncertainty as well as injected corner noise. A local covariance cannot fully
describe planar PnP ambiguity or correlated detection bias; measured consistency
results must therefore be inspected, not assumed.

## Part C — EKF

Prediction is

\[
\hat x^-_{k+1}=f(\hat x_k,u_k,\Delta t),\qquad
P^-_{k+1}=FP_kF^T+G\Sigma_uG^T+Q_c\Delta t.
\]

Wheel travel covariance follows the supplied Part B model:
`var(dL)=k*abs(dL)`, `var(dR)=k*abs(dR)`. It is transformed into displacement/
heading covariance and includes the variance of two independent noisy encoder
readings. Division by `dt²` produces input-velocity covariance for `G`.

`k=2e-4 m`; the additional continuous process diagonal is
`[0.003² m²/s, 0.003² m²/s, radians(0.3)² rad²/s]`. This allows unmodelled planar
motion and contact effects. The initial covariance diagonal is
`[0.08² m², 0.08² m², radians(6)² rad²]`. These are explicit modelling choices,
not values fitted to the true trajectory during a run.

With innovation \(\nu=z-h(\hat x^-)\), angular components are wrapped before
calculating \(S=HP^-H^T+R\). The NIS is \(\nu^T S^{-1}\nu\). Thresholds
16.27 (3-dimensional pose) and 13.82 (2-dimensional range/bearing) are approximately
99.9% chi-square quantiles; they reject extreme innovations. Linear systems are
solved directly rather than forming explicit inverses.

The covariance correction uses Joseph form:

\[
P^+=(I-KH)P^-(I-KH)^T+KRK^T.
\]

This avoids the numerical fragility of `(I-KH)P` and preserves covariance
symmetry. With no valid marker observation, the estimator performs prediction
only. The independent odometry-only baseline integrates exactly the same
encoder increments but receives no camera correction.

## Part D — state feedback and docking

The LiDAR map supplies an inflated-grid A* path. The local reference advances
along the closest path segment by a 0.28 m lookahead. Its heading points toward
the lookahead location, avoiding premature turns to the next segment tangent.
This is a receding path reference, not a fixed time-scheduled trajectory.

The supplied body's error coordinates are retained:

\[
\begin{bmatrix}e_x\\e_y\end{bmatrix}=
\begin{bmatrix}\cos\hat\theta&\sin\hat\theta\\-\sin\hat\theta&\cos\hat\theta\end{bmatrix}
\begin{bmatrix}x_r-\hat x\\y_r-\hat y\end{bmatrix},\quad
e_\theta=\operatorname{wrap}(\theta_r-\hat\theta).
\]

During tracking, the core feedback law is

\[
v=(v_r\cos e_\theta+k_1e_x)\cos^2\alpha,\qquad
\omega=k_2v_re_y+k_3\sin e_\theta,
\]

where \(\alpha\) is the bearing to the lookahead point and `v_r=0.7*max_v`.
The cosine factor reduces forward motion while turning. At large heading error
the robot rotates in place first, with hysteresis (enter above 0.65 rad, leave
below 0.35 rad). Unlike the original `v_ref=0` scaffold, this controller can
correct a lateral target error.

Within 0.28 m of the goal, a distance/bearing controller reduces speed. Within
position tolerance it turns to the goal heading. A consecutive dwell with
recent accepted camera data is required before declaring docking complete.
Loss of a condition resets the dwell counter.

Commands are bounded in forward/yaw speed. Wheel-speed conversion applies one
common scale if either wheel exceeds its limit, preserving curvature. A body-
frame LiDAR corridor reduces translation below 0.50 m and stops it below
0.24 m. Excessive position uncertainty stops all motion. These guards are
simulation safeguards, not a certified real-robot safety controller.

The controller has been tuned for this scene and tested conditions. It is not
LQR, and no global stability claim is made for the switched, saturated,
replanning system. A fixed-time tracking study and broader gain sweep would be
additional work, not evidence already supplied here.

## Part E — evaluation

The backend supplies true pose only to `Evaluator`. Every sample records
estimated/true/odometry/reference states, all nine covariance entries, wheel and
body commands, innovation information and controller mode. Position error is
Euclidean; heading errors are wrapped. NEES uses the full covariance, including
cross terms. The plotted ellipses have 2σ principal-axis radii, not a claimed
95% joint confidence region.

Three seeds per condition are a small reproducibility study. Noise and camera
dropout random streams are separated, so disabling the camera does not change
the encoder noise sequence. The conditions are:

| Scenario | Changes |
|---|---|
| nominal | LiDAR sigma 0.008 m; encoder sigma 0.0005 rad/sample; added corner sigma 0.25 pixels |
| noisy | All injected sensor noise scales ×3; independent 20% camera-frame dropout |
| dropout | Camera unavailable from 6–14 s, plus independent 15% frame dropout |
| disturbed | +y force 1.5 N from 9–9.5 s; obstacle A shifted +0.12 m in y |
| no_camera | All camera frames absent; deliberate negative completion test |

The push is modest and partly resisted by contact friction. These trials do not
establish recovery from large slips. Full-pose camera estimates can also make
the filter less accurate than ideal wheel odometry in some runs. Results report
that directly; neither successful docking nor a positive covariance proves an
accurate or statistically consistent filter.

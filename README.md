# Kalman Filter

**A Rust library implementing both scalar and 2D Kalman filters for optimal linear state estimation under Gaussian noise.** It provides the classic predict-update recursion for fusing noisy measurements with a dynamics model to produce minimum-variance state estimates.

## Why It Matters

The Kalman filter is the optimal recursive estimator for linear systems with Gaussian noise — it minimizes the mean-square error of the state estimate. It's used in GPS navigation, aerospace guidance (Apollo used it for Moon landing), robotics (SLAM), finance (volatility estimation), signal processing (noise cancellation), and sensor fusion (combining IMU + GPS + odometer). The "filter" recursively processes one observation at a time, maintaining constant O(1) memory and O(1) computation per step — no history needed.

## How It Works

Each cycle has two phases:

**Predict** — Project the state forward using the dynamics model:
```
x̂⁻ = F · x̂        (state prediction: F is the state transition)
P⁻ = F · P · Fᵀ + Q  (covariance prediction: Q is process noise)
```

For the scalar filter, F=1 (constant model), so `x̂⁻ = x̂` and `P⁻ = P + Q`. The process noise Q represents model uncertainty — how much we expect the true state to drift between measurements.

**Update** — Incorporate the measurement z:
```
y = z - H · x̂⁻     (innovation: difference between measurement and prediction)
S = H · P⁻ · Hᵀ + R  (innovation covariance: total uncertainty of innovation)
K = P⁻ · Hᵀ / S     (Kalman gain: how much to trust the measurement)
x̂ = x̂⁻ + K · y     (corrected state estimate)
P = (I - K · H) · P⁻  (corrected covariance: uncertainty reduced)
```

The **Kalman gain** K ∈ [0, 1] is the key: when measurement noise R is small relative to prediction uncertainty P, K → 1 (trust the sensor); when R is large, K → 0 (trust the model). The filter automatically adjusts this balance at each step.

The **2D variant** tracks `[position, velocity]` with a constant-velocity model: `F = [[1, dt], [0, 1]]`. It observes only position (H = [1, 0]) but infers velocity from the sequence of measurements. The covariance matrix is 2×2, capturing both the uncertainty in position/velocity and their correlation.

All operations are O(1) for the scalar filter and O(1) (fixed 2×2 matrices) for the 2D filter.

## Quick Start

```rust
use kalman_filter::KalmanFilter;

// Track a temperature that should be constant ~20°C
// Model: constant (F=1), noisy sensor (R=0.5), small drift (Q=0.01)
let mut kf = KalmanFilter::new(20.0,  // initial estimate
                                1.0,   // initial uncertainty
                                0.01,  // process noise
                                0.5);  // measurement noise

let readings = [19.8, 20.3, 19.9, 20.1, 20.2, 19.7];
let estimates = kf.filter(&readings);

for (i, (z, est)) in readings.iter().zip(estimates.iter()).enumerate() {
    println!("t={}: measured={:.1}°C, estimated={:.2}°C", i, z, est);
}
```

```rust
use kalman_filter::KalmanFilter2D;

// Track a moving object: position and velocity
let mut kf = KalmanFilter2D::new(
    [0.0, 1.0],                        // [pos=0, vel=1 m/s]
    [[1.0, 0.0], [0.0, 1.0]],          // initial covariance
    [[0.01, 0.0], [0.0, 0.01]],        // process noise
    0.5,                                // measurement noise
);

for z in &[1.0, 2.1, 3.0, 4.1, 5.0] {
    kf.predict(1.0);  // 1-second timestep
    kf.update(*z);
    println!("pos={:.2} vel={:.2}", kf.x[0], kf.x[1]);
}
```

## API

- **`KalmanFilter`** — Scalar (1D) filter.
  - `new(x0, p0, q, r)` — Initialize state, uncertainty, process noise, measurement noise.
  - `predict()` — Advance state via dynamics model. O(1).
  - `update(z)` — Incorporate measurement z. O(1).
  - `step(z) -> f64` — Full predict+update cycle. O(1).
  - `filter(&[f64]) -> Vec<f64>` — Run over observation sequence. O(n).
- **`KalmanFilter2D`** — 2D state [position, velocity], 1D observation.
  - `predict(dt)` — Constant-velocity model with timestep dt.
  - `update(z)` — Position-only measurement update.

## Architecture Notes

Part of the SuperInstance estimation and control toolkit. Pairs with the [Mahalanobis distance](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for outlier detection before filtering, the [Jacobian system](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for extending to the Extended Kalman Filter (EKF), and the [Itô calculus](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for stochastic differential equation models.

## License

MIT

//! Kalman filter implementation for linear state estimation.

/// A simple Kalman filter for scalar state with scalar observations.
pub struct KalmanFilter {
    /// State estimate
    pub x: f64,
    /// Estimate covariance
    pub p: f64,
    /// Process noise variance
    pub q: f64,
    /// Measurement noise variance
    pub r: f64,
    /// State transition (typically 1.0 for constant model)
    pub f: f64,
    /// Observation model (typically 1.0)
    pub h: f64,
}

impl KalmanFilter {
    pub fn new(x0: f64, p0: f64, q: f64, r: f64) -> Self {
        Self { x: x0, p: p0, q, r, f: 1.0, h: 1.0 }
    }

    /// Predict step: advance state using process model.
    pub fn predict(&mut self) {
        self.x = self.f * self.x;
        self.p = self.f * self.p * self.f + self.q;
    }

    /// Update step: incorporate observation.
    pub fn update(&mut self, z: f64) {
        let y = z - self.h * self.x;                    // innovation
        let s = self.h * self.p * self.h + self.r;      // innovation covariance
        let k = self.p * self.h / s;                     // Kalman gain
        self.x = self.x + k * y;
        self.p = (1.0 - k * self.h) * self.p;
    }

    /// Full predict + update cycle. Returns updated state estimate.
    pub fn step(&mut self, z: f64) -> f64 {
        self.predict();
        self.update(z);
        self.x
    }

    /// Run filter over a sequence of observations.
    pub fn filter(&mut self, observations: &[f64]) -> Vec<f64> {
        observations.iter().map(|&z| self.step(z)).collect()
    }
}

/// Vector (multi-dimensional) Kalman filter using nalgebra-style arrays.
/// Simplified: 2D state, 1D observation.
pub struct KalmanFilter2D {
    pub x: [f64; 2],
    pub p: [[f64; 2]; 2],
    pub q: [[f64; 2]; 2],
    pub r: f64,
}

impl KalmanFilter2D {
    pub fn new(x0: [f64; 2], p0: [[f64; 2]; 2], q: [[f64; 2]; 2], r: f64) -> Self {
        Self { x: x0, p: p0, q, r }
    }

    /// Constant-velocity predict: x = [pos, vel], F = [[1,dt],[0,1]]
    pub fn predict(&mut self, dt: f64) {
        let pos = self.x[0] + dt * self.x[1];
        let vel = self.x[1];
        self.x = [pos, vel];

        // P = F*P*F' + Q
        let p00 = self.p[0][0] + dt * (self.p[0][1] + self.p[1][0]) + dt * dt * self.p[1][1] + self.q[0][0];
        let p01 = self.p[0][1] + dt * self.p[1][1] + self.q[0][1];
        let p10 = self.p[1][0] + dt * self.p[1][1] + self.q[1][0];
        let p11 = self.p[1][1] + self.q[1][1];
        self.p = [[p00, p01], [p10, p11]];
    }

    /// Update with position observation z.
    pub fn update(&mut self, z: f64) {
        let y = z - self.x[0];
        let s = self.p[0][0] + self.r;
        let k0 = self.p[0][0] / s;
        let k1 = self.p[1][0] / s;

        self.x[0] += k0 * y;
        self.x[1] += k1 * y;

        self.p[0][0] -= k0 * self.p[0][0];
        self.p[0][1] -= k0 * self.p[0][1];
        self.p[1][0] -= k1 * self.p[0][0];
        self.p[1][1] -= k1 * self.p[0][1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_converges() {
        let mut kf = KalmanFilter::new(0.0, 1.0, 0.01, 0.1);
        let estimates = kf.filter(&[1.0, 1.1, 0.9, 1.05, 0.95]);
        assert!((estimates.last().unwrap() - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_2d_tracking() {
        let mut kf = KalmanFilter2D::new(
            [0.0, 1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.01, 0.0], [0.0, 0.01]],
            0.5,
        );
        for z in &[1.0, 2.1, 3.0, 4.1, 5.0] {
            kf.predict(1.0);
            kf.update(*z);
        }
        assert!(kf.x[0] > 4.0);
        assert!(kf.x[1] > 0.5);
    }
}

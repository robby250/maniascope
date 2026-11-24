//! Small shared kernels of the calculator.
use crate::py;

/// Speed (Hz) of a gap; below the knee it tapers to 0 (flam ≈ chord).
#[inline]
pub fn v(d: f64, knee: f64, taper: f64) -> f64 {
    if d >= knee {
        return 1.0 / d;
    }
    let x = d.max(0.0) / knee;
    // The tapers in use: a square and a root without a pow call.
    (if taper == 2.0 { x * x } else if taper == 0.5 { x.sqrt() } else { x.powf(taper) }) / knee
}

/// `v` at taper 2 with the gap before it (`prev`, 0 = none); `share` (FLAM_RUN) of the run's speed is kept. A lone flam is a chord: it moves at the speed of the gap before it.
/// A run of flams is a stream: no slower than the knee's speed, and than its pairs come (1 / two gaps). The plain taper fell
/// toward 0 for every gap under the knee, so past 22 a second a faster chart or rate rated lower (user 2026-11-22: REFORM
/// Theta, Iota and Kappa met at 2×, raw 48.8 / 49.9 at 1.75× / 2×; 61 % of Kappa's rows are under the knee at 1.0×).
#[inline]
pub fn v_run(d: f64, prev: f64, knee: f64, share: f64) -> f64 {
    let own = v(d, knee, 2.0);
    if d >= knee || prev <= 0.0 {
        return own;
    }
    own.max(share * (1.0 / knee).min(1.0 / prev).max(1.0 / (d + prev)))
}

#[inline]
pub fn smooth(x: f64, lo: f64, hi: f64) -> f64 {
    let x = (x - lo) / (hi - lo);
    if !(x > 0.0) {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    x * x * (3.0 - 2.0 * x)
}

/// Columns strictly between the outermost fingers of `mask` (13 → 2).
#[inline]
pub fn inner(mask: u32, keys: usize) -> u32 {
    let m = mask & ((1u64 << keys) - 1) as u32;
    if m == 0 {
        return 0;
    }
    let lo = m.trailing_zeros();
    let hi = 31 - m.leading_zeros();
    if hi <= lo + 1 {
        return 0;
    }
    (((1u64 << hi) - 1) as u32) & !(((1u64 << (lo + 1)) - 1) as u32)
}

/// 0..1: how far an event gap is off a reference (see skill_calc._off_grid).
#[inline]
pub fn off_grid(g: f64, reference: f64, pulse: bool) -> f64 {
    let mut r = g / reference;
    if r < 1.0 && !pulse {
        r = 1.0 / r;
    }
    let x = if r >= 1.0 { (r / 1f64.max(py::round(r))).log2().abs() } else { r.log2().abs() };
    1f64.min(2.2 * 0f64.max((x - py::round(x)).abs() - 0.06))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inner_mask() {
        assert_eq!(inner(0b101, 7), 0b010);
        assert_eq!(inner(0b1000001, 7), 0b0111110);
        assert_eq!(inner(0b11, 7), 0);
        assert_eq!(inner(0, 7), 0);
    }
}

#[cfg(test)]
mod run_tests {
    use super::*;

    /// A run under the flam knee never slows as it speeds up; a lone flam is a chord at the speed of the gap before it.
    #[test]
    fn a_run_under_the_flam_knee_keeps_rising_with_speed() {
        let knee = 0.045;
        let mut last = 0.0;
        for step in 0..60 {
            let d = 0.06 * 0.95f64.powi(step);
            let speed = v_run(d, d, knee, 1.0);
            assert!(speed >= last - 1e-9, "{d}: {speed} under {last}");
            last = speed;
        }
        assert!(last > 2.0 / knee, "a run at a twentieth of the knee is far past the knee's speed: {last}");
        assert_eq!(v_run(0.01, 0.3, knee, 1.0), 1.0 / 0.3);
        assert_eq!(v_run(0.01, 0.3, knee, 0.0), v(0.01, knee, 2.0));
    }
}

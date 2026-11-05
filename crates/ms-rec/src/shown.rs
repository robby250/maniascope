//! Displayed stars: the fitted rating on osu!'s scale per keymode (calib/anchor.json, written by `calib anchor`). Only numbers
//! a player reads go through it; dans, predictions, lists and fits keep the fitted units, so nothing else moves.
use std::sync::OnceLock;

/// osu!-scale stars for a fitted rating of a `keys`-key chart or player: exp(ln_a(k)) · s^b(k), k clamped to the fitted 4–10.
pub fn stars(keys: i64, s: f64) -> f64 {
    static CURVES: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    let (ln_a, b) = CURVES.get_or_init(|| {
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../calib/anchor.json")).expect("anchor.json");
        let list = |k: &str| v[k].as_array().expect("anchor curve").iter().filter_map(serde_json::Value::as_f64).collect();
        (list("ln_a"), list("b"))
    });
    let x = (keys as f64).clamp(4., 10.) - 7.;
    let at = |c: &[f64]| c.iter().fold(0., |acc, v| acc * x + v);
    if s > 0. { at(ln_a).exp() * s.powf(at(b)) } else { s }
}

#[cfg(test)]
mod tests {
    /// A calculator ship moves the fitted scale: the anchor must be refitted with it (`calib anchor`).
    #[test]
    fn the_anchor_was_fitted_on_this_calculator() {
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../calib/anchor.json")).unwrap();
        assert_eq!(v["calc"], crate::engine::CALC_ID, "rerun calib anchor for the new calculator");
        assert!(super::stars(7, 6.) < super::stars(7, 9.), "monotone");
        assert!((super::stars(7, 6.7) - 5.16).abs() < 0.2, "{}", super::stars(7, 6.7));
    }
}

//! Displayed stars: the fitted rating on osu!'s scale per keymode (calib/anchor.json, written by `calib anchor`). Only numbers
//! a player reads go through it; dans, predictions, lists and fits keep the fitted units, so nothing else moves.
use std::sync::OnceLock;

/// osu!-scale stars for a fitted rating of a `keys`-key chart or player (k clamped to the fitted 4–10).
pub fn stars(keys: i64, s: f64) -> f64 {
    static ANCHOR: OnceLock<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = OnceLock::new();
    let (poly, range) = ANCHOR.get_or_init(|| {
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../calib/anchor.json")).expect("anchor.json");
        (serde_json::from_value(v["poly"].clone()).expect("anchor poly"), serde_json::from_value(v["range"].clone()).expect("anchor range"))
    });
    curve(poly, range, keys, s)
}

/// The anchor curve: ln stars = a polynomial in l = ln s whose coefficients (`poly`, lowest power first) are quadratics in
/// k − 7, inside the keymode's fitted `range` of l (4K first). Past its top it continues along the tangent (a cubic would turn
/// down); below its bottom stars stay proportional to the rating, so 0 reads 0.
pub fn curve(poly: &[Vec<f64>], range: &[Vec<f64>], keys: i64, s: f64) -> f64 {
    if s <= 0. {
        return s;
    }
    let k = keys.clamp(4, 10);
    let x = k as f64 - 7.;
    let co: Vec<f64> = poly.iter().map(|c| c.iter().fold(0., |acc, v| acc * x + v)).collect();
    let (lo, hi) = (range[(k - 4) as usize][0], range[(k - 4) as usize][1]);
    let (l, c) = (s.ln(), s.ln().clamp(lo, hi));
    let slope = if l < lo { 1. } else { co.iter().enumerate().skip(1).rev().fold(0., |acc, (j, v)| acc * c + j as f64 * v) };
    (co.iter().rev().fold(0., |acc, v| acc * c + v) + slope * (l - c)).exp()
}

#[cfg(test)]
mod tests {
    /// A calculator ship moves the fitted scale: the anchor must be refitted with it (`calib anchor`).
    #[test]
    fn the_anchor_was_fitted_on_this_calculator() {
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../calib/anchor.json")).unwrap();
        assert_eq!(v["calc"], crate::engine::CALC_ID, "rerun calib anchor for the new calculator");
        for k in 4..=10 {
            assert!((1..300).all(|i| super::stars(k, i as f64 / 5.) < super::stars(k, i as f64 / 5. + 0.2)), "{k}K rises with the rating");
        }
        // moves with each calc's fitted scale (21660e31a4f6: LN_BASE 30 raised 7K units, 5.16 → 4.92)
        assert!((super::stars(7, 6.7) - 4.92).abs() < 0.2, "{}", super::stars(7, 6.7));
    }
}

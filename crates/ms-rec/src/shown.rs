//! Displayed stars (`ms_calc::anchor`): kept here under its old name for the app's callers.
pub use ms_calc::anchor::{curve, stars};

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

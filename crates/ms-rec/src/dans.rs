//! Dan equivalent of a chart: where it falls among the community dan charts (calib/dans.json). Anchors are the dan charts
//! themselves, rated by this calculator, not another calculator's scale (user 2026-10-01). Only chart content decides the
//! series and the place; titles never enter (user: "no cheating").
//!
//! Series: LN-heavy charts are placed among LN dans, the rest among rice dans; 4K charts with sustained same-column runs
//! are also placed among vibro dans. A chart mostly made of vibro runs shows the vibro dan alone: its mashing reads as fast
//! rice/LN to the calculator (user 2026-10-01), unless the notes outside the runs are dense on their own: REFORM
//! Iota/Kappa are 26/43 % vibro yet hard rice besides (user 2026-10-11). Vibro dans are placed by vibro speed together with
//! stars: stars alone put Vibro 2–8 within 8.5–9.2★ out of order.
use ms_data::codec::from_json;
use ms_calc::val::Val;
use ms_model::predict::{Consts, Feats};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

const TABLE: &str = include_str!("../../../calib/dans.json");
const COURSES: &str = include_str!("../../../calib/dan_courses.tsv");
pub const VIBRO_SHARE: f64 = 0.10; // share of notes in vibro runs: 117/120 vibro dan charts, 0.5% of ranked 4K (2026-10-01)
const LN_SHARE: f64 = 0.22;   // LN dans' charts hold >= 25% LN notes, rice dans' <= 19% (2026-10-01)
const LN_MAJORITY: f64 = 0.5; // from here holds are most notes: the LN dan alone

pub fn series(keys: i64, kind: &str) -> Option<&'static str> {
    match (keys, kind) {
        (4, "rice") => Some("4K REFORM"),
        (4, "ln") => Some("4K LN"),
        (6, "rice") => Some("6K Regular"),
        (6, "ln") => Some("6K LN"),
        (7, "rice") => Some("7K Regular"),
        (7, "ln") => Some("7K LN"),
        (10, "rice") => Some("10K Regular"),
        _ => None,
    }
}

pub fn short(series: &str) -> &'static str {
    match series {
        "7K Regular" => "Reg ",
        "7K LN" | "4K LN" => "LN ",
        "6K Regular" => "6K ",
        "6K LN" => "6K LN ",
        "10K Regular" => "10K ",
        _ => "",
    }
}

type Tiers = Vec<(String, f64)>;

struct Data {
    series: HashMap<String, Tiers>,
    weights: (f64, f64),
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let v: Value = serde_json::from_str(TABLE).unwrap_or(Value::Null);
        // Another calculator's ratings would misplace every chart.
        let v = if v["calc"] == crate::engine::CALC_ID { v } else { Value::Null };
        let series = v["series"].as_object().map_or_else(HashMap::new, |o| o.iter().map(|(k, tiers)| {
            (k.clone(), tiers.as_array().map_or_else(Vec::new, |t| t.iter().filter_map(|x| Some((x["tier"].as_str()?.to_string(), x["value"].as_f64()?))).collect()))
        }).collect());
        Data { series, weights: (v["vibro_weights"][0].as_f64().unwrap_or(1.), v["vibro_weights"][1].as_f64().unwrap_or(0.)) }
    })
}

/// (accuracy that clears the tier's dan, in stable's units?) from the `pass` column of calib/dan_courses.tsv: "96" stable
/// accuracy, "97v2" ScoreV2 (counts MAX above 300 like lazer). Each course's
/// own rule (checked 2026-11-11): Jinjin 7K regular 96 %, LN 95 % (forum t/304618); REFORM 96 % ("like other official
/// Reform Dan courses", set 1156299); CT's 10K 96 % normal clear (set 2084293); 4K LN v2 97 % ScoreV2 (set 891143). AXian's
/// 6K sets (2526107/128/129/142, 2531711, 2532062/64/67) name none: 96 % as the other regular dans. Staravia's Vibro
/// courses (563163/564238) ask only for a pass at HP 8.4–9.2 ("getting bad accuracy will not be very fatal"); the other
/// Vibro Dan Courses (973783/973787) state Beginner Clear 93 %, Expert 96 %: 93 %.
pub fn pass(series: &str, tier: &str) -> Option<(f64, bool)> {
    let row = COURSES.lines().skip(1).map(|l| l.split('\t').collect::<Vec<_>>()).find(|f| f.len() > 8 && f[0] == series && f[2] == tier)?;
    let (number, v2) = row[8].strip_suffix("v2").map_or((row[8], false), |n| (n, true));
    Some((number.parse::<f64>().ok()? / 100., !v2))
}

pub fn tiers(series: &str) -> Option<&'static Tiers> {
    data().series.get(series)
}

/// {tier: [(features at 1.0×, the skills it demands)]} of the series' dan songs (a course without songs of its own split at
/// its song boundaries), in tier order of the table.
pub fn songs(series: &str, c: &Consts) -> HashMap<String, Vec<(Feats, Vec<&'static str>)>> {
    // Read for Stats only: held for the app's life the parsed songs were 12 MB.
    let table: Value = serde_json::from_str(TABLE).unwrap_or(Value::Null);
    table["songs"][series].as_object().map_or_else(HashMap::new, |o| o.iter().map(|(tier, list)| {
        let feats = list.as_array().map_or_else(Vec::new, |l| l.iter().filter_map(|f| match from_json(f) {
            Val::Dict(d) => Some((Feats::of(&d, c).ok()?, ms_model::skills::demanded(&d))),
            _ => None,
        }).collect());
        (tier.clone(), feats)
    }).collect())
}

/// Rating at continuous tier index p (position's inverse), linear between and beyond the anchors.
pub fn rating_at(series: &str, p: f64) -> Option<f64> {
    let t = tiers(series).filter(|t| !t.is_empty())?;
    if t.len() == 1 {
        return Some(t[0].1.exp());
    }
    let j = (p.floor().max(0.) as usize).min(t.len() - 2);
    Some((t[j].1 + (p - j as f64) * (t[j + 1].1 - t[j].1)).exp())
}

pub use ms_calc::structure::{vibro_only, vibro_runs, Vibro};

fn value(series: &str, rating: f64, vibro: Option<&Vibro>) -> f64 {
    value_with(data().weights, series, rating, vibro)
}

/// The quantity a series' anchors are in: log stars, or for vibro dans weighted log speed + log stars.
pub fn value_with(weights: (f64, f64), series: &str, rating: f64, vibro: Option<&Vibro>) -> f64 {
    if series == "4K Vibro" {
        return weights.0 * vibro.map_or(1f64, |v| v.speed.max(1.)).ln() + weights.1 * rating.max(1e-3).ln();
    }
    rating.max(1e-3).ln()
}

/// Continuous tier index (0 = first anchor) of value v, linear between and beyond the anchors.
pub fn position(t: &[(String, f64)], v: f64) -> f64 {
    if t.len() == 1 {
        return 0.;
    }
    let j = if v <= t[0].1 { 0 } else if v >= t[t.len() - 1].1 { t.len() - 2 } else { (0..t.len() - 1).find(|&i| t[i].1 <= v && v <= t[i + 1].1).unwrap_or(0) };
    let (lo, hi) = (t[j].1, t[j + 1].1);
    if hi - lo < 1e-9 { j as f64 + 0.5 } else { j as f64 + (v - lo) / (hi - lo) }
}

/// Nearest dan, with −/+ when the chart sits more than a quarter of the way to the next one.
pub fn place(series: &str, rating: f64, vibro: Option<&Vibro>) -> Option<String> {
    named(series, position(tiers(series).filter(|t| !t.is_empty())?, value(series, rating, vibro)))
}

/// The dan at fractional tier p of the series.
pub fn named(series: &str, p: f64) -> Option<String> {
    let t = tiers(series).filter(|t| !t.is_empty())?;
    let prefix = short(series);
    if p < -0.5 {
        return Some(format!("below {prefix}{}", t[0].0));
    }
    if p > t.len() as f64 - 0.5 {
        return Some(format!("{prefix}{}+", t[t.len() - 1].0));
    }
    let j = (p.round_ties_even().max(0.) as usize).min(t.len() - 1);
    let off = p - j as f64;
    Some(format!("{prefix}{}{}", t[j].0, if off > 0.25 { "+" } else if off < -0.25 { "−" } else { "" }))
}

pub fn kind(ln_share: f64) -> &'static str {
    if ln_share >= LN_SHARE { "ln" } else { "rice" }
}

pub fn label(keys: i64, rating: f64, ln_share: f64, vibro: Option<&Vibro>) -> Option<String> {
    // Vibro last: the rice/LN dan is the main reading (user 2026-10-01).
    let vibro = vibro.filter(|_| keys == 4);
    // Mixed charts (LN dan share but mostly taps) read on both, the tap dans first (user 2026-10-01: a 29 % LN rice
    // stamina chart showed only "LN Gamma").
    let kinds: Vec<&str> = if !(LN_SHARE..LN_MAJORITY).contains(&ln_share) { vec![kind(ln_share)] } else { vec!["rice", "ln"] };
    let mut parts: Vec<String> = kinds.iter().filter(|_| !vibro.is_some_and(vibro_only)).filter_map(|k| place(series(keys, k)?, rating, None)).collect();
    parts.extend(vibro.filter(|v| v.share >= VIBRO_SHARE).and_then(|v| place("4K Vibro", rating, Some(v))));
    if parts.is_empty() { None } else { Some(parts.join(" · ")) }
}

/// The dan series kind a Stats skill is placed on: LN skills on the LN dans.
pub fn ability_kind(skill: &str) -> &'static str {
    ms_calc::taxonomy::family_label(skill.split('/').next().unwrap_or("")).map_or("rice", |f| f.1)
}

/// The dan a Stats value reads as: values come from the player's predicted accuracy on that skill's dan songs
/// (stats::dan_value), so the rating places directly. None for vibro (its dan is stats::Group::dan) and keymodes without dans.
pub fn ability(keys: i64, value: f64, skill: &str) -> Option<String> {
    if skill.ends_with("vibro") {
        return None;
    }
    place(series(keys, ability_kind(skill))?, value, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dan_tier_has_its_courses_pass_mark() {
        assert_eq!(pass("7K LN", "1st"), Some((0.95, true)));
        assert_eq!(pass("4K LN", "1st"), Some((0.97, false))); // ScoreV2
        assert_eq!(pass("4K REFORM", "Theta"), Some((0.96, true)));
        assert_eq!(pass("4K Vibro", "Vibro W"), Some((0.93, true)));
        for (series, tiers) in &data().series {
            for (tier, _) in tiers {
                assert!(pass(series, tier).is_some(), "{series} {tier}");
            }
        }
    }

    #[test]
    fn ladder_reads_its_own_anchors() {
        let t = tiers("7K Regular").expect("7K dans");
        assert_eq!(place("7K Regular", t[3].1.exp(), None), Some(format!("Reg {}", t[3].0)));
        assert!((rating_at("7K Regular", 3.).unwrap() - t[3].1.exp()).abs() < 1e-9);
    }

    #[test]
    fn a_long_vibro_run_counts_and_a_short_one_does_not() {
        let k = ms_calc::params::Knobs::default();
        let run = |n: usize| ms_calc::chart::Chart { keys: 4, t: (0..n).map(|i| i as f64 * 50.).collect(), end: (0..n).map(|i| i as f64 * 50.).collect(), col: vec![0; n], ..Default::default() };
        assert_eq!(vibro_runs(&run(10), 1., &k).share, 1.);
        assert_eq!(vibro_runs(&run(4), 1., &k).share, 0.);
    }
}

#[cfg(test)]
mod shipped_tests {
    use super::*;

    #[test]
    fn shipped_dan_table_is_stamped_for_this_calculator() {
        // A calculator change without a new table hides every dan reading.
        assert!(tiers("7K Regular").is_some_and(|t| !t.is_empty()));
    }
}

//! recdata.feats_batch: the stored feature dict of a chart at a rate, from the calculator's result.
use crate::background::background_map;
use crate::chart;
use crate::compute;
use crate::model;
use crate::params::Params;
use crate::py::round_digits;
use crate::val::{Dict, Key, Val};
use serde_json::Value;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

type Slot<T> = Arc<OnceLock<T>>;

/// The last few calculations by key. A caller that finds one running waits for it instead of repeating it.
struct Recent<T> {
    slots: Mutex<VecDeque<(u64, Slot<T>)>>,
    size: usize,
}

impl<T: Clone> Recent<T> {
    const fn new(size: usize) -> Self {
        Recent { slots: Mutex::new(VecDeque::new()), size }
    }

    fn get(&self, key: u64, make: impl FnOnce() -> T) -> T {
        let slot = {
            let mut slots = self.slots.lock().unwrap();
            match slots.iter().find(|s| s.0 == key) {
                Some(s) => s.1.clone(),
                None => {
                    let slot: Slot<T> = Arc::default();
                    slots.push_back((key, slot.clone()));
                    if slots.len() > self.size {
                        slots.pop_front();
                    }
                    slot
                }
            }
        };
        slot.get_or_init(make).clone()
    }
}

static SELECTED: Recent<Result<Arc<Dict>, String>> = Recent::new(4);

/// compute() for the selected chart. The card and the prediction ask for the same analysis at the same moment, from two
/// threads: this is one calculation of it. Batches stay on compute(): they would only churn these slots.
pub fn compute_selected(data: &[u8], parsed: &chart::Chart, rate: f64, sv: bool, prm: &Params) -> Result<Arc<Dict>, String> {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    (data, rate.to_bits(), sv, prm.id()).hash(&mut hash);
    SELECTED.get(hash.finish(), || compute::compute(parsed, rate, sv, prm).map(Arc::new))
}

/// Skills whose share of the overall a chart row stores (`sk`): what Stats, skill picks and the card read.
/// calib/difficulty.json: the frozen chart-only calibration.
pub fn standard_params() -> Params {
    let model: Value = serde_json::from_str(include_str!("../../../calib/difficulty.json")).expect("difficulty.json");
    // MS_KNOBS (JSON): rule knobs for a calibration screen (open loop: the old constants re-tuned with each new system). The app
    // never sets it; a screened pack goes to its own MANIASCOPE_REC.
    let knobs = std::env::var("MS_KNOBS").ok().and_then(|k| serde_json::from_str(&k).ok()).unwrap_or(Value::Null);
    Params::new(model, &knobs).expect("difficulty.json parameters")
}

/// The calibration a past calculator used: calib/carry.json holds what differs from today's, null for a key it lacked.
pub fn params_of(calc: &str) -> Result<Params, String> {
    fn merge(to: &mut Value, patch: &Value) {
        let (Some(to), Some(patch)) = (to.as_object_mut(), patch.as_object()) else { return };
        for (k, v) in patch {
            match (v.is_object(), to.get_mut(k)) {
                (_, _) if v.is_null() => { to.remove(k); }
                (true, Some(at)) => merge(at, v),
                _ => { to.insert(k.clone(), v.clone()); }
            }
        }
    }
    let patches: Value = serde_json::from_str(include_str!("../../../calib/carry.json")).map_err(|e| e.to_string())?;
    let mut model = standard_params().model;
    merge(&mut model, patches.get(calc).ok_or_else(|| format!("no calibration kept for {calc}"))?);
    Params::new(model, &Value::Null) // today's rules: the calibration differs, the rules can't be rebuilt
}

/// Rows stored under a calibration with display units (score_units, before 2026-11-04) back in rating units.
pub fn restore_feature(f: &mut Dict) {
    if let Some(Val::Dict(pre)) = f.pop("preunit_features") {
        for (k, v) in pre.0 {
            if let Key::S(k) = k {
                f.set(&k, v);
            }
        }
        f.pop("preunit_overall");
    }
}

/// A chart's stored features moved to `after`'s calibration, without its file; None if it doesn't move.
/// `sk`/`stam` are shares of the overall, so they move with it.
///
/// The joint stage reads the chart's per-hand flow, which older rows lack: `flow` finds it. No chart, no joint move.
pub fn recalibrated(f: &Dict, after: &Params, rate: f64, flow: &dyn Fn() -> Option<crate::flow::Flow>) -> Option<Dict> {
    let mut f = f.clone();
    restore_feature(&mut f);
    let keys = f.f("keys") as usize;
    let raw = f.fget("raw_overall", f.f("overall"));
    // The joint stage reads the chart's level: the one under `after`'s rules (it was read before the move, so wide keymodes'
    // carried rows got their level-dependent columns at the old rules' level, 2026-11-04).
    with_joint(&mut f, after, rate, raw, flow);
    // The rating `after` gives, computed, not the stored one times after/before: a fresh analysis matches final_rating to
    // ±1 %, but a row carried several times kept what each older calibration's ratio left behind (10K +.21 log, 2026-11-04).
    let (overall, baseline) = (
        model::final_rating(after, keys, raw, f.dict("execution")),
        model::baseline_rating(after, keys, raw, f.dict("execution")),
    );
    if !overall.is_finite() || overall <= 0. || (overall - f.f("overall")).abs() < 1e-12 {
        return None;
    }
    f.set("overall", overall);
    f.set("baseline_overall", baseline);
    Some(f)
}

/// A chart's rating under `after`'s rules over `before`'s (the chord-rule checks in `calib chordtest`).
pub fn rule_ratio(c: &crate::chart::Chart, rate: f64, before: &Params, after: &Params) -> Option<f64> {
    let overall = |p: &Params| crate::compute::compute(c, rate, true, p).ok()?.dict("scores").map(|s| s.f("overall"));
    Some(overall(after)? / overall(before)?).filter(|r| r.is_finite() && *r > 0.)
}

/// One stored row carried, with its chart when that can be found.
/// A row analysed under other rules (its `rules` stamp; none before 2026-11-05) is analysed again when its chart is found,
/// since a ratio would carry the old skill shares and stamina along. Without its file only the calibration moves and the
/// old stamp stays, so a later carry that finds the file still analyses it. The stamp is per row, not per calculator pair:
/// a row carried without its file once (stable charts until 2026-11-05) sat under each newer calculator's id with the
/// old analysis, which a calculator-level check can't see.
pub fn carried(f: &Dict, after: &Params, rate: f64, chart: &dyn Fn() -> Option<std::rc::Rc<crate::chart::Chart>>) -> Option<Dict> {
    let stale = !matches!(f.get("rules"), Some(Val::Str(s)) if *s == rules_stamp(after));
    let found = if stale { chart() } else { None };
    let Some(c) = found else {
        return recalibrated(f, after, rate, &|| chart().map(|c| crate::flow::flow(&c, rate, &after.k)));
    };
    let res = compute::compute(&c, rate, true, after).ok()?;
    let mut fresh = features_of(&res, after);
    for kept in ["stars", "hits"] { // lazer's own star rating, read from the file's bytes when the row was made
        if let Some(v) = f.get(kept) {
            fresh.set(kept, v.clone());
        }
    }
    Some(fresh)
}

/// The rules a row was analysed under: the rule knobs' hash (calibration moves without a new analysis).
pub fn rules_stamp(prm: &Params) -> String {
    use sha2::Digest;
    let knobs = serde_json::to_string(&prm.k.dump()).unwrap_or_default();
    sha2::Sha256::digest(knobs.as_bytes()).iter().take(6).map(|b| format!("{b:02x}")).collect()
}

fn with_joint(f: &mut Dict, after: &Params, rate: f64, raw: f64, flow: &dyn Fn() -> Option<crate::flow::Flow>) {
    let joint = &after.model["joint"]["modes"][(f.f("keys") as i64).to_string()];
    let length = f.fget("length", 0.);
    let calibration = f.get("calibration_features").map_or_else(Vec::new, Val::list_f);
    let od = f.fget("od", 8.);
    let Some(e) = f.dict_mut("execution").filter(|_| joint.is_object()) else { return };
    let Some(vector) = e.get("residual_vector").map(Val::list_f).filter(|v| !v.is_empty()) else { return };
    let stored = crate::flow::Flow::stored(e);
    // The chart's own flow first: a stored one is from the rules the row was analysed under (the middle key moved 2026-11-03).
    let Some(found) = flow().or(stored) else { return };
    found.store(e);
    e.set("joint_factor", model::joint_shift(joint, &crate::flow::joint_row(&vector, length, raw, rate, found, &crate::flow::extra_inputs(&calibration, e, od))).exp());
}

/// recdata._feats_at for a calculated result.
pub fn features_of(res: &Dict, prm: &Params) -> Dict {
    let demand = res;
    let scores = demand.dict("scores").cloned().unwrap_or_default();
    let o = scores.get("overall").filter(|v| v.truthy()).cloned().unwrap_or(Val::Int(0));
    let overall = o.f();
    let notes = res.get("notes").cloned().unwrap_or(Val::Int(0));
    let mut sk = Dict::new();
    if overall != 0. {
        for k in crate::taxonomy::kept().map(|s| s.name()) {
            if let Some(v) = scores.get(k).filter(|v| v.truthy()) {
                sk.set(k, round_digits(v.f() / overall, 4));
            }
        }
    }
    let mut f = Dict::new();
    f.set("keys", res.get("keys").cloned().unwrap_or(Val::Int(0)));
    f.set("overall", o);
    f.set("od", res.get("od").cloned().unwrap_or(Val::F(0.)));
    f.set("ln", res.f("ln_notes") / notes.f().max(1.));
    f.set("notes", notes);
    f.set("length", res.get("length").cloned().unwrap_or(Val::Int(0)));
    // Sustained share: the stamina rating (minutes-long windows) over the overall, 1 for an unrated chart.
    let stamina = if overall != 0. { scores.fget("stamina", overall) / overall } else { 1. };
    f.set("stam", stamina);
    f.set("sk", Val::Dict(sk));
    f.set("description", if res.fget("notes", 0.) >= 2. { crate::card::describe(res, &prm.k).0 } else { "Sparse".to_string() });
    let or_empty = |k: &str| res.get(k).cloned().unwrap_or_else(|| Val::Dict(Dict::new()));
    f.set("endurance", or_empty("endurance"));
    f.set("raw_overall", res.get("raw_overall").cloned().unwrap_or_else(|| Val::F(overall)));
    f.set("calibration_features", res.get("calibration_features").cloned().unwrap_or(Val::List(Vec::new())));
    f.set("execution", or_empty("execution"));
    if let Some(s) = res.get("structure").filter(|v| v.truthy()) {
        f.set("nps", s.clone());
    }
    f.set("baseline_overall", demand.get("baseline_overall").cloned().unwrap_or_else(|| Val::F(overall)));
    f.set("tech_profile", or_empty("tech_profile"));
    f.set("archetype_profile", or_empty("archetype_profile"));
    f.set("rules", rules_stamp(prm));
    f
}

/// (lazer stars, judgement count) at a ranked rate; None for a chart rosu-pp cannot read.
fn lazer_stars(map: &rosu_pp::Beatmap, rate: f64) -> Option<(f64, u32)> {
    let mods = if rate == 0.75 { 256 } else if rate == 1.5 { 64 } else { 0 };
    let d = rosu_pp::Difficulty::new().mods(mods).lazer(true);
    match if [0.75, 1., 1.5].contains(&rate) { d } else { d.clock_rate(rate) }.calculate(map) {
        rosu_pp::any::DifficultyAttributes::Mania(a) => Some((a.stars, a.n_objects + a.n_hold_notes)),
        _ => None,
    }
}

/// lazer's star rating and hit count of a chart at any rate (the card's pp at a rate the analysis didn't star).
pub fn stars_at(data: &[u8], rate: f64) -> Option<(f64, u32)> {
    lazer_stars(&rosu_pp::Beatmap::from_bytes(data).ok()?, rate)
}

/// {rate: features} of one chart's bytes.
pub fn chart_features(data: &[u8], rates: &[f64], prm: &Params) -> Result<Vec<(f64, Dict)>, String> {
    features_with(data, rates, prm, &|parsed, rate| compute::compute(parsed, rate, true, prm).map(Arc::new))
}

/// chart_features of the selected chart, on the card's calculation.
pub fn selected_features(data: &[u8], rate: f64, prm: &Params) -> Result<Vec<(f64, Dict)>, String> {
    features_with(data, &[rate], prm, &|parsed, rate| compute_selected(data, parsed, rate, true, prm))
}

fn features_with(data: &[u8], rates: &[f64], prm: &Params, calculate: &dyn Fn(&chart::Chart, f64) -> Result<Arc<Dict>, String>) -> Result<Vec<(f64, Dict)>, String> {
    let parsed = chart::parse(data).map_err(|e| match e {
        chart::ChartError::Chart(m) | chart::ChartError::Value(m) => m,
    })?;
    let mut out = Vec::with_capacity(rates.len());
    for &rate in rates {
        let res = calculate(&parsed, rate)?;
        out.push((rate, features_of(&res, prm)));
    }
    if out.iter().any(|x| matches!(x.0, r if r == 0.75 || r == 1. || r == 1.5)) {
        if let Ok(map) = rosu_pp::Beatmap::from_bytes(data) {
            for (rate, f) in &mut out {
                if let Some((stars, hits)) = lazer_stars(&map, *rate) {
                    f.set("stars", stars);
                    f.set("hits", Val::Int(i64::from(hits)));
                }
            }
        }
    }
    Ok(out)
}

/// chart_features for many charts on `threads` background threads (0 = default); a broken chart is its own error.
pub fn features_batch(jobs: &[(Vec<u8>, Vec<f64>)], threads: usize, prm: &Params) -> Vec<Result<Vec<(f64, Dict)>, String>> {
    background_map(jobs, threads, |(data, rates)| chart_features(data, rates, prm))
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn callers_at_the_same_moment_share_one_calculation() {
        static RECENT: Recent<u64> = Recent::new(2);
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        let threads: Vec<_> = (0..8).map(|_| std::thread::spawn(|| RECENT.get(1, || {
            RUNS.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(50));
            7
        }))).collect();
        assert!(threads.into_iter().all(|t| t.join().unwrap() == 7));
        assert_eq!(RUNS.load(Ordering::SeqCst), 1);
        // Another key leaves it, the slots stay bounded.
        [2, 3].iter().for_each(|&k| { RECENT.get(k, || k); });
        assert_eq!(RECENT.slots.lock().unwrap().len(), 2);
        assert_eq!(RECENT.get(1, || 9), 9);
    }

    #[test]
    fn the_selected_chart_keeps_the_batch_numbers() {
        let mut text = "osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[HitObjects]\n".to_string();
        (0..400).for_each(|i| text += &format!("{},192,{},1,0,0:0:0:0:\n", 64 + 128 * (i % 4), 1000 + i * 90));
        let prm = standard_params();
        let c = chart::parse(text.as_bytes()).unwrap();
        assert_eq!(*compute_selected(text.as_bytes(), &c, 1.1, true, &prm).unwrap(), compute::compute(&c, 1.1, true, &prm).unwrap());
        assert_eq!(selected_features(text.as_bytes(), 1.1, &prm).unwrap(), chart_features(text.as_bytes(), &[1.1], &prm).unwrap());
        // A different calculator is a different key.
        let mut model = prm.model.clone();
        model["joint"]["limit"] = serde_json::json!(0.1);
        assert_ne!(Params::new(model, &Value::Null).unwrap().id(), prm.id());
        assert_eq!(standard_params().id(), prm.id());
        // Stars at any rate (the card's pp), between the rated rates and equal to them on them.
        let (s1, s15, s11) = (stars_at(text.as_bytes(), 1.).unwrap().0, stars_at(text.as_bytes(), 1.5).unwrap().0, stars_at(text.as_bytes(), 1.1).unwrap().0);
        assert!(s1 < s11 && s11 < s15, "{s1} {s11} {s15}");
        assert_eq!(stars_at(text.as_bytes(), 1.5).unwrap().0, rosu_stars_dt(text.as_bytes()));
    }

    fn rosu_stars_dt(data: &[u8]) -> f64 {
        match rosu_pp::Difficulty::new().mods(64).lazer(true).calculate(&rosu_pp::Beatmap::from_bytes(data).unwrap()) {
            rosu_pp::any::DifficultyAttributes::Mania(a) => a.stars,
            _ => unreachable!(),
        }
    }
}

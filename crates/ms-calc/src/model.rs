//! Chart-only fitted stages: the population link (modes) and the families fit (joint) with its descriptors.
use crate::py;
use crate::val::{Dict, Val};
use crate::params::Params;
use serde_json::Value;

pub const FEATURES: [&str; 12] = [
    "ln", "chordjack", "delay", "minijack", "technical", "sv", "endurance", "cj_endurance", "wide_jacks", "od_rice",
    "od_ln", "od_jack",
];
pub const EXTRA_FEATURES: [&str; 6] = ["mash", "pseudo_mash", "ln_flow", "smooth_switch", "peak_pressure", "transition_pressure"];

fn num(v: &Value, default: f64) -> f64 {
    v.as_f64().unwrap_or(default)
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()).unwrap_or_default()
}
fn mode<'a>(m: &'a Value, stage: Option<&str>, keys: usize) -> &'a Value {
    let s = match stage {
        Some(st) => &m[st],
        None => m,
    };
    &s["modes"][keys.to_string()]
}
fn present(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Object(o) => !o.is_empty(),
        _ => true,
    }
}

/// Hand-local fatigue (difficulty_model.endurance): ({load, peak}, timeline).
pub fn endurance(fines: &[&[Vec<f64>; 3]], bin_seconds: f64, phases: usize) -> ((f64, f64), Vec<f64>) {
    let mut summaries: Vec<f64> = Vec::new();
    let mut tracks: Vec<Vec<f64>> = Vec::new();
    for fine in fines {
        for side in &fine[..2] {
            let values: Vec<f64> = side.chunks(phases).map(|c| py::sum(c.iter().copied())).collect();
            let n = 1i64.max(py::round(30.0 / bin_seconds) as i64) as usize;
            let mut rolling = 0.0f64;
            let mut peak = 0.0f64;
            for (i, &v) in values.iter().enumerate() {
                rolling += v - if i >= n { values[i - n] } else { 0.0 };
                peak = peak.max(rolling / n as f64);
            }
            if peak <= 0.0 {
                tracks.push(vec![0.0; values.len()]);
                summaries.push(0.0);
                continue;
            }
            let (mut state, mut weighted, mut total) = (0.0f64, 0.0f64, 0.0f64);
            let mut track = Vec::with_capacity(values.len());
            for &v in &values {
                let load = 3f64.min(0f64.max(v / peak));
                let target = 0f64.max(load - 0.2).powf(1.3);
                let tau = if target > state { 45.0 } else { 7.0 + 20.0 * 1f64.min(load) };
                let decay = (-bin_seconds / tau).exp();
                state = target + (state - target) * decay;
                track.push(state);
                weighted += v * state;
                total += v;
            }
            tracks.push(track);
            summaries.push(if total != 0.0 { weighted / total } else { 0.0 });
        }
    }
    if summaries.is_empty() {
        return ((0.0, 0.0), Vec::new());
    }
    let per: Vec<f64> = summaries.chunks(2).map(|c| c.iter().cloned().fold(f64::NEG_INFINITY, |m, x| if x > m { x } else { m })).collect();
    let nruns = fines.len();
    let timeline: Vec<f64> = (0..tracks[0].len())
        .map(|i| py::sum((0..tracks.len()).step_by(2).map(|j| tracks[j][i].max(tracks[j + 1][i]))) / 1usize.max(nruns) as f64)
        .collect();
    let peak = timeline.iter().cloned().fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x })).unwrap_or(0.0);
    ((py::sum(per.iter().copied()) / per.len() as f64, peak), timeline)
}

/// The skills `vector` reads from physical_sk (compute rates only these before execution::attribute).
pub const VECTOR_SKILLS: [&str; 8] = ["ln", "chordjack", "jackspeed", "delay", "dump", "minijack", "technical", "sv"];

/// difficulty_model.vector.
pub fn vector(res: &Dict, od: f64) -> [f64; 12] {
    let mut sk: Vec<(String, f64)> = Vec::new();
    if let Some(p) = res.dict("physical_sk").filter(|d| !d.0.is_empty()) {
        for (k, v) in &p.0 {
            if let crate::val::Key::S(k) = k {
                sk.push((k.clone(), 0f64.max(1.5f64.min(v.f()))));
            }
        }
    } else {
        let sc = res.dict("scores").unwrap();
        let o = sc.f("overall");
        for (k, v) in &sc.0 {
            if let crate::val::Key::S(k) = k {
                sk.push((k.clone(), 0f64.max(1.5f64.min(v.f() / 0.05f64.max(o)))));
            }
        }
    }
    let get = |k: &str| sk.iter().find(|(kk, _)| kk == k).map(|x| x.1);
    let ln = get("ln").unwrap_or(0.0);
    let cj = get("chordjack").unwrap_or(0.0).max(get("jackspeed").unwrap_or(0.0));
    let load = res.dict("endurance").map_or(0.0, |e| e.f("load"));
    let strict = (40.0 / 10f64.max(64.0 - 3.0 * 0f64.max(10f64.min(od)))).ln();
    [
        ln,
        cj,
        get("delay").unwrap_or_else(|| get("dump").unwrap_or(0.0)),
        get("minijack").unwrap_or(0.0),
        get("technical").unwrap_or(0.0),
        get("sv").unwrap_or(0.0),
        load,
        load * cj,
        res.fget("jack_width", 0.0),
        strict * 0f64.max(1.0 - ln.max(cj)),
        strict * ln,
        strict * cj,
    ]
}

/// Each keymode's scale: the population link from raw demand to its rating (`modes.<keys>.scale`).
fn scale(prm: &Params, keys: usize) -> f64 {
    num(&mode(&prm.model, None, keys)["scale"], 1.0)
}

/// The rating before the families shift. A scroll-read chart keeps the ratio of its raw demand to its no-SV demand.
pub fn baseline_rating(prm: &Params, keys: usize, raw: f64, execution: Option<&Dict>) -> f64 {
    if let Some(Val::Dict(physical)) = execution.and_then(|e| e.get("scroll_base")) {
        let praw = physical.f("raw");
        if praw > 0.0 {
            return raw / praw * baseline_rating(prm, keys, praw, physical.dict("execution"));
        }
    }
    raw * scale(prm, keys)
}

pub fn final_rating(prm: &Params, keys: usize, raw: f64, execution: Option<&Dict>) -> f64 {
    let base = baseline_rating(prm, keys, raw, execution);
    final_from_base(base, execution)
}

fn final_from_base(base: f64, execution: Option<&Dict>) -> f64 {
    base * execution.map_or(1.0, |e| e.fget("joint_factor", 1.0))
}

/// The joint stage: stages fitted one over another each kept its credit, so one refit over all their inputs corrects what they
/// left (calib joint). Sparse: `columns` index `flow::joint_row`.
const JOINT_LIMIT: f64 = 0.35;

pub fn joint_shift(model: &Value, row: &[f64]) -> f64 {
    let wide;
    let row = if model["spec"].is_object() { wide = crate::flow::extended(row, &model["spec"]); &wide[..] } else { row };
    let (lo, hi, mean, sd, coeff) = (nums(&model["low"]), nums(&model["high"]), nums(&model["mean"]), nums(&model["sd"]), nums(&model["coeff"]));
    let at = model["columns"].as_array().map_or_else(Vec::new, |c| c.iter().filter_map(Value::as_u64).map(|c| c as usize).collect());
    let shift: f64 = (0..at.len()).filter(|&i| sd[i] >= 1e-9).map(|i| coeff[i] * (row[at[i]].clamp(lo[i], hi[i]) - mean[i]) / sd[i]).sum(); // a column flat in the fit reads 0
    let limit = model["limit"].as_f64().unwrap_or(JOINT_LIMIT); // the families fit carries every stage's share, so it reaches further
    shift.clamp(-limit, limit)
}

// ---- descriptors (the families fit reads them) ----

pub const SR_SKILLS: [&str; 25] = [
    "stream", "jumpstream", "handstream", "delay", "chordstream", "bracket", "chordjack", "minijack", "longjack",
    "jackspeed", "ln", "release", "inverse", "hybrid", "shield", "jumptrill", "splittrill", "trill1h", "anchor",
    "technical", "patterntech", "rhythmtech", "vibro", "stamina", "sv",
];
pub const SR_EXECUTION: [&str; 11] = [
    "mash", "pseudo_mash", "ln_flow", "peak_pressure", "transition_pressure", "vibro_easy", "vibro_control",
    "jumptrill_easy", "roll_control", "roll_anchor", "jumptrill_sustain",
];
pub const SR_NAMES_LEN: usize = 25 + 14 + 11 + 10;

/// structural_residual.NAMES: the descriptors of `sr_vector`, in order.
pub fn sr_names() -> Vec<String> {
    let skills = ["stream", "jumpstream", "handstream", "delay", "chordstream", "bracket", "chordjack", "minijack", "longjack", "jackspeed", "ln", "release", "inverse", "hybrid", "shield",
                  "jumptrill", "splittrill", "trill1h", "anchor", "technical", "patterntech", "rhythmtech", "vibro", "stamina", "sv"];
    let plain = ["hold_fraction", "sustain_ratio", "log_duration", "log_density_per_key", "chord_width", "overlap", "repeat", "burst", "active", "rhythm_variation", "all_columns",
                 "fatigue_load", "fatigue_peak", "wide_jack_pressure"];
    let execution = ["mash", "pseudo_mash", "ln_flow", "peak_pressure", "transition_pressure", "vibro_easy", "vibro_control", "jumptrill_easy", "roll_control", "roll_anchor", "jumptrill_sustain"];
    let crossed = ["chordstream:bracket", "chordjack:overlap", "chordjack:width", "chordjack:sustain", "minijack:density", "ln:release", "ln:hybrid", "ln:flow", "rice:anchor", "trill:control"];
    skills.iter().map(|s| format!("skill:{s}")).chain(plain.iter().map(|s| s.to_string())).chain(execution.iter().map(|s| format!("execution:{s}"))).chain(crossed.iter().map(|s| s.to_string())).collect()
}

/// structural_residual.vector from the apply() feature view.
/// structural_residual.vector on a feature dict: stable compact descriptors; scalar calibration cancels from skill ratios.
pub fn feature_vector(f: &Dict) -> Vec<f64> {
    let empty = Dict::new();
    let sk: Vec<(String, f64)> = f.dict("sk").map(|d| d.0.iter().filter_map(|(k, v)| match k {
        crate::val::Key::S(k) => Some((k.clone(), v.f())),
        _ => None,
    }).collect()).unwrap_or_default();
    let cf = f.get("calibration_features").map(|v| v.list_f()).unwrap_or_default();
    sr_vector(&sk, f.dict("nps").unwrap_or(&empty), f.dict("execution").unwrap_or(&empty), f.dict("endurance").unwrap_or(&empty), f.fget("rate", 1.0), f.fget("length", 0.0),
              f.fget("notes", 0.0), f.f("keys") as usize, f.fget("ln", 0.0), f.fget("stam", 1.0), &cf)
}

pub fn sr_vector(sk_raw: &[(String, f64)], st: &Dict, ex: &Dict, en: &Dict, rate: f64, length: f64, notes: f64, keys: usize, ln_frac: f64, stam: f64, cf: &[f64]) -> Vec<f64> {
    let sk: Vec<(&str, f64)> = sk_raw.iter().map(|(k, v)| (k.as_str(), py::round_digits(0f64.max(1.5f64.min(*v)), 4))).collect();
    let g = |k: &str| sk.iter().find(|(kk, _)| *kk == k).map_or(0.0, |x| x.1);
    let truthy = |k: &str| st.get(k).filter(|v| v.truthy()).map(|v| v.f());
    let seconds = 0.001f64.max(truthy("play_span").unwrap_or(length / 1000.0) / rate);
    let density = truthy("nps").unwrap_or(notes / seconds) / 1usize.max(keys) as f64;
    let width = 0f64.max(st.fget("chord", 1.0)) / 1usize.max(keys) as f64;
    let overlap = 0f64.max(st.fget("overlap", 0.0));
    let load = 0f64.max(en.fget("load", 0.0));
    let (ln, cj) = (g("ln"), g("chordjack"));
    let rice = g("stream").max(g("delay")).max(g("dump"));
    let mut values: Vec<f64> = SR_SKILLS.iter().map(|k| g(k)).collect();
    values.extend([
        ln_frac,
        2f64.min(stam),
        (seconds / 120.0).ln_1p(),
        density.ln_1p(),
        width,
        overlap,
        st.fget("repeat", 0.0),
        5f64.min(st.fget("burst", 1.0)),
        st.fget("active", 0.0),
        5f64.min(st.fget("rhythm", 0.0)),
        st.fget("all_keys", 0.0),
        load,
        en.fget("peak", 0.0),
        if cf.len() > 8 { cf[8] } else { 0.0 },
    ]);
    values.extend(SR_EXECUTION.iter().map(|k| ex.fget(k, 0.0)));
    values.extend([
        g("chordstream") * g("bracket"),
        cj * overlap,
        cj * width,
        cj * load,
        g("minijack") * density.ln_1p(),
        ln * g("release"),
        ln * g("hybrid"),
        ln * ex.fget("ln_flow", 0.0),
        rice * g("anchor"),
        g("jumptrill") * ex.fget("roll_control", 0.0),
    ]);
    values.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect()
}

/// difficulty_model.apply on the result dict.
pub fn apply(prm: &Params, res: &mut Dict, od: f64, rate: f64) {
    let keys = res.f("keys") as usize;
    let raw = res.dict("scores").unwrap().f("overall");
    res.set("raw_overall", raw);
    let cf = vector(res, od);
    res.set("calibration_features", Val::floats(&cf));
    if raw != 0.0 && res.has("structure") {
        let sk: Vec<(String, f64)> = res.dict("scores").unwrap().0.iter().map(|(k, v)| {
            let crate::val::Key::S(k) = k else { unreachable!() };
            (k.clone(), py::round_digits(v.f() / raw, 4))
        }).collect();
        let notes = res.f("notes");
        let empty = Dict::new();
        let stam = res.dict("scores").unwrap().fget("stamina", raw) / raw;
        let rv = sr_vector(
            &sk,
            res.dict("structure").unwrap_or(&empty),
            res.dict("execution").unwrap_or(&empty),
            res.dict("endurance").unwrap_or(&empty),
            rate,
            res.fget("length", 0.0),
            notes,
            keys,
            res.f("ln_notes") / 1f64.max(notes),
            stam,
            &cf,
        );
        let joint = &prm.model["joint"]["modes"][keys.to_string()];
        if present(joint) {
            let e = res.dict("execution").unwrap();
            let f = crate::flow::Flow { broken: e.fget("flow_broken", 0.0), skip: e.fget("flow_skip", 0.0), change: e.fget("flow_change", 0.0), fast_ln: e.fget("fast_ln", 0.0), split: e.fget("flow_split", 0.0) };
            let factor = joint_shift(joint, &crate::flow::joint_row(&rv, res.fget("length", 0.0), raw, rate, f, &crate::flow::extra_inputs(&cf, e, od))).exp(); // `od`, not res["od"]: that is set below, so the window column read OD 8 (fixed 2026-11-04)
            res.dict_mut("execution").unwrap().set("joint_factor", factor);
        }
        res.dict_mut("execution").unwrap().set("residual_vector", Val::floats(&rv));
    }
    let base = baseline_rating(prm, keys, raw, res.dict("execution"));
    res.set("baseline_overall", base);
    let multiplier = if raw != 0.0 { final_from_base(base, res.dict("execution")) / 0.000001f64.max(raw) } else { 1.0 };
    res.dict_mut("scores").unwrap().map_values(|v| v * multiplier);
    scale_archetypes(res, multiplier);
    if let Some(Val::List(tl)) = res.get_mut("timeline") {
        for v in tl.iter_mut() {
            *v = Val::F(v.f() * multiplier);
        }
    }
    res.set("od", od);
    let sc = res.dict("scores").unwrap().f("overall");
    res.set("structural_correction", sc - res.f("baseline_overall"));
}

pub fn scale_archetypes(res: &mut Dict, k: f64) {
    if let Some(Val::List(items)) = res.get_mut("archetypes") {
        for a in items.iter_mut() {
            if let Val::Dict(d) = a {
                let r = d.f("rating");
                d.set("rating", r * k);
            }
        }
    }
}

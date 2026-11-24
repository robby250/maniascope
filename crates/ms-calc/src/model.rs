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
pub fn endurance(fines: &[&[Vec<f64>; 3]], bin_seconds: f64, phases: usize, ref_q: f64) -> ((f64, f64), Vec<f64>) {
    endurance_of(&endurance_series(fines, phases), bin_seconds, ref_q)
}

/// Each run side's load per bin, the endurance's only input (calib curves keeps it to re-rate endurance knobs).
pub fn endurance_series(fines: &[&[Vec<f64>; 3]], phases: usize) -> Vec<Vec<f64>> {
    fines.iter().flat_map(|fine| fine[..2].iter().map(|side| side.chunks(phases).map(|c| py::sum(c.iter().copied())).collect())).collect()
}

/// `ref_q`: the quantile of the side's 30 s rolling loads its loads are measured against (1 = its hardest 30 s).
pub fn endurance_of(series: &[Vec<f64>], bin_seconds: f64, ref_q: f64) -> ((f64, f64), Vec<f64>) {
    let mut summaries: Vec<f64> = Vec::new();
    let mut tracks: Vec<Vec<f64>> = Vec::new();
    for values in series {
        {
            let n = 1i64.max(py::round(30.0 / bin_seconds) as i64) as usize;
            let mut rolling = 0.0f64;
            let mut means = Vec::with_capacity(values.len());
            for (i, &v) in values.iter().enumerate() {
                rolling += v - if i >= n { values[i - n] } else { 0.0 };
                means.push(rolling / n as f64);
            }
            let peak = if ref_q >= 1. { means.iter().copied().fold(0.0f64, f64::max) } else {
                let mut m: Vec<f64> = means.into_iter().filter(|&x| x > 0.).collect();
                m.sort_by(f64::total_cmp);
                m.get(((m.len() as f64 - 1.) * ref_q).round().max(0.) as usize).copied().unwrap_or(0.)
            };
            if peak <= 0.0 {
                tracks.push(vec![0.0; values.len()]);
                summaries.push(0.0);
                continue;
            }
            let rising = (-bin_seconds / 45.0).exp();
            let (mut state, mut weighted, mut total) = (0.0f64, 0.0f64, 0.0f64);
            let mut track = Vec::with_capacity(values.len());
            for &v in values {
                let load = 3f64.min(0f64.max(v / peak));
                let target = 0f64.max(load - 0.2).powf(1.3);
                // rising: one decay for the whole chart, made once (it called exp per bin)
                let decay = if target > state { rising } else { (-bin_seconds / (7.0 + 20.0 * 1f64.min(load))).exp() };
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
    let nruns = series.len() / 2;
    let timeline: Vec<f64> = (0..tracks[0].len())
        .map(|i| py::sum((0..tracks.len()).step_by(2).map(|j| tracks[j][i].max(tracks[j + 1][i]))) / 1usize.max(nruns) as f64)
        .collect();
    let peak = timeline.iter().cloned().fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x })).unwrap_or(0.0);
    ((py::sum(per.iter().copied()) / per.len() as f64, peak), timeline)
}

/// The skills `vector` reads from physical_sk (rated first, after the slices, before the card's modifiers).
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
/// Share of the families correction a chart at `level` keeps: all of it up to `width` under its keymode's fitted `top`
/// (99.5 % of the keymode's levels; a pooled top faded 10K's ordinary charts), none from the top on, where the fitted
/// constant (`top_shift`) takes over. The public scores hardly reach the top, so there the correction extrapolates and reordered the hardest
/// charts (REFORM Iota under Theta, 2026-11-14; user: "at that difficulty … osu star rating rates them well"): above it the
/// raw demand alone orders them. `width` 0: no fade.
pub fn family_fade(level: f64, top: f64, width: f64) -> f64 {
    if width <= 0. { 1. } else { ((top - level) / width).clamp(0., 1.) }
}

/// Share of the fitted top constant a chart at `level` keeps: all of it up to the top, then less over `width` levels
/// past it, where no public score reaches (user 2026-11-16: "a curve at the top where you slowly remove the lift").
/// `width` 0: held.
pub fn top_taper(level: f64, top: f64, width: f64) -> f64 {
    if width <= 0. { 1. } else { (1. - (level - top).max(0.) / width).clamp(0., 1.) }
}

/// Share of the top constant a chart played at log rate `log_rate` gets when the bundle ties it to speed (`top_rate` = the
/// share kept at 1.0× and slower): from that share at 1.0× to all of it at DT. Over each keymode's top the public scores are DT plays of shorter charts (4K dev:
/// no 1.0× row past level 1.07), so the constant they fit is what speeding a hard chart up costs, not what a 1.0× marathon
/// past them does (REFORM finals 2026-11-16).
pub fn top_rate_weight(log_rate: f64, floor: f64) -> f64 {
    floor + (1. - floor) * (log_rate / 1.5f64.ln()).clamp(0., 1.)
}

pub fn joint_shift(model: &Value, row: &[f64]) -> f64 {
    let (lo, hi, mean, sd, coeff) = (nums(&model["low"]), nums(&model["high"]), nums(&model["mean"]), nums(&model["sd"]), nums(&model["coeff"]));
    let at = model["columns"].as_array().map_or_else(Vec::new, |c| c.iter().filter_map(Value::as_u64).map(|c| c as usize).collect());
    // Products read the level inside its fitted range: level² kept falling past level's edge while density sat at its own,
    // so REFORM Iota (level 1.44 > 1.38) fell under Theta (round 27). Only the level, only in products: clamping every
    // product input, or the arms' level weight too, cost 4K DevSelect +.0015; refitting with it pooled Theta/Iota again.
    let mut pin = row.to_vec();
    if let Some(i) = at.iter().position(|&c| c == crate::flow::LEVEL).filter(|_| row.len() > crate::flow::LEVEL) {
        pin[crate::flow::LEVEL] = row[crate::flow::LEVEL].clamp(lo[i], hi[i]);
    }
    let wide;
    let row = if model["spec"].is_object() { wide = crate::flow::extended_with(row, &pin, &model["spec"]); &wide[..] } else { row };
    let shift: f64 = (0..at.len()).filter(|&i| sd[i] >= 1e-9).map(|i| coeff[i] * (row[at[i]].clamp(lo[i], hi[i]) - mean[i]) / sd[i]).sum(); // a column flat in the fit reads 0
    let top = model["top"].as_f64(); // the keymode's own top level (the families fit writes it with `fade`)
    // faded toward the fitted top constant (`top_shift`, calib joint's "top" column), not 0: Eta under 4K's top kept ×1.15 while Theta over it got
    // ×1.0, so the order of the hardest REFORM courses followed how much correction each kept (user 2026-11-15)
    let keep = top.zip(model["fade"].as_f64()).map_or(1., |(top, w)| family_fade(row[crate::flow::LEVEL], top, w));
    let mut taper = top.zip(model["taper"].as_f64()).map_or(1., |(top, w)| top_taper(row[crate::flow::LEVEL], top, w));
    if let Some(floor) = model["top_rate"].as_f64() {
        taper *= row.get(crate::flow::LEVEL + 1 + crate::flow::EXTRA.len()).map_or(1., |&r| top_rate_weight(r, floor));
    }
    let shift = keep * shift + (1. - keep) * taper * model["top_shift"].as_f64().unwrap_or(0.);
    model["limit"].as_f64().map_or(shift, |l| shift.clamp(-l, l)) // the families fit writes its limit (calib joint FAMILY_LIMIT)
}

// ---- descriptors (the families fit reads them) ----

pub const SR_SKILLS: [&str; 25] = [
    "stream", "jumpstream", "handstream", "delay", "chordstream", "bracket", "chordjack", "minijack", "longjack",
    "jackspeed", "ln", "ln_release", "ln_inverse", "ln_tech", "shield", "jumptrill", "splittrill", "trill1h", "anchor",
    "technical", "patterntech", "rhythmtech", "vibro", "stamina", "sv",
];
pub const SR_EXECUTION: [&str; 11] = [
    "mash", "pseudo_mash", "ln_flow", "peak_pressure", "transition_pressure", "vibro_easy", "vibro_control",
    "jumptrill_easy", "roll_control", "roll_anchor", "jumptrill_sustain",
];
pub const SR_NAMES_LEN: usize = 25 + 14 + 11 + 10;

/// structural_residual.NAMES: the descriptors of `sr_vector`, in order.
pub fn sr_names() -> Vec<String> {
    let skills = ["stream", "jumpstream", "handstream", "delay", "chordstream", "bracket", "chordjack", "minijack", "longjack", "jackspeed", "ln", "ln_release", "ln_inverse", "ln_tech", "shield",
                  "jumptrill", "splittrill", "trill1h", "anchor", "technical", "patterntech", "rhythmtech", "vibro", "stamina", "sv"];
    let plain = ["hold_fraction", "sustain_ratio", "log_duration", "log_density_per_key", "chord_width", "overlap", "repeat", "burst", "active", "rhythm_variation", "all_columns",
                 "fatigue_load", "fatigue_peak", "wide_jack_pressure"];
    let execution = ["mash", "pseudo_mash", "ln_flow", "peak_pressure", "transition_pressure", "vibro_easy", "vibro_control", "jumptrill_easy", "roll_control", "roll_anchor", "jumptrill_sustain"];
    let crossed = ["chordstream:bracket", "chordjack:overlap", "chordjack:width", "chordjack:sustain", "minijack:density", "ln:release", "ln:tech", "ln:flow", "rice:anchor", "trill:control"];
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
    let own = |k: &str| sk.iter().find(|(kk, _)| *kk == k).map_or(0.0, |x| x.1);
    // A base skill's column reads the strongest of its rows (the base keeps what its slices leave, plan 2026-11-11).
    let g = |k: &str| crate::skills::Sk::from_name(k).filter(|s| crate::taxonomy::measured(*s, keys)).map_or(0.0, |s| crate::taxonomy::rows_of_base(s).map(|r| own(r.name())).fold(own(k), f64::max));
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
        ln * g("ln_release"),
        ln * g("ln_tech"),
        ln * ex.fget("ln_flow", 0.0),
        rice * g("anchor"),
        g("jumptrill") * ex.fget("roll_control", 0.0),
    ]);
    values.into_iter().map(|v| if v.is_finite() { v } else { 0.0 }).collect()
}

/// The skill ratios of a chart's taps view (the press rating's row): no skill of the LN family, the others on the press level.
/// By name ("ln…") LN Shields stayed in, and the row's LN column reads the strongest of the family (user 2026-11-22: AXION
/// [Pew's Insane], led by LN Shields, read a lower regular dan than with Hold Off; correction ×1.006 against ×1.057).
fn press_skills(sk: &[(String, f64)], scale: f64) -> Vec<(String, f64)> {
    let ln = |k: &str| crate::skills::Sk::from_name(k).is_some_and(|s| crate::taxonomy::of(s).family == "ln");
    sk.iter().map(|(k, v)| (k.clone(), if ln(k) { 0.0 } else { py::round_digits(v * scale, 4) })).collect()
}

/// difficulty_model.apply on the result dict.
pub fn apply(prm: &Params, res: &mut Dict, od: f64, rate: f64) {
    let keys = res.f("keys") as usize;
    let raw = res.dict("scores").unwrap().f("overall");
    res.set("raw_overall", raw);
    let cf = vector(res, od);
    res.set("calibration_features", Val::floats(&cf));
    let mut press_factor = None;
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
            let f = crate::flow::Flow::stored(e).unwrap_or_default();
            let factor = joint_shift(joint, &crate::flow::joint_row(&rv, res.fget("length", 0.0), raw, rate, f, &crate::flow::extra_inputs(&cf, e, od))).exp(); // `od`, not res["od"]: that is set below, so the window column read OD 8 (fixed 2026-11-04)
            res.dict_mut("execution").unwrap().set("joint_factor", factor);
            // The press rating (compute: the chart's holds as taps, on the raw scale) takes the correction of that chart: this
            // one's row with no hold in it (no LN skill, hold share, LN flow, presses under a hold) at the press level.
            // dan charts with ≥ 20 % holds it lands within 2.7 % of the Hold Off chart's own correction on average; the chart's
            // own factor missed by 9 % (4K +11 %, 10K −14 %), and before 2026-11-22 the press rating had no correction at all
            // (a chart without holds read its raw rating on the regular dans).
            let press_raw = res.fget("press_overall", raw);
            if (press_raw - raw).abs() > 1e-12 && press_raw > 0.0 {
                let skp = press_skills(&sk, raw / press_raw);
                let (mut ex, mut cfp, mut fp) = (res.dict("execution").unwrap().clone(), cf.clone(), f);
                ex.set("ln_flow", 0.0);
                for name in ["ln", "od_ln"] {
                    if let Some(v) = FEATURES.iter().position(|x| *x == name).and_then(|i| cfp.get_mut(i)) {
                        *v = 0.0;
                    }
                }
                (fp.fast_ln, fp.hold_tap) = (0.0, [0.0; 2]);
                let rvp = sr_vector(&skp, res.dict("structure").unwrap_or(&empty), &ex, res.dict("endurance").unwrap_or(&empty), rate, res.fget("length", 0.0), notes, keys, 0.0, stam, &cfp);
                press_factor = Some(joint_shift(joint, &crate::flow::joint_row(&rvp, res.fget("length", 0.0), press_raw, rate, fp, &crate::flow::extra_inputs(&cfp, &ex, od))).exp());
            }
        }
        res.dict_mut("execution").unwrap().set("residual_vector", Val::floats(&rv));
    }
    let base = baseline_rating(prm, keys, raw, res.dict("execution"));
    res.set("baseline_overall", base);
    let multiplier = if raw != 0.0 { final_from_base(base, res.dict("execution")) / 0.000001f64.max(raw) } else { 1.0 };
    res.dict_mut("scores").unwrap().map_values(|v| v * multiplier);
    // the press rating on the ratings' scale: with its own correction, else (no holds, no families fit) the chart's
    if let Some(press) = res.get("press_overall").map(Val::f) {
        res.set("press_overall", press_factor.map_or(press * multiplier, |f| baseline_rating(prm, keys, press, res.dict("execution")) * f));
    }
    if let Some(Val::List(tl)) = res.get_mut("timeline") {
        for v in tl.iter_mut() {
            *v = Val::F(v.f() * multiplier);
        }
    }
    res.set("od", od);
    let sc = res.dict("scores").unwrap().f("overall");
    res.set("structural_correction", sc - res.f("baseline_overall"));
}

/// The note correction's weight at a level's place (`notenet::Pop::place`): NOTE_W, falling to NOTE_W_TOP over `notenet::TAPER_FROM` … `TAPER_TO`.
pub fn note_weight(prm: &Params, place: f64) -> f64 {
    let (k, from, to) = (&prm.k, crate::notenet::TAPER_FROM, crate::notenet::TAPER_TO);
    k.NOTE_W + (k.NOTE_W_TOP - k.NOTE_W) * ((place - from) / (to - from)).clamp(0.0, 1.0)
}

/// The population's note model on top of `apply`: every rating × e^(w · the net's disagreement with the calculator's overall),
/// `note` = the net's rating of the chart-rate. The weights' search: params.rs NOTE_W.
pub fn note_correction(prm: &Params, res: &mut Dict, net: &crate::notenet::Pop, note: f64) {
    let keys = res.f("keys") as usize;
    let overall = res.dict("scores").map_or(0.0, |s| s.f("overall"));
    let w = note_weight(prm, net.place(keys, overall).unwrap_or(0.0));
    let factor = (w * net.over(keys, note, overall, res.fget("notes", 0.0))).exp();
    res.set("note_rating", note);
    res.set("note_factor", factor);
    if factor == 1.0 {
        return;
    }
    res.dict_mut("scores").unwrap().map_values(|v| v * factor);
    if let Some(press) = res.get("press_overall").map(Val::f) {
        res.set("press_overall", press * factor);
    }
    if let Some(Val::List(tl)) = res.get_mut("timeline") {
        for v in tl.iter_mut() {
            *v = Val::F(v.f() * factor);
        }
    }
    let sc = res.dict("scores").unwrap().f("overall");
    res.set("structural_correction", sc - res.f("baseline_overall"));
}

#[cfg(test)]
mod press_tests {
    use super::*;

    #[test]
    fn the_taps_view_of_a_chart_keeps_no_skill_of_the_ln_family() {
        let sk: Vec<(String, f64)> = [("stream", 0.5), ("ln", 0.6), ("ln_release", 0.4), ("shield", 0.6), ("technical", 0.7)].map(|(k, v)| (k.to_string(), v)).into();
        let taps = press_skills(&sk, 1.2);
        let at = |k: &str| taps.iter().find(|x| x.0 == k).unwrap().1;
        assert_eq!((at("ln"), at("ln_release"), at("shield")), (0., 0., 0.), "LN Shields is an LN skill too");
        assert_eq!((at("stream"), at("technical")), (0.6, 0.84));
    }
}

//! Full analysis of a chart at a rate (skill_calc.compute/_compute).
use crate::card;
use crate::chart::Chart;
use crate::demand::{self, Geometry, Parts, Run, P};
use crate::execution::{self, Info};
use crate::gestures;
use crate::model;
use crate::params::Params;
use crate::pattern;
use crate::py;
use crate::scroll;
use crate::skills::{self, Series, Sk};
use crate::structure;
use crate::val::{Dict, Key, Val};

pub const MODEL_VERSION: &str = "3.5-technical-presence";
const SV_NAMES: [&str; 5] = ["Fast SV", "Slowjam SV", "Accel SV", "Stutter SV", "SV"];

pub struct Trace {
    pub total: Vec<f64>,
    pub series: Series,
    pub switch: Vec<f64>,
    pub parts: Parts,
    pub result: Dict,
}

#[derive(Default)]
pub struct Shared {
    pub trace: Option<Vec<(bool, Trace)>>,
    pub geometry: Option<Geometry>,
    pub plain_runs: Option<Vec<Run>>,
    pub execution: Option<Info>,
    pub gestures: Option<gestures::Analysis>,
    pub tech_profile: Option<[f64; 3]>,
    pub quadstream: Option<(Vec<f64>, f64)>,
    pub endurance: Option<((f64, f64), Vec<f64>)>,
    pub structure: Option<Dict>,
}

/// Rates every skill series. A series `rated` already holds unchanged keeps its rating: the pass after
/// execution::attribute changes few series, and rating them all again cost a fifth of a chart's analysis.
fn series_dict_scores(sc: &mut Dict, series: &Series, judged: &skills::Judged, keys: usize, prm: &Params, rated: &mut Vec<(Vec<f64>, f64)>, only: Option<&[&str]>) {
    for &key in &series.keys {
        if key == Sk::switch || only.is_some_and(|o| !o.contains(&key.name())) {
            continue;
        }
        let s = series.get(key);
        let r = match rated.iter().find(|(v, _)| v.as_slice() == s) { // the rating reads only the series: equal series share it
            Some(hit) => hit.1,
            None => {
                let r = skills::rating(s, judged, keys, &prm.k);
                rated.push((s.to_vec(), r));
                r
            }
        };
        sc.set(key.name(), r);
    }
}

pub fn compute_inner(chart: &Chart, rate: f64, sv: bool, shared: &mut Shared, prm: &Params) -> Result<Dict, String> {
    let k = &prm.k;
    if !(rate > 0.0) || !rate.is_finite() {
        return Err(format!("invalid rate: {rate}"));
    }
    let n = chart.len();
    let keys = chart.keys;
    let mut res = Dict::new();
    res.set("model", MODEL_VERSION);
    res.set("sv_name", "SV");
    res.set("keys", keys);
    res.set("rate", rate);
    res.set("notes", n);
    res.set("ln_notes", chart.t.iter().zip(&chart.end).filter(|(t, e)| e > t).count());
    let mut sc = Dict::new();
    sc.set("overall", 0.0);
    for s in skills::skills(keys) {
        sc.set(s.name(), 0.0);
    }
    res.set("scores", sc.clone());
    res.set("timeline", Val::List(Vec::new()));
    res.set("sv", if chart.sv.is_empty() { "constant" } else if sv { "included" } else { "ignored" });
    if n < 2 {
        return Ok(res);
    }
    let read = if sv && !prm.off("sv_reading") { Some(scroll::reading(chart, rate, prm)?) } else { None };
    let any_strain = read.as_ref().is_some_and(|r| r.0.iter().any(|&x| x != 0.0));
    let runs: Vec<Run> = if !sv && shared.plain_runs.is_some() {
        shared.plain_runs.take().unwrap() // the one plain pass (compute) reads it once
    } else {
        let want_plain = sv && !chart.sv.is_empty() && any_strain;
        let g = demand::geometry(chart, rate, k);
        let mut rs = Vec::new();
        let mut plains = Vec::new();
        for cl in if keys % 2 == 1 { vec![true, false] } else { vec![true] } {
            let (run, plain) = demand::demand(
                chart, &g, cl, read.as_ref().map(|r| &r.0[..]), read.as_ref().map(|r| &r.1[..]), want_plain, k,
            );
            rs.push(run);
            if let Some(p) = plain {
                plains.push(p);
            }
        }
        shared.geometry = Some(g);
        if want_plain {
            shared.plain_runs = Some(plains);
        }
        rs
    };
    if shared.endurance.is_none() {
        let fines: Vec<&[Vec<f64>; 3]> = runs.iter().map(|r| &r.fine).collect();
        shared.endurance = Some(if prm.off("endurance") {
            ((0.0, 0.0), vec![0.0; runs[0].totals.len()])
        } else {
            model::endurance(&fines, k.BIN, k.PHASES)
        });
    }
    let (total, mut parts): (Vec<f64>, Parts) = if runs.len() == 2 {
        let t: Vec<f64> = runs[0].totals.iter().zip(&runs[1].totals).map(|(l, r)| (l + r) / 2.0).collect();
        let mut p = runs[0].parts.clone();
        for (x, y) in p.data.iter_mut().zip(&runs[1].parts.data) {
            *x = (*x + y) / 2.0;
        }
        (t, p)
    } else {
        let r = runs.into_iter().next().unwrap();
        (r.totals, r.parts)
    };
    let tracing = shared.trace.is_some();
    let g = shared.geometry.as_ref().unwrap();
    let judged = skills::Judged::new(&g.times, &g.ends, total.len(), shared.endurance.as_ref().unwrap().1.clone(), k);
    let overall = skills::rating(&total, &judged, keys, k);
    sc.set("overall", overall);
    // Sustained demand: the same rating on minutes-long windows, so a uniformly hard chart keeps it near the overall.
    sc.set("stamina", skills::stamina_rating(&total, &judged, keys, k));
    let mut series = skills::attribute(&parts, &total, keys, false, k);
    let mut rated = Vec::new();
    // Before execution::attribute only model::vector's skills are read (physical_sk); the rest are rated once, after it.
    series_dict_scores(&mut sc, &series, &judged, keys, prm, &mut rated, Some(&model::VECTOR_SKILLS));
    let mut psk = Dict::new();
    for (key, v) in &sc.0 {
        if let Key::S(s) = key {
            if model::VECTOR_SKILLS.contains(&s.as_str()) {
                psk.set(s, v.f() / 0.05f64.max(overall));
            }
        }
    }
    res.set("physical_sk", psk);
    if shared.execution.is_none() {
        shared.execution = Some(execution::analyse(chart, shared.geometry.as_ref().unwrap(), k.BIN, k.CENTRE_SPLIT > 0.0));
    }
    if shared.gestures.is_none() {
        // Off: empty series (every reader is length-guarded), zero features, no roll plans.
        shared.gestures = Some(if prm.off("gestures") {
            gestures::Analysis::default()
        } else {
            gestures::analyse(chart, rate, k.BIN)
        });
    }
    if shared.quadstream.is_none() {
        shared.quadstream =
            Some(if prm.off("quadstream") { (vec![0.0; total.len()], 0.0) } else { pattern::quadstream(chart, rate, k.BIN) });
    }
    let (changed, extra) = if prm.off("execution") {
        (parts.clone(), Vec::new())
    } else {
        execution::attribute(
            &parts, &total, &mut series, chart, rate, shared.execution.as_ref().unwrap(), shared.gestures.as_ref().unwrap(),
            shared.quadstream.as_ref().unwrap(), k,
        )
    };
    parts = changed;
    let mut ex = Dict::new();
    for (name, v) in extra {
        ex.set(name, v);
    }
    res.set("execution", ex);
    if shared.tech_profile.is_none() {
        shared.tech_profile = Some(pattern::tech_profile(chart, rate, shared.execution.as_ref().unwrap(), k));
    }
    let tp = shared.tech_profile.unwrap_or([0.0; 3]);
    let mut tpd = Dict::new();
    tpd.set("version", pattern::TECH_PROFILE_VERSION);
    tpd.set("technical", tp[0]);
    tpd.set("patterntech", tp[1]);
    tpd.set("rhythmtech", tp[2]);
    if !prm.off("tech_profile") {
        res.set("tech_profile", tpd);
    }
    let switch = series.remove(Sk::switch);
    series_dict_scores(&mut sc, &series, &judged, keys, prm, &mut rated, None);
    res.set("archetypes", Val::List(card::archetypes(&series, &sc, &judged, keys, &switch, k)));
    res.set("tech_where", Val::List(card::tech_where(&series, &parts, chart.t[0] / 1000.0, rate, k)));
    let sv_tot = py::sum(parts.get(P::sv).iter().copied());
    if sv_tot != 0.0 {
        let shares: Vec<f64> = demand::SV_KINDS.iter().map(|&p| py::sum(parts.get(p).iter().copied()) / sv_tot).collect();
        if let Some(i) = shares.iter().position(|&f| f >= k.SV_KIND_SHARE) {
            res.set("sv_name", SV_NAMES[i]);
        }
    }
    let tl = skills::local_levels(&total, k.LOCAL_WINDOW, k);
    let rk = rate.powf(k.RATE_G);
    sc.map_values(|v| v * rk);
    res.set("scores", sc);
    model::scale_archetypes(&mut res, rk);
    res.set("timeline", Val::floats(&tl.iter().map(|x| x * rk).collect::<Vec<f64>>()));
    let ((load, peak), ftl) = shared.endurance.clone().unwrap();
    let mut en = Dict::new();
    en.set("load", load);
    en.set("peak", peak);
    res.set("endurance", en);
    res.set("fatigue_timeline", Val::floats(&ftl));
    let mut rs: Vec<(f64, u32)> = Vec::new();
    for i in 0..n {
        let t = chart.t[i];
        match rs.last_mut() {
            Some(x) if x.0 == t => x.1 |= 1 << chart.col[i],
            _ => rs.push((t, 1 << chart.col[i])),
        }
    }
    let mut pressure = 0.0f64;
    let mut pairs = 0usize;
    for w2 in rs.windows(2) {
        let gap = (w2[1].0 - w2[0].0) / 1000.0 / rate;
        if 0.0 < gap && gap < 0.25 {
            pressure += ((w2[0].1 & w2[1].1).count_ones() as f64 / keys as f64).powf(2.0) * 2f64.min(0.125 / gap);
            pairs += 1;
        }
    }
    res.set("jack_width", pressure / 1usize.max(pairs) as f64);
    let has_bracket = res.dict("scores").unwrap().has("bracket");
    let mut bracket_gate = 1.0f64;
    if has_bracket {
        let half = keys / 2;
        let all = ((1u64 << keys) - 1) as u32;
        // Odd keymodes: the middle key with each hand in turn, averaged (user 2026-11-03: "a thumb is a finger"); it was in neither.
        let lo = ((1u64 << half) - 1) as u32;
        let lefts: Vec<[u32; 2]> = if keys % 2 == 1 && k.CENTRE_SPLIT > 0.0 { vec![[(lo << 1) | 1, all & !((lo << 1) | 1)], [lo, all & !lo]] } else { vec![[lo, lo << (keys - half)]] };
        bracket_gate = 0.;
        for &hands in &lefts {
            let (mut moves, mut interleaved) = (0usize, 0usize);
            for w2 in rs.windows(2) {
                for h in hands {
                    let (pa, pb) = (w2[0].1 & h, w2[1].1 & h);
                    if pa != 0 && pb != 0 && pa & pb == 0 && pa.count_ones().max(pb.count_ones()) >= 2 {
                        moves += 1;
                        interleaved += (crate::kern::inner(pa, keys) & pb != 0 || crate::kern::inner(pb, keys) & pa != 0) as usize;
                    }
                }
            }
            bracket_gate += if moves > 0 { 1f64.min(interleaved as f64 / (k.BRACKET_IL_FULL * moves as f64)) } else { 1.0 } / lefts.len() as f64;
        }
    }
    let sv_physical = sv && !chart.sv.is_empty() && any_strain;
    if sv_physical {
        let physical = compute_inner(chart, rate, false, shared, prm)?;
        res.set("no_sv_overall", physical.dict("scores").unwrap().f("overall"));
        let summary = read.as_ref().unwrap().2;
        let mut rd = Dict::new();
        for (name, v) in scroll::FEATURES.iter().zip(summary) {
            rd.set(name, v);
        }
        res.set("reading", rd);
        let mut base = Dict::new();
        base.set("raw", physical.f("raw_overall"));
        base.set("features", physical.get("calibration_features").unwrap().clone());
        base.set("execution", physical.get("execution").unwrap().clone());
        res.dict_mut("execution").unwrap().set("scroll_base", base);
    }
    if shared.structure.is_none() {
        shared.structure = Some(structure::structure(chart, rate, k));
    }
    res.set("structure", shared.structure.clone().unwrap());
    let length = chart.last_end() - chart.t[0];
    res.set("length", if length.fract() == 0.0 && length.abs() < 9e15 { Val::Int(length as i64) } else { Val::F(length) });
    if prm.model["joint"]["modes"][chart.keys.to_string()].is_object() {
        if let Some(e) = res.dict_mut("execution") {
            crate::flow::flow(chart, rate).store(e);
        }
    }
    model::apply(prm, &mut res, chart.od, rate);
    if has_bracket && bracket_gate < 1.0 {
        let scd = res.dict_mut("scores").unwrap();
        let bv = scd.f("bracket") * bracket_gate;
        scd.set("bracket", bv);
        let cs = scd.f("chordstream");
        if let Some(Val::List(items)) = res.get_mut("archetypes") {
            for a in items.iter_mut() {
                if let Val::Dict(d) = a {
                    if matches!(d.get("parts"), Some(Val::Tuple(p)) if p.iter().any(|x| matches!(x, Val::Str(s) if s == "bracket"))) {
                        let r = d.f("rating");
                        d.set("rating", r * bracket_gate);
                    }
                }
            }
            items.retain(|a| match a {
                Val::Dict(d) => !matches!(d.get("name"), Some(Val::Str(s)) if s == "Bracket") || bv >= k.REFINE_Q * cs,
                _ => true,
            });
        }
    }
    if tracing {
        let tr = Trace { total, series, switch, parts, result: res.clone() };
        let t = shared.trace.as_mut().unwrap();
        t.retain(|(key, _)| *key != sv);
        t.push((sv, tr));
    }
    Ok(res)
}

/// Full analysis of a parsed chart at a constant rate (skill_calc.compute).
pub fn compute(chart: &Chart, rate: f64, sv: bool, prm: &Params) -> Result<Dict, String> {
    let mut result = compute_inner(chart, rate, sv, &mut Shared::default(), prm)?;
    let mut ap = Dict::new();
    ap.set("version", 3i64);
    ap.set("ln_chordstream", if prm.off("archetype_profile") { 0.0 } else { card::profile(chart, rate) });
    result.set("archetype_profile", ap);
    Ok(result)
}

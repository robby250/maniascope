//! Full analysis of a chart at a rate (skill_calc.compute/_compute).
use crate::card;
use crate::chart::Chart;
use crate::demand::{self, Geometry, Parts, Run, P};
use crate::execution::{self, Info};
use crate::gestures;
use crate::model;
use crate::params::Params;
use crate::pattern;
use crate::scroll;
use crate::skills::{self, Series, Sk};
use crate::structure;
use crate::val::{Dict, Key, Val};

pub const MODEL_VERSION: &str = "3.5-technical-presence";

pub struct Trace {
    pub total: Vec<f64>,
    pub series: Series,
    pub switch: Vec<f64>,
    pub parts: Parts,
    pub result: Dict,
    /// Each slice's evidence density where it can slice (skills::slice).
    pub densities: Vec<(Sk, Vec<f64>)>,
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
    pub end_series: Option<Vec<Vec<f64>>>,
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
    let _runs = crate::structure::RunScope::open();
    let k = &prm.k;
    if !(rate > 0.0) || !rate.is_finite() {
        return Err(format!("invalid rate: {rate}"));
    }
    // analysed as played: notes the player strikes as one gesture (Chart::mashed) are one row
    let (written, played) = (chart, chart.played(rate, k));
    let chart = played.as_ref().unwrap_or(chart);
    let n = chart.len();
    let keys = chart.keys;
    let mut res = Dict::new();
    res.set("model", MODEL_VERSION);
    res.set("keys", keys);
    res.set("rate", rate);
    res.set("notes", n);
    res.set("ln_notes", chart.t.iter().zip(&chart.end).filter(|(t, e)| e > t).count());
    let mut sc = Dict::new();
    sc.set("overall", 0.0);
    for s in crate::taxonomy::rated(keys) {
        sc.set(s.name(), 0.0);
    }
    res.set("scores", sc.clone());
    res.set("timeline", Val::List(Vec::new()));
    res.set("sv", if chart.sv.is_empty() { "constant" } else if sv { "included" } else { "ignored" });
    if n < 2 {
        // The structure stamp marks the analysis current: without it a 0/1-note chart was analysed again forever and the
        // library bar stopped short (a player 2026-11-06: "10,099 / 10,126" in every version).
        res.set("structure", structure::structure(chart, rate, k));
        return Ok(res);
    }
    let (read, runs) = demand_pass(chart, rate, sv, keys, shared, prm)?;
    let any_strain = read.as_ref().is_some_and(|r| r.0.iter().any(|&x| x != 0.0));
    if shared.end_series.is_none() {
        shared.end_series = Some(model::endurance_series(&runs.iter().map(|r| &r.fine).collect::<Vec<_>>(), k.PHASES));
    }
    shared.endurance = Some(if prm.off("endurance") {
        ((0.0, 0.0), vec![0.0; runs[0].totals.len()])
    } else {
        model::endurance_of(shared.end_series.as_ref().unwrap(), k.BIN, k.END_REF_Q)
    });
    let (mut total, mut parts): (Vec<f64>, Parts) = if runs.len() == 2 {
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
    if shared.execution.is_none() {
        shared.execution = Some(execution::analyse(chart, shared.geometry.as_ref().unwrap(), k.BIN));
    }
    // a mashed passage is struck, not read note by note: its demand eases by MASH_EASE of the share of its written rows
    // that merged into one gesture (Chart::mashed; user 2026-11-15: A Fool Moon Night's all-key mash sections ~98.5–99 % on
    // OD 4). The mash detector's share eased ordinary 4K patterns too (dump TEST +.0003).
    if k.MASH_EASE > 0.0 && played.is_some() {
        let (g, bin) = (shared.geometry.as_ref().unwrap(), k.BIN);
        let rows_per_bin = |c: &Chart| {
            let mut n = vec![0.0f64; total.len()];
            for w in 0..c.len() {
                if w == 0 || c.t[w] != c.t[w - 1] {
                    let b = ((c.t[w] / 1000.0 / rate - g.times[0]) / bin).max(0.0) as usize;
                    if let Some(x) = n.get_mut(b) { *x += 1.0; }
                }
            }
            n
        };
        let (wr, pr) = (rows_per_bin(written), rows_per_bin(chart));
        let m: Vec<f64> = wr.iter().zip(&pr).map(|(w, p)| if *w > 0.0 { (1.0 - p / w).max(0.0) } else { 0.0 }).collect();
        // the wider the great window, the more a mash's spread still lands in it (user: "OD4 so it's very easy to manip")
        let window = crate::gestures::great_window(chart.od) / crate::gestures::great_window(8.0);
        let ease: Vec<f64> = (0..total.len()).map(|b| 1.0 - (k.MASH_EASE * window * m.get(b).copied().unwrap_or(0.0)).min(0.9)).collect();
        total.iter_mut().zip(&ease).for_each(|(t, e)| *t *= e);
        for &p in parts.keys.clone().iter().filter(|&&p| !demand::non_demand(p)) {
            parts.get_mut(p).iter_mut().zip(&ease).for_each(|(v, e)| *v *= e);
        }
    }
    let tracing = shared.trace.is_some();
    let g = shared.geometry.as_ref().unwrap();
    let judged = skills::Judged::new(&g.times, &g.ends, total.len(), shared.endurance.as_ref().unwrap().1.clone(), k);
    if k.SURF.iter().any(|&c| c != 0.0) || [k.OD_POW, k.OD_POW_L, k.SWITCH_W, k.SWITCH_L, k.VIB_L, k.OD_LN, k.OD_LN_L, k.LN_TOP, k.LN_TOP_MIX, k.SUSTAIN_W].iter().any(|&c| c != 0.0) {
        let first = skills::rating(&total, &judged, keys, k);
        let sustain = if k.SUSTAIN_W != 0.0 && first > 0.0 { skills::window_rating(&total, &judged, keys, k.SUSTAIN_WINDOW, k) / first } else { 0.0 };
        surfaces(&mut total, &mut parts, first, keys, k);
        // share of the chart's notes inside vibro runs: top players' vibro surplus grows with it (panel 2026-11-17: run share
        // < .1 −.26, .25–.5 −.84, ≥ .5 −1.64), so speed charts with short bursts (REFORM Kappa/Iota Speed, .14/.09) keep theirs
        let run_share = if k.VIB_L != 0.0 { structure::vibro_notes(chart, rate, k).iter().map(Vec::len).sum::<usize>() as f64 / n.max(1) as f64 } else { 0.0 };
        level_terms(&mut total, &mut parts, first, chart.od, run_share, sustain, keys, k);
    }
    let overall = skills::rating(&total, &judged, keys, k);
    sc.set("overall", overall);
    // The press rating: the chart with its holds as taps (each bin without the LN parts' share, releases not judged). The regular
    // dans read it: an LN chart's regular dan is what hitting its notes asks, not its hold control (user 2026-11-19: the LN
    // Stellium course read "Reg Stellium+"). No second analysis: one more rating of the series.
    let press = if parts.keys.contains(&P::ln) && parts.get(P::ln).iter().any(|&v| v != 0.0) {
        let rice: Vec<f64> = total.iter().zip(parts.get(P::ln)).map(|(t, l)| (t - l).max(0.0)).collect();
        skills::rating(&rice, &skills::Judged::new(&g.times, &g.times, total.len(), judged.fatigue.clone(), k), keys, k)
    } else {
        overall
    };
    res.set("press_overall", press);
    // Sustained demand: the same rating on minutes-long windows, so a uniformly hard chart keeps it near the overall.
    sc.set("stamina", skills::stamina_rating(&total, &judged, keys, k));
    let mut series = skills::attribute(&parts, &total, keys, false, k);
    let mut rated = Vec::new();
    if shared.gestures.is_none() {
        // Off: empty series (every reader is length-guarded), zero features, no roll plans.
        shared.gestures = Some(if prm.off("gestures") {
            gestures::Analysis::default()
        } else {
            gestures::analyse(chart, rate, k)
        });
    }
    if shared.quadstream.is_none() {
        shared.quadstream =
            Some(if prm.off("quadstream") { (vec![0.0; total.len()], 0.0) } else { pattern::quadstream(chart, rate, k.BIN) });
    }
    let (changed, extra, own) = if prm.off("execution") {
        (parts.clone(), Vec::new(), Vec::new())
    } else {
        execution::attribute(
            &parts, &total, &mut series, chart, rate, shared.execution.as_ref().unwrap(), shared.gestures.as_ref().unwrap(),
            shared.quadstream.as_ref().unwrap(), k,
        )
    };
    parts = changed;
    let quad: &[f64] = shared.quadstream.as_ref().map_or(&[], |q| &q.0);
    let densities = skills::slice(&mut series, &parts, &total, &skills::Evidence { own: &own, quad, shift: shared.gestures.as_ref().map_or(&[][..], |g| &g.series[crate::gestures::s::VIBRO_SHIFT]), interleaved: interleaved(chart) }, keys, overall, k);
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
    res.set("tags", card::tags(&series, &sc, &judged, keys, k));
    res.set("tech_where", Val::List(card::tech_where(&series, &parts, chart.t[0] / 1000.0, rate, k)));
    let tl = skills::local_levels(&total, k.LOCAL_WINDOW, k);
    res.set("scores", sc);
    res.set("timeline", Val::floats(&tl));
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
            crate::flow::flow(written, rate, &prm.k).store(e);
        }
    }
    model::apply(prm, &mut res, chart.od, rate);
    if tracing {
        let tr = Trace { total, series, switch, parts, result: res.clone(), densities };
        let t = shared.trace.as_mut().unwrap();
        t.retain(|(key, _)| *key != sv);
        t.push((sv, tr));
    }
    Ok(res)
}

/// The share of a hand's disjoint chord-group moves that interleave (one group's fingers inside the other's span), averaged
/// over the hand splits (odd keymodes: the middle key with each hand in turn, user 2026-11-03: "a thumb is a finger"); 1 with
/// no such moves. Disjoint moves that never interleave are chordstream (user 2026-10-11: cigarettes).
pub fn interleaved(chart: &Chart) -> f64 {
    let keys = chart.keys;
    let mut rs: Vec<(f64, u32)> = Vec::new();
    for i in 0..chart.len() {
        match rs.last_mut() {
            Some(x) if x.0 == chart.t[i] => x.1 |= 1 << chart.col[i],
            _ => rs.push((chart.t[i], 1 << chart.col[i])),
        }
    }
    let half = keys / 2;
    let all = ((1u64 << keys) - 1) as u32;
    let lo = ((1u64 << half) - 1) as u32;
    let lefts: Vec<[u32; 2]> = if keys % 2 == 1 { vec![[(lo << 1) | 1, all & !((lo << 1) | 1)], [lo, all & !lo]] } else { vec![[lo, lo << (keys - half)]] };
    let mut out = 0.0;
    for &hands in &lefts {
        let (mut moves, mut inter) = (0usize, 0usize);
        for w2 in rs.windows(2) {
            for h in hands {
                let (pa, pb) = (w2[0].1 & h, w2[1].1 & h);
                if pa != 0 && pb != 0 && pa & pb == 0 && pa.count_ones().max(pb.count_ones()) >= 2 {
                    moves += 1;
                    inter += (crate::kern::inner(pa, keys) & pb != 0 || crate::kern::inner(pb, keys) & pa != 0) as usize;
                }
            }
        }
        out += if moves > 0 { inter as f64 / moves as f64 } else { 1.0 } / lefts.len() as f64;
    }
    out
}

/// Full analysis of a parsed chart at a constant rate (skill_calc.compute).
pub fn compute(chart: &Chart, rate: f64, sv: bool, prm: &Params) -> Result<Dict, String> {
    let mut result = compute_inner(chart, rate, sv, &mut Shared::default(), prm)?;
    let mut ap = Dict::new();
    ap.set("version", 3i64);
    result.set("archetype_profile", ap);
    Ok(result)
}

type Reading = (Vec<f64>, Vec<[f64; 5]>, [f64; 4]);

/// The SV reading and the demand runs of a pass (the plain pass takes the runs the SV pass left).
fn demand_pass(chart: &Chart, rate: f64, sv: bool, keys: usize, shared: &mut Shared, prm: &Params) -> Result<(Option<Reading>, Vec<Run>), String> {
    let k = &prm.k;
    let read = if sv && !prm.off("sv_reading") { Some(scroll::reading(chart, rate, prm)?) } else { None };
    let any_strain = read.as_ref().is_some_and(|r| r.0.iter().any(|&x| x != 0.0));
    let runs = if !sv && shared.plain_runs.is_some() {
        shared.plain_runs.take().unwrap() // the one plain pass (compute) reads it once
    } else {
        let want_plain = sv && !chart.sv.is_empty() && any_strain;
        let g = demand::geometry(chart, rate, k);
        let mut rs = Vec::new();
        let mut plains = Vec::new();
        let one = |cl: bool| demand::demand(chart, &g, cl, read.as_ref().map(|r| &r.0[..]), read.as_ref().map(|r| &r.1[..]), want_plain, k);
        // odd keymodes: the middle key with each hand, two independent passes on two threads (a fifth of a 7K chart's time)
        let both = if keys % 2 == 1 {
            std::thread::scope(|s| {
                let right = s.spawn(|| one(false));
                vec![one(true), right.join().expect("demand pass")]
            })
        } else {
            vec![one(true)]
        };
        for (run, plain) in both {
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
    Ok((read, runs))
}

/// Pattern strength surfaces (plan 2026-11-16 curves): each demand part × e^(SURF[5p..5p+5]·(x, x², L, x·L, L²)) with
/// x = ln(keys/7) and L = ln(first-pass overall / SURF_REF); each bin's total moves by its parts' weighted share, so all-zero
/// coefficients leave it exactly as it was.
/// Plan 2026-11-16 B6/B7, over L = ln(first-pass overall / LEVEL_REF) (default 0 = off):
/// - hit window: every bin × (great window / OD 8's)^−(OD_POW + OD_POW_L·L) (a tight window costs more at the top?);
/// - pattern switching: bin × (1 + SWITCH_W·e^(SWITCH_L·L)·(1 − cos(parts of b, parts of b−1))), the change of pattern mix
///   between neighbouring bins.
fn level_terms(total: &mut [f64], parts: &mut Parts, first: f64, od: f64, run_share: f64, sustain: f64, keys: usize, k: &crate::params::Knobs) {
    let l = (first.max(1e-6) / k.LEVEL_REF).ln();
    let nb = parts.nbins;
    let mut m = vec![1.0f64; nb];
    // Sustained load (own 7K plays 2026-11-18: maps whose 180 s rating stays near the overall score .31 ± .06 pp under the forecast,
    // n 293; short or peaky maps don't): every bin × (1 + SUSTAIN_W·(keys/7)^SUSTAIN_X·e^(SUSTAIN_L·L)·max(0, sustain − SUSTAIN_R)),
    // sustain = the SUSTAIN_WINDOW-second rating / the overall.
    if k.SUSTAIN_W != 0.0 {
        let f = 1.0 + k.SUSTAIN_W * (keys as f64 / 7.0).powf(k.SUSTAIN_X) * (k.SUSTAIN_L * l).exp() * (sustain - k.SUSTAIN_R).max(0.0);
        m.iter_mut().for_each(|v| *v *= f);
    }
    if k.OD_POW != 0.0 || k.OD_POW_L != 0.0 {
        let window = crate::gestures::great_window(od) / crate::gestures::great_window(8.0);
        m.iter_mut().for_each(|v| *v *= window.powf(-(k.OD_POW + k.OD_POW_L * l)));
    }
    if k.SWITCH_W != 0.0 {
        let demand: Vec<_> = parts.keys.iter().copied().filter(|&p| !demand::non_demand(p)).collect();
        let w = k.SWITCH_W * (k.SWITCH_L * l).exp();
        for b in 1..nb {
            let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
            for &p in &demand {
                let (x, y) = (parts.data[p as usize * nb + b], parts.data[p as usize * nb + b - 1]);
                ab += x * y; aa += x * x; bb += y * y;
            }
            if aa > 0.0 && bb > 0.0 {
                m[b] *= 1.0 + w * (1.0 - ab / (aa * bb).sqrt()).max(0.0);
            }
        }
    }
    if k.VIB_L != 0.0 && parts.keys.contains(&P::vib) {
        let vib = &parts.data[P::vib as usize * nb..(P::vib as usize + 1) * nb];
        m.iter_mut().zip(vib).for_each(|(v, s)| *v *= (k.VIB_L * l * run_share * s.clamp(0.0, 1.0)).exp());
    }
    let (before, mut after) = (parts_sum(parts), vec![0.0f64; nb]);
    let ln_m = if k.OD_LN != 0.0 || k.OD_LN_L != 0.0 {
        (crate::gestures::great_window(od) / crate::gestures::great_window(8.0)).powf(-(k.OD_LN + k.OD_LN_L * l))
    } else { 1.0 };
    // LN ease at the top only (zero below the hinge), strongest in LN-heavy stretches: top players and the per-note net agree
    // LN-heavy stretches play easier than rated (study 2026-11-18), own replays the reverse at lower levels (LN_BASE)
    let top = (l - k.LN_TOP_L0).max(0.0);
    let mut ln_f = vec![ln_m; nb];
    let mix = k.LN_TOP_MIX * (keys as f64 / 7.0).powf(k.LN_TOP_MIX_X);
    if top > 0.0 && (k.LN_TOP != 0.0 || mix != 0.0) {
        let share: Vec<f64> = (0..nb).map(|b| {
            let (mut lnv, mut all) = (0.0, 0.0);
            for &p in parts.keys.iter().filter(|&&p| !demand::non_demand(p)) {
                let v = parts.data[p as usize * nb + b];
                all += v;
                if matches!(p, P::ln | P::ln_rel) { lnv += v }
            }
            if all > 0.0 { lnv / all } else { 0.0 }
        }).collect();
        let w = (k.LN_TOP_WIN / k.BIN).round() as usize;
        let mut acc = 0.0;
        for b in 0..nb {
            acc += share[b] - if w > 0 && b >= w { share[b - w] } else { 0.0 };
            let s = if w == 0 { 1.0 } else { acc / (b + 1).min(w) as f64 };
            // top players: LN-heavy stretches easier, mixed ones (rice under held LNs) harder (7K panel by level × LN share)
            ln_f[b] = ln_m * (-k.LN_TOP * top * s.max(0.0).powf(k.LN_TOP_P)).exp(); // the rolling sum drifts below 0: NaN at a fractional power
            m[b] *= (mix * top * (4.0 * s * (k.LN_TOP_MIX_C - s) / (k.LN_TOP_MIX_C * k.LN_TOP_MIX_C)).max(0.0)).exp();
        }
    }
    for &p in parts.keys.clone().iter().filter(|&&p| !demand::non_demand(p)) {
        let is_ln = matches!(p, P::ln | P::ln_rel);
        parts.data[p as usize * nb..(p as usize + 1) * nb].iter_mut().zip(&m).enumerate().for_each(|(b, (v, f))| { *v *= f * if is_ln { ln_f[b] } else { 1.0 }; after[b] += *v; });
    }
    // the total follows its demand parts (an LN-only factor moves only their share of it)
    total.iter_mut().enumerate().for_each(|(b, t)| if before[b] > 0.0 { *t *= after[b] / before[b] } else { *t *= m[b] });
}

fn parts_sum(parts: &Parts) -> Vec<f64> {
    let nb = parts.nbins;
    let mut s = vec![0.0f64; nb];
    for &p in parts.keys.iter().filter(|&&p| !demand::non_demand(p)) {
        parts.data[p as usize * nb..(p as usize + 1) * nb].iter().enumerate().for_each(|(b, v)| s[b] += v);
    }
    s
}

fn surfaces(total: &mut [f64], parts: &mut Parts, first: f64, keys: usize, k: &crate::params::Knobs) {
    let (x, l) = ((keys as f64 / 7.0).ln(), (first.max(1e-6) / k.SURF_REF).ln());
    let basis = [x, x * x, l, x * l, l * l];
    let nb = parts.nbins;
    let (mut before, mut after) = (vec![0.0f64; nb], vec![0.0f64; nb]);
    for &p in parts.keys.clone().iter().filter(|&&p| !demand::non_demand(p)) {
        let c = &k.SURF[5 * p as usize..5 * p as usize + 5];
        let m = c.iter().zip(&basis).map(|(c, b)| c * b).sum::<f64>().exp();
        let s = &mut parts.data[p as usize * nb..(p as usize + 1) * nb];
        for (b, v) in s.iter_mut().enumerate() {
            before[b] += *v;
            *v *= m;
            after[b] += *v;
        }
    }
    for (b, t) in total.iter_mut().enumerate() {
        if before[b] > 0.0 {
            *t *= after[b] / before[b];
        }
    }
}

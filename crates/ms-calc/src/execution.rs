//! Structural execution evidence (execution.py).
use crate::chart::Chart;
use crate::demand::{Geometry, Parts, P};
use crate::gestures;
use crate::kern::{off_grid, smooth};
use crate::params::Knobs;
use crate::pattern;
use crate::py;
use crate::skills::{attribute as sk_attribute, Series, Sk};

/// execution.analyse output (per-bin averages).
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub mash: Vec<f64>,
    pub pseudo: Vec<f64>,
    pub regular: Vec<f64>,
    pub switch: Vec<f64>,
    pub regular_cal: Vec<f64>,
    pub switch_cal: Vec<f64>,
    pub count: Vec<f64>,
    pub release_sync: f64,
    pub flow: f64,
}

pub fn analyse(chart: &Chart, g: &Geometry, bin_seconds: f64) -> Info {
    let (times, ends, cols) = (&g.times, &g.ends, &g.columns);
    let n = times.len();
    if n == 0 {
        return Info::default();
    }
    let first = times[0];
    let max_e = ends.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bins = ((max_e - first) / bin_seconds) as usize + 1;
    let width = 2.0 * 0.015f64.max(0.045f64.min((64.0 - 3.0 * chart.od) / 1000.0));
    // groups as [start, end) note ranges (consecutive notes)
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut gstart = 0usize;
    let mut mask = 0u32;
    for i in 0..n {
        if i > gstart && (times[i] - times[gstart] > width || mask >> cols[i] & 1 == 1) {
            groups.push((gstart, i));
            gstart = i;
            mask = 0;
        }
        mask |= 1 << cols[i];
    }
    groups.push((gstart, n));
    let ng = groups.len();
    let masks: Vec<u32> = groups.iter().map(|&(a, z)| (a..z).fold(0u32, |m, i| m | 1 << cols[i])).collect();
    let centers: Vec<f64> = groups.iter().map(|&(a, z)| py::sum(times[a..z].iter().copied()) / (z - a) as f64).collect();
    let spans: Vec<f64> = groups.iter().map(|&(a, z)| times[z - 1] - times[a]).collect();
    let mut note_mash = vec![0.0f64; n];
    let mut note_pseudo = vec![0.0f64; n];
    for i in 1..ng.saturating_sub(1) {
        let gaps = [centers[i] - centers[i - 1], centers[i + 1] - centers[i]];
        let (mn, mx) = (gaps[0].min(gaps[1]), gaps[0].max(gaps[1]));
        if mn < 0.045 || mx > 0.22 || mx > 1.65 * mn {
            continue;
        }
        let coverage = (groups[i - 1..=i + 1].iter().map(|&(a, z)| z - a).sum::<usize>()) as f64 / (3 * chart.keys) as f64;
        let similarity = py::sum((i - 1..=i).map(|j| {
            let (a, b) = (masks[j], masks[j + 1]);
            (a & b).count_ones() as f64 / 1u32.max((a | b).count_ones()) as f64
        })) / 2.0;
        let half_gap = 0.001f64.max(py::sum(gaps) / 2.0);
        let compact = py::sum((i - 1..=i + 1).map(|j| 1.0 - smooth(spans[j] / half_gap, 0.30, 0.47))) / 3.0;
        let mut value = smooth(coverage, 0.65, 0.95) * smooth(similarity, 0.55, 0.9) * compact;
        let (a, z) = groups[i];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut any = false;
        for x in a..z {
            if ends[x] - times[x] > 0.04 {
                any = true;
                lo = lo.min(ends[x]);
                hi = hi.max(ends[x]);
            }
        }
        if any {
            value *= 1.0 - 0.85 * smooth(hi - lo, width, 0.25);
        }
        for x in a..z {
            note_mash[x] = value;
            note_pseudo[x] = if spans[i] > 0.006 { value } else { 0.0 };
        }
    }
    let mut mash = vec![0.0f64; bins];
    let mut pseudo = vec![0.0f64; bins];
    let mut regular = vec![0.0f64; bins];
    let mut switch = vec![0.0f64; bins];
    let mut regular_cal = vec![0.0f64; bins];
    let mut switch_cal = vec![0.0f64; bins];
    let mut count = vec![0.0f64; bins];
    let rows = &g.rows;
    let half = chart.keys / 2;
    let roll = pattern::smooth_rolls(rows, chart.keys);
    // Odd keymodes: the middle key with each hand in turn, averaged, as demand and smooth_rolls do (user 2026-11-03: "a
    // thumb is a finger … do 3"); it went to the right hand only.
    let splits: Vec<u32> = if chart.keys % 2 == 1 { vec![((1u64 << (half + 1)) - 1) as u32, ((1u64 << half) - 1) as u32] } else { vec![((1u64 << half) - 1) as u32] };
    let mut pulses = vec![(0.0f64, 0.0f64); rows.len()];
    for &lo_mask in &splits {
        for (ri, p) in hand_pulses(rows, lo_mask, &roll).into_iter().enumerate() {
            pulses[ri].0 += p.0 / splits.len() as f64;
            pulses[ri].1 += p.1 / splits.len() as f64;
        }
    }
    let mut tail_count = 0.0f64;
    let mut synced = 0.0f64;
    for (ri, row) in rows.iter().enumerate() {
        let t = row.t;
        let (p, q) = (row.start, row.end);
        let b = (bins - 1).min(((t - first) / bin_seconds) as usize);
        let (pulse, original_pulse) = pulses[ri];
        let gap = if ri > 0 { t - rows[ri - 1].t } else { 1.0 };
        let repeat = gap < 0.15 && row.mask & rows[if ri > 0 { ri - 1 } else { rows.len() - 1 }].mask != 0;
        let rep = if repeat { 0.8 } else { 0.0 };
        regular[b] += pulse;
        switch[b] += pulse.max(rep);
        regular_cal[b] += original_pulse;
        switch_cal[b] += original_pulse.max(rep);
        if q - p == 1 {
            mash[b] += 0.0 + note_mash[p];
            pseudo[b] += 0.0 + note_pseudo[p];
        } else {
            mash[b] += py::sum(note_mash[p..q].iter().copied()) / (q - p) as f64;
            pseudo[b] += py::sum(note_pseudo[p..q].iter().copied()) / (q - p) as f64;
        }
        count[b] += 1.0;
        for x in p..q {
            let (nt, end) = (times[x], ends[x]);
            if end - nt < 0.06 {
                continue;
            }
            tail_count += 1.0;
            let at = times.partition_point(|&h| h < end);
            let mut nearest = if at > 0 { (times[at - 1] - end).abs() } else { 1.0 };
            if at < n && (at == 0 || (times[at] - end).abs() < nearest) {
                nearest = (times[at] - end).abs();
            }
            synced += 1.0 - smooth(nearest, 0.003, 0.025);
        }
    }
    for v in [&mut mash, &mut pseudo, &mut regular, &mut switch, &mut regular_cal, &mut switch_cal] {
        for (x, &c) in v.iter_mut().zip(&count) {
            *x /= 1f64.max(c);
        }
    }
    let release_sync = synced / 1f64.max(tail_count);
    let irregular =
        py::sum(regular_cal.iter().zip(&count).map(|(r, c)| r * c)) / 1f64.max(py::sum(count.iter().copied()));
    let flow = release_sync * (1.0 - 1f64.min(irregular)) * tail_count / 1f64.max(n as f64);
    Info { mash, pseudo, regular, switch, regular_cal, switch_cal, count, release_sync, flow }
}

/// Per row (pulse, unsmoothed pulse) of the hand-to-hand timing, the left hand playing `lo_mask`'s columns.
fn hand_pulses(rows: &[crate::demand::Row], lo_mask: u32, roll: &[Vec<f64>; 2]) -> Vec<(f64, f64)> {
    let mut last: [Option<f64>; 2] = [None, None];
    let mut history: [std::collections::VecDeque<f64>; 2] = Default::default();
    let mut recent: Vec<f64> = Vec::with_capacity(6);
    let mut out = Vec::with_capacity(rows.len());
    for (ri, row) in rows.iter().enumerate() {
        let t = row.t;
        let has_lo = row.mask & lo_mask != 0;
        let has_hi = row.mask & !lo_mask != 0;
        let mut pulse: Option<f64> = None;
        let mut original_pulse: Option<f64> = None;
        for hand in [0usize, 1] {
            if (hand == 0 && !has_lo) || (hand == 1 && !has_hi) {
                continue;
            }
            let g = last[hand].map(|l| t - l);
            match g {
                Some(g) if g != 0.0 && 0.006 < g && g < 0.5 => {
                    recent.clear();
                    recent.extend(history[hand].iter());
                    recent.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let reference = if recent.len() >= 2 { recent[recent.len() / 2] } else { g };
                    let original_value = off_grid(g, reference, true);
                    original_pulse = Some(original_pulse.map_or(original_value, |m: f64| if original_value > m { original_value } else { m }));
                    let mut chosen = reference;
                    for k in 0..recent.len().saturating_sub(1) {
                        let (v, following) = (recent[k], recent[k + 1]);
                        if v > reference {
                            break;
                        }
                        if following - v < 0.004 {
                            chosen = v;
                            break;
                        }
                    }
                    let value = if chosen == reference { original_value } else { off_grid(g, chosen, true) };
                    let aw = value * (1.0 - roll[hand][ri]);
                    pulse = Some(pulse.map_or(aw, |m: f64| if aw > m { aw } else { m }));
                    if history[hand].len() == 6 {
                        history[hand].pop_front();
                    }
                    history[hand].push_back(g);
                }
                _ => history[hand].clear(),
            }
            last[hand] = Some(t);
        }
        out.push((pulse.unwrap_or(0.0), original_pulse.unwrap_or(0.0)));
    }
    out
}

/// Per bin: share of notes inside vibro runs (`structure::column_runs`), bins from the first note like every other series
/// (absolute song time put every run of a chart starting late into its last bin).
pub fn column_runs(chart: &Chart, rate: f64, bins: usize, k: &Knobs) -> Vec<f64> {
    let mut inrun = vec![0.0f64; bins];
    let mut count = vec![0.0f64; bins];
    let first = chart.t.first().copied().unwrap_or(0.0);
    let bin = |i: usize| (bins - 1).min(((chart.t[i] - first) * 0.001 / rate / k.BIN) as usize);
    (0..chart.len()).for_each(|i| count[bin(i)] += 1.0);
    crate::structure::column_runs(chart, rate, k.VIBRO_GAP, k).iter().flatten().for_each(|&i| inrun[bin(i)] += 1.0);
    inrun.iter().zip(&count).map(|(&i, &c)| if c != 0.0 { i / c } else { 0.0 }).collect()
}

/// Reassign overlapping descriptions (execution.attribute). Mutates `series`; returns the changed
/// parts and the execution features (ordered).
#[allow(clippy::too_many_arguments)]
pub fn attribute(
    parts: &Parts,
    total: &[f64],
    series: &mut Series,
    chart: &Chart,
    rate: f64,
    info: &Info,
    motion: &gestures::Analysis,
    quad: &(Vec<f64>, f64),
    k: &Knobs,
) -> (Parts, Vec<(&'static str, f64)>, Vec<f64>) {
    let m = &info.mash;
    // Rice: the base skills of the rice and chord fit families (mash and rolls take from them).
    let rice: Vec<Sk> = crate::taxonomy::TABLE.iter().filter(|r| r.role == crate::taxonomy::Role::Base && matches!(r.fit, "rice" | "chords") && series.has(r.sk)).map(|r| r.sk).collect();
    let mut changed = parts.clone();
    let zipn = |a: usize, b: usize| a.min(b);
    {
        let pins = parts.get(P::pins);
        let n = zipn(pins.len(), m.len());
        let out = changed.get_mut(P::pins);
        for b in 0..n {
            out[b] = pins[b] * (1.0 - m[b]);
        }
        let psw = parts.get(P::psw);
        let out = changed.get_mut(P::psw);
        for b in 0..psw.len().min(info.switch.len()).min(m.len()) {
            out[b] = psw[b] * info.switch[b] * (1.0 - m[b]);
        }
        let rodd = parts.get(P::rodd);
        let out = changed.get_mut(P::rodd);
        for b in 0..rodd.len().min(info.regular.len()).min(m.len()) {
            let gate = 1f64.min(info.regular[b] / k.RT_FULL);
            out[b] = rodd[b] * gate * (1.0 - m[b]);
        }
    }
    let corrected = sk_attribute(&changed, total, chart.keys, true, k);
    for name in [Sk::technical, Sk::patterntech, Sk::rhythmtech, Sk::switch] {
        series.set(name, corrected.get(name));
    }
    let sv_raw = parts.get(P::sv_raw);
    let nosv: Vec<f64> = total.iter().zip(sv_raw).map(|(t, s)| 0f64.max(t - s / k.BIN)).collect();
    let mash_series: Vec<f64> = nosv.iter().zip(m).map(|(x, s)| x * s).collect();
    series.set(Sk::mash, &mash_series);
    let scale = |series: &mut Series, s: Sk, f: &dyn Fn(usize) -> f64, n: usize| {
        if series.has(s) {
            let v: Vec<f64> = series.get(s).iter().take(n).enumerate().map(|(b, x)| x * f(b)).collect();
            series.set(s, &v);
        }
    };
    for &s in &rice {
        let n = series.get(s).len().min(m.len());
        scale(series, s, &|b| 1.0 - m[b], n);
    }
    let ln_shared = info.release_sync;
    let n = series.get(Sk::ln).len().min(m.len());
    scale(series, Sk::ln, &|b| 1.0 - m[b] * ln_shared, n);
    let sum_total = py::sum(total.iter().copied());
    let weight = 1f64.max(sum_total);
    let mash = py::sum(total.iter().zip(m).map(|(t, v)| t * v)) / weight;
    let pseudo = py::sum(total.iter().zip(&info.pseudo).map(|(t, v)| t * v)) / weight;
    let smooth_switch = py::sum(parts.get(P::psw).iter().zip(&info.switch_cal).map(|(v, g)| v * (1.0 - g)))
        / 1f64.max(py::sum(parts.get(P::rows).iter().copied()));
    let mean = sum_total / 1usize.max(total.len()) as f64;
    let mm = 1f64.max(mean);
    let peak = py::sum(total.iter().map(|&t| t * 0f64.max(1f64.max(t / mm).ln()))) / weight;
    let surge = py::sum(total.windows(2).map(|w| {
        let d = 0f64.max(w[1] - w[0]);
        d * (d / mm).ln_1p()
    })) / weight;
    let mut extra: Vec<(&'static str, f64)> = vec![
        ("mash", mash),
        ("pseudo_mash", pseudo),
        ("ln_flow", info.flow),
        ("smooth_switch", smooth_switch),
        ("peak_pressure", peak),
        ("transition_pressure", surge),
    ];
    let gs = &motion.series;
    let runs = column_runs(chart, rate, total.len(), k);
    let rows = parts.get(P::rows);
    let vib = &gs[gestures::s::VIBRO];
    let nown = vib.len().min(runs.len()).min(rows.len());
    let own: Vec<f64> = (0..nown)
        .map(|b| {
            let o = (1f64.min(vib[b] / 0.8)).max(runs[b]); // the same chord re-struck, or a column's vibro run
            if o >= 0.02 { o } else { 0.0 }
        })
        .collect();
    // Rolled rice is jumptrill: the rolled share of each rice skill moves to it (shares still sum to the bin's total).
    let rolled = &gs[gestures::s::ROLLED];
    for b in 0..rolled.len().min(total.len()) {
        let strength = rolled[b];
        if strength <= 0.0 {
            continue;
        }
        let moved: f64 = rice.iter().filter_map(|&s| series.get(s).get(b)).sum::<f64>() * strength;
        if let Some(jt) = series.get_mut(Sk::jumptrill).get_mut(b) {
            *jt += moved;
        }
        for &s in &rice {
            if let Some(x) = series.get_mut(s).get_mut(b) {
                *x *= 1.0 - strength;
            }
        }
    }
    for (name, v) in gestures::FEATURES.iter().zip(motion.features) {
        extra.push((name, v));
    }
    extra.push(("full_chord_stream_share", quad.1));
    (changed, extra, own)
}

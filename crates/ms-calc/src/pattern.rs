//! Local execution patterns used for recognition (pattern_control.py).
use crate::chart::Chart;
use crate::demand::Row;
use crate::execution::Info;
use crate::kern::off_grid;
use crate::params::Knobs;
use crate::py;

#[inline]
fn smooth(x: f64, lo: f64, hi: f64) -> f64 {
    let x = 1f64.min(0f64.max((x - lo) / (hi - lo)));
    x * x * (3.0 - 2.0 * x)
}

/// Full-column chords between single notes: (per-bin coverage × support, whole-chart share).
pub fn quadstream(chart: &Chart, rate: f64, rows: &[Row], bin_seconds: f64) -> (Vec<f64>, f64) {
    if chart.len() == 0 {
        return (Vec::new(), 0.0);
    }
    let first = chart.t[0];
    // (chart ms, columns, holds a long note: over 3 ms of real time, the LN rule of demand)
    let rows: Vec<(f64, u32, bool)> = rows
        .iter()
        .map(|r| (chart.t[r.start], r.mask, (r.start..r.end).any(|i| chart.end[i] - chart.t[i] > 3.0 * rate)))
        .collect();
    let conv = |d: f64| d * 0.001 / rate / bin_seconds;
    let bins = conv(chart.last_end() - first) as usize + 1;
    let mut hits = vec![0.0f64; bins];
    let mut counts = vec![0.0f64; bins];
    let mut episodes = vec![0.0f64; bins];
    let mut marked = vec![0.0f64; rows.len()];
    for i in 1..rows.len().saturating_sub(1) {
        let (a, b, c) = (rows[i - 1], rows[i], rows[i + 1]);
        if b.2 || a.2 || c.2 || b.1.count_ones() as usize != chart.keys || a.1.count_ones() != 1 || c.1.count_ones() != 1 {
            continue;
        }
        let ga = (b.0 - a.0) * 0.001 / rate;
        let gb = (c.0 - b.0) * 0.001 / rate;
        if ga.min(gb) <= 0.006 || ga.max(gb) > 0.23 {
            continue;
        }
        let strength = (1.0 - smooth(ga.max(gb), 0.16, 0.23)) * (1.0 - smooth((ga / gb).ln().abs(), 0.15, 0.6));
        for j in [i - 1, i, i + 1] {
            marked[j] = marked[j].max(strength);
        }
        episodes[(bins - 1).min(conv(b.0 - first) as usize)] += strength;
    }
    for (r, &v) in rows.iter().zip(&marked) {
        let b = (bins - 1).min(conv(r.0 - first) as usize);
        let n = r.1.count_ones() as f64;
        counts[b] += n;
        hits[b] += n * v;
    }
    let out = (0..bins)
        .map(|b| {
            let (lo, hi) = (b.saturating_sub(2), bins.min(b + 3));
            let coverage = py::sum(hits[lo..hi].iter().copied()) / 1f64.max(py::sum(counts[lo..hi].iter().copied()));
            let support = 1f64.min(py::sum(episodes[lo..hi].iter().copied()) / 3.0);
            coverage * support
        })
        .collect();
    (out, py::sum(hits.iter().copied()) / 1f64.max(py::sum(counts.iter().copied())))
}

/// Notes inside stable adjacent-finger rolls, per hand: [hand][row].
pub fn smooth_rolls(rows: &[Row], keys: usize) -> [Vec<f64>; 2] {
    let assignments = crate::demand::splits(keys);
    let share = 1.0 / assignments.len() as f64;
    let mut output = [vec![0.0f64; rows.len()], vec![0.0f64; rows.len()]];
    for &cl in assignments {
        let masks = crate::demand::hand_masks(keys, cl);
        for h in 0..2 {
            let hm = masks[h];
            // (row index, t, lowest column, count)
            let local: Vec<(usize, f64, i64, u32)> = rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.mask & hm != 0)
                .map(|(i, r)| (i, r.t, (r.mask & hm).trailing_zeros() as i64, (r.mask & hm).count_ones()))
                .collect();
            let mut run: Vec<(usize, f64, i64, u32)> = Vec::new();
            let mut step: Option<i64> = None;
            let close = |run: &Vec<(usize, f64, i64, u32)>, out: &mut Vec<f64>| {
                if run.len() >= 3 {
                    for x in run {
                        out[x.0] += share;
                    }
                }
            };
            for &row in &local {
                let (_, t, c0, cnt) = row;
                if cnt != 1 {
                    close(&run, &mut output[h]);
                    run.clear();
                    step = None;
                    continue;
                }
                let delta = if let Some(l) = run.last() { c0 - l.2 } else { 0 };
                let gap = if let Some(l) = run.last() { t - l.1 } else { 1.0 };
                let prevgap = if run.len() > 1 { run[run.len() - 1].1 - run[run.len() - 2].1 } else { gap };
                if !run.is_empty()
                    && delta.abs() == 1
                    && (step.is_none() || Some(delta) == step)
                    && 0.006 < gap
                    && gap < 0.16
                    && (gap - prevgap).abs() < 0.003f64.max(0.12 * prevgap)
                {
                    run.push(row);
                    step = Some(delta);
                } else {
                    close(&run, &mut output[h]);
                    run = if !run.is_empty() && delta.abs() == 1 && 0.006 < gap && gap < 0.16 {
                        vec![*run.last().unwrap(), row]
                    } else {
                        vec![row]
                    };
                    step = if run.len() > 1 { Some(delta) } else { None };
                }
            }
            close(&run, &mut output[h]);
        }
    }
    output
}

pub const TECH_PROFILE_VERSION: i64 = 2;

/// Whole-map technical character: [technical, patterntech, rhythmtech] (zeros without rows).
pub fn tech_profile(chart: &Chart, rows: &[Row], info: &Info, k: &Knobs) -> [f64; 3] {
    let counts = &info.count;
    if counts.is_empty() || py::sum(counts.iter().copied()) == 0.0 {
        return [0.0; 3];
    }
    let n = rows.len();
    // both hand splits of an odd keymode, averaged, as demand reads them (the middle key sat with the right hand only)
    let splits = crate::demand::splits(chart.keys);
    let (mut inserted, mut switched) = (vec![0.0f64; n], vec![0.0f64; n]);
    for &cl in splits {
        let trills = crate::demand::trills(&rows, chart.keys, cl, k);
        let (ins, sw) = crate::demand::pattern_evidence(&rows, &trills, k);
        for r in 0..n {
            inserted[r] += ins[r] / splits.len() as f64;
            switched[r] += sw[r] / splits.len() as f64;
        }
    }
    let nb = counts.len();
    let mut pattern = vec![0.0f64; nb];
    let mut global_rhythm = vec![0.0f64; nb];
    let first = rows[0].t;
    let mut previous: Option<f64> = None;
    let mut previous_gap = 0.0f64;
    let mut pulse = 0.0f64;
    for (r, row) in rows.iter().enumerate() {
        let t = row.t;
        let b = (nb - 1).min(((t - first) / k.BIN) as usize);
        pattern[b] += (inserted[r] + switched[r] * info.switch[b]) * (1.0 - info.mash[b]);
        let gap = previous.map_or(0.0, |p| t - p);
        if 0.0 < gap && gap < 0.5 {
            if pulse != 0.0 || (0.0 < previous_gap && previous_gap < 0.5) {
                global_rhythm[b] += off_grid(gap, if pulse != 0.0 { pulse } else { previous_gap }, pulse != 0.0);
            }
            if previous_gap != 0.0 && (gap / previous_gap).log2().abs() < 0.06 {
                pulse = gap;
            }
        } else {
            pulse = 0.0;
        }
        previous = Some(t);
        previous_gap = gap;
    }
    let rhythm: Vec<f64> = (0..nb).map(|b| (info.regular[b] * counts[b]).min(global_rhythm[b]) * (1.0 - info.mash[b])).collect();
    let window = |values: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(values.len());
        let mut total = 0.0f64;
        let h = 2;
        for i in 0..values.len() + h {
            if i < values.len() {
                total += values[i];
            }
            if i > 2 * h {
                total -= values[i - 2 * h - 1];
            }
            if i >= h {
                out.push(0f64.max(total));
            }
        }
        out
    };
    let (den, pr, rr) = (window(counts), window(&pattern), window(&rhythm));
    let mut totals = [0.0f64; 3]; // technical, patterntech, rhythmtech
    for b in 0..nb {
        let n = counts[b];
        let pt = 1f64.min(pr[b] / 1f64.max(den[b]) / k.PT_FULL);
        let rt = 1f64.min(rr[b] / 1f64.max(den[b]) / k.RT_FULL);
        totals[1] += n * pt;
        totals[2] += n * rt;
        totals[0] += n * (1.0 - (1.0 - pt) * (1.0 - rt));
    }
    let total = py::sum(counts.iter().copied());
    totals.map(|v| py::round_digits(v / total, 6))
}


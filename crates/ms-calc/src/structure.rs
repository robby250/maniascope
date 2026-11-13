//! Chart-only geometry descriptors (chart_structure.py), numpy semantics within 1e-9.
use crate::chart::Chart;
use crate::params::Knobs;
use crate::py;
use crate::val::{Dict, Val};
use sha2::{Digest, Sha256};

/// A stored analysis of another version is redone: 3 = rows carry the LN types in `sk` (2026-11-06).
pub const STRUCTURE_VERSION: i64 = 5; // 4: vibro_runs / vibro_rest (2026-11-11); 5: one vibro run rule (2026-11-13)

/// numpy pairwise summation (np.add.reduce on a contiguous float64 array).
pub fn np_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = -0.0f64;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    np_sum(&a[..n2]) + np_sum(&a[n2..])
}

pub fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

pub fn np_std(a: &[f64]) -> f64 {
    let m = np_mean(a);
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (np_sum(&sq) / a.len() as f64).sqrt()
}

/// np.percentile(a, q), method 'linear'.
pub fn np_percentile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    let virt = (n - 1) as f64 * (q / 100.0);
    let prev = virt.floor();
    let pi = (prev as usize).min(n - 1);
    let ni = (pi + 1).min(n - 1);
    let t = virt - prev;
    let (a0, b0) = (s[pi], s[ni]);
    let d = b0 - a0;
    if t >= 0.5 {
        b0 - d * (1.0 - t)
    } else {
        a0 + d * t
    }
}

pub fn structure(chart: &Chart, rate: f64, k: &Knobs) -> Dict {
    let mut d = Dict::new();
    d.set("version", STRUCTURE_VERSION);
    let n = chart.len();
    if n < 2 {
        d.set("nps", Val::None);
        return d;
    }
    let heads = &chart.t;
    let span = heads[n - 1] - heads[0];
    let max_end = chart.end.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let play_span = max_end - heads[0];
    if span <= 0.0 {
        d.set("nps", Val::None);
        return d;
    }
    let mut row_times: Vec<f64> = Vec::new();
    let mut masks: Vec<u32> = Vec::new();
    for i in 0..n {
        let t = heads[i];
        // defaultdict keyed by time: notes are time-sorted, so a new key is a new last row
        match row_times.last() {
            Some(&x) if x == t => *masks.last_mut().unwrap() |= 1 << chart.col[i],
            _ => {
                row_times.push(t);
                masks.push(1 << chart.col[i]);
            }
        }
    }
    let sizes: Vec<f64> = masks.iter().map(|m| m.count_ones() as f64).collect();
    let overlap: Vec<f64> = masks
        .windows(2)
        .map(|w| (w[0] & w[1]).count_ones() as f64 / w[0].count_ones().max(w[1].count_ones()) as f64)
        .collect();
    let repeated = masks.windows(2).filter(|w| w[0] == w[1]).count() as f64 / 1usize.max(masks.len() - 1) as f64;
    let runs = column_runs(chart, rate, k.VIBRO_GAP, k);
    let fast_count: usize = runs.iter().map(Vec::len).sum();
    let longest = runs.iter().map(|r| (heads[r[r.len() - 1]] - heads[r[0]]) / 1000.0 / rate).fold(0.0f64, f64::max);
    // Layout plus coarse normalized phrase timing (variety only, never accuracy merging).
    let num = 64usize.min(n);
    let step = (n - 1) as f64 / (num - 1) as f64;
    let mut sample: Vec<usize> = (0..num).map(|i| (i as f64 * step + 0.0) as usize).collect();
    *sample.last_mut().unwrap() = n - 1;
    let mut h = Sha256::new();
    h.update([chart.keys as u8]);
    h.update(&chart.col);
    h.update(chart.t.iter().zip(&chart.end).map(|(t, e)| (e > t) as u8).collect::<Vec<u8>>());
    for &i in &sample {
        let v = py::round((heads[i] - heads[0]) / span * 200.0) as i16;
        h.update(v.to_le_bytes());
    }
    let family: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()[..24].to_string();
    let idx: Vec<usize> = heads.iter().map(|&x| ((x - heads[0]) / 5000.0) as usize).collect();
    let nb = idx.iter().copied().max().unwrap() + 1;
    let mut bin_counts = vec![0.0f64; nb];
    for &i in &idx {
        bin_counts[i] += 1.0;
    }
    for b in &mut bin_counts {
        *b /= 5.0;
    }
    let mean = n as f64 / (span / 1000.0);
    let cycles = [1usize, 2, 4, 8]
        .iter()
        .map(|&lag| {
            let c = (lag..masks.len()).filter(|&i| masks[i] == masks[i - lag]).count();
            c as f64 / 1usize.max(masks.len().saturating_sub(lag)) as f64
        })
        .fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x }))
        .unwrap_or(0.0);
    let flat = 0f64.max(1.0 - np_std(&bin_counts) / 1f64.max(mean) / 0.35);
    let size_mean = np_mean(&sizes);
    let wide = 1f64.min(0f64.max(size_mean - 1.5) / 1.5);
    let mut drill = 0f64.max((cycles - 0.65) / 0.35) * flat * wide;
    let ge_keys: Vec<f64> = sizes.iter().map(|&s| (s >= chart.keys as f64) as u8 as f64).collect();
    let all_keys = np_mean(&ge_keys);
    drill = drill.max(0f64.max((all_keys - 0.45) / 0.55) * flat);
    let ge3: Vec<f64> = sizes.iter().map(|&s| (s >= 3.0) as u8 as f64).collect();
    let active_v: Vec<f64> = bin_counts.iter().map(|&b| (b > 0.4 * mean) as u8 as f64).collect();
    let rhythm = if row_times.len() > 1 {
        let diff: Vec<f64> = row_times.windows(2).map(|w| w[1] - w[0]).collect();
        np_std(&diff) / 1f64.max(np_mean(&diff))
    } else {
        0.0
    };
    d.set("nps", mean * rate);
    d.set("head_span", span / 1000.0);
    d.set("play_span", play_span / 1000.0);
    d.set("family", family);
    d.set("chord", size_mean);
    d.set("wide", np_mean(&ge3));
    d.set("repeat", repeated);
    d.set("overlap", if overlap.is_empty() { 0.0 } else { np_mean(&overlap) });
    d.set("burst", np_percentile(&bin_counts, 95.0) / mean.max(1e-6));
    d.set("active", np_mean(&active_v));
    d.set("rhythm", rhythm);
    d.set("vibro_share", fast_count as f64 / n as f64);
    let v = vibro_runs(chart, rate, k);
    (d.set("vibro_runs", v.share), d.set("vibro_rest", v.rest));
    d.set("vibro_longest", longest);
    d.set("vibro_focused", fast_count as f64 / n as f64 > 0.6 && repeated > 0.65 && longest > 3.0);
    d.set("periodicity", cycles);
    d.set("all_keys", all_keys);
    d.set("drill", 1f64.min(drill));
    d
}

pub const VIBRO_ONLY: f64 = 0.25; // with VIBRO_REST: 43/48 vibro dan courses, 0/45 REFORM, 0/79 LN (column_runs, 2026-11-13)
pub const VIBRO_REST: f64 = 11.5; // notes/s outside vibro runs: vibro dan courses <= 11.1, REFORM >= 11.9 at 25 %+ vibro

/// The one column run (user 2026-11-13: jacks are sustained strain on one column and hand; a run the hand leaves is not
/// a jack): a column re-struck under `gap` s apart at `rate` with no other column of its hand struck between (chords on the
/// hit rows don't end it; a stream or roll through the hand does), VIBRO_MIN hits or more. Vibro reads it at VIBRO_GAP,
/// Longjacks at LJ_GAP. A hold re-struck that fast is mashed like a tap (Vibro Dansen's 42 ms holds). Note indices per
/// run, in time order. Execution, demand, structure, flow and the dans all read it.
/// Labelled sets (2026-11-13, share of notes in runs at .25 s): REFORM jack songs .53, other REFORM songs .09, 7K jack
/// practice .05, stream/speed .02; without the hand condition streams read .3–.9 and REFORM others .95.
pub fn column_runs(chart: &Chart, rate: f64, gap: f64, k: &Knobs) -> Vec<Vec<usize>> {
    let gap = 1000.0 * gap * rate; // chart ms
    let hand: Vec<usize> = (0..chart.keys).map(|c| crate::flow::hand(c, chart.keys, true).0).collect();
    let mut open: Vec<Vec<usize>> = vec![Vec::new(); chart.keys];
    let mut left = vec![false; chart.keys]; // another column of its hand was struck since this column's last hit
    let mut out = Vec::new();
    let mut i = 0;
    while i < chart.len() {
        let mut m = i + 1;
        while m < chart.len() && chart.t[m] - chart.t[i] <= 3.0 {
            m += 1;
        }
        for x in i..m {
            let c = chart.col[x] as usize;
            let run = &mut open[c];
            if run.last().is_some_and(|&a| left[c] || chart.t[x] - chart.t[a] >= gap) {
                let done = std::mem::take(run);
                if done.len() >= k.VIBRO_MIN {
                    out.push(done);
                }
            }
            open[c].push(x);
            left[c] = false;
        }
        let row = chart.col[i..m].iter().fold(0u32, |a, &c| a | 1 << c);
        for c in 0..chart.keys {
            if row >> c & 1 == 0 && chart.col[i..m].iter().any(|&x| hand[x as usize] == hand[c]) {
                left[c] = true;
            }
        }
        i = m;
    }
    out.extend(open.into_iter().filter(|r| r.len() >= k.VIBRO_MIN));
    out
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vibro {
    pub share: f64,
    pub speed: f64,
    pub rest: f64,
}

/// Share of notes in vibro runs (`column_runs`), their median hits per second, and the other notes per second of chart.
/// Rolls count too: their columns repeat just as fast.
pub fn vibro_runs(chart: &Chart, rate: f64, k: &Knobs) -> Vibro {
    let runs = column_runs(chart, rate, k.VIBRO_GAP, k);
    let inrun: usize = runs.iter().map(Vec::len).sum();
    let mut gaps: Vec<f64> = runs.iter().flat_map(|r| r.windows(2).map(|w| (chart.t[w[1]] - chart.t[w[0]]) / rate)).collect();
    gaps.sort_by(f64::total_cmp);
    let n = chart.len();
    let span = if n == 0 { 0. } else { (chart.t[n - 1] - chart.t[0]) / rate / 1000. };
    Vibro { share: inrun as f64 / n.max(1) as f64, speed: if gaps.is_empty() { 0. } else { 1000. / gaps[gaps.len() / 2] },
            rest: if span != 0. { (n - inrun) as f64 / span } else { 0. } }
}

/// Mostly vibro runs with little else: the rice/LN dan would name the wrong skill.
pub fn vibro_only(v: &Vibro) -> bool {
    v.share >= VIBRO_ONLY && v.rest < VIBRO_REST
}


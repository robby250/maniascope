//! Per-hand flow and pattern change (user 2026-11-01: "on 10k per hand patterns and how good they flow matterns A LOT more than on
//! 7k", "mental and physical endurance are different"). Descriptors only: what they weigh is fitted per keymode.
use crate::chart::Chart;

const LINK: f64 = 0.25;     // s: a hand's next note further than this starts a new phrase
const FLOOR: f64 = 0.04;    // s: faster than this counts the same
const MEMORY: usize = 32;   // rows: a pair of rows seen this recently is no new pattern
const FAST_LN: f64 = 0.05;  // s: LN heads this close on adjacent columns roll

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Flow {
    pub broken: f64,  // speed-weighted share of a hand's notes out of finger order: not a roll, not a trill
    pub skip: f64,    // speed-weighted fingers jumped over per note
    pub change: f64,  // share of rows making a pair the last MEMORY rows didn't have
    pub fast_ln: f64, // share of LN heads rolling onto a neighbour
    pub split: f64,   // speed-weighted share of a hand's rows that are a chord with a finger left out inside it (`1011`)
    pub imbalance: f64, // density²-weighted |L − R| / (L + R) of 1 s bins: one hand carrying the load (MinaCalc HandBalance)
    pub even: f64,      // speed-weighted share of notes whose neighbour finger repeats at the same interval (Sunny unevenness)
    pub coupling: f64,  // jack load × cross-column share of 0.5 s bins, Sunny's √S·T^1.5 over √S
    pub roll: f64,      // speed-weighted share of notes continuing a one-step roll at even timing (MinaCalc WideRangeRoll)
    pub flam: f64,      // jack-speed-weighted share of notes 1–30 ms off another column's note (MinaCalc WideRangeJJ)
    pub ohj: f64,       // speed-weighted share of a hand's jumps following a jump sharing a finger (MinaCalc OHJ, CJOHAnchor)
    pub ring: f64,      // share of ring-finger notes with a neighbour finger pressing within 0.15 s (enslaving: the ring is least independent)
    pub polyrate: f64,  // share of the slower hand's notes in 1 s stretches where the hands run at different rates (bimanual interference)
    pub isolated: f64,  // share of notes more than 0.4 s after any note: timed without a rhythmic reference (scalar timing)
    pub variety: f64,   // re-judge (masking audit): Interlude Variety on rhythm
    pub chaos: f64,     // re-judge: MinaCalc Chaos
    pub jack_comp: f64, // re-judge: Interlude jack compensation
    pub vibro_run: f64, // share of notes re-tapping their column at ≥ 9/s more than 2 s into such a run (arm-tension vibro burns out)
    pub gaps: [f64; 4], // share of a hand's presses after a gap of < 90, 90–130, 130–180, 180–260 ms on that hand (rest: ≥ 260)
    pub hold_tap: [f64; 2], // share of a hand's presses while it holds a long note: after ≥ 180 ms, faster
    pub shape_load: f64, // per hand press: distinct shapes of that hand over the last 2 s × its presses/s over the last 30 s
}

impl Flow {
    /// Kept with the chart's other execution numbers: the next calibration needn't read the file again.
    pub fn store(&self, execution: &mut crate::val::Dict) {
        for (name, v) in [("flow_broken", self.broken), ("flow_skip", self.skip), ("flow_change", self.change), ("fast_ln", self.fast_ln), ("flow_split", self.split), ("imbalance", self.imbalance), ("even", self.even), ("coupling", self.coupling), ("roll", self.roll), ("flam", self.flam), ("ohj", self.ohj), ("ring", self.ring), ("polyrate", self.polyrate), ("isolated", self.isolated), ("vibro_run", self.vibro_run), ("variety", self.variety), ("chaos", self.chaos), ("jack_comp", self.jack_comp)].into_iter().chain(GAP_NAMES.into_iter().zip(self.gaps)).chain(HOLD_NAMES.into_iter().zip(self.hold_tap)).chain([("shape_load", self.shape_load)]) {
            execution.set(name, v);
        }
    }

    /// The flow `store` kept, when it holds every descriptor (an older row lacks the newest: read the chart again).
    pub fn stored(e: &crate::val::Dict) -> Option<Flow> {
        e.get("shape_load")?;
        let g = |k: &str| e.fget(k, 0.);
        Some(Flow { broken: g("flow_broken"), skip: g("flow_skip"), change: g("flow_change"), fast_ln: g("fast_ln"), split: g("flow_split"), imbalance: g("imbalance"), even: g("even"), coupling: g("coupling"), roll: g("roll"), flam: g("flam"), ohj: g("ohj"), ring: g("ring"), polyrate: g("polyrate"), isolated: g("isolated"), vibro_run: g("vibro_run"), variety: g("variety"), chaos: g("chaos"), jack_comp: g("jack_comp"), gaps: GAP_NAMES.map(g), hold_tap: HOLD_NAMES.map(g), shape_load: g("shape_load") })
    }
}

pub const NAMES: [&str; 25] = ["flow_broken", "flow_skip", "mental", "fast_ln", "flow_split", "imbalance", "even", "coupling", "roll", "flam", "ohj", "ring", "polyrate", "isolated", "vibro_run", "variety", "chaos", "jack_comp", GAP_NAMES[0], GAP_NAMES[1], GAP_NAMES[2], GAP_NAMES[3], HOLD_NAMES[0], HOLD_NAMES[1], "shape_load"];
pub const HOLD_NAMES: [&str; 2] = ["hold_tap_slow", "hold_tap_fast"];
/// The per-hand gap profile (open loop round 3): rate pairs of one chart put the error on the gap a hand moves at, + at
/// 180–260 ms and under 90, − at 130–180 (`calib motifs --rates`, 4K and 7K): the speed curve's shape, not a pattern.
pub const GAP_NAMES: [&str; 4] = ["gap_90", "gap_130", "gap_180", "gap_260"];
const GAP_EDGES: [f64; 4] = [0.09, 0.13, 0.18, 0.26];

/// How much of a gap counts as under `edge`: 1 → 0 over ±10 % of it (log scale). A hard edge let a 2 % rate step move a
/// hand's presses across a band at once: +9 % rating from 0.92× to 0.94× (rate_sweep_is_smooth_and_uncapped, 2026-11-06).
fn under(gap: f64, edge: f64) -> f64 {
    (0.5 - (gap / edge).ln() / 0.2).clamp(0., 1.)
}
pub const DIRECT: usize = crate::model::SR_NAMES_LEN + NAMES.len();

/// Under about 5–7 stars processing is the limit, on physically hard maps the arms are (user 2026-11-01).
fn mental_weight(stars: f64) -> f64 {
    ((9. - stars) / 3.).clamp(0., 1.)
}

/// The joint stage's inputs: the residual stage's, the new descriptors, then each over log rate and log stars. `raw` is the demand
/// before any fitted stage, so taking a stage out doesn't move the inputs.
pub fn joint_row(residual: &[f64], length_ms: f64, raw: f64, rate: f64, f: Flow, extra: &[f64]) -> Vec<f64> {
    let mut v = residual.to_vec();
    v.resize(crate::model::SR_NAMES_LEN, 0.);
    let stars = raw.max(0.05);
    v.extend([f.broken, f.skip, f.change * (1. - (-length_ms / 1000. / rate / 120.).exp()) * mental_weight(stars), f.fast_ln, f.split, f.imbalance, f.even, f.coupling, f.roll, f.flam, f.ohj, f.ring, f.polyrate, f.isolated, f.vibro_run, f.variety, f.chaos, f.jack_comp]);
    v.extend(f.gaps);
    v.extend(f.hold_tap);
    v.push(f.shape_load);
    let (r, l) = (rate.ln(), stars.ln() - 1.8);
    v.extend((0..DIRECT).map(|i| v[i] * r).collect::<Vec<_>>());
    v.extend((0..DIRECT).map(|i| v[i] * l).collect::<Vec<_>>());
    v.push(l);
    v.extend(extra);
    v.push(r); // log rate alone, after the extras: its curvature is a pair column (R11)
    v
}

/// Mechanism inputs the old stages read (`modes`, `execution`, `gestures`) that the residual vector lacks, appended after
/// the level so older joint bundles keep their column indices. The OD terms are the hit window's share of each mechanism.
pub const EXTRA: [&str; 10] = ["od_rice", "od_ln", "od_jack", "wide_jacks", "endurance", "cj_endurance", "smooth_switch", "vibro_sustain", "vibro_locked", "window"];

/// `window`: log of the great window's width against OD 8's, (64 − 3·OD) ms (lazer's mania windows don't scale with rate).
/// Wider windows make every pattern easier to hit accurately; it was 7K's `score_units` term only.
pub fn extra_inputs(calibration: &[f64], execution: &crate::val::Dict, od: f64) -> Vec<f64> {
    EXTRA.iter().map(|name| match (*name, crate::model::FEATURES.iter().position(|f| f == name)) {
        ("window", _) => ((64. - 3. * od.clamp(0., 10.)) / 40.).ln(),
        (_, Some(i)) => calibration.get(i).copied().unwrap_or(0.),
        (_, None) => execution.fget(name, 0.),
    }).collect()
}

/// Index of the chart's level (log raw stars − 1.8) in the joint row.
pub const LEVEL: usize = 3 * DIRECT;

/// Each mechanism column again, weighted by how far the chart is past the point where the arms, not processing, set the
/// limit: a logistic step at level `l0` with width `s` (rung C1; `mental_weight` is the same idea for one term). Bounded,
/// unlike the ×stars columns, so a 12★ map can't stretch a weight fitted on 6★ ones.
/// The families fit's columns beyond the joint row, as its bundle's `spec` says, in fit order: one arms block per regime
/// step `[l0, s]`, then products of two row inputs `[a, b]` (level² is the raw→rating curve's bend), then each skill past
/// `dominant` (the pattern that defines the chart). The fit and the app both build columns here, so they can't drift.
pub fn extended(row: &[f64], spec: &serde_json::Value) -> Vec<f64> {
    let mut out = row.to_vec();
    let pair = |v: &serde_json::Value| (v[0].as_f64().unwrap_or(0.), v[1].as_f64().unwrap_or(0.));
    for step in spec["arms"].as_array().into_iter().flatten() {
        let (l0, s) = pair(step);
        out.extend(arms(row, l0, s));
    }
    for p in spec["pairs"].as_array().into_iter().flatten() {
        let (a, b) = pair(p);
        out.push(row[a as usize] * row[b as usize]);
    }
    if let Some(t) = spec["dominant"].as_f64() {
        out.extend(row[..crate::model::SR_SKILLS.len()].iter().map(|v| (v - t).max(0.)));
    }
    out
}

pub fn arms(row: &[f64], l0: f64, s: f64) -> Vec<f64> {
    let w = 1. / (1. + (-(row[LEVEL] - l0) / s).exp());
    row[..DIRECT].iter().chain(&row[LEVEL + 1..LEVEL + 1 + EXTRA.len()]).map(|v| v * w).collect()
}

/// Which hand plays a column and where on it. The middle key of an odd keymode goes with the left hand or the right one,
/// both read and averaged (user 2026-11-03: "a thumb is a finger … do 3"); it used to sit outside both hands' order.
pub fn hand(col: usize, keys: usize, centre_left: bool) -> (usize, f64) {
    (if col < keys / 2 || (centre_left && keys % 2 == 1 && col == keys / 2) { 0 } else { 1 }, col as f64)
}

/// The chart's rows: (seconds at `rate`, column mask).
pub fn rows(chart: &Chart, rate: f64) -> Vec<(f64, u32)> {
    let scale = 0.001 / rate;
    let mut rows: Vec<(f64, u32)> = Vec::new();
    let mut i = 0;
    while i < chart.len() {
        let t = chart.t[i];
        let mut mask = 0u32;
        while i < chart.len() && chart.t[i] == t {
            mask |= 1 << chart.col[i];
            i += 1;
        }
        rows.push((t * scale, mask));
    }
    rows
}

pub fn flow(chart: &Chart, rate: f64) -> Flow {
    let scale = 0.001 / rate;
    let rows = rows(chart, rate);
    let splits: &[bool] = if chart.keys % 2 == 1 { &[true, false] } else { &[true] };
    let (mut broken, mut skip, mut split, mut imbalance, mut ohj, mut ring, mut polyrate, mut variety, mut chaos) = (0., 0., 0., 0., 0., 0., 0., 0., 0.);
    for &centre_left in splits {
        let (b, s) = order(chart, scale, centre_left);
        let part = splits.len() as f64;
        broken += b / part;
        skip += s / part;
        split += gapped(&rows, chart.keys, centre_left) / part;
        imbalance += hand_imbalance(&rows, chart.keys, centre_left) / part;
        ohj += jump_chains(&rows, chart.keys, centre_left) / part;
        ring += ring_conflicts(&rows, chart.keys, centre_left) / part;
        polyrate += hand_rates(&rows, chart.keys, centre_left) / part;
        variety += rhythm_variety(&rows, chart.keys, centre_left) / part;
        chaos += gap_chaos(&rows, chart.keys, centre_left) / part;
    }
    let even = evenness(chart, scale);
    let (coupling, roll, flam) = (coupling(chart, scale), rolls(chart, scale), flams(chart, scale));
    let (isolated, vibro_run, jack_comp) = (isolation(&rows), vibro_runs(chart, scale), jack_compensation(chart, scale));
    let (mut gaps, mut hold_tap, holds, mut shape_load) = ([0.; 4], [0.; 2], held(chart), 0.);
    for &centre_left in splits {
        shape_load += shape_loads(&rows, chart.keys, centre_left) / splits.len() as f64;
        let g = gap_profile(&rows, chart.keys, centre_left);
        (0..4).for_each(|i| gaps[i] += g[i] / splits.len() as f64);
        let h = hold_taps(&rows, &holds, chart.keys, centre_left);
        (0..2).for_each(|i| hold_tap[i] += h[i] / splits.len() as f64);
    }
    let (mut changed, mut seen) = (0usize, Vec::with_capacity(MEMORY));
    for pair in rows.windows(2) {
        let p = (pair[0].1, pair[1].1);
        changed += usize::from(!seen.contains(&p));
        if seen.len() == MEMORY {
            seen.remove(0);
        }
        seen.push(p);
    }
    let heads: Vec<(f64, u8)> = (0..chart.len()).filter(|&n| chart.end[n] > chart.t[n]).map(|n| (chart.t[n] * scale, chart.col[n])).collect();
    let rolled = heads.windows(2).filter(|w| w[1].0 - w[0].0 <= FAST_LN && w[1].0 > w[0].0 && w[1].1.abs_diff(w[0].1) == 1).count();
    Flow { broken, skip, change: changed as f64 / rows.len().max(2).saturating_sub(1) as f64, fast_ln: rolled as f64 / heads.len().max(1) as f64, split, imbalance, even, coupling, roll, flam, ohj, ring, polyrate, isolated, vibro_run, variety, chaos, jack_comp, gaps, hold_tap, shape_load }
}

/// Each row's columns held by a long note pressed before it and released after it (aligned with `flow::rows`).
pub fn held(chart: &Chart) -> Vec<u32> {
    let (mut out, mut active, mut i): (Vec<u32>, Vec<(f64, u8)>, usize) = (Vec::new(), Vec::new(), 0);
    while i < chart.len() {
        let t = chart.t[i];
        active.retain(|a| a.0 > t);
        out.push(active.iter().fold(0, |m, a| m | 1 << a.1));
        while i < chart.len() && chart.t[i] == t {
            if chart.end[i] > t {
                active.push((chart.end[i], chart.col[i]));
            }
            i += 1;
        }
    }
    out
}

/// Presses on a hand that holds a long note at the time (round 3, `calib motifs`: 7K taps under a same-hand hold play
/// harder than rated at 180 ms and slower, TRAIN t up to +11, where a speed-weighted demand is near zero): shares of all
/// hand presses, [gap ≥ 180 ms, faster].
fn hold_taps(rows: &[(f64, u32)], held: &[u32], keys: usize, centre_left: bool) -> [f64; 2] {
    let (mut out, mut n, mut last) = ([0.; 2], 0., [f64::NEG_INFINITY; 2]);
    let side = |m: u32| (0..keys).filter(|&c| m >> c & 1 == 1).fold([0u32; 2], |mut h, c| { h[hand(c, keys, centre_left).0] |= 1 << c; h });
    for (&(t, mask), &hold) in rows.iter().zip(held) {
        let (hands, holding) = (side(mask), side(hold));
        for h in (0..2).filter(|&h| hands[h] != 0) {
            n += 1.;
            if holding[h] != 0 {
                let fast = under(t - last[h], 0.18);
                (out[0], out[1]) = (out[0] + 1. - fast, out[1] + fast);
            }
            last[h] = t;
        }
    }
    out.map(|v| if n > 0. { v / n } else { 0. })
}

/// Chord shapes a hand must be ready for while it has been busy (round 3 R22, own 10K replays, `calib rows`: at equal local
/// demand a note loses more where its hand's shapes keep changing and the last 30 s were dense; natives up to +1.2 pp, 7K→10K
/// converts +2–5 pp, which play harder than rated, the loss growing through the map). Per hand press: distinct shapes of that
/// hand over the last 2 s × that hand's presses per second over the last 30 s, averaged over presses.
fn shape_loads(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    let (mut total, mut n) = (0., 0.);
    for h in 0..2 {
        let mine: u32 = (0..keys).filter(|&c| hand(c, keys, centre_left).0 == h).fold(0, |m, c| m | 1 << c);
        let hr: Vec<(f64, u32)> = rows.iter().filter(|r| r.1 & mine != 0).map(|r| (r.0, r.1 & mine)).collect();
        let (mut lo2, mut lo30, mut seen) = (0, 0, std::collections::HashMap::<u32, u32>::new());
        for (i, &(t, m)) in hr.iter().enumerate() {
            *seen.entry(m).or_default() += 1;
            while hr[lo2].0 < t - 2. {
                let k = hr[lo2].1;
                let left = seen.get_mut(&k).map_or(0, |c| { *c -= 1; *c });
                if left == 0 {
                    seen.remove(&k);
                }
                lo2 += 1;
            }
            while hr[lo30].0 < t - 30. {
                lo30 += 1;
            }
            total += seen.len() as f64 * (i - lo30) as f64 / 30.;
            n += 1.;
        }
    }
    if n > 0. { total / n } else { 0. }
}

/// Each hand's presses by the gap since that hand's last press; shares of all presses.
fn gap_profile(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> [f64; 4] {
    let (mut out, mut n, mut last) = ([0.; 4], 0., [f64::NEG_INFINITY; 2]);
    for &(t, mask) in rows {
        let mut hands = [false; 2];
        (0..keys).filter(|&c| mask >> c & 1 == 1).for_each(|c| hands[hand(c, keys, centre_left).0] = true);
        for h in (0..2).filter(|&h| hands[h]) {
            n += 1.;
            let mut below = 0.;
            for (b, &e) in GAP_EDGES.iter().enumerate() {
                let s = under(t - last[h], e);
                out[b] += s - below;
                below = s;
            }
            last[h] = t;
        }
    }
    out.map(|v| if n > 0. { v / n } else { 0. })
}

/// Split hand chords: the hand must hold a finger up between two pressing ones (finger enslaving). Replays, 7K and 10K alike:
/// about +1 pp of accuracy lost on each such note against an adjacent chord at the same speed and size (2026-11-04).
/// A chart-level hand-shape entropy (2 s, Hick–Hyman) was tried beside it and didn't help the fit (2026-11-04).
fn gapped(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    let (mut total, mut split, mut last) = (0., 0., [f64::NEG_INFINITY; 2]);
    for &(t, mask) in rows {
        let mut hands = [0u32; 2];
        (0..keys).filter(|&c| mask >> c & 1 == 1).for_each(|c| hands[hand(c, keys, centre_left).0] |= 1 << c);
        for h in 0..2 {
            if hands[h] == 0 {
                continue;
            }
            let weight = 1. / (t - last[h]).max(FLOOR);
            last[h] = t;
            total += weight;
            if (hands[h] & !(hands[h] << 1)).count_ones() > 1 {
                split += weight;
            }
        }
    }
    if total > 0. { split / total } else { 0. }
}

/// One hand carrying a 1 s stretch: |L − R| / (L + R), weighted by the stretch's notes squared so dense parts decide.
fn hand_imbalance(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    let mut bins: std::collections::BTreeMap<i64, [f64; 2]> = std::collections::BTreeMap::new();
    let start = rows.first().map_or(0., |r| r.0); // bins from the first note: the chart's offset can't move them
    for &(t, mask) in rows {
        let bin = bins.entry((t - start).floor() as i64).or_default();
        (0..keys).filter(|&c| mask >> c & 1 == 1).for_each(|c| bin[hand(c, keys, centre_left).0] += 1.);
    }
    let (mut num, mut den) = (0., 0.);
    for [l, r] in bins.into_values() {
        let n = l + r;
        num += n * (l - r).abs();
        den += n * n;
    }
    if den > 0. { num / den } else { 0. }
}

/// Sunny's unevenness, inverted: a note whose neighbouring finger on the same hand is repeating at the same interval
/// (|Δ| < 20 ms, both under 0.5 s) moves with it as one motion. Speed-weighted share.
fn evenness(chart: &Chart, scale: f64) -> f64 {
    let keys = chart.keys;
    let (mut last, mut gap) = (vec![f64::NEG_INFINITY; keys], vec![f64::INFINITY; keys]);
    let (mut total, mut even) = (0., 0.);
    for n in 0..chart.len() {
        let (c, t) = (chart.col[n] as usize, chart.t[n] * scale);
        let d = t - last[c];
        if d <= 0.5 {
            let weight = 1. / d.max(FLOOR);
            total += weight;
            let paired = [c.wrapping_sub(1), c + 1].into_iter().filter(|&o| o < keys && same_hand(o, c, keys)).any(|o| t - last[o] <= d && (gap[o] - d).abs() < 0.02);
            if paired {
                even += weight;
            }
        }
        gap[c] = d;
        last[c] = t;
    }
    if total > 0. { even / total } else { 0. }
}

fn same_hand(a: usize, b: usize, keys: usize) -> bool {
    (a < keys / 2) == (b < keys / 2) || (keys % 2 == 1 && (a == keys / 2 || b == keys / 2))
}

/// Sunny's same-column × cross-column coupling: per 0.5 s bin S = Σ 1/jack gap, X = Σ 1/gap of notes off the previous
/// note's column, T = X/(X+S+1); √S·T^1.5 summed over √S summed: jack-heavy stretches that also move across columns.
fn coupling(chart: &Chart, scale: f64) -> f64 {
    let mut bins: std::collections::BTreeMap<i64, [f64; 2]> = std::collections::BTreeMap::new();
    let mut last = vec![f64::NEG_INFINITY; chart.keys];
    let (mut prev_t, mut prev_c) = (f64::NEG_INFINITY, usize::MAX);
    let start = chart.t.first().map_or(0., |t| t * scale);
    for n in 0..chart.len() {
        let (c, t) = (chart.col[n] as usize, chart.t[n] * scale);
        let bin = bins.entry(((t - start) * 2.).floor() as i64).or_default();
        if t - last[c] <= 0.5 {
            bin[0] += 1. / (t - last[c]).max(FLOOR);
        }
        if c != prev_c && t > prev_t && t - prev_t <= LINK {
            bin[1] += 1. / (t - prev_t).max(FLOOR);
        }
        last[c] = t;
        (prev_t, prev_c) = (t, c);
    }
    let (mut num, mut den) = (0., 0.);
    for [s, x] in bins.into_values() {
        num += s.sqrt() * (x / (x + s + 1.)).powf(1.5);
        den += s.sqrt();
    }
    if den > 0. { num / den } else { 0. }
}

/// Interlude's variety, on rhythm: each hand's gap to its previous row in quarter-octave buckets; per row, the distinct
/// (hand, bucket) pairs within ±0.75 s, averaged, /10.
fn rhythm_variety(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    let mut marks: Vec<(f64, i32)> = Vec::new();
    let mut last = [f64::NEG_INFINITY; 2];
    for &(t, mask) in rows {
        for h in 0..2 {
            if (0..keys).any(|c| mask >> c & 1 == 1 && hand(c, keys, centre_left).0 == h) {
                let gap = t - last[h];
                let bucket = if gap > LINK { -1 } else { ((gap.max(FLOOR) * 1000.).log2() * 4.).round() as i32 };
                marks.push((t, bucket * 2 + h as i32));
                last[h] = t;
            }
        }
    }
    let (mut count, mut lo, mut hi, mut total) = (std::collections::HashMap::<i32, usize>::new(), 0, 0, 0.);
    for i in 0..marks.len() {
        while hi < marks.len() && marks[hi].0 <= marks[i].0 + 0.75 {
            *count.entry(marks[hi].1).or_default() += 1;
            hi += 1;
        }
        while marks[lo].0 < marks[i].0 - 0.75 {
            let e = count.get_mut(&marks[lo].1).expect("counted");
            *e -= 1;
            if *e == 0 {
                count.remove(&marks[lo].1);
            }
            lo += 1;
        }
        total += count.len() as f64;
    }
    total / marks.len().max(1) as f64 / 10.
}

/// MinaCalc's Chaos: on each hand, the longer of two consecutive gaps over the shorter; how far it sits from a whole number
/// (1:1, 2:1 read clean, 3:2 or 4:3 poly-ish), speed-weighted.
fn gap_chaos(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    let (mut last, mut gap) = ([f64::NEG_INFINITY; 2], [f64::INFINITY; 2]);
    let (mut total, mut chaos) = (0., 0.);
    for &(t, mask) in rows {
        for h in 0..2 {
            if !(0..keys).any(|c| mask >> c & 1 == 1 && hand(c, keys, centre_left).0 == h) {
                continue;
            }
            let g = t - last[h];
            if g <= LINK && gap[h] <= LINK {
                let ratio = g.max(gap[h]) / g.min(gap[h]).max(1e-3);
                let frac = ratio.fract();
                let weight = 1. / g.max(FLOOR);
                total += weight;
                chaos += weight * frac.min(1. - frac);
            }
            (last[h], gap[h]) = (t, g);
        }
    }
    if total > 0. { chaos / total } else { 0. }
}

/// Interlude's jack compensation: a re-tap of a column through a same-hand neighbour (c·o·c); its jack gap over its stream
/// gap, log2, √, ≤ 1, is how far the stream (not the jack) sets it. Here the jack's share, 1 − that, speed-weighted.
fn jack_compensation(chart: &Chart, scale: f64) -> f64 {
    let keys = chart.keys;
    let mut last = vec![f64::NEG_INFINITY; keys];
    let (mut total, mut jack) = (0., 0.);
    for n in 0..chart.len() {
        let (c, t) = (chart.col[n] as usize, chart.t[n] * scale);
        let jd = t - last[c];
        if jd <= 0.5 {
            let weight = 1. / jd.max(FLOOR);
            total += weight;
            let between = (0..keys).filter(|&o| o != c && same_hand(o, c, keys) && last[o] > last[c] && last[o] < t).map(|o| t - last[o]).fold(f64::INFINITY, f64::min);
            if between.is_finite() {
                jack += weight * (1. - (jd / between).log2().max(0.).sqrt().min(1.));
            }
        }
        last[c] = t;
    }
    if total > 0. { jack / total } else { 0. }
}

/// MinaCalc's rolls: a single note one column on from the previous single, the same way as the step before, its gap within
/// 20 % of that step's: a roll the hand sweeps, not a stream it reads.
fn rolls(chart: &Chart, scale: f64) -> f64 {
    let mut singles: Vec<(f64, i32)> = Vec::new();
    let mut i = 0;
    while i < chart.len() {
        let j = (i..chart.len()).find(|&j| chart.t[j] != chart.t[i]).unwrap_or(chart.len());
        if j - i == 1 {
            singles.push((chart.t[i] * scale, chart.col[i] as i32));
        }
        i = j;
    }
    let (mut total, mut roll) = (0., 0.);
    for w in singles.windows(3) {
        let (g1, g2) = (w[1].0 - w[0].0, w[2].0 - w[1].0);
        if g2 > LINK {
            continue;
        }
        let weight = 1. / g2.max(FLOOR);
        total += weight;
        let (s1, s2) = (w[1].1 - w[0].1, w[2].1 - w[1].1);
        if s1 == s2 && s1.abs() == 1 && (g2 - g1).abs() <= 0.2 * g1 {
            roll += weight;
        }
    }
    if total > 0. { roll / total } else { 0. }
}

/// MinaCalc's flammed jumpjacks: a note 1–30 ms off another column's note (a chord played as a flam), weighted by how fast
/// its column jacks; share of the jack-weighted notes.
fn flams(chart: &Chart, scale: f64) -> f64 {
    let mut last = vec![f64::NEG_INFINITY; chart.keys];
    let (mut total, mut flam) = (0., 0.);
    for n in 0..chart.len() {
        let (c, t) = (chart.col[n] as usize, chart.t[n] * scale);
        let jd = t - last[c];
        last[c] = t;
        if jd > 0.5 {
            continue;
        }
        let weight = 1. / jd.max(FLOOR);
        total += weight;
        let near = |m: usize| chart.col[m] as usize != c && (chart.t[m] * scale - t).abs() > 0.0005 && (chart.t[m] * scale - t).abs() <= 0.03;
        let lo = (0..n).rev().take_while(|&m| t - chart.t[m] * scale <= 0.03).any(near);
        let hi = (n + 1..chart.len()).take_while(|&m| chart.t[m] * scale - t <= 0.03).any(near);
        if lo || hi {
            flam += weight;
        }
    }
    if total > 0. { flam / total } else { 0. }
}

/// One-hand jump chains: a hand's row of ≥ 2 notes right after (≤ LINK) another ≥ 2-note row of that hand sharing a column.
fn jump_chains(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    let (mut last, mut prev) = ([f64::NEG_INFINITY; 2], [0u32; 2]);
    let (mut total, mut chain) = (0., 0.);
    for &(t, mask) in rows {
        let mut hands = [0u32; 2];
        (0..keys).filter(|&c| mask >> c & 1 == 1).for_each(|c| hands[hand(c, keys, centre_left).0] |= 1 << c);
        for h in 0..2 {
            if hands[h] == 0 {
                continue;
            }
            let gap = t - last[h];
            let weight = 1. / gap.max(FLOOR);
            total += weight;
            if gap <= LINK && hands[h].count_ones() > 1 && prev[h].count_ones() > 1 && hands[h] & prev[h] != 0 {
                chain += weight;
            }
            (last[h], prev[h]) = (t, hands[h]);
        }
    }
    if total > 0. { chain / total } else { 0. }
}

/// The ring finger of each hand on 6K and up (second column from the hand's outer edge; 4K and 5K hands have none): its notes
/// with a neighbour finger (middle or little) pressing within 0.15 s, a share of its notes. Enslaving is strongest between
/// adjacent fingers and the ring finger is the least independent (PMC5167409).
fn ring_conflicts(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    if keys < 6 {
        return 0.;
    }
    let cols: [Vec<usize>; 2] = [0, 1].map(|h| (0..keys).filter(|&c| hand(c, keys, centre_left).0 == h).collect());
    let rings: [usize; 2] = [cols[0][1], cols[1][cols[1].len() - 2]];
    let mut last = vec![f64::NEG_INFINITY; keys];
    let (mut total, mut hit) = (0., 0.);
    for &(t, mask) in rows {
        for h in 0..2 {
            let r = rings[h];
            if mask >> r & 1 == 1 {
                total += 1.;
                let near = [r.wrapping_sub(1), r + 1].into_iter().filter(|&o| o < keys && hand(o, keys, centre_left).0 == h);
                if near.into_iter().any(|o| t - last[o] <= 0.15) { // a chord with the neighbour is one motion, not a conflict
                    hit += 1.;
                }
            }
        }
        (0..keys).filter(|&c| mask >> c & 1 == 1).for_each(|c| last[c] = t);
    }
    if total > 0. { hit / total } else { 0. }
}

/// Bimanual interference: per 1 s stretch (from the first note) each hand's row rate; where both hands play (≥ 2/s) at different
/// rates, the slower hand's rows count by 1 − rate ratio. Share of all hand rows. The faster limb degrades the slower one
/// (Panzer 2021).
fn hand_rates(rows: &[(f64, u32)], keys: usize, centre_left: bool) -> f64 {
    let start = rows.first().map_or(0., |r| r.0);
    let mut bins: std::collections::BTreeMap<i64, [f64; 2]> = std::collections::BTreeMap::new();
    for &(t, mask) in rows {
        let bin = bins.entry((t - start).floor() as i64).or_default();
        for h in 0..2 {
            if (0..keys).any(|c| mask >> c & 1 == 1 && hand(c, keys, centre_left).0 == h) {
                bin[h] += 1.;
            }
        }
    }
    let (mut total, mut slow) = (0., 0.);
    for [l, r] in bins.into_values() {
        total += l + r;
        if l >= 2. && r >= 2. {
            slow += l.min(r) * (1. - l.min(r) / l.max(r));
        }
    }
    if total > 0. { slow / total } else { 0. }
}

/// Notes more than 0.4 s after the previous row: timed alone, the timing spread grows with the interval (scalar timing).
fn isolation(rows: &[(f64, u32)]) -> f64 {
    let alone = rows.windows(2).filter(|w| w[1].0 - w[0].0 > 0.4).map(|w| w[1].1.count_ones() as f64).sum::<f64>();
    let notes = rows.iter().map(|r| r.1.count_ones() as f64).sum::<f64>();
    if notes > 0. { alone / notes } else { 0. }
}

/// A column re-tapped at ≥ 9/s is vibro; a note more than 2 s into such an unbroken run counts (arm-tension vibro tires fast,
/// user 2026-11-05: "very fatiguing and intense"; wrist vibro doesn't, so the fit's level arms decide who it costs).
fn vibro_runs(chart: &Chart, scale: f64) -> f64 {
    let (mut last, mut began) = (vec![f64::NEG_INFINITY; chart.keys], vec![0.; chart.keys]);
    let mut long = 0.;
    for n in 0..chart.len() {
        let (c, t) = (chart.col[n] as usize, chart.t[n] * scale);
        if t - last[c] > 1. / 9. {
            began[c] = t;
        } else if t - began[c] > 2. {
            long += 1.;
        }
        last[c] = t;
    }
    long / chart.len().max(1) as f64
}

/// (broken, skip) of both hands' finger order, the middle key of an odd keymode with the left hand or not.
fn order(chart: &Chart, scale: f64, centre_left: bool) -> (f64, f64) {
    let mut hands: [Vec<(f64, f64)>; 2] = Default::default();
    let mut i = 0;
    while i < chart.len() {
        let t = chart.t[i];
        let (mut sum, mut count) = ([0.; 2], [0.; 2]);
        while i < chart.len() && chart.t[i] == t {
            let (h, at) = hand(chart.col[i] as usize, chart.keys, centre_left);
            sum[h] += at;
            count[h] += 1.;
            i += 1;
        }
        for h in 0..2 {
            if count[h] > 0. {
                hands[h].push((t * scale, sum[h] / count[h]));
            }
        }
    }
    let (mut total, mut broken, mut skip) = (0., 0., 0.);
    for notes in &hands {
        for w in notes.windows(3) {
            let (gap1, gap2) = (w[1].0 - w[0].0, w[2].0 - w[1].0);
            let (s1, s2) = (w[1].1 - w[0].1, w[2].1 - w[1].1);
            if gap1 > LINK || gap2 > LINK || s1 == 0. || s2 == 0. {
                continue; // a jack has no order; its cost is elsewhere
            }
            let weight = 1. / gap2.max(FLOOR);
            total += weight;
            skip += weight * (s2.abs() - 1.).max(0.);
            if s1 * s2 < 0. && (w[2].1 - w[0].1).abs() > 1e-9 {
                broken += weight;
            }
        }
    }
    let share = |v: f64| if total > 0. { v / total } else { 0. };
    (share(broken), share(skip))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chart(keys: usize, cols: &[u8], step: f64) -> Chart {
        let t: Vec<f64> = (0..cols.len()).map(|i| i as f64 * step).collect();
        Chart { keys, end: t.clone(), t, col: cols.to_vec(), ..Chart::default() }
    }

    fn repeated(pattern: &[u8], n: usize) -> Vec<u8> {
        pattern.iter().copied().cycle().take(n).collect()
    }

    /// A long note on column 0 from 0 to 1000 ms: column 1's taps inside it read as held, the one after it not.
    #[test]
    fn taps_under_a_long_note_are_held() {
        let c = Chart { keys: 4, t: vec![0., 200., 400., 1200.], end: vec![1000., 200., 400., 1200.], col: vec![0, 1, 1, 1], ..Default::default() };
        assert_eq!(held(&c), vec![0, 1, 1, 0]);
    }

    /// 7K: column 0 held 0–2000 ms while column 1 (same hand) taps every 250 ms: slow taps under a hold; column 5 taps
    /// alone on the other hand.
    #[test]
    fn taps_beside_a_held_finger_count_per_hand() {
        let mut c = Chart { keys: 7, t: vec![0.], end: vec![2000.], col: vec![0], ..Chart::default() };
        for i in 1..8 {
            for col in [1, 5] {
                c.t.push(i as f64 * 250.);
                c.end.push(i as f64 * 250.);
                c.col.push(col);
            }
        }
        let h = flow(&c, 1.).hold_tap;
        assert!((h[0] - 7. / 15.).abs() < 1e-9 && h[1] == 0., "{h:?}");
    }

    /// 4K hands alternating every 100 ms: each hand presses every 200 ms (the 180–260 band); one hand alone at 100 ms
    /// lands in 90–130. At rate 2 the alternation moves down to 90–130.
    #[test]
    fn shape_load_grows_with_changing_shapes_under_load() {
        let same = flow(&chart(10, &repeated(&[0, 1], 600), 100.), 1.).shape_load;
        let varied = flow(&chart(10, &repeated(&[0, 1, 2, 3, 0, 2, 1, 3], 600), 100.), 1.).shape_load;
        let sparse = flow(&chart(10, &repeated(&[0, 1, 2, 3, 0, 2, 1, 3], 600), 1000.), 1.).shape_load;
        assert!(varied > 1.5 * same && varied > 5. * sparse, "{same} {varied} {sparse}");
    }

    #[test]
    fn the_gap_profile_reads_each_hands_own_gap() {
        let alt = flow(&chart(4, &repeated(&[0, 3], 200), 100.), 1.).gaps;
        assert!(alt[3] > 0.98 && alt[..3].iter().all(|v| *v == 0.), "{alt:?}");
        let one = flow(&chart(4, &repeated(&[0, 1], 200), 100.), 1.).gaps;
        assert!(one[1] > 0.98, "{one:?}");
        assert!(flow(&chart(4, &repeated(&[0, 3], 200), 100.), 2.).gaps[1] > 0.98);
    }

    #[test]
    fn the_same_notes_out_of_order_read_broken() {
        let rolled = flow(&chart(10, &repeated(&[0, 1, 2, 3, 4], 400), 100.), 1.);
        let broken = flow(&chart(10, &repeated(&[0, 2, 1, 4, 3], 400), 100.), 1.);
        assert!(broken.broken > rolled.broken + 0.3 && broken.skip > rolled.skip, "{rolled:?} {broken:?}");
    }

    #[test]
    fn the_middle_key_of_an_odd_keymode_is_in_a_hand() {
        // 7K: rolls through the middle key against the same notes out of order. Left out of both hands, each hand kept two
        // columns, which have no order to break.
        let roll = flow(&chart(7, &repeated(&[1, 2, 3, 4, 5], 300), 100.), 1.);
        let broken = flow(&chart(7, &repeated(&[1, 3, 2, 5, 4], 300), 100.), 1.);
        assert!(broken.broken > roll.broken + 0.2, "{roll:?} {broken:?}");
    }

    #[test]
    fn one_hand_alone_is_imbalanced_and_alternating_hands_are_not() {
        assert!((flow(&chart(4, &repeated(&[0, 1], 400), 100.), 1.).imbalance - 1.).abs() < 1e-9);
        assert!(flow(&chart(4, &repeated(&[0, 3], 400), 100.), 1.).imbalance < 0.05);
    }

    #[test]
    fn neighbours_jacking_together_read_even() {
        // [01] jumpjacks: both fingers repeat at one interval; a 0-2 split on 7K has no neighbour moving with it.
        let rows = |pair: [u8; 2]| {
            let cols: Vec<u8> = std::iter::repeat_n(pair, 200).flatten().collect();
            let t: Vec<f64> = (0..cols.len()).map(|i| (i / 2) as f64 * 150.).collect();
            Chart { keys: 7, end: t.clone(), t, col: cols, ..Chart::default() }
        };
        assert!(flow(&rows([0, 1]), 1.).even > 0.9);
        assert_eq!(flow(&rows([0, 2]), 1.).even, 0.);
    }

    /// Notes from (ms, column) pairs, sorted.
    fn notes(keys: usize, mut n: Vec<(f64, u8)>) -> Chart {
        n.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t: Vec<f64> = n.iter().map(|p| p.0).collect();
        Chart { keys, end: t.clone(), col: n.iter().map(|p| p.1).collect(), t, ..Chart::default() }
    }

    #[test]
    fn jacks_moving_across_columns_couple_and_a_lone_jack_does_not() {
        assert_eq!(flow(&chart(4, &[0; 200], 100.), 1.).coupling, 0.);
        assert!(flow(&chart(4, &repeated(&[0, 1], 100), 100.), 1.).coupling > 0.1);
    }

    #[test]
    fn a_flammed_jumpjack_reads_flammed_and_a_chord_or_trill_does_not() {
        let flammed = notes(4, (0..200).flat_map(|i| [(i as f64 * 120., 0), (i as f64 * 120. + 8., 1)]).collect());
        let trill = flow(&chart(4, &repeated(&[0, 1], 100), 60.), 1.);
        let (f, chord) = (flow(&flammed, 1.), flow(&notes(4, (0..200).flat_map(|i| [(i as f64 * 120., 0), (i as f64 * 120., 1)]).collect()), 1.));
        assert!(f.flam > 0.9 && chord.flam == 0. && trill.flam == 0.);
    }

    #[test]
    fn a_ring_finger_right_after_its_neighbour_conflicts_and_one_in_a_chord_does_not() {
        // 7K left hand 0 1 2 (+ middle key): the ring finger is column 1.
        assert!(flow(&chart(7, &repeated(&[0, 1], 200), 100.), 1.).ring > 0.9);
        assert_eq!(flow(&notes(7, (0..200).flat_map(|i| [(i as f64 * 300., 0), (i as f64 * 300., 1)]).collect()), 1.).ring, 0.);
        assert_eq!(flow(&chart(4, &repeated(&[0, 1], 200), 100.), 1.).ring, 0.);
    }

    #[test]
    fn hands_at_different_rates_interfere_and_matched_hands_do_not() {
        let hands = |right: f64| notes(4, (0..40).map(|i| (i as f64 * 100., 0)).chain((0..(4000. / right) as usize).map(|i| (i as f64 * right + 10., 3))).collect());
        assert!(flow(&hands(200.), 1.).polyrate > 0.1);
        assert!(flow(&hands(100.), 1.).polyrate < 1e-9);
    }

    #[test]
    fn sparse_notes_are_isolated_and_a_stream_is_not() {
        assert!((flow(&chart(4, &repeated(&[0, 1], 50), 600.), 1.).isolated - 0.98).abs() < 0.03);
        assert_eq!(flow(&chart(4, &repeated(&[0, 1], 50), 100.), 1.).isolated, 0.);
    }

    #[test]
    fn a_long_vibro_run_burns_and_a_short_one_does_not() {
        let long = flow(&chart(4, &[0; 400], 80.), 1.).vibro_run; // 32 s at 12.5/s
        let short = flow(&chart(4, &[0; 20], 80.), 1.).vibro_run; // 1.6 s
        assert!(long > 0.9 && short == 0., "{long} {short}");
    }

    #[test]
    fn an_even_sweep_rolls_and_a_split_stream_does_not() {
        assert!(flow(&chart(4, &repeated(&[0, 1, 2, 3], 50), 50.), 1.).roll > 0.4);
        assert_eq!(flow(&chart(4, &repeated(&[0, 2, 1, 3], 50), 50.), 1.).roll, 0.);
    }

    #[test]
    fn one_hand_jumps_in_a_row_chain_and_alternating_hands_do_not() {
        let jumps = |hands: &[[u8; 2]]| notes(7, (0..200).flat_map(|i| hands[i % hands.len()].map(|c| (i as f64 * 150., c))).collect());
        assert!(flow(&jumps(&[[0, 1]]), 1.).ohj > 0.9);
        assert_eq!(flow(&jumps(&[[0, 1], [4, 5]]), 1.).ohj, 0.);
    }

    #[test]
    fn a_trill_flows() {
        assert_eq!(flow(&chart(7, &repeated(&[0, 1], 200), 100.), 1.).broken, 0.);
    }

    #[test]
    fn two_columns_a_hand_have_no_order_to_break() {
        assert_eq!(flow(&chart(4, &repeated(&[0, 1, 0, 0, 1, 1], 300), 100.), 1.).broken, 0.);
    }

    #[test]
    fn repeating_one_pattern_stops_being_new() {
        let same = flow(&chart(7, &repeated(&[0, 1, 2], 600), 100.), 1.);
        let cols: Vec<u8> = (0..600u32).scan(1u32, |x, _| { *x = x.wrapping_mul(1664525).wrapping_add(1013904223); Some((*x >> 24) as u8 % 7) }).collect();
        assert!(same.change < 0.02 && flow(&chart(7, &cols, 100.), 1.).change > same.change + 0.1);
    }

    #[test]
    fn long_notes_rolling_onto_a_neighbour_are_counted() {
        let mut c = chart(7, &repeated(&[0, 1, 2], 90), 30.);
        c.end = c.t.iter().map(|t| t + 200.).collect();
        assert!(flow(&c, 1.).fast_ln > 0.5);
        assert_eq!(flow(&chart(7, &repeated(&[0, 1, 2], 90), 30.), 1.).fast_ln, 0.);
    }

    #[test]
    fn a_hand_chord_with_a_finger_left_out_reads_split() {
        // 10K left hand: [0,2] leaves the ring finger up between two pressing ones, [0,1] doesn't; the right hand plays [5,6].
        let rows = |pairs: &[[u8; 2]]| {
            let cols: Vec<u8> = pairs.iter().cycle().take(400).flatten().copied().collect();
            let t: Vec<f64> = (0..cols.len()).map(|i| (i / 2) as f64 * 100.).collect();
            Chart { keys: 10, end: t.clone(), t, col: cols, ..Chart::default() }
        };
        assert!((flow(&rows(&[[0, 2], [5, 6]]), 1.).split - 0.5).abs() < 0.01);
        assert_eq!(flow(&rows(&[[0, 1], [5, 6]]), 1.).split, 0.);
    }
}

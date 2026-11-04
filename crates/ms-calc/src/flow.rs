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
}

impl Flow {
    /// Kept with the chart's other execution numbers: the next calibration needn't read the file again.
    pub fn store(&self, execution: &mut crate::val::Dict) {
        for (name, v) in [("flow_broken", self.broken), ("flow_skip", self.skip), ("flow_change", self.change), ("fast_ln", self.fast_ln), ("flow_split", self.split)] {
            execution.set(name, v);
        }
    }
}

pub const NAMES: [&str; 5] = ["flow_broken", "flow_skip", "mental", "fast_ln", "flow_split"];
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
    v.extend([f.broken, f.skip, f.change * (1. - (-length_ms / 1000. / rate / 120.).exp()) * mental_weight(stars), f.fast_ln, f.split]);
    let (r, l) = (rate.ln(), stars.ln() - 1.8);
    v.extend((0..DIRECT).map(|i| v[i] * r).collect::<Vec<_>>());
    v.extend((0..DIRECT).map(|i| v[i] * l).collect::<Vec<_>>());
    v.push(l);
    v.extend(extra);
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
fn hand(col: usize, keys: usize, centre_left: bool) -> (usize, f64) {
    (if col < keys / 2 || (centre_left && keys % 2 == 1 && col == keys / 2) { 0 } else { 1 }, col as f64)
}

pub fn flow(chart: &Chart, rate: f64) -> Flow {
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
    let splits: &[bool] = if chart.keys % 2 == 1 { &[true, false] } else { &[true] };
    let (mut broken, mut skip, mut split) = (0., 0., 0.);
    for &centre_left in splits {
        let (b, s) = order(chart, scale, centre_left);
        broken += b / splits.len() as f64;
        skip += s / splits.len() as f64;
        split += gapped(&rows, chart.keys, centre_left) / splits.len() as f64;
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
    Flow { broken, skip, change: changed as f64 / rows.len().max(2).saturating_sub(1) as f64, fast_ln: rolled as f64 / heads.len().max(1) as f64, split }
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

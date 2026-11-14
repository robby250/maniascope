//! Per-note records of a chart (`calib rows` for replays, the app's per-note forecast for charts): chart-only inputs for every
//! note, plus the replay's judgement fields when a play supplies them (left 0/NaN otherwise).
use std::collections::{HashMap, VecDeque};

pub const W: usize = 76;
pub const COLUMNS: [&str; W] = [
    "play", "keys", "rate", "od", "t", "frac", "col", "hand", "finger", "ln", "held", "j", "err", "signed", "row", "size", "n_hand", "n_other",
    "row_gap", "hand_gap", "col_gap", "prev_width", "kept", "added", "dropped", "jacked", "other_changed", "other_kept", "span", "skips", "nb_kept", "same2",
    "surp_row", "surp_hand", "uniq_1s", "rows_1s", "notes_1s", "hand_rows_1s", "bits_1s", "demand", "cj", "recent_miss", "recent_lost", "prev_size", "row_changed", "centre",
    "fail", "row_miss", "next_gap", "next_kept", "next_added", "fingers_05", "hand_notes_05", "chord_run", "mix_run", "rep_run", "move", "gap_ratio", "col_run", "col_1s", "col_next", "hand_bits_1s", "shapes_2s", "shape_bits", "hold_len", "tail_j", "tail_err", "load_2", "load_8", "load_30", "load_120", "all_8", "all_30", "all_120", "hold_30", "block",
];

/// Surprise of the next chord given the last, over a sliding window: −log2 of an order-1 estimate, backed off to chord frequency.
/// A repeated or looping pattern costs near 0 bits, a generated one costs close to the keymode's whole alphabet.
struct Markov {
    seen: VecDeque<(u32, u32)>,
    pair: HashMap<(u32, u32), u32>,
    from: HashMap<u32, u32>,
    freq: HashMap<u32, u32>,
    alphabet: f64,
    window: usize,
}

impl Markov {
    fn new(alphabet: f64, window: usize) -> Markov {
        Markov { seen: VecDeque::new(), pair: HashMap::new(), from: HashMap::new(), freq: HashMap::new(), alphabet, window }
    }

    fn step(&mut self, prev: u32, cur: u32) -> f64 {
        let n = self.seen.len() as f64;
        let get = |m: &HashMap<u32, u32>, k: u32| f64::from(m.get(&k).copied().unwrap_or(0));
        let base = (get(&self.freq, cur) + 0.5) / (n + 0.5 * self.alphabet);
        let p = (f64::from(self.pair.get(&(prev, cur)).copied().unwrap_or(0)) + base) / (get(&self.from, prev) + 1.);
        *self.pair.entry((prev, cur)).or_default() += 1;
        *self.from.entry(prev).or_default() += 1;
        *self.freq.entry(cur).or_default() += 1;
        self.seen.push_back((prev, cur));
        if self.seen.len() > self.window {
            let (a, b) = self.seen.pop_front().unwrap();
            *self.pair.get_mut(&(a, b)).unwrap() -= 1;
            *self.from.get_mut(&a).unwrap() -= 1;
            *self.freq.get_mut(&b).unwrap() -= 1;
        }
        -p.log2()
    }
}

fn centroid(m: u32) -> f64 {
    if m == 0 { return 0.; }
    (0..32).filter(|c| m >> c & 1 == 1).sum::<u32>() as f64 / f64::from(m.count_ones())
}

#[derive(Clone, Copy)]
pub struct Input {
    pub t: f64, // real seconds
    pub col: usize,
    pub ln: bool,
    pub end: f64,
    pub j: usize,
    pub err: f64, // real ms, signed; NaN on a miss
    pub fail: u8, // a miss: 1 key stayed down, 2 key stayed up, 3 pressed but the press went elsewhere
    pub tail_j: usize, // a hold's release judgement (0–5), 6 for a tap
    pub tail_err: f64, // real ms, signed; NaN without one
}

/// Per-note `demand` (the calc's total strain in the note's bin) and `cj` (its chordjack share), by real seconds; `first` is
/// the chart's first note in the same seconds.
pub fn demand_of<'a>(total: &'a [f64], cj: &'a [f64], first: f64) -> impl Fn(f64) -> (f64, f64) + 'a {
    let bin = crate::features::standard_params().k.BIN;
    move |t: f64| {
        let b = (((t - first) / bin).max(0.) as usize).min(total.len().saturating_sub(1));
        let all = total.get(b).copied().unwrap_or(0.);
        (all, if all > 0. { cj.get(b).copied().unwrap_or(0.) / all } else { 0. })
    }
}

/// A chart's records at a rate with no play: what the per-note forecast reads.
pub fn chart_records(c: &crate::chart::Chart, rate: f64) -> Result<Vec<[f32; W]>, String> {
    let mut shared = crate::compute::Shared { trace: Some(Vec::new()), ..Default::default() };
    crate::compute::compute_inner(c, rate, true, &mut shared, &crate::features::standard_params())?;
    let (_, trace) = shared.trace.iter().flatten().find(|x| x.0).ok_or("no trace")?;
    let cj = trace.series.keys.iter().find(|k| k.name() == "chordjack").map_or(&[][..], |&k| trace.series.get(k));
    let notes: Vec<Input> = (0..c.t.len()).map(|i| Input { t: c.t[i] / 1000. / rate, col: c.col[i] as usize, ln: c.end[i] > c.t[i], end: c.end[i] / 1000. / rate,
                                                             j: 0, err: f64::NAN, fail: 0, tail_j: 6, tail_err: f64::NAN }).collect();
    let first = notes.first().map_or(0., |n| n.t);
    let mut out = features(c.keys, &notes, &demand_of(&trace.total, cj, first));
    out.iter_mut().for_each(|r| (r[2], r[3]) = (rate as f32, c.od as f32));
    Ok(out)
}

/// The record of each note, sorted by time; `demand` and `cj` are read per note by the caller's closure.
pub fn features(keys: usize, notes: &[Input], at: &dyn Fn(f64) -> (f64, f64)) -> Vec<[f32; W]> {
    let hand: Vec<usize> = (0..keys).map(|c| usize::from(c >= keys / 2 + keys % 2)).collect();
    let bit = |c: usize| 1u32 << c;
    let mut rows: Vec<(usize, usize, f64, u32)> = Vec::new(); // start, end, t, mask
    let mut i = 0;
    while i < notes.len() {
        let mut j = i + 1;
        while j < notes.len() && notes[j].t - notes[i].t <= 0.003 {
            j += 1;
        }
        rows.push((i, j, notes[i].t, notes[i..j].iter().fold(0, |m, n| m | bit(n.col))));
        i = j;
    }
    let hand_mask = [(0..keys).filter(|&c| hand[c] == 0).fold(0u32, |m, c| m | bit(c)), (0..keys).filter(|&c| hand[c] == 1).fold(0u32, |m, c| m | bit(c))];
    let mut row_chain = Markov::new(2f64.powi(keys as i32) - 1., 128);
    let mut hand_chain = [Markov::new(f64::from(hand_mask[0].count_ones()).exp2() - 1., 64), Markov::new(f64::from(hand_mask[1].count_ones()).exp2() - 1., 64)];
    let (mut prev_hand, mut prev2_hand, mut prev_t) = ([0u32; 2], [0u32; 2], [f64::NEG_INFINITY; 2]);
    let mut last_col = vec![f64::NEG_INFINITY; keys];
    let (mut prev_mask, mut prev_row_t, mut prev_size) = (0u32, f64::NEG_INFINITY, 0usize);
    let mut bits: VecDeque<(f64, f64, u32)> = VecDeque::new();
    let mut hits: VecDeque<(f64, bool, f64)> = VecDeque::new();
    let mut hand_times: [VecDeque<f64>; 2] = [VecDeque::new(), VecDeque::new()];
    let times: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let mut out = Vec::with_capacity(notes.len());
    let (first, last) = (notes[0].t, notes[notes.len() - 1].t);
    let hand_rows: [Vec<(f64, u32)>; 2] = std::array::from_fn(|h| rows.iter().filter(|r| r.3 & hand_mask[h] != 0).map(|r| (r.2, r.3 & hand_mask[h])).collect());
    let mut cursor = [0usize; 2];
    let (mut chord_run, mut mix_run, mut rep_run, mut prev_gap) = ([0f32; 2], [0f32; 2], [0f32; 2], [0f64; 2]);
    let mut recent: [VecDeque<(f64, usize)>; 2] = [VecDeque::new(), VecDeque::new()];
    let mut hand_info: [VecDeque<(f64, f64, u32)>; 2] = [VecDeque::new(), VecDeque::new()];
    let by_col: Vec<Vec<f64>> = (0..keys).map(|c| notes.iter().filter(|n| n.col == c).map(|n| n.t).collect()).collect();
    let mut col_run = vec![0f32; keys]; // notes in a row on one finger, under 300 ms apart
    // The chart's own load at several time scales (ms-calc2 standing alone, plan S7): exponentially decayed notes per second
    // per hand over 2/8/30/120 s, both hands over 8/30/120 s, held LN seconds per hand over 30 s; each note counts itself.
    const TAU: [f64; 4] = [2., 8., 30., 120.];
    let (mut load, mut load_all, mut hold_load, mut load_t) = ([[0f64; 4]; 2], [0f64; 3], [0f64; 2], f64::NAN);
    for (ri, &(a, b, t, mask)) in rows.iter().enumerate() {
        if load_t.is_finite() {
            let dt = t - load_t;
            for k in 0..4 {
                let d = (-dt / TAU[k]).exp();
                load.iter_mut().for_each(|l| l[k] *= d);
                if k > 0 {
                    load_all[k - 1] *= d;
                }
                if k == 2 {
                    hold_load.iter_mut().for_each(|x| *x *= d);
                }
            }
        }
        load_t = t;
        for n in &notes[a..b] {
            let h = hand[n.col];
            load[h].iter_mut().for_each(|l| *l += 1.);
            load_all.iter_mut().for_each(|l| *l += 1.);
            if n.ln {
                hold_load[h] += n.end - n.t;
            }
        }
        let surp_row = row_chain.step(prev_mask, mask);
        bits.push_back((t, surp_row, mask));
        while bits.front().is_some_and(|x| x.0 < t - 1.) {
            bits.pop_front();
        }
        let bits_1s: f64 = bits.iter().map(|x| x.1).sum();
        let mut uniq: Vec<u32> = bits.iter().map(|x| x.2).collect();
        uniq.sort_unstable();
        uniq.dedup();
        while hits.front().is_some_and(|x| x.0 < t - 1.) {
            hits.pop_front();
        }
        let recent_miss = hits.iter().filter(|x| x.1).count();
        let recent_lost = if hits.is_empty() { 0. } else { hits.iter().map(|x| x.2).sum::<f64>() / hits.len() as f64 };
        let (lo, hi) = (times.partition_point(|&x| x < t - 0.5), times.partition_point(|&x| x <= t + 0.5));
        let notes_1s = rows[lo..hi].iter().map(|r| r.1 - r.0).sum::<usize>();
        let row_hand = [mask & hand_mask[0], mask & hand_mask[1]];
        let mut surp_hand = [0f64; 2];
        for h in 0..2 {
            if row_hand[h] != 0 {
                surp_hand[h] = hand_chain[h].step(prev_hand[h], row_hand[h]);
                hand_times[h].push_back(t);
                while hand_times[h].front().is_some_and(|&x| x < t - 1.) {
                    hand_times[h].pop_front();
                }
            }
        }
        let mut next = [(9f64, 0u32, 0u32); 2];
        for h in 0..2 {
            if row_hand[h] != 0 {
                recent[h].extend(notes[a..b].iter().filter(|n| hand[n.col] == h).map(|n| (t, n.col)));
                while recent[h].front().is_some_and(|x| x.0 < t - 0.5) {
                    recent[h].pop_front();
                }
                let was = prev_hand[h];
                chord_run[h] = if row_hand[h].count_ones() >= 2 { chord_run[h] + 1. } else { 0. };
                let mixed = row_hand[h] & was != 0 && row_hand[h] & !was != 0;
                mix_run[h] = if mixed { mix_run[h] + 1. } else { 0. };
                rep_run[h] = if row_hand[h] == was { rep_run[h] + 1. } else { 0. };
                if let Some(&(nt, nm)) = hand_rows[h].get(cursor[h] + 1) {
                    next[h] = (nt - t, (nm & row_hand[h]).count_ones(), (nm & !row_hand[h]).count_ones());
                }
                cursor[h] += 1;
                hand_info[h].push_back((t, surp_hand[h], row_hand[h]));
                while hand_info[h].front().is_some_and(|x| x.0 < t - 2.) {
                    hand_info[h].pop_front();
                }
            }
        }
        // Order-0 entropy of the hand's shapes over the last 2 s: how many chords the hand must be ready for.
        let shape_stats: [(f64, f32, f32); 2] = std::array::from_fn(|h| {
            let bits_1s: f64 = hand_info[h].iter().filter(|x| x.0 >= t - 1.).map(|x| x.1).sum();
            let mut m: Vec<u32> = hand_info[h].iter().map(|x| x.2).collect();
            m.sort_unstable();
            let n = m.len().max(1) as f64;
            let entropy: f64 = m.chunk_by(|a, b| a == b).map(|g| { let p = g.len() as f64 / n; -p * p.log2() }).sum();
            m.dedup();
            (bits_1s, m.len() as f32, entropy as f32)
        });
        let held: [bool; 2] = std::array::from_fn(|h| notes[..a].iter().rev().take(32).any(|n| n.ln && hand[n.col] == h && n.end > t + 0.003));
        let (demand, cj) = at(t);
        for n in &notes[a..b] {
            let h = hand[n.col];
            col_run[n.col] = if t - last_col[n.col] < 0.3 { col_run[n.col] + 1. } else { 0. };
            let times = &by_col[n.col];
            let (col_lo, col_at) = (times.partition_point(|&x| x < t - 1.), times.partition_point(|&x| x <= t + 0.001));
            let col_next = times.get(col_at).map_or(9., |x| (x - t).min(9.));
            let (mine, other) = (row_hand[h], row_hand[1 - h]);
            let was = prev_hand[h];
            let width = |m: u32| (m != 0).then(|| (31 - m.leading_zeros() - m.trailing_zeros() + 1) as f32).unwrap_or(0.);
            let nb = [n.col.wrapping_sub(1), n.col + 1].iter().filter(|&&c| c < keys && hand[c] == h && mine & was & bit(c) != 0).count();
            let finger = if h == 0 { n.col } else { keys - 1 - n.col };
            let centre = keys % 2 == 1 && n.col == keys / 2;
            out.push([0., keys as f32, 0., 0., t as f32, ((t - first) / (last - first).max(1.)) as f32, n.col as f32, h as f32, finger as f32, f32::from(u8::from(n.ln)), f32::from(u8::from(held[h])),
                n.j as f32, n.err.abs() as f32, n.err as f32, ri as f32, (b - a) as f32, mine.count_ones() as f32, other.count_ones() as f32,
                (t - prev_row_t).min(9.) as f32, (t - prev_t[h]).min(9.) as f32, (t - last_col[n.col]).min(9.) as f32, was.count_ones() as f32, (mine & was).count_ones() as f32, (mine & !was).count_ones() as f32, (was & !mine).count_ones() as f32,
                f32::from(u8::from(was & bit(n.col) != 0)), f32::from(u8::from(other != 0 && other != prev_hand[1 - h])), (other & prev_hand[1 - h]).count_ones() as f32,
                width(mine), width(mine) - mine.count_ones() as f32, nb as f32, f32::from(u8::from(mine == prev2_hand[h])),
                surp_row as f32, surp_hand[h] as f32, uniq.len() as f32, bits.len() as f32, notes_1s as f32, hand_times[h].len() as f32, bits_1s as f32, demand as f32, cj as f32,
                recent_miss as f32, recent_lost as f32, prev_size as f32, f32::from(u8::from(mask != prev_mask)), f32::from(u8::from(centre)),
                f32::from(n.fail), notes[a..b].iter().filter(|x| x.j == 5).count() as f32, next[h].0.min(9.) as f32, next[h].1 as f32, next[h].2 as f32,
                { let mut f: Vec<usize> = recent[h].iter().map(|x| x.1).collect(); f.sort_unstable(); f.dedup(); f.len() as f32 }, recent[h].len() as f32, chord_run[h], mix_run[h], rep_run[h],
                (centroid(mine) - centroid(was)).abs() as f32, if prev_gap[h] > 0. { ((t - prev_t[h]) / prev_gap[h]).ln().abs() as f32 } else { 0. }, col_run[n.col], (col_at - col_lo) as f32, col_next as f32, shape_stats[h].0 as f32, shape_stats[h].1, shape_stats[h].2, if n.ln { (n.end - t) as f32 } else { 0. }, n.tail_j as f32, n.tail_err as f32,
                (load[h][0] / TAU[0]) as f32, (load[h][1] / TAU[1]) as f32, (load[h][2] / TAU[2]) as f32, (load[h][3] / TAU[3]) as f32,
                (load_all[0] / TAU[1]) as f32, (load_all[1] / TAU[2]) as f32, (load_all[2] / TAU[3]) as f32, (hold_load[h] / TAU[2]) as f32, 0.]);
        }
        for n in &notes[a..b] {
            let lost = 1. - [305., 300., 200., 100., 50., 0.][n.j] / 305.;
            hits.push_back((t, n.j == 5, lost));
            last_col[n.col] = t;
        }
        for h in 0..2 {
            if row_hand[h] != 0 {
                prev_gap[h] = (t - prev_t[h]).min(9.);
                prev2_hand[h] = prev_hand[h];
                prev_hand[h] = row_hand[h];
                prev_t[h] = t;
            }
        }
        prev_mask = mask;
        prev_row_t = t;
        prev_size = b - a;
    }
    out
}


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
type Counts<K> = HashMap<K, u32, std::hash::BuildHasherDefault<crate::val::Fx>>;

pub(crate) struct Markov {
    seen: VecDeque<(u32, u32)>,
    pair: Pairs,
    from: Tally,
    freq: Tally,
    alphabet: f64,
    window: usize,
}

/// Counts by (previous, current) chord mask: a table while the alphabet squared fits one (a hand's chords, rows up to 8K), the
/// map for the rest and for a mask past the alphabet (a hash per row of every chain was a fifth of the builder's time).
struct Pairs {
    side: usize,
    table: Vec<u32>,
    map: Counts<(u32, u32)>,
}
impl Pairs {
    fn new(alphabet: f64) -> Pairs {
        let side = if alphabet < 256. { alphabet as usize + 1 } else { 0 };
        Pairs { side, table: vec![0; side * side], map: Counts::default() }
    }
    fn at(&mut self, a: u32, b: u32) -> &mut u32 {
        if (a as usize) < self.side && (b as usize) < self.side { &mut self.table[a as usize * self.side + b as usize] } else { self.map.entry((a, b)).or_default() }
    }
}

/// log2 of a small whole number, from a table made once (the windows' counts: several a row).
fn log2_of(n: usize) -> f64 {
    static TABLE: std::sync::LazyLock<Vec<f64>> = std::sync::LazyLock::new(|| (0..1024).map(|i| f64::from(i).log2()).collect());
    TABLE.get(n).copied().unwrap_or_else(|| (n as f64).log2())
}

/// Counts by chord mask: a plain table while the keymode's masks fit one (≤ 12 keys), a map beyond.
enum Tally {
    Table(Vec<u32>),
    Map(Counts<u32>),
}
impl Tally {
    fn new(alphabet: f64) -> Tally {
        if alphabet < 4096. { Tally::Table(vec![0; alphabet as usize + 1]) } else { Tally::Map(Counts::default()) }
    }
    fn at(&mut self, k: u32) -> &mut u32 {
        if let Tally::Table(t) = self {
            if k as usize >= t.len() {
                // a mask past the alphabet (never from a chart of that keymode): the map from here on
                *self = Tally::Map(t.iter().enumerate().filter(|x| *x.1 != 0).map(|(i, v)| (i as u32, *v)).collect());
            }
        }
        match self {
            Tally::Table(t) => &mut t[k as usize],
            Tally::Map(m) => m.entry(k).or_default(),
        }
    }
}

impl Markov {
    pub(crate) fn new(alphabet: f64, window: usize) -> Markov {
        Markov { seen: VecDeque::new(), pair: Pairs::new(alphabet), from: Tally::new(alphabet), freq: Tally::new(alphabet), alphabet, window }
    }

    pub(crate) fn step(&mut self, prev: u32, cur: u32) -> f64 {
        let n = self.seen.len() as f64;
        let base = (f64::from(*self.freq.at(cur)) + 0.5) / (n + 0.5 * self.alphabet);
        let pair = self.pair.at(prev, cur);
        let p = (f64::from(*pair) + base) / (f64::from(*self.from.at(prev)) + 1.);
        *pair += 1;
        *self.from.at(prev) += 1;
        *self.freq.at(cur) += 1;
        self.seen.push_back((prev, cur));
        if self.seen.len() > self.window {
            let (a, b) = self.seen.pop_front().unwrap();
            *self.pair.at(a, b) -= 1;
            *self.from.at(a) -= 1;
            *self.freq.at(b) -= 1;
        }
        -p.log2()
    }
}

fn centroid(m: u32) -> f64 {
    if m == 0 { return 0.; }
    let (mut sum, mut left) = (0u32, m);
    while left != 0 {
        sum += left.trailing_zeros();
        left &= left - 1;
    }
    f64::from(sum) / f64::from(m.count_ones())
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
pub fn demand_of<'a>(total: &'a [f64], cj: &'a [f64], first: f64, bin: f64) -> impl Fn(f64) -> (f64, f64) + 'a {
    // `bin` = the pass's BIN knob, from the caller: reading it off `standard_params()` here parsed the whole calibration
    // once a chart (73 of the note model's 100 ms on 84 charts, 2026-11-23)
    move |t: f64| {
        let b = (((t - first) / bin).max(0.) as usize).min(total.len().saturating_sub(1));
        let all = total.get(b).copied().unwrap_or(0.);
        (all, if all > 0. { cj.get(b).copied().unwrap_or(0.) / all } else { 0. })
    }
}

/// A chart's records at a rate with no play: what the per-note forecast reads.
pub fn chart_records(c: &crate::chart::Chart, rate: f64) -> Result<Vec<[f32; W]>, String> {
    let (mut shared, prm) = (crate::compute::Shared { trace: Some(Vec::new()), ..Default::default() }, crate::features::standard_params());
    crate::compute::compute_inner(c, rate, true, &mut shared, &prm)?;
    let (_, trace) = shared.trace.iter().flatten().find(|x| x.0).ok_or("no trace")?;
    let cj = trace.series.keys.iter().find(|k| k.name() == "chordjack").map_or(&[][..], |&k| trace.series.get(k));
    Ok(records_of(c, rate, &trace.total, cj, prm.k.BIN, Need::ALL))
}

/// The same records from a pass's own series (`total` = its strain per bin, `cj` = the chordjack series).
pub fn records_of(c: &crate::chart::Chart, rate: f64, total: &[f64], cj: &[f64], bin: f64, need: Need) -> Vec<[f32; W]> {
    let notes: Vec<Input> = (0..c.t.len()).map(|i| Input { t: c.t[i] / 1000. / rate, col: c.col[i] as usize, ln: c.end[i] > c.t[i], end: c.end[i] / 1000. / rate,
                                                             j: 0, err: f64::NAN, fail: 0, tail_j: 6, tail_err: f64::NAN }).collect();
    let first = notes.first().map_or(0., |n| n.t);
    let mut out = features_for(c.keys, &notes, &demand_of(total, cj, first, bin), need);
    out.iter_mut().for_each(|r| (r[2], r[3]) = (rate as f32, c.od as f32));
    out
}

/// The costlier column groups a reader of the records wants; a group left out stays 0 in every record.
/// `chains`: surp_row, surp_hand, bits_1s, hand_bits_1s, shapes_2s, shape_bits (the chord chains and the shape counts);
/// `loads`: load_2 … load_120, all_8 … all_120, hold_30 (four exp a row).
#[derive(Clone, Copy)]
pub struct Need {
    pub chains: bool,
    pub loads: bool,
}

impl Need {
    pub const ALL: Need = Need { chains: true, loads: true };

    /// What reading these columns takes.
    pub fn of(cols: &[usize]) -> Need {
        let any = |names: &[&str]| cols.iter().any(|&c| names.contains(&COLUMNS[c]));
        Need { chains: any(&["surp_row", "surp_hand", "bits_1s", "hand_bits_1s", "shapes_2s", "shape_bits"]), loads: any(&["load_2", "load_8", "load_30", "load_120", "all_8", "all_30", "all_120", "hold_30"]) }
    }
}

/// Slack at a window's edge, seconds: a note exactly a window away is inside it (a jack exactly 300 ms on is not one), whatever
/// the last bits of the times say. Without it a chart moved in time counted its edge notes in or out by rounding and the note
/// model rated the two apart (5·10⁻⁵, 2026-11-23). Chart times are milliseconds over the rate: no two are this close.
const EDGE: f64 = 1e-6;

/// The record of each note, sorted by time; `demand` and `cj` are read per note by the caller's closure.
pub fn features(keys: usize, notes: &[Input], at: &dyn Fn(f64) -> (f64, f64)) -> Vec<[f32; W]> {
    features_for(keys, notes, at, Need::ALL)
}

/// The same, with only the column groups `need` names.
pub fn features_for(keys: usize, notes: &[Input], at: &dyn Fn(f64) -> (f64, f64), need: Need) -> Vec<[f32; W]> {
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
    // the last second's bits as running sums (summed afresh at every row, they were most of the chains' cost)
    let (mut bits_sum, mut hand_sum) = (0f64, [0f64; 2]);
    let mut hand_bits: [VecDeque<(f64, f64)>; 2] = [VecDeque::new(), VecDeque::new()];
    // Running counts of what the windows hold (they were sorted and counted afresh at every row: half the builder's time).
    let (mut row_seen, mut uniq) = (Tally::new(2f64.powi(keys as i32) - 1.), 0usize);
    let mut shape_seen = [Tally::new(2f64.powi(keys as i32) - 1.), Tally::new(2f64.powi(keys as i32) - 1.)];
    let (mut shapes, mut shape_sum) = ([0usize; 2], [0f64; 2]); // distinct shapes, Σ count · log2 count
    let clog = |c: u32| if c > 1 { f64::from(c) * log2_of(c as usize) } else { 0. };
    let (mut win_lo, mut win_hi) = (0usize, 0usize);
    let row_notes: Vec<usize> = std::iter::once(0).chain(rows.iter().scan(0, |n, r| { *n += r.1 - r.0; Some(*n) })).collect(); // notes before each row
    let mut hits: VecDeque<(f64, bool, f64)> = VecDeque::new();
    let played = notes.iter().any(|n| n.j != 0); // a chart with no play loses nothing: its windows of lost accuracy stay empty
    let mut ln_end = [f64::NEG_INFINITY; 2]; // per hand: the latest release of the holds so far
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
    let mut col_win = vec![(0usize, 0usize); keys]; // per column: first note of the last second, first note after this one
    let mut col_run = vec![0f32; keys]; // notes in a row on one finger, under 300 ms apart
    // The chart's own load at several time scales (ms-calc2 standing alone, plan S7): exponentially decayed notes per second
    // per hand over 2/8/30/120 s, both hands over 8/30/120 s, held LN seconds per hand over 30 s; each note counts itself.
    const TAU: [f64; 4] = [2., 8., 30., 120.];
    let (mut load, mut load_all, mut hold_load, mut load_t) = ([[0f64; 4]; 2], [0f64; 3], [0f64; 2], f64::NAN);
    for (ri, &(a, b, t, mask)) in rows.iter().enumerate() {
        if need.loads && load_t.is_finite() {
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
                ln_end[h] = ln_end[h].max(n.end);
            }
        }
        let surp_row = if need.chains { row_chain.step(prev_mask, mask) } else { 0. };
        bits.push_back((t, surp_row, mask));
        bits_sum += surp_row;
        let seen = row_seen.at(mask);
        uniq += usize::from(*seen == 0);
        *seen += 1;
        while bits.front().is_some_and(|x| x.0 < t - 1. - EDGE) {
            let (_, was, gone) = bits.pop_front().unwrap();
            bits_sum -= was;
            *row_seen.at(gone) -= 1;
            uniq -= usize::from(*row_seen.at(gone) == 0);
        }
        if bits.len() == 1 {
            bits_sum = surp_row; // alone in its second: the running sum starts clean
        }
        let bits_1s = bits_sum;
        while hits.front().is_some_and(|x| x.0 < t - 1. - EDGE) {
            hits.pop_front();
        }
        let recent_miss = if played { hits.iter().filter(|x| x.1).count() } else { 0 };
        let recent_lost = if !played || hits.is_empty() { 0. } else { hits.iter().map(|x| x.2).sum::<f64>() / hits.len() as f64 };
        while times[win_lo] < t - 0.5 - EDGE {
            win_lo += 1;
        }
        while win_hi < times.len() && times[win_hi] <= t + 0.5 + EDGE {
            win_hi += 1;
        }
        let notes_1s = row_notes[win_hi] - row_notes[win_lo];
        let row_hand = [mask & hand_mask[0], mask & hand_mask[1]];
        let mut surp_hand = [0f64; 2];
        for h in 0..2 {
            if row_hand[h] != 0 {
                surp_hand[h] = if need.chains { hand_chain[h].step(prev_hand[h], row_hand[h]) } else { 0. };
                hand_times[h].push_back(t);
                while hand_times[h].front().is_some_and(|&x| x < t - 1. - EDGE) {
                    hand_times[h].pop_front();
                }
            }
        }
        let mut next = [(9f64, 0u32, 0u32); 2];
        for h in 0..2 {
            if row_hand[h] != 0 {
                recent[h].extend(notes[a..b].iter().filter(|n| hand[n.col] == h).map(|n| (t, n.col)));
                while recent[h].front().is_some_and(|x| x.0 < t - 0.5 - EDGE) {
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
                if !need.chains {
                    continue;
                }
                hand_info[h].push_back((t, surp_hand[h], row_hand[h]));
                hand_bits[h].push_back((t, surp_hand[h]));
                hand_sum[h] += surp_hand[h];
                let c = shape_seen[h].at(row_hand[h]);
                shape_sum[h] += clog(*c + 1) - clog(*c);
                shapes[h] += usize::from(*c == 0);
                *c += 1;
                while hand_info[h].front().is_some_and(|x| x.0 < t - 2. - EDGE) {
                    let c = shape_seen[h].at(hand_info[h].pop_front().unwrap().2);
                    shape_sum[h] += clog(*c - 1) - clog(*c);
                    *c -= 1;
                    shapes[h] -= usize::from(*c == 0);
                }
            }
        }
        // Order-0 entropy of the hand's shapes over the last 2 s: how many chords the hand must be ready for.
        let shape_stats: [(f64, f32, f32); 2] = std::array::from_fn(|h| {
            if !need.chains {
                return (0., 0., 0.);
            }
            while hand_bits[h].front().is_some_and(|x| x.0 < t - 1. - EDGE) {
                hand_sum[h] -= hand_bits[h].pop_front().unwrap().1;
            }
            if hand_bits[h].is_empty() {
                hand_sum[h] = 0.;
            }
            let bits_1s = hand_sum[h];
            // −Σ p log2 p with p = count / n: log2 n − Σ count · log2 count / n
            let n = hand_info[h].len().max(1) as f64;
            (bits_1s, shapes[h] as f32, (log2_of(hand_info[h].len().max(1)) - shape_sum[h] / n).max(0.) as f32)
        });
        // per hand, once a row: fingers used in the last half second, and how far the hand's centre moved
        let fingers: [f32; 2] = std::array::from_fn(|h| recent[h].iter().fold(0u32, |m, x| m | bit(x.1)).count_ones() as f32);
        let moved: [f32; 2] = std::array::from_fn(|h| (centroid(row_hand[h]) - centroid(prev_hand[h])).abs() as f32);
        let row_missed = notes[a..b].iter().filter(|x| x.j == 5).count() as f32;
        let held: [bool; 2] = std::array::from_fn(|h| ln_end[h] > t + 0.003 && notes[..a].iter().rev().take(32).any(|n| n.ln && hand[n.col] == h && n.end > t + 0.003));
        let (demand, cj) = at(t);
        let gap_change: [f32; 2] = std::array::from_fn(|h| if prev_gap[h] > 0. { ((t - prev_t[h]) / prev_gap[h]).ln().abs() as f32 } else { 0. });
        for n in &notes[a..b] {
            let h = hand[n.col];
            col_run[n.col] = if t - last_col[n.col] < 0.3 - EDGE { col_run[n.col] + 1. } else { 0. };
            let times = &by_col[n.col];
            let (mut col_lo, mut col_at) = col_win[n.col];
            while times[col_lo] < t - 1. - EDGE {
                col_lo += 1;
            }
            while col_at < times.len() && times[col_at] <= t + 0.001 {
                col_at += 1;
            }
            col_win[n.col] = (col_lo, col_at);
            let col_next = times.get(col_at).map_or(9., |x| (x - t).min(9.));
            let (mine, other) = (row_hand[h], row_hand[1 - h]);
            let was = prev_hand[h];
            let width = |m: u32| if m != 0 { (31 - m.leading_zeros() - m.trailing_zeros() + 1) as f32 } else { 0. };
            let nb = [n.col.wrapping_sub(1), n.col + 1].iter().filter(|&&c| c < keys && hand[c] == h && mine & was & bit(c) != 0).count();
            let finger = if h == 0 { n.col } else { keys - 1 - n.col };
            let centre = keys % 2 == 1 && n.col == keys / 2;
            out.push([0., keys as f32, 0., 0., t as f32, ((t - first) / (last - first).max(1.)) as f32, n.col as f32, h as f32, finger as f32, f32::from(u8::from(n.ln)), f32::from(u8::from(held[h])),
                n.j as f32, n.err.abs() as f32, n.err as f32, ri as f32, (b - a) as f32, mine.count_ones() as f32, other.count_ones() as f32,
                (t - prev_row_t).min(9.) as f32, (t - prev_t[h]).min(9.) as f32, (t - last_col[n.col]).min(9.) as f32, was.count_ones() as f32, (mine & was).count_ones() as f32, (mine & !was).count_ones() as f32, (was & !mine).count_ones() as f32,
                f32::from(u8::from(was & bit(n.col) != 0)), f32::from(u8::from(other != 0 && other != prev_hand[1 - h])), (other & prev_hand[1 - h]).count_ones() as f32,
                width(mine), width(mine) - mine.count_ones() as f32, nb as f32, f32::from(u8::from(mine == prev2_hand[h])),
                surp_row as f32, surp_hand[h] as f32, uniq as f32, bits.len() as f32, notes_1s as f32, hand_times[h].len() as f32, bits_1s as f32, demand as f32, cj as f32,
                recent_miss as f32, recent_lost as f32, prev_size as f32, f32::from(u8::from(mask != prev_mask)), f32::from(u8::from(centre)),
                f32::from(n.fail), row_missed, next[h].0.min(9.) as f32, next[h].1 as f32, next[h].2 as f32,
                fingers[h], recent[h].len() as f32, chord_run[h], mix_run[h], rep_run[h],
                moved[h], gap_change[h], col_run[n.col], (col_at - col_lo) as f32, col_next as f32, shape_stats[h].0 as f32, shape_stats[h].1, shape_stats[h].2, if n.ln { (n.end - t) as f32 } else { 0. }, n.tail_j as f32, n.tail_err as f32,
                (load[h][0] / TAU[0]) as f32, (load[h][1] / TAU[1]) as f32, (load[h][2] / TAU[2]) as f32, (load[h][3] / TAU[3]) as f32,
                (load_all[0] / TAU[1]) as f32, (load_all[1] / TAU[2]) as f32, (load_all[2] / TAU[3]) as f32, (hold_load[h] / TAU[2]) as f32, 0.]);
        }
        for n in &notes[a..b] {
            if played {
                hits.push_back((t, n.j == 5, 1. - [305., 300., 200., 100., 50., 0.][n.j] / 305.));
            }
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


//! Per-action effort → per-bin demand (skill_calc._demand_geometry/_demand and their row evidence).
use crate::chart::Chart;
use crate::kern::{inner, off_grid, smooth, v};
use crate::params::Knobs;
use crate::py;

/// Demand parts: one row of the [P::N × nbins] matrix each (order = skill_calc's all_keys).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum P {
    stream, jumpstream, handstream, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
    chordstream, jack, technical, bracket, dump, odd, rodd, pins, psw, oht, anc, rows, sv_raw,
    ln_rel, mj, sus, lj, shield, inv, hyb, vib, sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake,
    cover, stag, lnend, used,
}
pub const NP: usize = 40;
pub const P_NAMES: [&str; NP] = [
    "stream", "jumpstream", "handstream", "jumptrill", "splittrill", "jackspeed", "chordjack", "ln", "sv", "delay",
    "chordstream", "jack", "technical", "bracket", "dump", "odd", "rodd", "pins", "psw", "oht", "anc", "rows",
    "sv_raw", "ln_rel", "mj", "sus", "lj", "shield", "inv", "hyb", "vib", "sv_fast", "sv_slow", "sv_accel",
    "sv_stutter", "sv_brake", "cover", "stag", "lnend", "used",
];
/// Every part in declaration order (P_NAMES' order).
pub const ALL_P: [P; NP] = {
    use P::*;
    [
        stream, jumpstream, handstream, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
        chordstream, jack, technical, bracket, dump, odd, rodd, pins, psw, oht, anc, rows, sv_raw,
        ln_rel, mj, sus, lj, shield, inv, hyb, vib, sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake,
        cover, stag, lnend, used,
    ]
};
pub const PARTS_4K: [P; 9] = [P::stream, P::jumpstream, P::handstream, P::jumptrill, P::splittrill, P::jackspeed, P::chordjack, P::ln, P::sv];
pub const PARTS_5K: [P; 8] = [P::delay, P::stream, P::chordstream, P::jumptrill, P::jack, P::chordjack, P::ln, P::sv];
pub const SV_KINDS: [P; 5] = [P::sv_fast, P::sv_slow, P::sv_accel, P::sv_stutter, P::sv_brake];
const TAIL: [P; 30] = [
    P::technical, P::bracket, P::dump, P::odd, P::rodd, P::pins, P::psw, P::oht, P::anc, P::rows, P::sv_raw, P::ln_rel, P::mj, P::sus, P::lj, P::shield, P::inv, P::hyb,
    P::vib, P::sv_fast, P::sv_slow, P::sv_accel, P::sv_stutter, P::sv_brake, P::cover, P::stag, P::lnend, P::used,
    // padding never read (TAIL_LEN)
    P::stream, P::stream,
];
const TAIL_LEN: usize = 28;

/// Output order of the parts dict for a keymode.
pub fn part_keys(wide: bool) -> Vec<P> {
    let mut out: Vec<P> = if wide { PARTS_5K.to_vec() } else { PARTS_4K.to_vec() };
    out.extend_from_slice(&TAIL[..TAIL_LEN]);
    out
}

/// A closed burst of same-column re-hits: short ones are minijacks at each re-hit's speed weight, longer ones sustained jacks.
fn add_burst(comp: &mut [f64], burst: &[(usize, f64)], max: usize) {
    for &(b, w) in burst {
        if burst.len() <= max { comp[b * NP + P::mj as usize] += w } else { comp[b * NP + P::sus as usize] += 1.0 }
    }
}

#[inline]
pub fn non_demand(p: P) -> bool {
    matches!(p, P::odd | P::rodd | P::pins | P::psw | P::oht | P::anc | P::rows | P::sv_raw | P::mj | P::sus | P::lj | P::shield | P::inv | P::hyb | P::vib | P::cover | P::stag | P::lnend | P::used)
}

/// Part series, keyed like the Python dict (`keys` = its order).
#[derive(Clone, Debug)]
pub struct Parts {
    pub nbins: usize,
    pub data: Vec<f64>,
    pub keys: Vec<P>,
}
impl Parts {
    #[inline]
    pub fn get(&self, p: P) -> &[f64] {
        let o = p as usize * self.nbins;
        &self.data[o..o + self.nbins]
    }
    #[inline]
    pub fn get_mut(&mut self, p: P) -> &mut [f64] {
        let o = p as usize * self.nbins;
        &mut self.data[o..o + self.nbins]
    }
    pub fn has(&self, p: P) -> bool {
        self.keys.contains(&p)
    }
}

/// One `_demand` run: (parts, totals, fine per side).
#[derive(Clone, Debug)]
pub struct Run {
    pub parts: Parts,
    pub totals: Vec<f64>,
    pub fine: [Vec<f64>; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub t: f64,
    pub mask: u32,
    pub start: usize,
    pub end: usize,
}
impl Row {
    #[inline]
    pub fn size(&self) -> u32 {
        self.mask.count_ones()
    }
}

/// Hand-independent rows and repeat evidence for one chart/rate calculation.
pub struct Geometry {
    pub times: Vec<f64>,
    pub ends: Vec<f64>,
    pub columns: Vec<u8>,
    pub vib: Vec<bool>,
    /// in a Longjacks run (`structure::column_runs`)
    pub lj: Vec<bool>,
    pub rows: Vec<Row>,
    pub roll: Vec<bool>,
}

pub fn geometry(chart: &Chart, rate: f64, k: &Knobs) -> Geometry {
    let inv = 0.001 / rate;
    let n = chart.len();
    let times: Vec<f64> = chart.t.iter().map(|&t| t * inv).collect();
    let ends: Vec<f64> = chart.end.iter().map(|&e| e * inv).collect();
    let columns = chart.col.clone();
    let mut vib = vec![false; n];
    let mut lj = vec![false; n];
    for r in crate::structure::column_runs(chart, rate, k) {
        r.notes.iter().for_each(|&i| if r.vibro { vib[i] = true } else { lj[i] = true });
    }
    let mut rows = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let mut m = i + 1;
        while m < n && times[m] - times[i] <= 0.003 {
            m += 1;
        }
        let mask = columns[i..m].iter().fold(0u32, |a, &c| a | 1 << c);
        rows.push(Row { t: times[i], mask, start: i, end: m });
        i = m;
    }
    let nr = rows.len();
    let mut roll = vec![false; nr];
    let mut run: Vec<usize> = Vec::new();
    let mut run_step: Option<i64> = None;
    let keys = chart.keys as i64;
    for r in 0..=nr {
        let cs = if r < nr { rows[r].mask } else { 0 };
        let single = cs.count_ones() == 1;
        let step = if single && !run.is_empty() && rows[r - 1].mask.count_ones() == 1 {
            Some((cs.trailing_zeros() as i64 - rows[r - 1].mask.trailing_zeros() as i64).rem_euclid(keys))
        } else {
            None
        };
        if let Some(s) = step {
            if (s == 1 || s == keys - 1) && (run.len() < 2 || Some(s) == run_step) {
                run.push(r);
                run_step = step;
                continue;
            }
        }
        if run.len() >= k.ROLL_MIN {
            for &x in &run {
                roll[x] = true;
            }
        }
        run.clear();
        if single {
            run.push(r);
        }
        run_step = None;
    }
    Geometry { times, ends, columns, vib, lj, rows, roll }
}

/// Row families for Pattern Tech switches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trill {
    None,
    Jumptrill,
    Splittrill,
}

/// Pattern Tech evidence per row (inserted repeats, chained switches).
pub fn pattern_evidence(rows: &[Row], trill: &[Trill], k: &Knobs) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len();
    let rep: Vec<bool> =
        (0..n).map(|r| r > 0 && rows[r].mask & rows[r - 1].mask != 0 && rows[r].t - rows[r - 1].t < 0.25).collect();
    let mut ins = vec![0.0; n];
    let mut sws = vec![0.0; n];
    let mut last: Option<f64> = None;
    for r in 0..n {
        let t = rows[r].t;
        let g = if r > 0 { t - rows[r - 1].t } else { 1.0 };
        if rep[r] && r >= 3 && !(rep[r - 1] || rep[r - 2] || r + 1 < n && rep[r + 1]) {
            ins[r] = smooth(1.0 / g, 6.0, 12.0) * smooth(1.0 / (rows[r - 1].t - rows[r - 2].t), 6.0, 12.0);
        }
        if r >= 3
            && r + 2 < n
            && trill[r] != trill[r - 1]
            && trill[r - 1] == trill[r - 2]
            && trill[r - 2] == trill[r - 3]
            && trill[r] == trill[r + 1]
            && trill[r + 1] == trill[r + 2]
        {
            let sw = smooth(1.0 / g, 5.0, 10.0);
            if let Some(l) = last {
                if t - l < k.PT_CHAIN {
                    sws[r] = sw;
                }
            }
            if sw > 0.3 {
                last = Some(t);
            }
        }
    }
    (ins, sws)
}

/// 1H Trill runs with the other hand playing in between: the hand's own rows alternate two single columns at a steady
/// gap ≤ 0.2 s, and the other hand plays on at most 3 of the run's rows (its own or between) and chords at most one trill row (more is a chord pattern or bracket) and never alternates two columns itself (that is
/// a split trill or bracket). hand_patterns only finds trills where the other hand is silent: the user's 2,201 7K charts
/// never rated 1H Trill above .28 (2026-11-10, "0 1h trill plays").
fn mixed_trills(rows: &[Row], hand_mask: [u32; 2], oht: &mut [bool]) {
    for (h, &hm) in hand_mask.iter().enumerate() {
        let other = hand_mask[1 - h];
        let mine: Vec<usize> = (0..rows.len()).filter(|&r| rows[r].mask & hm != 0).collect();
        let mut between: Vec<u32> = Vec::new();
        for w in mine.windows(4) {
            let m = [0, 1, 2, 3].map(|x| rows[w[x]].mask & hm);
            if m.iter().any(|x| x & (x - 1) != 0) || m[0] == m[1] || m[0] != m[2] || m[1] != m[3] {
                continue;
            }
            let gaps = [0, 1, 2].map(|x| rows[w[x + 1]].t - rows[w[x]].t);
            let (mx, mn) = gaps.iter().fold((f64::MIN, f64::MAX), |a, &g| (a.0.max(g), a.1.min(g)));
            if mx > 0.2 || mx > 1.6 * mn {
                continue;
            }
            between.clear();
            between.extend((w[0]..=w[3]).map(|r| rows[r].mask & other).filter(|&x| x != 0));
            let alternating = between.len() >= 3 && between.windows(3).all(|t| t[0] == t[2] && t[0] != t[1] && t[0] & (t[0] - 1) == 0);
            // Chords on two of the four trill rows are a chord pattern (4K "[13]2[14]2" chordjack), not one hand trilling.
            let chords = w.iter().filter(|&&r| rows[r].mask & other != 0).count();
            if between.len() <= 3 && !alternating && chords <= 1 {
                w.iter().for_each(|&r| oht[r] = true);
            }
        }
    }
}

/// Row membership of 1H Trill and Anchor.
fn hand_patterns(rows: &[Row], hand_mask: [u32; 2], keys: usize) -> (Vec<bool>, Vec<bool>) {
    let n = rows.len();
    let mut oht = vec![false; n];
    let mut anc = vec![false; n];
    for hm in hand_mask {
        for r in 3..n {
            let (a, b) = (rows[r - 3].mask & hm, rows[r - 2].mask & hm);
            if a != 0
                && b != 0
                && a & (a - 1) == 0
                && b & (b - 1) == 0
                && a == rows[r - 1].mask & hm
                && b == rows[r].mask & hm
                && a != b
                && ((rows[r - 3].mask != a) as u32
                    + (rows[r - 2].mask != b) as u32
                    + (rows[r - 1].mask != a) as u32
                    + (rows[r].mask != b) as u32)
                    <= 1
            {
                let g = [rows[r - 1].t - rows[r - 2].t, rows[r].t - rows[r - 1].t, 0.0];
                let g0 = rows[r - 2].t - rows[r - 3].t;
                let gaps = [g0, g[0], g[1]];
                let mx = gaps.iter().cloned().fold(f64::MIN, f64::max);
                let mn = gaps.iter().cloned().fold(f64::MAX, f64::min);
                if mx <= 0.2 && mx <= 1.6 * mn {
                    for x in &mut oht[r - 3..=r] {
                        *x = true;
                    }
                }
            }
        }
    }
    let mut others = vec![0usize; keys];
    // every column's rows in one walk over the rows (it walked them once per column)
    let mut occs: Vec<Vec<usize>> = vec![Vec::new(); keys];
    for (r, row) in rows.iter().enumerate() {
        let mut m = row.mask;
        while m != 0 {
            if let Some(o) = occs.get_mut(m.trailing_zeros() as usize) {
                o.push(r);
            }
            m &= m - 1;
        }
    }
    for c in 0..keys {
        let bit = 1u32 << c;
        let occ = &occs[c];
        let mut run: Vec<usize> = occ.first().copied().into_iter().collect();
        for (x, &a) in occ.iter().enumerate() {
            let z = occ.get(x + 1).copied();
            if let Some(z) = z {
                if (2..=4).contains(&(z - a)) && rows[z].t - rows[a].t < 0.5 {
                    run.push(z);
                    continue;
                }
            }
            if run.len() >= 5 && run.len() * 3 >= run[run.len() - 1] - run[0] + 1 {
                others.iter_mut().for_each(|o| *o = 0);
                for r in run[0]..=run[run.len() - 1] {
                    let m = rows[r].mask & !bit;
                    for x in 0..keys {
                        if m >> x & 1 == 1 {
                            others[x] += 1;
                        }
                    }
                }
                let distinct = others.iter().filter(|&&o| o > 0).count();
                let mx = others.iter().copied().max().unwrap_or(0);
                if distinct >= 3 && mx as f64 <= 0.6 * run.len() as f64 {
                    for x in &mut anc[run[0]..=run[run.len() - 1]] {
                        *x = true;
                    }
                }
            }
            run.clear();
            run.extend(z);
        }
    }
    (oht, anc)
}

pub fn hand_of(keys: usize, centre_left: bool) -> Vec<u8> {
    let half = keys / 2;
    (0..keys).map(|c| if c < half || (keys % 2 == 1 && c == half && centre_left) { 0 } else { 1 }).collect()
}

/// Extra outputs of a run on an SV chart: the same physical work without the reading part.
pub struct Plain {
    pub run: Run,
}

/// A row's work that no hand split changes: its gap, share, bins, cross-hand and technical effort, SV reading and rhythm
/// evidence. An odd keymode's two splits read one copy (each made its own).
pub struct BaseRow {
    gap: f64,
    prev_gap: f64,
    share: f64,
    b: usize,
    fb: usize,
    cross: f64,
    tech: f64,
    read: f64,
    top: usize,
    odd_v: f64,
    rodd: f64,
}

/// The hand-independent part of a `demand` run: every row's `BaseRow` and each bin's pattern information.
pub struct Base {
    rows: Vec<BaseRow>,
    info: Vec<f64>,
}

pub fn base(chart: &Chart, g: &Geometry, strain: Option<&[f64]>, k: &Knobs) -> Base {
    let keys = chart.keys;
    let row_share: Vec<f64> = (0..=keys).map(|c| (c as f64).powf(-k.ROW_SHARE)).collect();
    let n = g.times.len();
    let (times, columns, roll) = (&g.times, &g.columns, &g.roll);
    let t0 = times[0];
    let max_end = g.ends.iter().cloned().fold(f64::MIN, f64::max);
    let (bin, phases) = (k.BIN, k.PHASES);
    let nbins = ((max_end - t0) / bin) as usize + 1;
    let nfine = nbins * phases;
    let zero = vec![0.0; if strain.is_none() { n } else { 0 }];
    let reading: &[f64] = strain.unwrap_or(&zero);
    let w_cross = if keys >= 5 { k.W_CROSS_WIDE } else { k.W_CROSS };
    let flam = k.FLAM;
    let mut prev_row_t = 0.0f64;
    let mut prev_gap = 0.0f64;
    let mut pp_gap = 0.0f64;
    let mut grid = 0.0f64;
    let mut ev_t = 0.0f64;
    let mut ev_gap = 0.0f64;
    let mut ev_grid = 0.0f64;
    let mut ev_seen = false;
    let mut after_roll = false;
    let mut rows = Vec::with_capacity(g.rows.len());
    for (ri, row) in g.rows.iter().enumerate() {
        let (i, j) = (row.start, row.end);
        let t = times[i];
        let size = j - i;
        let gap = if ri > 0 { t - prev_row_t } else { 0.0 };
        let share = row_share.get(size).copied().unwrap_or_else(|| (size as f64).powf(-k.ROW_SHARE))
            * if k.ROW_FAST != 0.0 && size > 1 && ri > 0 { (size as f64).powf(-k.ROW_FAST * (1.0 - smooth(gap, k.ROW_FAST_LO, k.ROW_FAST_HI))) } else { 1.0 };
        let b = ((t - t0) / bin) as usize;
        let fb = (nfine - 1).min(((t - t0) * phases as f64 / bin) as usize);
        let cross = if gap != 0.0 { w_cross * v(gap, flam, 2.0) } else { 0.0 };
        // gap / prev_gap's octaves: read up to four times a row (irregularity, grid, roll), one log2
        let octaves = std::cell::Cell::new(f64::NAN);
        let ratio = || {
            if octaves.get().is_nan() {
                octaves.set((gap / prev_gap).log2().abs());
            }
            octaves.get()
        };
        let mut rodd = 0.0;
        let mut tech = 0.0;
        if gap != 0.0 && gap < 0.5 {
            let irr = if gap < 0.04 {
                1.0
            } else if prev_gap != 0.0 && prev_gap < 0.5 {
                let x = ratio();
                let frac = (x - py::round(x)).abs();
                if x < 0.08 {
                    0.0
                } else {
                    1f64.min(2.2 * 0f64.max(frac - 0.06) + 0.25 * x.min(1.0))
                }
            } else {
                0.0
            };
            tech = k.W_TECH * v(gap, flam, 2.0) * irr;
        }
        let mut read = 0.0;
        let mut top = i;
        for ni in i..j {
            if reading[ni] > reading[top] {
                top = ni;
            }
        }
        if gap != 0.0 {
            read = k.W_SV * reading[top] * (6.0 * v(gap, flam, 2.0)).sqrt();
        }
        let mut odd_v = 0.0;
        if gap != 0.0 && gap < 0.04 {
            let nxt = if j < n { times[j] - t } else { 1.0 };
            odd_v = if !(prev_gap != 0.0 && prev_gap < 0.06 || nxt < 0.06) { 1.0 } else { 0.0 };
        } else if gap != 0.0
            && (0.06..0.5).contains(&gap)
            && (grid != 0.0 || (prev_gap != 0.0 && (0.06..0.5).contains(&prev_gap) && !(pp_gap != 0.0 && pp_gap < 0.06)))
            && !(prev_gap != 0.0 && prev_gap < 0.06 && pp_gap != 0.0 && pp_gap < 0.06)
        {
            let x = if grid != 0.0 { (gap / grid).log2().abs() } else { ratio() };
            odd_v = 1f64.min(2.2 * 0f64.max((x - py::round(x)).abs() - 0.06));
        }
        if gap != 0.0 && (0.06..0.5).contains(&gap) && prev_gap != 0.0 && ratio() < 0.06 {
            grid = gap;
        }
        let nxt = if j < n { times[j] - t } else { 0.0 };
        let rolling = roll[ri]
            && gap != 0.0
            && (prev_gap != 0.0 && ratio() < 0.06 || (nxt != 0.0 && (gap / nxt).log2().abs() < 0.06));
        if ri == 0 || gap >= k.R_EVENT {
            let eg = if ev_seen { t - ev_t } else { 0.0 };
            let judged = !(rolling || after_roll);
            let evref = if ev_grid != 0.0 { ev_grid } else { ev_gap };
            if judged && eg != 0.0 && eg < 0.5 && (ev_grid != 0.0 || (ev_gap != 0.0 && ev_gap < 0.5)) {
                rodd = (if gap != 0.0 && gap < 0.04 { odd_v } else { 0.0 }).max(off_grid(eg, evref, ev_grid != 0.0));
            } else if judged && gap != 0.0 && gap < 0.04 {
                rodd = odd_v;
            }
            if rolling {
            } else if eg != 0.0 && eg < 0.5 && ev_gap != 0.0 && (eg / ev_gap).log2().abs() < 0.06 {
                ev_grid = eg;
            } else if eg == 0.0 || eg >= 0.5 {
                ev_grid = 0.0;
            }
            ev_t = t;
            ev_gap = eg;
            after_roll = rolling;
            ev_seen = true;
        } else {
            rodd = odd_v;
        }
        rows.push(BaseRow { gap, prev_gap, share, b, fb, cross, tech, read, top, odd_v, rodd });
        pp_gap = prev_gap;
        prev_gap = gap;
        prev_row_t = t;
    }
    // Pattern information (replay study 2026-11-17: Sea of Stars' gap to Onus/Blossom follows the surprise bits per second of its
    // chord sequence at high density): each row's −log2 under an order-1 chain over the last 128 rows (notefeat's), per keymode bit
    // (÷ keys), summed per bin; the bin's total × e^(PATTERN_INFO · max(0, that − PATTERN_INFO_B)).
    let info: Vec<f64> = if k.PATTERN_INFO == 0.0 { Vec::new() } else {
        let mut chain = crate::notefeat::Markov::new(2f64.powi(keys as i32) - 1., 128);
        let mut out = vec![0.0f64; nbins];
        let (mut i, mut prev) = (0usize, 0u32);
        while i < n {
            let mut j = i;
            let mut mask = 0u32;
            while j < n && times[j] - times[i] <= 0.003 {
                mask |= 1 << columns[j];
                j += 1;
            }
            let b = (nbins - 1).min(((times[i] - t0) / bin) as usize);
            out[b] += chain.step(prev, mask) / keys as f64;
            prev = mask;
            i = j;
        }
        out
    };
    Base { rows, info }
}

/// One `_demand` run (see skill_calc._demand).
pub fn demand(chart: &Chart, g: &Geometry, centre_left: bool, strain: Option<&[f64]>, kinds: Option<&[[f64; 5]]>, want_plain: bool, k: &Knobs) -> (Run, Option<Run>) {
    demand_with(chart, g, &base(chart, g, strain, k), centre_left, kinds, want_plain, k)
}

/// `demand` on a `base` made once for the chart's hand splits.
pub fn demand_with(chart: &Chart, g: &Geometry, base: &Base, centre_left: bool, kinds: Option<&[[f64; 5]]>, want_plain: bool, k: &Knobs) -> (Run, Option<Run>) {
    crate::tick::lap("d.before");
    let keys = chart.keys;
    let wide = keys >= 5;
    let x = (keys as f64 / 7.0).ln(); // keymode curves: quadratics in ln(keys/7), 0 at 7K (plan 2026-11-11: no hinge at 7K)
    let v_fast = k.V_FAST * (k.V_FAST_A * x + k.V_FAST_B * x * x).exp();
    // One curve over keymodes (user 2026-11-03: "gradients over the 4k-10k spectrum").
    let w_jack = k.W_JACK * (7.0 / keys as f64).powf(k.W_JACK_KEYS);
    let w_group = k.W_GROUP * (k.GROUP_A * x + k.GROUP_B * x * x).exp();
    let jack_share = k.JACK_SHARE * (k.JACK_SHARE_A * x + k.JACK_SHARE_B * x * x).exp();
    let jack_p = k.JACK_P * (keys as f64 / 4.0).powf(k.JACK_P_KEYS);
    let run_finger: [f64; 32] = std::array::from_fn(|r| ((1 + r) as f64 / k.JACK_RUN_REF).powf(k.JACK_RUN_P));
    let run_arm: [f64; 32] = std::array::from_fn(|r| ((1 + r) as f64 / k.JACK_RUN_REF).powf(k.JACK_ARM_RUN_P));
    let arm_v = k.JACK_ARM_V * (keys as f64 / 4.0).powf(k.JACK_ARM_KEYS);
    let arm_at = k.JACK_ARM * arm_v * (arm_v / 12.0).powf(jack_p);
    // Per-row and per-note powers with few distinct inputs, computed once (same calls, same values).
    let by_count = |e: f64| -> Vec<f64> { (0..=keys).map(|c| (c as f64).powf(e)).collect() };
    let jack_split = by_count(-jack_share);
    let shapes_t = k.SHAPES_T * (10.0 / keys as f64).powf(k.SHAPES_KEYS);
    let mix_edge = k.MIX_V * (7.0 / keys as f64).powf(k.MIX_KEYS);
    let vf_jacked = 1.0 - k.VF_JACK * (4.0 / keys as f64).powf(k.VF_JACK_KEYS);
    let rhd_n = k.RHD_N * (keys as f64 / 7.0).powf(k.RHD_X);
    // max(1, x)^p is exactly 1 for x ≤ 1: no pow call.
    let above_one = |x: f64, p: f64| if x > 1.0 { x.powf(p) } else { 1.0 };
    let hand_of = hand_of(keys, centre_left);
    let n = g.times.len();
    let times = &g.times;
    let ends = &g.ends;
    let columns = &g.columns;
    let t0 = times[0];
    let max_end = ends.iter().cloned().fold(f64::MIN, f64::max);
    let bin = k.BIN;
    let phases = k.PHASES;
    let nbins = ((max_end - t0) / bin) as usize + 1;
    let nfine = nbins * phases;
    // per-column state in fixed arrays, a column masked to 0–31 (as the rows' u32 masks hold it): no bounds check per note
    const C: usize = 31;
    let mut order = [0usize; 32];
    let mut seen = 0u32;
    let mut last_head = [0.0f64; 32];
    let mut last_end = [0.0f64; 32];
    let mut last_size = [0usize; 32];
    let mut hold_weight = [0.0f64; 32];
    let mut nseen = 0usize;
    let mut previous_hand = [0.0f64; 2];
    let mut previous_mask = [0u32; 2];
    let mut hand_mask = [0u32; 2];
    for c in 0..keys {
        hand_mask[hand_of[c] as usize] |= 1 << c;
    }
    let mut held_until = f64::MIN; // the latest release so far: past it no column is held and the hold scan finds nothing
    let mut jrun = [0usize; 32];
    let mut lj_row = [0u32; 32]; // the row of each column's last hit
    let mut burst: Vec<(usize, f64)> = Vec::new(); // bin, the re-hit's minijack weight
    crate::tick::lap("d.setup");
    let rows = &g.rows;
    let nr = rows.len();
    let mut trill = vec![Trill::None; nr];
    for r in 3..nr {
        let (a, bb, cc, dd) = (rows[r - 3].mask, rows[r - 2].mask, rows[r - 1].mask, rows[r].mask);
        if a.count_ones().min(bb.count_ones()).min(cc.count_ones()).min(dd.count_ones()) >= 2
            && rows[r].t - rows[r - 3].t < 3.0 * k.TRILL_GAP
        {
            let hands = |m: u32| -> u8 {
                let mut s = 0u8;
                for c in 0..keys {
                    if m >> c & 1 == 1 {
                        s |= 1 << hand_of[c];
                    }
                }
                s
            };
            let (ha, hb, hc, hd) = (hands(a), hands(bb), hands(cc), hands(dd));
            let kind = if ha.count_ones() == 1 && hb.count_ones() == 1 && ha != hb && ha == hc && hb == hd {
                Trill::Jumptrill
            } else if !wide && cc == a && dd == bb && a & bb == 0 && ha.count_ones() == 2 && hb.count_ones() == 2 {
                Trill::Splittrill
            } else {
                Trill::None
            };
            if kind != Trill::None {
                for x in &mut trill[r - 3..=r] {
                    *x = kind;
                }
            }
        }
    }
    crate::tick::lap("d.trill");
    let (p_ins, p_sw) = pattern_evidence(rows, &trill, k);
    crate::tick::lap("d.evidence");
    let (mut oht, anc) = hand_patterns(rows, hand_mask, keys);
    crate::tick::lap("d.hand_patterns");
    if k.OHT_MIX > 0.0 {
        mixed_trills(rows, hand_mask, &mut oht);
    }
    crate::tick::lap("d.mixed_trills");
    let keys_out = part_keys(wide);
    let mut comp = vec![0.0f64; NP * nbins];
    let mut hands = vec![0.0f64; 3 * nbins];
    let mut fine = vec![0.0f64; 3 * nfine];
    let mut plain_x = if want_plain { vec![0.0f64; nbins] } else { Vec::new() };
    let mut plain_fine_x = if want_plain { vec![0.0f64; nfine] } else { Vec::new() };
    let mut rows_by_size = [vec![0usize; nbins], vec![0usize; nbins], vec![0usize; nbins]];
    let o = |p: P| p as usize; // comp is bin-major: a row's parts sit side by side (they were NP series nbins apart)
    let mut prev_row_t = 0.0f64;
    let mut prev_row_mask = 0u32;
    let mut pending: Vec<(f64, usize)> = Vec::with_capacity(48);
    let mut shapes_seen: [Vec<(f64, u32)>; 2] = [Vec::new(), Vec::new()];
    let mut shapes_from = [0usize; 2];
    let mut hand_recent: [Vec<f64>; 2] = [Vec::new(), Vec::new()]; // a hand's note heads in the last RHD_SPAN s
    let mut shape_count: [Vec<(u32, u32)>; 2] = [Vec::new(), Vec::new()]; // (shape, times in the window)
    let flam = k.FLAM;
    crate::tick::lap("d.alloc");
    for (ri, row) in rows.iter().enumerate() {
        let (i, j) = (row.start, row.end);
        let t = times[i];
        let size = j - i;
        let &BaseRow { gap, prev_gap, share, b, fb, cross, tech, read, top, odd_v, rodd } = &base.rows[ri];
        let tr: Option<usize> = match trill[ri] {
            Trill::None => None,
            Trill::Jumptrill => Some(o(P::jumptrill)),
            Trill::Splittrill => Some(o(P::splittrill)),
        };
        rows_by_size[size.min(3) - 1][b] += 1;
        comp[b * NP + o(P::technical)] += tech;
        comp[b * NP + o(P::sv)] += read;
        comp[b * NP + o(P::sv_raw)] += read;
        if read != 0.0 {
            if let Some(kinds) = kinds.filter(|x| !x.is_empty()) {
                for (q, &f) in SV_KINDS.iter().zip(kinds[top].iter()) {
                    comp[b * NP + o(*q)] += read * f;
                }
            }
        }
        hands[b * 3 + 2] += cross + tech + read;
        fine[fb * 3 + 2] += cross + tech + read;
        if want_plain {
            plain_x[b] += cross + tech + 0.0;
            plain_fine_x[fb] += cross + tech + 0.0;
        }
        comp[b * NP + o(P::rodd)] += rodd; // 0 where the row adds none
        comp[b * NP + o(P::rows)] += 1.0;
        comp[b * NP + o(P::pins)] += p_ins[ri];
        comp[b * NP + o(P::psw)] += p_sw[ri];
        comp[b * NP + o(P::oht)] += oht[ri] as u8 as f64;
        comp[b * NP + o(P::anc)] += anc[ri] as u8 as f64;
        let row_mask = row.mask;
        let any_vib = g.vib[i..j].iter().any(|&x| x);
        comp[b * NP + o(P::vib)] += any_vib as u8 as f64;
        if gap != 0.0 && gap < 0.25 && row_mask & prev_row_mask != 0 {
            // a minijack in full up to MJ_FULL, none from MJ_NONE: slow re-hits read Minijacks first though the user saw none
            // and only while few columns re-hit: a re-hit of most of the keys is a chordjack/mash (user 2026-11-17: 10K
            // Wonder tale, 7–8-key chords twice at 128 ms, read Minijacks first)
            let wide_rehit = smooth((row_mask & prev_row_mask).count_ones() as f64 / keys as f64, k.MJ_WIDE_LO, k.MJ_WIDE_HI);
            burst.push((b, ((k.MJ_NONE - gap) / (k.MJ_NONE - k.MJ_FULL)).clamp(0.0, 1.0) * (1.0 - wide_rehit)));
        } else {
            add_burst(&mut comp, &burst, k.MJ_BURST);
            burst.clear();
        }
        let mut n_hand = [0usize; 2];
        let mut n_jack = [0usize; 2];
        let row_hand = [row_mask & hand_mask[0], row_mask & hand_mask[1]];
        let mut mirror = [1.0f64; 2];
        let mut eh = [0.0f64; 2];
        if row_mask.count_ones() as usize == size {
            for h in 0..2 {
                n_hand[h] = row_hand[h].count_ones() as usize;
                n_jack[h] = (row_hand[h] & previous_mask[h]).count_ones() as usize;
            }
        } else {
            // a column twice in one row (two notes within 3 ms): counted per note
            for ni in i..j {
                let c = columns[ni] as usize;
                let h = hand_of[c] as usize;
                n_hand[h] += 1;
                n_jack[h] += (previous_mask[h] >> c & 1) as usize;
            }
        }
        // Hardest-7K study (2026-11-18): top players lose more on column re-hits only while the same hand is dense (Absolute
        // Zero, Decipher ≥ 14 notes a hand in 0.5 s; Victory's re-hits in sparse hands cost nothing extra); each alone failed.
        let mut hand_dense = [0.0f64; 2];
        if k.RHD_W != 0.0 {
            for h in 0..2 {
                hand_recent[h].extend(std::iter::repeat_n(t, n_hand[h]));
                let gone = hand_recent[h].partition_point(|&x| x < t - k.RHD_SPAN);
                hand_recent[h].drain(..gone);
                hand_dense[h] = (hand_recent[h].len() as f64 / rhd_n - 1.0).max(0.0);
            }
        }
        if n_jack[0] != 0 && n_jack[1] != 0 {
            let mh = if n_jack[0] <= n_jack[1] { 0 } else { 1 };
            mirror[mh] = k.MIRROR;
            // A partial mirror (some fingers of a hand jack, others move), on every keymode.
            let mut partial = false;
            for h in 0..2 {
                let old_width = previous_mask[h].count_ones() as usize;
                partial = partial || !(n_jack[h] == n_hand[h] && n_hand[h] == old_width);
            }
            if partial {
                let speed = v(t - previous_hand[0], 0.07, 0.5).min(v(t - previous_hand[1], 0.07, 0.5));
                mirror[mh] += (1.0 - k.MIRROR) * smooth(speed, 7.0, 9.0);
            }
        }
        let mut jack_effort = [0.0f64; 2];
        let mut inserted = [0.0f64; 2];
        let mut jack_arm = [f64::INFINITY; 2];
        for h in 0..2 {
            if n_jack[h] != 0 {
                let gap = t - previous_hand[h];
                let rest = w_jack * jack_split.get(n_jack[h]).copied().unwrap_or_else(|| (n_jack[h] as f64).powf(-jack_share)) * mirror[h];
                let vj = 1.0 / gap.max(0.02);
                jack_arm[h] = rest * arm_at * (vj / arm_v).powf(k.JACK_ARM_Q.max(0.0));
                jack_effort[h] = rest * vj * (vj / 12.0).powf(jack_p);
                inserted[h] = k.W_IJACK + (1.0 - k.W_IJACK) * smooth(vj, 7.0, 11.0);
            }
        }
        // How many chords the hand must have ready: user's duplicated-4K map is 44 NPS with few shapes and plays far easier than a
        // generated map at the same speed. Grows past what a 7K hand can reach, so wider keymodes pay it on their own.
        let mut many = [0.0f64; 2];
        if k.W_SHAPES != 0.0 {
            for h in 0..2 {
                if n_hand[h] >= 2 {
                    shapes_seen[h].push((t, row_hand[h]));
                    match shape_count[h].iter_mut().find(|s| s.0 == row_hand[h]) {
                        Some(s) => s.1 += 1,
                        None => shape_count[h].push((row_hand[h], 1)),
                    }
                }
                // rows come in time order: the expired shapes are a prefix, read from `shapes_from` (no shift per row)
                while shapes_from[h] < shapes_seen[h].len() && shapes_seen[h][shapes_from[h]].0 < t - 2.0 {
                    let x = shapes_seen[h][shapes_from[h]];
                    shapes_from[h] += 1;
                    let at = shape_count[h].iter().position(|s| s.0 == x.1).expect("a counted shape");
                    shape_count[h][at].1 -= 1;
                    if shape_count[h][at].1 == 0 {
                        shape_count[h].swap_remove(at);
                    }
                }
                if n_hand[h] >= 2 {
                    let count = shape_count[h].len() as f64; // the distinct shapes in the window (it scanned the window per row)
                    // up to shapes_t shapes the log is of exactly 1: 0, no call
                    many[h] = if count > shapes_t { k.W_SHAPES * (count / shapes_t).max(1.0).log2() } else { k.W_SHAPES * 0.0 };
                }
            }
        }
        let mut mix = [0.0f64; 2];
        for h in 0..2 {
            if k.W_MIX != 0.0 && row_hand[h] & previous_mask[h] != 0 && row_hand[h] & !previous_mask[h] != 0 {
                mix[h] = k.W_MIX * smooth(v(t - previous_hand[h], 0.07, 0.5), mix_edge, mix_edge * 1.3);
            }
        }
        pending.clear();
        let mut jack_w = 0.0f64;
        for ni in i..j {
            let e = ends[ni];
            let c = columns[ni] as usize & C;
            let h = (hand_mask[1] >> c & 1) as usize;
            let jacked = previous_mask[h] >> c & 1 == 1;
            // the hand's latest head on another column (a jacked note skips the columns of its row and the hand's last one):
            // every such head is before this row, so the smallest gap is the gap to the latest
            let mut others = seen & hand_mask[h] & !(1 << c) & !(if jacked { row_mask | previous_mask[h] } else { 0 });
            let mut dh = 0.0f64;
            if others != 0 {
                let mut head = f64::MIN;
                while others != 0 {
                    head = head.max(last_head[others.trailing_zeros() as usize & C]);
                    others &= others - 1;
                }
                dh = t - head;
            }
            let was = seen >> c & 1 == 1;
            let dj = if was { (t - last_head[c]).min(1.0) } else { 1.0 };
            let vdj = v(dj, flam, 2.0);
            let vdh = if dh != 0.0 { v(dh, flam, 2.0) } else { 0.0 };
            let e_hand = if dh != 0.0 {
                k.W_HAND * (vdh * vdj).sqrt() * share * above_one(vdh / v_fast, k.SPEED_P)
            } else {
                0.0
            };
            eh[h] += e_hand;
            let mut e_jack = if jacked {
                let r = jrun[c].min(31);
                (jack_effort[h] * run_finger[r]).min(jack_arm[h] * run_arm[r])
            } else {
                0.0
            };
            if jacked && prev_row_mask >> c & 1 == 0 {
                e_jack *= inserted[h];
            }
            // User: "when chords switch, it gets much harder, the higher the keycount is". Rejected on replays: entropy per row,
            // neighbour left up (W_ISO, overlaps the groups), fast jack premium, reviving V_FAST on 10K.
            let groups = (row_hand[h] & !(row_hand[h] << 1)).count_ones() as f64;
            let cost = 1. + mix[h] + many[h] + if size as f64 >= k.GROUP_SIZE { w_group * (groups - 1.) } else { 0. };
            // V_FINGER over keymodes (·(keys/7)^x) rejected 2026-11-07 (r6c): DevSelect better, DevConfirm 4K and 10K worse.
            let dense_rehit = 1.0 + k.RHD_W * (vdj / k.V_FINGER - 1.0).max(0.0) * hand_dense[h];
            let rice = (e_hand + e_jack) * above_one(vdj / k.V_FINGER, if jacked { vf_jacked } else { 1.0 }) * dense_rehit * cost;
            let rice = if keys % 2 == 1 && c == keys / 2 { rice * (1.0 + k.W_CENTRE) } else { rice };
            let mut held_same = 0.0f64;
            let mut held_other = 0.0f64;
            if held_until > t + 0.003 {
                for &oc in &order[..nseen] {
                    let oc = oc & C;
                    if oc != c && last_head[oc] < t - 0.003 && last_end[oc] > t + 0.003 {
                        if hand_mask[h] >> oc & 1 == 1 {
                            held_same += hold_weight[oc];
                        } else {
                            held_other += hold_weight[oc];
                        }
                    }
                }
            }
            let mut e_ln = (rice + k.LN_BASE * share)
                * (k.W_HOLD_SAME * if held_same > 0.0 { held_same.powf(0.7) } else { 0.0 } + k.W_HOLD_OTHER * held_other.min(1.0));
            if e - t <= 0.003 && held_same + held_other >= 0.5 && gap != 0.0 && gap < 0.15 {
                comp[b * NP + o(P::hyb)] += 1.0;
            }
            if was && last_end[c] - last_head[c] <= 0.003 && e - t > 0.003 && t - last_head[c] < 0.12 {
                comp[b * NP + o(P::shield)] += 1.0;
            }
            if was && last_end[c] - last_head[c] > 0.003 {
                let gg = t - last_end[c];
                if gg < 0.25 {
                    let rep = k.W_REPRESS
                        * v(gg, 0.04, 0.5)
                        * (1.0 - smooth(gg, 0.12, 0.25))
                        * smooth(last_end[c] - last_head[c], 0.06, 0.25)
                        * share;
                    e_ln += rep;
                    if gg < 0.12 && last_end[c] - last_head[c] >= 0.1 {
                        comp[b * NP + o(P::inv)] += 1.0;
                    }
                }
            }
            hands[b * 3 + h] += rice + e_ln;
            fine[fb * 3 + h] += rice + e_ln;
            comp[b * NP + o(P::ln)] += e_ln;
            jrun[c] = if jacked { jrun[c] + 1 } else { 0 };
            // a Longjacks note: a tap in a column run whose row shares only this column with the run's previous row (more
            // shared columns are a chordjack)
            if g.lj[ni] && e - t <= 0.003 && last_end[c] - last_head[c] <= 0.003 && (row_mask & lj_row[c]).count_ones() == 1 {
                comp[b * NP + o(P::lj)] += 1.0;
            }
            lj_row[c] = row_mask;
            if wide && e_jack != 0.0 && (jrun[c] >= 2 || n_hand[h] >= 2) {
                jack_w += 1.0;
                let part = if n_jack[h] >= 2 || n_hand[h] >= 2 { P::chordjack } else { P::jack };
                comp[b * NP + tr.unwrap_or(o(part))] += rice;
            } else if wide && e_jack != 0.0 {
                pending.push((rice, h));
            } else if !wide && e_jack != 0.0 && prev_row_mask >> c & 1 == 1 {
                jack_w += 1.0;
                let part = if size >= 2 && was && last_size[c] >= 2 { P::chordjack } else { P::jackspeed };
                comp[b * NP + tr.unwrap_or(o(part))] += rice;
            } else {
                pending.push((rice, h));
            }
        }
        comp[b * NP + o(P::odd)] += odd_v;
        let split = k.W_SPLIT * eh[0].min(eh[1]);
        hands[b * 3 + 2] += split;
        fine[fb * 3 + 2] += split;
        if want_plain {
            plain_x[b] += split;
            plain_fine_x[fb] += split;
        }
        let fast = 1.0 - smooth(if gap != 0.0 { gap } else { 1.0 }, k.DELAY_GAP - 0.01, k.DELAY_GAP + 0.01);
        if wide {
            if k.W_ALT != 0.0 {
                for h in 0..2 {
                    if row_hand[h] != 0
                        && previous_mask[h] != 0
                        && n_jack[h] == 0
                        && previous_hand[h] == prev_row_t
                        && (inner(previous_mask[h], keys) & row_hand[h] != 0
                            || inner(row_hand[h], keys) & previous_mask[h] != 0)
                    {
                        let mut alt = 0.0;
                        for &(amount, hh) in &pending {
                            if hh == h {
                                alt += amount;
                            }
                        }
                        alt *= k.W_ALT;
                        hands[b * 3 + h] += alt;
                        fine[fb * 3 + h] += alt;
                        pending.push((alt, h));
                    }
                }
            }
            let jf = jack_w / size as f64;
            let mut cs_sum = 0.0;
            let mut ds_sum = 0.0;
            for &(amount, h) in &pending {
                if n_hand[h] >= 2 {
                    cs_sum += amount;
                } else {
                    ds_sum += amount;
                }
            }
            cs_sum += if size >= 3 { cross + split } else { 0.0 };
            ds_sum += if size < 3 { cross + split } else { 0.0 };
            comp[b * NP + tr.unwrap_or(o(P::chordstream))] += cs_sum * (1.0 - jf);
            comp[b * NP + tr.unwrap_or(o(P::delay))] += ds_sum * (1.0 - jf) * fast;
            comp[b * NP + tr.unwrap_or(o(P::stream))] += ds_sum * (1.0 - jf) * (1.0 - fast);
            let part = if size >= 2 { P::chordjack } else { P::jack };
            comp[b * NP + tr.unwrap_or(o(part))] += (cs_sum + ds_sum) * jf;
            for h in 0..2 {
                let old_width = previous_mask[h].count_ones() as usize;
                if n_hand[h] != 0
                    && n_jack[h] == 0
                    && previous_mask[h] != 0
                    && previous_hand[h] == prev_row_t
                    && n_hand[h].max(old_width) >= 2
                {
                    let mut other = 0.0;
                    for &(amount, hh) in &pending {
                        if hh == h {
                            other += amount;
                        }
                    }
                    comp[b * NP + o(P::bracket)] += other * (1.0 - jf);
                }
            }
        } else {
            let jf = if size >= 2 { 1f64.min(2.0 * jack_w / size as f64) } else { jack_w };
            let mut other = 0.0;
            for &(amount, _) in &pending {
                other += amount;
            }
            other = other + cross + split;
            comp[b * NP + tr.unwrap_or(o(P::stream))] += other * (1.0 - jf);
            let part = if size >= 2 { P::chordjack } else { P::jackspeed };
            comp[b * NP + tr.unwrap_or(o(part))] += other * jf;
            if tr.is_none() {
                let mut ds_sum = 0.0;
                for &(amount, h) in &pending {
                    if n_hand[h] < 2 {
                        ds_sum += amount;
                    }
                }
                ds_sum += if size < 3 { cross + split } else { 0.0 };
                // dumps are fast rice off an even grid (bursts, mixed snaps: 769 4K charts tagged "dump" against 900 others,
                // user 2026-11-14: FREEDOM DiVE [Black Another]'s even 1/4 jumpstream read Dump)
                let dump = ds_sum * (1.0 - jf) * fast;
                if dump != 0.0 {
                    // the log only where there is fast rice to call a dump (elsewhere the product is 0 whatever it says)
                    let uneven = if gap != 0.0 && prev_gap != 0.0 { smooth((gap / prev_gap).ln().abs(), 0.1, 0.3) } else { 0.0 };
                    comp[b * NP + o(P::dump)] += dump * uneven;
                }
            }
        }
        for ni in i..j {
            let c = columns[ni] as usize & C;
            let e = ends[ni];
            if seen >> c & 1 == 0 {
                order[nseen & C] = c;
                nseen += 1;
                seen |= 1 << c;
            }
            last_head[c] = t;
            last_end[c] = e;
            held_until = held_until.max(e);
            last_size[c] = size;
            hold_weight[c] = smooth(e - t, 0.06, 0.25);
        }
        for h in 0..2 {
            if row_hand[h] != 0 {
                previous_hand[h] = t;
                previous_mask[h] = row_hand[h];
            }
        }
        prev_row_t = t;
        prev_row_mask = row_mask;
    }
    crate::tick::lap("d.main");
    // releases and staggered LN actions, per hand
    let mut acts: [Vec<(f64, u8)>; 2] = [Vec::new(), Vec::new()];
    // only a long note's release reads them: a chart without one builds and sorts nothing
    let any_ln = (0..n).any(|ni| ends[ni] - times[ni] > 0.003);
    if any_ln {
        // heads come in (time, column) order; only the releases need sorting, then one merge (it sorted every action)
        let mut heads: [Vec<(f64, u8)>; 2] = [Vec::with_capacity(n), Vec::with_capacity(n)];
        let mut tails: [Vec<(f64, u8)>; 2] = [Vec::new(), Vec::new()];
        for ni in 0..n {
            let (t, e, c) = (times[ni], ends[ni], columns[ni]);
            let h = hand_of[c as usize] as usize;
            heads[h].push((t, c));
            if e - t > 0.003 {
                tails[h].push((e, c));
            }
        }
        for h in 0..2 {
            let (hd, tl) = (&mut heads[h], &mut tails[h]);
            if !hd.is_sorted_by(|x, y| x <= y) {
                hd.sort_by(|x, y| x.partial_cmp(y).unwrap());
            }
            tl.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
            let out = &mut acts[h];
            out.reserve(hd.len() + tl.len());
            let (mut x, mut y) = (0, 0);
            while x < hd.len() || y < tl.len() {
                if y == tl.len() || (x < hd.len() && hd[x] <= tl[y]) {
                    out.push(hd[x]);
                    x += 1;
                } else {
                    out.push(tl[y]);
                    y += 1;
                }
            }
        }
    }
    let ats: [Vec<f64>; 2] = [acts[0].iter().map(|x| x.0).collect(), acts[1].iter().map(|x| x.0).collect()];
    crate::tick::lap("d.rel.acts");
    let sync_share = by_count(-k.ROW_SHARE);
    let mut holds: Vec<Vec<(f64, f64)>> = vec![Vec::new(); keys]; // per column, in time order (a column's holds don't overlap)
    let mut used = vec![0u32; nbins]; // columns pressed or held in each bin: what the held share is a share of
    for ni in 0..n {
        let bin_of = |x: f64| (nbins - 1).min(((x - t0) / bin).max(0.0) as usize);
        let (first, last) = (bin_of(times[ni]), bin_of(ends[ni].max(times[ni])));
        used[first..=last].iter_mut().for_each(|m| *m |= 1 << columns[ni]);
        if ends[ni] - times[ni] > 0.003 {
            holds[columns[ni] as usize].push((times[ni], ends[ni]));
        }
    }
    for (b, m) in used.iter().enumerate() {
        comp[b * NP + o(P::used)] = f64::from(m.count_ones()) * bin;
    }
    crate::tick::lap("d.rel.used");
    for ni in 0..n {
        let (t, e, c) = (times[ni], ends[ni], columns[ni]);
        let ln_len = e - t;
        if ln_len <= 0.003 {
            continue;
        }
        let h = hand_of[c as usize] as usize;
        let at = &ats[h];
        // the release's own neighbours only: actions within 5 ms (in sync) and the last one before them, back to 0.18 s
        // (it walked the whole ±0.18 s window; same tests on the same values)
        let hi = at.partition_point(|&x| x <= e + 0.006);
        let mut sync = 1usize;
        let mut prev_d = 0.0f64;
        for oi in (0..hi).rev() {
            let ot = at[oi];
            if ot < e - 0.18 {
                break;
            }
            if acts[h][oi].1 == c {
                continue;
            }
            if (ot - e).abs() <= 0.005 {
                sync += 1;
            }
            if ot < e - 0.005 {
                prev_d = e - ot;
                break;
            }
        }
        let base = k.W_REL * (k.LN_BASE + if prev_d != 0.0 { v(prev_d, flam, 2.0) } else { 0.0 });
        // No Release: a tail held past its time is Perfect, so the release asks for no timing
        let eff = if chart.no_release { 0.0 } else { base * sync_share.get(sync).copied().unwrap_or_else(|| (sync as f64).powf(-k.ROW_SHARE)) * smooth(ln_len, 0.06, 0.25) };
        let b = (nbins - 1).min(((e - t0) / bin) as usize);
        let fb = (nfine - 1).min(((e - t0) * phases as f64 / bin) as usize);
        hands[b * 3 + h] += eff;
        fine[fb * 3 + h] += eff;
        comp[b * NP + o(P::ln)] += eff;
        comp[b * NP + o(P::ln_rel)] += eff;
        // skills::attribute's LN shares: releases, the staggered ones (another finger of the hand holds on), held time
        comp[b * NP + o(P::lnend)] += 1.0;
        let staggered = (0..keys).filter(|&oc| oc != c as usize && hand_of[oc] as usize == h).any(|oc| {
            let v = &holds[oc];
            let i = v.partition_point(|x| x.0 < e - 0.02);
            i > 0 && v[i - 1].1 > e + 0.02
        });
        if staggered {
            comp[b * NP + o(P::stag)] += 1.0;
        }
        let first = (nbins - 1).min(((t - t0) / bin).max(0.0) as usize);
        for hb in first..=b {
            let (lo, hi) = (t0 + hb as f64 * bin, if hb == nbins - 1 { f64::INFINITY } else { t0 + (hb + 1) as f64 * bin });
            comp[hb * NP + o(P::cover)] += 0f64.max(e.min(hi) - t.max(lo));
        }
    }
    crate::tick::lap("d.releases");
    add_burst(&mut comp, &burst, k.MJ_BURST);
    if !wide {
        for b in 0..nbins {
            let low = b.saturating_sub(2);
            let high = nbins.min(b + 3);
            let singles: usize = rows_by_size[0][low..high].iter().sum();
            let jumps: usize = rows_by_size[1][low..high].iter().sum();
            let chords: usize = rows_by_size[2][low..high].iter().sum();
            let count = singles + jumps + chords;
            if count != 0 && comp[b * NP + o(P::stream)] != 0.0 {
                let hs = 1f64.min(k.HS_SAT * chords as f64 / count as f64);
                let js = (1.0 - hs) * 1f64.min(3.0 * jumps as f64 / count as f64);
                let work = comp[b * NP + o(P::stream)];
                comp[b * NP + o(P::handstream)] = work * hs;
                comp[b * NP + o(P::jumpstream)] = work * js;
                comp[b * NP + o(P::stream)] = work * (1.0 - hs - js);
            }
        }
    }
    let offsets: Vec<usize> = keys_out.iter().filter(|&&p| !non_demand(p)).map(|&p| o(p)).collect();
    let info = &base.info;
    let info_w = k.PATTERN_INFO * (keys as f64 / 7.0).powf(k.PATTERN_INFO_X);
    let info_x = |b: usize| if info.is_empty() { 1.0 } else { (info_w * (info[b] - k.PATTERN_INFO_B).max(0.0)).exp() };
    let plain = if want_plain {
        let mut pc = comp.clone();
        for b in 0..nbins {
            for p in [P::sv, P::sv_raw].iter().chain(SV_KINDS.iter()) {
                pc[b * NP + o(*p)] = 0.0;
            }
        }
        let mut pt = vec![0.0f64; nbins];
        for b in 0..nbins {
            let (left, right, cv) = (hands[b * 3], hands[b * 3 + 1], plain_x[b]);
            let raw = left + right + cv; // the parts scale up to the total
            if raw > 0.0 {
                pt[b] = ((left * left + right * right).sqrt() + cv) / bin * info_x(b);
                let factor = pt[b] / raw;
                for &off in &offsets {
                    pc[b * NP + off] *= factor;
                }
            }
        }
        Some(Run {
            parts: Parts { nbins, data: by_part(&pc, nbins), keys: keys_out.clone() },
            totals: pt,
            fine: [side(&fine, 0), side(&fine, 1), plain_fine_x],
        })
    } else {
        None
    };
    let mut totals = vec![0.0f64; nbins];
    for b in 0..nbins {
        let (left, right, cv) = (hands[b * 3], hands[b * 3 + 1], hands[b * 3 + 2]);
        let raw = left + right + cv; // the parts scale up to the total
        if raw > 0.0 {
            totals[b] = ((left * left + right * right).sqrt() + cv) / bin * info_x(b);
            let factor = totals[b] / raw;
            for &off in &offsets {
                comp[b * NP + off] *= factor;
            }
        }
    }
    let fine = [side(&fine, 0), side(&fine, 1), side(&fine, 2)];
    crate::tick::lap("d.tail");
    (Run { parts: Parts { nbins, data: by_part(&comp, nbins), keys: keys_out }, totals, fine }, plain)
}

/// The bin-major work matrix as `Parts` holds it: one series per part.
fn by_part(comp: &[f64], nbins: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; NP * nbins];
    for (b, row) in comp.chunks_exact(NP).enumerate() {
        for (p, v) in row.iter().enumerate() {
            out[p * nbins + b] = *v;
        }
    }
    out
}

/// One side (left, right, cross) of the interleaved fine bins.
fn side(fine: &[f64], h: usize) -> Vec<f64> {
    fine.iter().skip(h).step_by(3).copied().collect()
}

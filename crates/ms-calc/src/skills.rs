//! Skill keys and the per-skill series (skill_calc.skills / _attribute).
use crate::demand::{Parts, P};
use crate::taxonomy::Density;
use crate::params::Knobs;
use crate::py;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Sk {
    stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
    chordstream, bracket, jack, quadstream, minijack, longjack, vibro, shield,
    ln_tech, ln_inverse, ln_release, ln_general,
    sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake, trill1h, anchor, patterntech, rhythmtech, technical,
    mash, stamina, overall, switch,
}
pub const NSK: usize = 37;
pub const SK_NAMES: [&str; NSK] = [
    "stream", "jumpstream", "handstream", "dump", "jumptrill", "splittrill", "jackspeed", "chordjack", "ln", "sv",
    "delay", "chordstream", "bracket", "jack", "quadstream", "minijack", "longjack", "vibro", "shield",
    "ln_tech", "ln_inverse", "ln_release", "ln_general", "sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake", "trill1h", "anchor",
    "patterntech", "rhythmtech", "technical", "mash", "stamina", "overall", "_switch",
];
pub const ALL_SK: [Sk; NSK] = {
    use Sk::*;
    [
        stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
        chordstream, bracket, jack, quadstream, minijack, longjack, vibro, shield,
        ln_tech, ln_inverse, ln_release, ln_general, sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake, trill1h, anchor, patterntech, rhythmtech, technical,
        mash, stamina, overall, switch,
    ]
};
impl Sk {
    pub fn name(self) -> &'static str {
        SK_NAMES[self as usize]
    }
    pub fn from_name(s: &str) -> Option<Sk> {
        SK_NAMES.iter().position(|&n| n == s).map(|i| ALL_SK[i])
    }
}

/// The skill with the same key as a demand part.
pub fn sk_of(p: P) -> Option<Sk> {
    Sk::from_name(crate::demand::P_NAMES[p as usize])
}

/// Ordered per-skill series (a Python dict of lists).
#[derive(Clone, Debug)]
pub struct Series {
    pub nbins: usize,
    pub keys: Vec<Sk>,
    pub data: Vec<f64>, // [NSK × nbins], rows of absent keys stay zero
    /// A key's list may be shorter than nbins (Python zip() truncation in execution.attribute).
    pub lens: [usize; NSK],
}
impl Series {
    pub fn new(nbins: usize, keys: Vec<Sk>) -> Series {
        Series { nbins, keys, data: vec![0.0; NSK * nbins], lens: [nbins; NSK] }
    }
    #[inline]
    pub fn get(&self, s: Sk) -> &[f64] {
        let o = s as usize * self.nbins;
        &self.data[o..o + self.lens[s as usize]]
    }
    #[inline]
    pub fn get_mut(&mut self, s: Sk) -> &mut [f64] {
        let o = s as usize * self.nbins;
        &mut self.data[o..o + self.lens[s as usize]]
    }
    /// Replace a key's list (len ≤ nbins).
    pub fn set(&mut self, s: Sk, v: &[f64]) {
        let o = s as usize * self.nbins;
        assert!(v.len() <= self.nbins);
        self.data[o..o + v.len()].copy_from_slice(v);
        self.data[o + v.len()..o + self.nbins].iter_mut().for_each(|x| *x = 0.0);
        self.lens[s as usize] = v.len();
    }
    pub fn has(&self, s: Sk) -> bool {
        self.keys.contains(&s)
    }
    pub fn remove(&mut self, s: Sk) -> Vec<f64> {
        self.keys.retain(|&k| k != s);
        let v = self.get(s).to_vec();
        self.get_mut(s).iter_mut().for_each(|x| *x = 0.0);
        v
    }
}

/// ±h bins sums.
pub fn win(v: &[f64], h: usize) -> Vec<f64> {
    let nb = v.len();
    let mut out = vec![0.0; nb];
    let mut acc = 0.0f64;
    for i in 0..nb + h {
        if i < nb {
            acc += v[i];
        }
        if i >= 2 * h + 1 {
            acc -= v[i - 2 * h - 1];
        }
        if i >= h {
            out[i - h] = acc;
        }
    }
    out
}

/// The demand parts of the keymode's base skills (taxonomy: the Base rows measured on it).
pub fn base_parts(keys: usize) -> Vec<P> {
    crate::taxonomy::TABLE.iter().filter(|r| r.role == crate::taxonomy::Role::Base && crate::taxonomy::measured(r.sk, keys))
        .map(|r| crate::demand::P_NAMES.iter().position(|n| *n == r.sk.name()).map(|i| crate::demand::ALL_P[i]).expect("a base skill is a demand part")).collect()
}

/// Per-skill demand series: every base skill = the bin's total × its part's share of the base parts (plan 2026-11-11: one
/// rule; it was total × part / the largest part, so the largest part took the bin's whole demand. A p-norm between the two
/// was screened: p = 1 fit 4K best and tied 7K). The modifiers
/// (technical, pattern, rhythm) are the same notes seen for one property. Slices come after execution (`slice`).
pub fn attribute(parts: &Parts, total: &[f64], keys: usize, technical_only: bool, k: &Knobs) -> Series {
    let nb = total.len();
    let pk = base_parts(keys);
    let pw: Vec<Vec<f64>> = pk.iter().map(|&p| win(parts.get(p), EVIDENCE_WIN)).collect();
    let p_tech = win(parts.get(P::technical), EVIDENCE_WIN);
    let p_rows = win(parts.get(P::rows), EVIDENCE_WIN);
    let rodd = win(parts.get(P::rodd), k.RT_WIN);
    let rrows = win(parts.get(P::rows), k.RT_WIN);
    let pp: Vec<f64> = parts.get(P::pins).iter().zip(parts.get(P::psw)).map(|(a, c)| a + c).collect();
    let podd = win(&pp, k.RT_WIN);
    let psw = win(parts.get(P::psw), k.RT_WIN);
    let nosv: Vec<f64> = total.iter().zip(parts.get(P::sv_raw)).map(|(t, s)| 0f64.max(t - s / k.BIN)).collect();
    let mut names: Vec<Sk> =
        if technical_only { vec![Sk::technical, Sk::patterntech, Sk::rhythmtech] } else { crate::taxonomy::rated(keys) };
    names.push(Sk::switch);
    names.retain(|&s| s != Sk::stamina);
    let mut se = Series::new(nb, names);
    let pk_sk: Vec<Sk> = pk.iter().map(|&p| sk_of(p).unwrap()).collect();
    let nbn = se.nbins;
    let at = |s: Sk, b: usize| s as usize * nbn + b;
    for b in 0..nb {
        let base = py::sum(pw.iter().map(|x| x[b]));
        if base + p_tech[b] == 0.0 || total[b] == 0.0 {
            continue;
        }
        if !technical_only && base > 0.0 {
            for (x, &s) in pw.iter().zip(&pk_sk) {
                se.data[at(s, b)] = total[b] * x[b] / base;
            }
        }
        if p_rows[b] != 0.0 && rrows[b] > 0.5 {
            let rt = 1f64.min(0f64.max(rodd[b]) / (k.RT_FULL * rrows[b]));
            let pt = 1f64.min(0f64.max(podd[b]) / (k.PT_FULL * rrows[b]));
            se.data[at(Sk::rhythmtech, b)] = nosv[b] * rt;
            se.data[at(Sk::patterntech, b)] = nosv[b] * pt;
            se.data[at(Sk::switch, b)] = nosv[b] * 1f64.min(0f64.max(psw[b]) / (k.PT_FULL * rrows[b]));
            se.data[at(Sk::technical, b)] = nosv[b] * (1.0 - (1.0 - rt) * (1.0 - pt));
        }
    }
    se
}

/// Bins each side of a bin that the base parts and every slice's evidence are summed over.
pub const EVIDENCE_WIN: usize = 2;

/// Per-bin evidence the parts don't hold: vibro runs (execution) and quad texture (pattern::quadstream).
pub struct Evidence<'a> {
    pub own: &'a [f64],
    pub quad: &'a [f64],
    /// compute::interleaved: the chart's share of interleaving disjoint chord-group moves.
    pub interleaved: f64,
}

/// A slice's evidence density in each bin (taxonomy::Density over ±EVIDENCE_WIN bins).
pub fn density(d: Density, parts: &Parts, series: &Series, total: &[f64], ev: &Evidence, keys: usize, k: &Knobs) -> Vec<f64> {
    let nb = total.len();
    let w = |p: P| win(parts.get(p), EVIDENCE_WIN);
    let ratio = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> { (0..nb).map(|i| if b[i] > 0.0 { a[i] / b[i] } else { 0.0 }).collect() };
    let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
    match d {
        Density::PerRow(p) => ratio(w(p), w(P::rows)),
        Density::Ratio(p, q) => ratio(w(p), w(q)),
        Density::Share(p) | Density::Interleaved(p) => {
            let base: Vec<f64> = base_parts(keys).iter().map(|&q| w(q)).fold(vec![0.0; nb], |acc, v| acc.iter().zip(&v).map(|(a, b)| a + b).collect());
            let gate = if matches!(d, Density::Interleaved(_)) { ev.interleaved } else { 1.0 };
            ratio(w(p), base).into_iter().map(|x| x * gate).collect()
        }
        Density::Technical => {
            let nosv: Vec<f64> = total.iter().zip(parts.get(P::sv_raw)).map(|(t, s)| 0f64.max(t - s / k.BIN)).collect();
            ratio(series.get(Sk::technical).to_vec(), nosv)
        }
        Density::VibroRuns => (0..nb).map(|i| at(ev.own, i)).collect(),
        Density::Quad => (0..nb).map(|i| at(ev.quad, i)).collect(),
    }
}

/// The chart level the ramps are stated at (a reference point: `calib ramps` refits lo and hi around it).
pub const RAMP_LEVEL: f64 = 5.0;

/// One ramp for every slice: 0 at or under `lo`, 1 at or over `hi`, linear between; both × (keys/7)^e × (level/RAMP_LEVEL)^g,
/// `level` the chart's raw overall (harder charts hold more of every specialty's evidence).
pub fn ramp(x: f64, r: [f64; 4], keys: usize, level: f64) -> f64 {
    let s = (keys as f64 / 7.0).powf(r[2]) * (level.max(0.1) / RAMP_LEVEL).powf(r[3]);
    let (lo, hi) = (r[0] * s, r[1] * s);
    if hi <= lo { (x >= hi) as u8 as f64 } else { ((x - lo) / (hi - lo)).clamp(0.0, 1.0) }
}

/// Slices, layered over their bases (plan 2026-11-12: the base keeps its whole demand, as in 1.9.0). A slice takes ramp(its
/// density) of what its density is measured against: a density per note row or a share of the bin (Minijacks, Vibro, Dump,
/// Bracket, 1H Trill, Anchors, Shields, Quadstream) of the bin's whole demand, as 1.9.0 did; a density inside a base's demand
/// (LN Tech/Inverse/Release, the SV kinds) of that base, where they share at most all of it and a Rest row (LN General)
/// holds what they leave. Slicing every type from its base's share left Minijacks nearly nothing on a jumpstream chart. A
/// slice of the bin takes its share out of the bases it slices in other families (Minijacks out of Chordjack).
/// → each slice's density in the bins it can take from (`calib ramps` fits the ramps on them).
pub fn slice(series: &mut Series, parts: &Parts, total: &[f64], ev: &Evidence, keys: usize, level: f64, k: &Knobs) -> Vec<(Sk, Vec<f64>)> {
    use crate::taxonomy::{Role, TABLE};
    let nb = series.nbins;
    let slices: Vec<(Sk, &[Sk], Density, Vec<f64>)> = TABLE.iter().filter(|r| series.has(r.sk)).filter_map(|r| match r.role {
        Role::Slice(of) => {
            let d = density(r.density?, parts, series, total, ev, keys, k);
            Some((r.sk, of, r.density?, d))
        }
        _ => None,
    }).collect();
    // the population a slice's ramp is read on: the bins it can take from (its bases' demand, or any demand for a slice of the bin)
    let seen: Vec<(Sk, Vec<f64>)> = slices.iter().map(|(s, of, d0, d)| {
        let can = |b: usize| if within(*d0) { of.iter().any(|&o| series.get(o).get(b).is_some_and(|x| *x > 0.0)) } else { total.get(b).is_some_and(|x| *x > 0.0) };
        (*s, (0..nb).filter(|&b| can(b)).map(|b| d[b]).collect())
    }).collect();
    let slices: Vec<(Sk, &[Sk], bool, Vec<f64>)> = slices.into_iter().map(|(s, of, d, x)| {
        let r = k.ramp(s);
        (s, of, within(d), x.into_iter().map(|x| ramp(x, r, keys, level)).collect())
    }).collect();
    for (s, _, _, r) in slices.iter().filter(|s| !s.2) {
        let v: Vec<f64> = (0..series.get(*s).len()).map(|b| total.get(b).copied().unwrap_or(0.0) * r[b]).collect();
        series.set(*s, &v);
    }
    let bases: Vec<Sk> = TABLE.iter().filter(|r| r.role == Role::Base && series.has(r.sk)).map(|r| r.sk).collect();
    // A slice of the bin over a base of another family takes its notes out of that base (Minijacks out of Chordjack, 1H Trill
    // out of Stream: 1.9.0 moved jack bursts out of every jack part); within its own family it stays layered.
    let family = |s: Sk| crate::taxonomy::of(s).family;
    for &base in &bases {
        let cut: Vec<f64> = (0..nb).map(|b| slices.iter().filter(|s| !s.2 && s.1.contains(&base) && family(s.0) != family(base)).map(|s| s.3[b]).sum::<f64>().min(1.0)).collect();
        if cut.iter().any(|c| *c > 0.0) {
            series.get_mut(base).iter_mut().zip(&cut).for_each(|(x, c)| *x *= 1.0 - c);
        }
    }
    let rests: Vec<(Sk, Sk)> = TABLE.iter().filter(|r| series.has(r.sk)).filter_map(|r| match r.role { Role::Rest(of) => Some((r.sk, of)), _ => None }).collect();
    // a Rest row starts as its whole base: bins no slice reaches keep it all
    for &(rest, of) in &rests {
        let v = series.get(of).to_vec();
        series.set(rest, &v);
    }
    for base in bases {
        let mine: Vec<&(Sk, &[Sk], bool, Vec<f64>)> = slices.iter().filter(|s| s.2 && s.1.contains(&base)).collect();
        if mine.is_empty() {
            continue;
        }
        for b in 0..nb.min(series.get(base).len()) {
            let whole = series.get(base)[b];
            let r: f64 = mine.iter().map(|s| s.3[b]).sum();
            if whole == 0.0 || r == 0.0 {
                continue;
            }
            let norm = r.max(1.0);
            for s in &mine {
                if let Some(x) = series.get_mut(s.0).get_mut(b) {
                    *x += whole * s.3[b] / norm;
                }
            }
            // a slice from another family takes its share out of this base (Minijacks out of Chordjack)
            let out: f64 = mine.iter().filter(|s| family(s.0) != family(base)).map(|s| s.3[b] / norm).sum();
            series.get_mut(base)[b] = whole * (1.0 - out);
            for &rest in &rests {
                if rest.1 == base {
                    if let Some(x) = series.get_mut(rest.0).get_mut(b) {
                        *x = whole * (1.0 - r / norm);
                    }
                }
            }
        }
    }
    seen
}

/// A density measured inside its base's demand (a ratio of two of its parts, or the technical share where the base is): the
/// slice takes that base's demand. Any other density counts note rows or the bin's share: the slice takes the bin's total.
fn within(d: Density) -> bool {
    matches!(d, Density::Ratio(..) | Density::Technical)
}

pub fn star(level: f64, k: &Knobs) -> f64 {
    k.STAR_A * level.powf(k.STAR_B)
}

// ---- rating: the level at which a player's expected accuracy over the chart's notes reaches the target ----
// One function rates the overall, every skill, stamina and the card's archetypes (user 2026-11-04: "i care about expected
// accuracy in the final diff rating on all skillsets"). It replaced the peak model (hardest 2/8/30/120 s windows), which
// let a short hard stretch rate a mostly easy chart as hard.

/// What every rating of a chart weights and scales its bins by: the judgements in each bin (accuracy is an average over
/// them; a long note is judged at its head and its release) and the hands' fatigue there (model::endurance).
#[derive(Clone, Debug, Default)]
pub struct Judged {
    pub notes: Vec<f64>,
    pub fatigue: Vec<f64>,
    /// (1 + w·fatigue)^p per bin, made once per chart: every skill's rating reads the same factor.
    factor: std::sync::OnceLock<Vec<f64>>,
}

impl Judged {
    pub fn new(times: &[f64], ends: &[f64], bins: usize, fatigue: Vec<f64>, k: &Knobs) -> Judged {
        let mut notes = vec![0.0f64; bins];
        if let (Some(&t0), Some(last)) = (times.first(), bins.checked_sub(1)) {
            let mut add = |t: f64| notes[(((t - t0) / k.BIN).max(0.0) as usize).min(last)] += 1.0;
            for (&t, &e) in times.iter().zip(ends) {
                add(t);
                if e > t {
                    add(e);
                }
            }
        }
        Judged { notes, fatigue, ..Default::default() }
    }

    pub fn with(notes: Vec<f64>, fatigue: Vec<f64>) -> Judged {
        Judged { notes, fatigue, ..Default::default() }
    }

    fn factor(&self, keys: usize, k: &Knobs) -> &[f64] {
        self.factor.get_or_init(|| {
            let fw = fatigue_weight(keys, k);
            (0..self.notes.len()).map(|i| pow(1.0 + fw * self.fatigue.get(i).copied().unwrap_or(0.0), k.ACC_POWER)).collect()
        })
    }
}

/// Each bin's level from the mean demand of its trailing `window` seconds (a chart's first seconds have less behind them).
/// At LOCAL_WINDOW it is the difficulty a note meets (the timeline); at STAMINA_WINDOW it is the sustained demand, so a
/// short chart has little of it.
pub fn local_levels(series: &[f64], window: f64, k: &Knobs) -> Vec<f64> {
    let w = 1usize.max((window / k.BIN) as usize);
    let mut acc = 0.0f64;
    (0..series.len())
        .map(|i| {
            acc = 0f64.max(acc + series[i] - if i >= w { series[i - w] } else { 0.0 });
            star(k.SCALE * (acc / w as f64).sqrt() / k.LOCAL_TOL, k)
        })
        .collect()
}

/// The skill whose expected accuracy over the notes reaches the target, as MinaCalc searches the rating that reaches its
/// goal accuracy. A note at level l loses 1 − exp(−(l/skill)^p) of its accuracy: little far below the skill, at most a
/// miss far above it, so a short hard burst costs its own notes and not the whole chart (a peak model's error).
pub fn rating(series: &[f64], judged: &Judged, keys: usize, k: &Knobs) -> f64 {
    search(&powered(series, k.LOCAL_WINDOW, judged, keys, k), &judged.notes, k)
}

/// The stamina rating: the same accuracy level with the sustained demand (STAMINA_WINDOW) in place of the local one.
pub fn stamina_rating(series: &[f64], judged: &Judged, keys: usize, k: &Knobs) -> f64 {
    search(&powered(series, k.STAMINA_WINDOW, judged, keys, k), &judged.notes, k)
}

/// x^p, exact multiplications for a whole p.
fn pow(x: f64, p: f64) -> f64 {
    if p.fract() == 0.0 { x.powi(p as i32) } else { x.powf(p) }
}

/// (level^p, judgements) of each bin with notes and demand, where level = `fatigued(local_levels(..))`, straight from the
/// demand: star(c·√m)^p = (A·c^B)^p · m^(B·p/2), one powf per bin and no level vector (most of a rating's cost was there).
fn powered(series: &[f64], window: f64, judged: &Judged, keys: usize, k: &Knobs) -> Vec<(f64, f64)> {
    let w = 1usize.max((window / k.BIN) as usize);
    let (p, factor) = (k.ACC_POWER, judged.factor(keys, k));
    let unit = pow(k.STAR_A * (k.SCALE / k.LOCAL_TOL).powf(k.STAR_B), p);
    let mut out = Vec::new();
    let mut acc = 0.0f64;
    for i in 0..series.len() {
        acc = 0f64.max(acc + series[i] - if i >= w { series[i - w] } else { 0.0 });
        let n = judged.notes.get(i).copied().unwrap_or(0.0);
        if acc > 0.0 && n > 0.0 {
            out.push((unit * (acc / w as f64).powf(k.STAR_B * p / 2.0) * factor[i], n));
        }
    }
    out
}

/// w = FATIGUE_W·(keys/7)^FATIGUE_W_KEYS.
pub fn fatigue_weight(keys: usize, k: &Knobs) -> f64 {
    k.FATIGUE_W * (keys as f64 / 7.0).powf(k.FATIGUE_W_KEYS)
}

/// `levels` × (1 + w·fatigue), bin by bin.
pub fn fatigued(mut levels: Vec<f64>, fatigue: &[f64], w: f64) -> Vec<f64> {
    if w != 0.0 {
        levels.iter_mut().zip(fatigue).for_each(|(l, f)| *l *= 1.0 + w * f);
    }
    levels
}

/// `rating` from levels already made (the SV-reading calibration builds them from its own window sums).
pub fn accuracy_rating(levels: &[f64], notes: &[f64], k: &Knobs) -> f64 {
    let lp: Vec<(f64, f64)> = levels.iter().zip(notes).filter(|(l, n)| **l > 0.0 && **n > 0.0).map(|(l, n)| (pow(*l, k.ACC_POWER), *n)).collect();
    search(&lp, notes, k)
}

/// The search behind every rating, on (level^p, judgements) of the bins with notes; `notes` holds every bin's judgements.
fn search(lp: &[(f64, f64)], notes: &[f64], k: &Knobs) -> f64 {
    let den: f64 = notes.iter().sum();
    let p = k.ACC_POWER;
    let mean = lp.iter().map(|(x, n)| n * x).sum::<f64>() / den.max(1e-9);
    if den <= 0.0 || mean <= 0.0 {
        return 0.0;
    }
    // v = ln(skill^p); loss(v) = Σ n·(1 − e^−x)/den with x = level^p·e^−v falls monotonically and, at a small target, almost
    // as e^−v: Newton on ln(loss) takes about 4 steps. exp, not expm1: a note losing almost nothing is off by ~1e-16 absolute,
    // nothing against the target's total, and expm1 cost twice as much (ratings move ≤ 1e-14).
    let at = |v: f64| {
        let scale = (-v).exp();
        let (loss, slope) = lp.iter().fold((0.0, 0.0), |(f, d), (x, n)| {
            let x = x * scale;
            // a bin losing almost nothing: 1 − e^−x by its series (error x⁵/120 < 1e-17), no exp
            let lost = if x < 1e-3 { x * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x / 24.0))) } else { 1.0 - (-x).exp() };
            (f + n * lost, d + n * x * (1.0 - lost))
        });
        (loss / den, -slope / den)
    };
    let (mut lo, mut hi) = (mean.ln() - 3.0 * p * 10f64.ln(), mean.ln() + 3.0 * p * 10f64.ln());
    if lp.iter().map(|(_, n)| n).sum::<f64>() / den <= k.ACC_TARGET {
        return (lo / p).exp(); // too few of the chart's notes to lose the target even all missed: the search's floor (it bisected 43 times to it)
    }
    if at(lo).0 <= k.ACC_TARGET {
        return (lo / p).exp(); // under the target even at the floor: the search bisected ~43 times to within 1e-10 of it (a tenth of all searches)
    }
    let mut v = mean.ln() - (-(1.0 - k.ACC_TARGET).ln()).ln(); // exact when every note meets the same level
    for _ in 0..100 {
        let (loss, slope) = at(v);
        if loss > k.ACC_TARGET { lo = v } else { hi = v }
        let step = if loss > 0.0 && slope < 0.0 { ((loss / k.ACC_TARGET).ln() * loss / slope).clamp(-3.0, 3.0) } else { 3.0 }; // no loss: skill far too high
        if step.abs() < 1e-6 {
            v -= step; // Newton is quadratic here: what this step leaves is ~step², below 1e-12
            break;
        }
        // a step leaving the bracket bisects it (never the converged point: that sits on the bracket's edge)
        v = if v - step > lo && v - step < hi { v - step } else { 0.5 * (lo + hi) };
        if hi - lo < 1e-10 {
            break;
        }
    }
    (v / p).exp()
}

#[cfg(test)]
mod rating_tests {
    use super::*;

    #[test]
    fn a_short_hard_burst_counts_by_its_notes_not_as_the_whole_chart() {
        let k = Knobs::default();
        let j = Judged::with(vec![1.0; 400], Vec::new());
        let easy = vec![1.0; 400];
        let mut burst = easy.clone();
        burst[200..216].iter_mut().for_each(|x| *x = 9.0);
        let hard = vec![9.0; 400];
        let r = |s: &[f64], k: &Knobs| rating(s, &j, 7, k);
        let (e, b, h) = (r(&easy, &k), r(&burst, &k), r(&hard, &k));
        // A peak model rates the burst chart near the hard one; on accuracy it sits much closer to the easy one.
        assert!(e < b && b < h && (b / e).ln() < 0.5 * (h / e).ln(), "{e} {b} {h}");
    }

    #[test]
    fn notes_weight_the_bins() {
        let k = Knobs::default();
        let series = [1.0, 1.0, 9.0, 9.0];
        let (sparse_hard, dense_hard) = ([4.0, 4.0, 1.0, 1.0], [1.0, 1.0, 4.0, 4.0]);
        let at = |notes: [f64; 4]| Judged::with(notes.to_vec(), Vec::new());
        assert!(rating(&series, &at(dense_hard), 7, &k) > rating(&series, &at(sparse_hard), 7, &k));
        assert_eq!(Judged::new(&[10.0, 10.2, 11.0, 11.1], &[10.0, 10.2, 11.0, 12.6], 4, Vec::new(), &k).notes, vec![2.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn tired_hands_make_the_same_notes_harder() {
        let mut k = Knobs::default();
        k.FATIGUE_W = 0.2;
        let series = vec![4.0; 100];
        let fresh = Judged::with(vec![1.0; 100], vec![0.0; 100]);
        let tired = Judged::with(vec![1.0; 100], vec![1.0; 100]);
        assert!(rating(&series, &tired, 7, &k) > 1.15 * rating(&series, &fresh, 7, &k));
    }
}

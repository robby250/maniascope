//! Skill keys and the per-skill series (skill_calc.skills / _attribute).
use crate::demand::{Parts, P, PARTS_4K, PARTS_5K};
use crate::params::Knobs;
use crate::py;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Sk {
    stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
    chordstream, bracket, jack, quadstream, minijack, longjack, vibro, release, inverse, hybrid, shield,
    ln_general, ln_tech, ln_inverse, ln_release,
    sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake, trill1h, anchor, patterntech, rhythmtech, technical,
    mash, stamina, overall, switch,
}
pub const NSK: usize = 40;
pub const SK_NAMES: [&str; NSK] = [
    "stream", "jumpstream", "handstream", "dump", "jumptrill", "splittrill", "jackspeed", "chordjack", "ln", "sv",
    "delay", "chordstream", "bracket", "jack", "quadstream", "minijack", "longjack", "vibro", "release", "inverse",
    "hybrid", "shield", "ln_general", "ln_tech", "ln_inverse", "ln_release", "sv_fast", "sv_slow", "sv_accel", "sv_stutter", "sv_brake", "trill1h", "anchor",
    "patterntech", "rhythmtech", "technical", "mash", "stamina", "overall", "_switch",
];
pub const ALL_SK: [Sk; NSK] = {
    use Sk::*;
    [
        stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv, delay,
        chordstream, bracket, jack, quadstream, minijack, longjack, vibro, release, inverse, hybrid, shield,
        ln_general, ln_tech, ln_inverse, ln_release, sv_fast, sv_slow, sv_accel, sv_stutter, sv_brake, trill1h, anchor, patterntech, rhythmtech, technical,
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

pub const JACK_PARTS: [Sk; 3] = [Sk::chordjack, Sk::jack, Sk::jackspeed];
pub const LN_KINDS: [Sk; 4] = [Sk::release, Sk::inverse, Sk::hybrid, Sk::shield];
/// The LN skills a player sees, each defined so its own KomeijiDove 7K LN Dan practice pack (Less 2026-11-04) rates it top:
/// General (LN without a specialty), Tech, Inverse, Release. Shields stay as they were.
pub const LN_SKILLS: [Sk; 4] = [Sk::ln_general, Sk::ln_tech, Sk::ln_inverse, Sk::ln_release];
/// Rated for the families fit's LN columns (model::sr_vector), never shown: Hybrid was deleted (Less 2026-11-04), Release and
/// Inverse became ln_release/ln_inverse. Redefining these would move every chart's stars.
pub const FIT_ONLY: [Sk; 3] = [Sk::release, Sk::inverse, Sk::hybrid];
pub const SV_KINDS: [Sk; 5] = [Sk::sv_fast, Sk::sv_slow, Sk::sv_accel, Sk::sv_stutter, Sk::sv_brake];
pub const TECH_KEYS: [Sk; 3] = [Sk::patterntech, Sk::rhythmtech, Sk::technical];

/// Skill keys measured for a keymode, in display order for ties.
pub fn skills(keys: usize) -> Vec<Sk> {
    use Sk::*;
    let base: Vec<Sk> = if keys < 5 {
        vec![stream, jumpstream, handstream, dump, jumptrill, splittrill, jackspeed, chordjack, ln, sv]
    } else {
        vec![delay, stream, chordstream, bracket, jumptrill, jack, chordjack, ln, sv]
    };
    let i = base.iter().position(|&s| s == ln).unwrap();
    let mut out = base[..i].to_vec();
    if keys == 4 {
        out.push(quadstream);
    }
    out.extend([minijack, longjack, vibro, ln]);
    out.extend(LN_KINDS);
    out.extend(LN_SKILLS);
    out.extend_from_slice(&base[i + 1..]);
    out.extend(SV_KINDS);
    out.extend([trill1h, anchor, patterntech, rhythmtech, technical, mash, stamina]);
    out
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

/// Per-skill demand series (skill_calc._attribute).
pub fn attribute(parts: &Parts, total: &[f64], keys: usize, technical_only: bool, k: &Knobs) -> Series {
    let nb = total.len();
    let pk: &[P] = if keys < 5 { &PARTS_4K } else { &PARTS_5K };
    let w = |p: P| win(parts.get(p), 2);
    let pw: Vec<Vec<f64>> = pk.iter().map(|&p| w(p)).collect();
    let p_tech = w(P::technical);
    let p_rows = w(P::rows);
    let rodd = win(parts.get(P::rodd), k.RT_WIN);
    let rrows = win(parts.get(P::rows), k.RT_WIN);
    let pp: Vec<f64> = parts.get(P::pins).iter().zip(parts.get(P::psw)).map(|(a, c)| a + c).collect();
    let podd = win(&pp, k.RT_WIN);
    let psw = win(parts.get(P::psw), k.RT_WIN);
    let nosv: Vec<f64> = total.iter().zip(parts.get(P::sv_raw)).map(|(t, s)| 0f64.max(t - s / k.BIN)).collect();
    let mut names: Vec<Sk> =
        if technical_only { vec![Sk::technical, Sk::patterntech, Sk::rhythmtech] } else { skills(keys) };
    names.push(Sk::switch);
    names.retain(|&s| s != Sk::stamina);
    let mut se = Series::new(nb, names);
    let extra = |p: P| if technical_only { Vec::new() } else { w(p) };
    let (mj, lj, vib, shield, inv, hyb, oht, anc, sus, ln_rel, dump, bracket) = (
        extra(P::mj), extra(P::lj), extra(P::vib), extra(P::shield), extra(P::inv), extra(P::hyb), extra(P::oht),
        extra(P::anc), extra(P::sus), extra(P::ln_rel), extra(P::dump), extra(P::bracket),
    );
    let ln_i = pk.iter().position(|&p| p == P::ln).unwrap();
    let pk_sk: Vec<Sk> = pk.iter().map(|&p| sk_of(p).unwrap()).collect();
    let has_dump = se.has(Sk::dump);
    let has_bracket = se.has(Sk::bracket);
    let nbn = se.nbins;
    let at = |s: Sk, b: usize| s as usize * nbn + b;
    let mut tech_share = vec![0.0f64; nb];
    for b in 0..nb {
        let tot = py::sum(pw.iter().map(|x| x[b])) + p_tech[b];
        if tot == 0.0 || total[b] == 0.0 {
            continue;
        }
        if !technical_only {
            let top = pw.iter().map(|x| x[b]).fold(f64::NEG_INFINITY, f64::max);
            for (x, &s) in pw.iter().zip(&pk_sk) {
                se.data[at(s, b)] = total[b] * x[b] / top;
            }
        }
        let rows = p_rows[b];
        let mut rt = 0.0;
        let mut pt = 0.0;
        if rows != 0.0 {
            if rrows[b] > 0.5 {
                rt = 1f64.min(0f64.max(rodd[b]) / (k.RT_FULL * rrows[b]));
                pt = 1f64.min(0f64.max(podd[b]) / (k.PT_FULL * rrows[b]));
            }
            se.data[at(Sk::rhythmtech, b)] = nosv[b] * rt;
            se.data[at(Sk::patterntech, b)] = nosv[b] * pt;
            se.data[at(Sk::switch, b)] =
                if rrows[b] > 0.5 { nosv[b] * 1f64.min(0f64.max(psw[b]) / (k.PT_FULL * rrows[b])) } else { 0.0 };
            se.data[at(Sk::technical, b)] = nosv[b] * (1.0 - (1.0 - rt) * (1.0 - pt));
        }
        if technical_only {
            continue;
        }
        let tb = total[b];
        if rows != 0.0 {
            se.data[at(Sk::minijack, b)] = tb * 1f64.min(mj[b] / (k.MJ_FULL * rows));
            se.data[at(Sk::longjack, b)] = tb * 1f64.min(lj[b] / (k.LJ_FULL * rows));
            se.data[at(Sk::vibro, b)] = tb * 1f64.min(vib[b] / (k.VIB_FULL * rows));
            se.data[at(Sk::shield, b)] = tb * 1f64.min(shield[b] / (k.SHIELD_FULL * rows));
            se.data[at(Sk::inverse, b)] = tb * 1f64.min(inv[b] / (k.INV_FULL * rows));
            se.data[at(Sk::hybrid, b)] = tb * 1f64.min(hyb[b] / (k.HYB_FULL * rows));
            se.data[at(Sk::trill1h, b)] = tb * 1f64.min(oht[b] / (k.OHT_FULL * rows));
            se.data[at(Sk::anchor, b)] = tb * 1f64.min(anc[b] / (k.ANC_FULL * rows));
        }
        let jr = mj[b] + sus[b];
        if jr != 0.0 {
            let burst = mj[b] / jr;
            for s in JACK_PARTS {
                if se.keys.contains(&s) {
                    let moved = se.data[at(s, b)] * burst;
                    se.data[at(s, b)] -= moved;
                    let m = &mut se.data[at(Sk::minijack, b)];
                    *m = m.max(moved);
                }
            }
        }
        if pw[ln_i][b] > 0.0 {
            se.data[at(Sk::release, b)] = se.data[at(Sk::ln, b)] * 1f64.min(ln_rel[b] / (k.REL_FULL * pw[ln_i][b]));
        }
        if has_dump {
            se.data[at(Sk::dump, b)] = tb * 1f64.min(dump[b] / (k.DUMP_FULL * tot));
        }
        if has_bracket {
            se.data[at(Sk::bracket, b)] = tb * 1f64.min(bracket[b] / (k.BRACKET_FULL * tot));
        }
        tech_share[b] = 1.0 - (1.0 - rt) * (1.0 - pt);
    }
    if !technical_only && se.has(Sk::ln_general) {
        // Shares of the LN demand, over ±RT_WIN bins. Inverse: the columns in use held most of the time (the gaps are the
        // notes). Release: releases while another finger of the hand keeps holding (finger independence on the release).
        // Tech: the technical share (irregular rhythm or awkward transitions) where LN is. General: LN without one of them.
        let (cover, used, stag, ends) = (win(parts.get(P::cover), k.RT_WIN), win(parts.get(P::used), k.RT_WIN), win(parts.get(P::stag), k.RT_WIN), win(parts.get(P::lnend), k.RT_WIN));
        let rel_full = k.LN_REL_FULL * (keys as f64 / 7.0).powf(k.LN_REL_KEYS);
        for b in 0..nb {
            let ln = se.data[at(Sk::ln, b)];
            if ln == 0.0 {
                continue;
            }
            let inverse = ((cover[b] / used[b].max(1e-9) - k.LN_INV_LO) / (k.LN_INV_HI - k.LN_INV_LO)).clamp(0.0, 1.0);
            let release = if ends[b] > 0.0 { 1f64.min(stag[b] / (rel_full * ends[b])) } else { 0.0 };
            let tech = 1f64.min(tech_share[b] / k.LN_TECH_FULL);
            se.data[at(Sk::ln_inverse, b)] = ln * inverse;
            se.data[at(Sk::ln_release, b)] = ln * release;
            se.data[at(Sk::ln_tech, b)] = ln * tech;
            se.data[at(Sk::ln_general, b)] = ln * (1.0 - inverse.max(release).max(tech));
        }
    }
    se
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

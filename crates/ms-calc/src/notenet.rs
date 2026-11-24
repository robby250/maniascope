//! The personal per-note forecast (plan 2026-11-13b S7): a difficulty for every judgement of a chart (each head, and each
//! LN release) from its chart-only record (`notefeat`), and P(a judgement's accuracy lost) = σ(d − θ) for the player's
//! ability θ at the play. Trained on one player's replays (s7net `notefit --save`), so the weights live in that player's data
//! folder, never in the build: other players keep the population forecast.
use crate::notefeat::{COLUMNS, W};
use serde_json::Value;

struct Layer {
    i: usize,
    o: usize,
    w: Vec<f32>, // [i][o]
    b: Vec<f32>,
}

impl Layer {
    fn of(v: &Value) -> Result<Layer, String> {
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("layer {k}"));
        let f = |k: &str| v[k].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.) as f32).collect::<Vec<f32>>()).ok_or(format!("layer {k}"));
        let (i, o, w, b) = (n("in")?, n("out")?, f("w")?, f("b")?);
        if w.len() != i * o || b.len() != o {
            return Err("layer shape".into());
        }
        Ok(Layer { i, o, w, b })
    }

    /// `out` (rows × o) = `x` (rows × i) · w + b, one SGEMM for every row.
    fn forward(&self, x: &[f32], rows: usize, out: &mut [f32]) {
        for r in out.chunks_exact_mut(self.o) {
            r.copy_from_slice(&self.b);
        }
        // SAFETY: x is rows×i, w is i×o, out is rows×o, all row-major and sized by the caller and Layer::of
        unsafe {
            matrixmultiply::sgemm(rows, self.i, self.o, 1., x.as_ptr(), self.i as isize, 1, self.w.as_ptr(), self.o as isize, 1, 1., out.as_mut_ptr(), self.o as isize, 1);
        }
    }
}

/// erf-based GELU, as the trainer's (burn): Abramowitz–Stegun 7.1.26 (|erf error| < 1.5e-7) with an f32 exp the compiler
/// vectorizes (2^n · a degree-6 polynomial on [−½, ½] ln 2; relative error < 3e-6 up to v = 20, from rounding −v·log2 e), so a block of hidden units runs as SIMD.
#[inline(always)]
fn gelu(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1. / (1. + 0.3275911 * z.abs());
    let poly = ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t + 0.254_829_6) * t;
    let y = 1. - poly * exp_neg(z * z);
    0.5 * x * (1. + y.copysign(z))
}

/// e^−v for v ≥ 0 (underflows to 0 past ~87).
#[inline(always)]
fn exp_neg(v: f32) -> f32 {
    let a = (-v).max(-87.) * std::f32::consts::LOG2_E;
    let n = a.round();
    let f = (a - n) * std::f32::consts::LN_2;
    let p = 1. + f * (1. + f * (0.5 + f * (1. / 6. + f * (1. / 24. + f * (1. / 120. + f * (1. / 720.))))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

pub struct Net {
    cols: Vec<usize>, // record columns read, in input order
    mu: Vec<f32>,
    sd: Vec<f32>,
    session: bool,
    others: usize, // inputs flagging the other players it was trained on: the app asks as its own player (zeros)
    l1: Layer,
    l2: Layer,
    l3: Layer,
}

impl Net {
    pub fn parse(text: &str) -> Result<Net, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let names: Vec<&str> = v["inputs"].as_array().ok_or("inputs")?.iter().filter_map(Value::as_str).collect();
        let cols = names.iter().map(|n| COLUMNS.iter().position(|c| c == n).ok_or(format!("unknown input {n}"))).collect::<Result<Vec<usize>, String>>()?;
        let nums = |k: &str| v[k].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.) as f32).collect::<Vec<f32>>()).ok_or(k.to_string());
        let (mu, sd) = (nums("mu")?, nums("sd")?);
        let session = v["session"].as_bool().unwrap_or(false);
        let (l1, l2, l3) = (Layer::of(&v["l1"])?, Layer::of(&v["l2"])?, Layer::of(&v["l3"])?);
        let others = v["others"].as_u64().unwrap_or(0) as usize;
        let want = cols.len() + 1 + if session { 2 } else { 0 } + others;
        if mu.len() != cols.len() || sd.len() != cols.len() || l1.i != want || l2.i != l1.o || l2.o != l1.o || l3.i != l1.o || l3.o != 1 {
            return Err("notenet shape".into());
        }
        Ok(Net { cols, mu, sd, session, others, l1, l2, l3 })
    }

    /// d of every judgement: each record's head, then (records with `ln` set, when `releases`: lazer judges them, stable doesn't)
    /// its release. `place`: the play's place in its session, (minutes since the session began, plays before it in the session).
    pub fn difficulties(&self, records: &[[f32; W]], place: (f64, f64), releases: bool) -> Vec<f32> {
        let ln = COLUMNS.iter().position(|c| *c == "ln").unwrap_or(0);
        let k = self.l1.i;
        let session = [(place.0.ln_1p() / 3. - 1.) as f32, (place.1.ln_1p() / 2. - 1.) as f32];
        let mut x: Vec<f32> = Vec::with_capacity(records.len() * 2 * k);
        for r in records {
            for tail in [false, true] {
                if tail && !(releases && r[ln] > 0.5) {
                    continue;
                }
                x.extend(self.cols.iter().zip(self.mu.iter().zip(&self.sd)).map(|(&c, (m, s))| (r[c] - m) / s));
                x.push(if tail { 1. } else { 0. });
                if self.session {
                    x.extend_from_slice(&session);
                }
                x.extend(std::iter::repeat_n(0., self.others));
            }
        }
        // a block of judgements at a time: its two hidden layers stay in cache (512 × 1024 × 4 B × 2 = 4 MB)
        run(&self.l1, &self.l2, &self.l3, &x, 1024)
    }
}

/// The net's outputs for every input row of `x` (rows × l1.i), l3.o a row, `block` rows a product.
fn run(l1: &Layer, l2: &Layer, l3: &Layer, x: &[f32], block: usize) -> Vec<f32> {
    let (k, h) = (l1.i, l1.o);
    let n = x.len() / k;
    let o = l3.o; // outputs a row
    #[allow(non_snake_case)]
    let BLOCK = block;
    let (mut h1, mut h2, mut out) = (vec![0f32; BLOCK * h], vec![0f32; BLOCK * h], vec![0f32; n * o]);
    for (b, xs) in x.chunks(BLOCK * k).enumerate() {
        let rows = xs.len() / k;
        let (h1, h2) = (&mut h1[..rows * h], &mut h2[..rows * h]);
        l1.forward(xs, rows, h1);
        h1.iter_mut().for_each(|v| *v = gelu(*v));
        l2.forward(h1, rows, h2);
        h2.iter_mut().zip(h1.iter()).for_each(|(v, a)| *v = gelu(*v) + a);
        l3.forward(h2, rows, &mut out[b * BLOCK * o..(b * BLOCK + rows) * o]);
    }
    out
}

/// The population's note model (plan 2026-11-22; `popfit --bins` → calib/popnet.json, in the build): a difficulty for every
/// half-second bin of a chart-rate, from the bin's summaries of the note columns (`notefeat`), the calculator's series there
/// and its rating of the chart-rate. A student of a per-note net trained on the public scores' accuracy: that net read a
/// record per note and cost seven times the calculator (R15); the student runs once a bin and holds its gain in 4K–6K.
/// Its rating of a chart-rate = e^(θ/3) of the ability θ at which the bins, weighted by their notes, lose 10 %; `curves` =
/// each keymode's usual rating (ln) over the calculator's (ln), which `model::note_correction` reads the net's disagreement
/// against.
pub struct Pop {
    shares: Vec<String>, // the calculator's skill series of the bin, as shares of its strain
    mean: Vec<usize>,    // note columns averaged over the bin's notes
    min: Vec<usize>,
    max: Vec<usize>,
    ctx: bool,           // strain and notes of the two bins before and the two after
    split: usize,        // its bins in one of the calculator's
    hands: bool,         // the means of each hand's notes apart, and each hand's notes
    clip: f32,           // inputs are held within this many sd
    bias: f32,
    bias2: f32,          // of the second output: the bin's difficulty for a player `FAR` abilities over the chart's
    mu: Vec<f32>,
    per_sd: Vec<f32>, // 1 / sd: a product per input, not a division (f32: the row's scaling runs as SIMD)
    l1: Layer,
    l2: Layer,
    l3: Layer,
    curves: Vec<(usize, Curve)>, // by keys: the keymodes the net saw enough of (popfit LINE_CHARTS)
}

/// A keymode's usual net rating at a calculator rating, both ln (`popfit` fits it): a cubic in u = the level's place in
/// [lo, hi] as −1 … 1 (the 1st–99th percentile of the fitted chart-rates), continued by the line's `slope` outside.
/// A line instead bent whole level bands (dev 2026-11-23: the net over its line by +.31 ln at ln calculator 2.5–3, −.16 at
/// 1–2), a change of scale the score check can't tell from none: the map bias follows the part left of the curve (4K slope
/// 1.0, 7K 1.4, straight from −.25 to +.25) and not the bend (slope .06 / −.04).
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub lo: f64,
    pub hi: f64,
    pub c: [f64; 4],
    pub slope: f64,
    pub bottom: f64, // where the fitted chart-rates end (their 0.1st and 99.9th percentile)
    pub top: f64,
}

impl Curve {
    pub fn at(&self, x: f64) -> f64 {
        let inside = x.clamp(self.lo, self.hi);
        let u = (2. * inside - self.lo - self.hi) / (self.hi - self.lo);
        self.c[0] + u * (self.c[1] + u * (self.c[2] + u * self.c[3])) + self.slope * (x - inside)
    }

    /// The share of the correction a chart of calculator rating `x` (ln) gets: all of it inside the span, less outside as the
    /// fitted chart-rates thin out, none past their end. Out there the net's disagreement is its own extrapolation against
    /// the line's: gate 2026-11-23 without this, REFORM Iota and Kappa (4K 27.8, 31.3; the span ends at 22.9) +13 % and
    /// +16 %, and the hardest 2× charts 2★ over osu!'s stars, which the calculator alone matched (15.33 against 15.28).
    pub fn reach(&self, x: f64) -> f64 {
        if x > self.hi {
            ((self.top - x) / (self.top - self.hi).max(1e-9)).clamp(0., 1.)
        } else if x < self.lo {
            ((x - self.bottom) / (self.lo - self.bottom).max(1e-9)).clamp(0., 1.)
        } else {
            1.
        }
    }
}

/// The student's inputs of the chart and the bin, before the shares and the summaries: level, keys, OD, the bin's notes (ln),
/// its time, its place in the chart, its strain (ln).
const CHART_INPUTS: usize = 7;
/// Abilities over the chart's own at which the student's second output is taken (`popfit --bins`, the same 2).
pub const FAR: f64 = 2.;

/// The level's place (`Curve` u) over which the correction's weight falls from NOTE_W to NOTE_W_TOP: the net has fewer scores
/// per chart at the top (its disagreement is twice as wide there, .105 against .044, and the map bias follows it half as
/// steeply). Dev 2026-11-23, .7 → .3 over (.3, .75) against (.1, .6): 4K .1782 / .1779, 7K .1557 / .1544: flat.
pub const TAPER_FROM: f64 = 0.3;
pub const TAPER_TO: f64 = 0.75;

/// Notes from which a chart gets the net's correction, in part and in full. The trainer drops scores without a lost
/// judgement (ln loss), so short charts read as hard: the net over its curve at the 99th percentile +3.6 ln at 100–200 notes,
/// +1.0 at 200–300, +.35 at 300–400 (2026-11-23); no judged map is under 300.
// ponytail: a ramp over the limit, not a cure; goes when the trainer's loss counts perfect scores (judgement-count likelihood).
pub const SHORT_NONE: f64 = 150.;
pub const SHORT_FULL: f64 = 300.;

impl Pop {
    pub fn parse(text: &str) -> Result<Pop, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let b = &v["bins"];
        let names = |k: &str| b[k].as_array().map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect::<Vec<String>>()).ok_or(format!("bins {k}"));
        let cols = |k: &str| names(k)?.iter().map(|n| COLUMNS.iter().position(|c| c == n).ok_or(format!("unknown column {n}"))).collect::<Result<Vec<usize>, String>>();
        let nums = |k: &str| v[k].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.)).collect::<Vec<f64>>()).ok_or(k.to_string());
        let (shares, mean, min, max, ctx) = (names("shares")?, cols("mean")?, cols("min")?, cols("max")?, b["ctx"].as_bool().unwrap_or(false));
        let (mu, sd) = (nums("mu")?, nums("sd")?);
        let (l1, l2, l3) = (Layer::of(&v["l1"])?, Layer::of(&v["l2"])?, Layer::of(&v["l3"])?);
        let curves: Vec<(usize, Curve)> = v["curves"].as_object().ok_or("curves")?.iter().filter_map(|(k, l)| {
            let c = l["c"].as_array()?;
            let curve = Curve { lo: l["lo"].as_f64()?, hi: l["hi"].as_f64()?, c: [c.first()?.as_f64()?, c.get(1)?.as_f64()?, c.get(2)?.as_f64()?, c.get(3)?.as_f64()?], slope: l["slope"].as_f64()?, bottom: l["bottom"].as_f64()?, top: l["top"].as_f64()? };
            (curve.hi > curve.lo && curve.slope > 0.).then_some((k.parse().ok()?, curve))
        }).collect();
        let hands = b["hands"].as_bool().unwrap_or(false);
        let k = CHART_INPUTS + shares.len() + mean.len() * if hands { 2 } else { 1 } + if hands { 2 } else { 0 } + min.len() + max.len() + if ctx { 8 } else { 0 };
        if v["level"].as_bool() != Some(true) || v["t0"].as_bool() != Some(true) || mu.len() != k || sd.len() != k || l1.i != k || l2.i != l1.o || l2.o != l1.o || l3.i != l1.o || l3.o != 2 || curves.is_empty() {
            return Err("popnet shape".into());
        }
        Ok(Pop { shares, mean, min, max, ctx, split: b["sub"].as_u64().unwrap_or(1).max(1) as usize, hands, clip: b["clip"].as_f64().ok_or("bins clip")? as f32, bias: b["bias"].as_f64().ok_or("bins bias")? as f32, bias2: b["bias2"].as_f64().ok_or("bins bias2")? as f32, mu: mu.iter().map(|m| *m as f32).collect(), per_sd: sd.iter().map(|s| (1. / s) as f32).collect(), l1, l2, l3, curves })
    }

    /// The net file's text: the build's calib/popnet.json, or MS_POPNET=FILE (a candidate net timed or screened without a build).
    fn text() -> &'static str {
        static TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        TEXT.get_or_init(|| std::env::var("MS_POPNET").ok().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_else(|| include_str!("../../../calib/popnet.json").to_string()))
    }

    /// The net in use; None while the file holds none (`{}`).
    pub fn shipped() -> Option<&'static Pop> {
        static NET: std::sync::OnceLock<Option<Pop>> = std::sync::OnceLock::new();
        NET.get_or_init(|| Pop::parse(Pop::text()).ok()).as_ref()
    }

    /// The net file, named by its content ("" while it holds no net).
    pub fn id() -> &'static str {
        static ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        ID.get_or_init(|| {
            use sha2::Digest;
            if Pop::shipped().is_none() { String::new() } else { sha2::Sha256::digest(Pop::text().as_bytes()).iter().take(6).map(|b| format!("{b:02x}")).collect() }
        })
    }

    /// The net's rating of a `keys`-key chart-rate with these records whose calculator rating is `overall`; `total` and
    /// `series` (by skill name) = the pass's strain per `bin` seconds. A note's bin counts from the first record, as the
    /// trainer's (`popfit --bins`): every number here is made the way it makes its inputs.
    pub fn rating<'a>(&self, records: &[[f32; W]], overall: f64, keys: usize, total: &[f64], series: &dyn Fn(&str) -> &'a [f64], bin: f64) -> (f64, f64) {
        // its own bins: `split` to one of the calculator's, whose series a bin reads where it falls
        let (nb, bin) = (total.len() * self.split, bin / self.split as f64);
        if records.is_empty() || nb == 0 {
            return (0., f64::NAN);
        }
        let col = |n: &str| COLUMNS.iter().position(|c| *c == n).unwrap_or(0);
        let (t, od) = (col("t"), col("od"));
        let first = f64::from(records[0][t]);
        let (hand, one) = (col("hand"), self.mean.len());
        let nm = one * if self.hands { 2 } else { 1 }; // the means' block: one hand after the other
        let (nn, na) = (nm + self.min.len(), nm + self.min.len() + self.max.len());
        let (mut n, mut agg, mut of_hand) = (vec![0f32; nb], vec![0f32; nb * na], vec![[0f32; 2]; nb]);
        for r in records {
            // to the millisecond: the f32 seconds put a note on a bin's edge on either side of it (a chart moved in time rated apart)
            let b = ((((f64::from(r[t]) - first) * 1000.).round() / 1000. / bin).max(0.) as usize).min(nb - 1);
            let a = &mut agg[b * na..(b + 1) * na];
            if n[b] == 0. {
                a[nm..nn].fill(f32::INFINITY);
                a[nn..].fill(f32::NEG_INFINITY);
            }
            let h = usize::from(self.hands && r[hand] > 0.5);
            self.mean.iter().zip(&mut a[h * one..(h + 1) * one]).for_each(|(&c, v)| *v += r[c]);
            of_hand[b][h] += 1.;
            self.min.iter().zip(&mut a[nm..nn]).for_each(|(&c, v)| *v = v.min(r[c]));
            self.max.iter().zip(&mut a[nn..]).for_each(|(&c, v)| *v = v.max(r[c]));
            n[b] += 1.;
        }
        crate::tick::lap("note bins");
        let level = ((overall.max(0.1).ln() - 1.5) / 0.7) as f32; // as popfit's level input
        let shares: Vec<&[f64]> = self.shares.iter().map(|s| series(s)).collect();
        let strain = |b: usize| (total[b / self.split] as f32).max(0.);
        let (mut x, mut w) = (Vec::with_capacity(nb * self.l1.i), Vec::with_capacity(nb));
        for b in (0..nb).filter(|&b| n[b] > 0.) {
            let start = x.len();
            x.extend([level, keys as f32 / 10., records[0][od] / 10., n[b].ln_1p(), (b as f64 * bin / 100.) as f32, b as f32 / nb as f32, strain(b).ln_1p()]);
            let all = total[b / self.split] as f32;
            x.extend(shares.iter().map(|s| if all > 0. { (s.get(b / self.split).copied().unwrap_or(0.) as f32 / all).clamp(0., 2.) } else { 0. }));
            let a = &agg[b * na..(b + 1) * na];
            if self.hands {
                (0..2).for_each(|h| x.extend(a[h * one..(h + 1) * one].iter().map(|v| v / of_hand[b][h].max(1.))));
                x.extend(of_hand[b].map(f32::ln_1p));
            } else {
                x.extend(a[..nm].iter().map(|v| v / n[b]));
            }
            x.extend(&a[nm..]);
            if self.ctx {
                for o in [-1i64, 1, -2, 2] {
                    let at = b as i64 + o;
                    x.extend(if at >= 0 && (at as usize) < nb { [strain(at as usize).ln_1p(), n[at as usize].ln_1p()] } else { [0., 0.] });
                }
            }
            x[start..].iter_mut().zip(self.mu.iter().zip(&self.per_sd)).for_each(|(v, (m, s))| *v = ((*v - m) * s).clamp(-self.clip, self.clip));
            w.push(n[b]);
        }
        crate::tick::lap("note inputs");
        // blocks small enough that the product stays on this thread (matrixmultiply splits one from (rows + out) × in =
        // 32,768 on): the bulk rating already runs a chart a core, and its threads inside each cost 10 % CPU
        let both = run(&self.l1, &self.l2, &self.l3, &x, (30_000 / self.l1.i.max(1)).saturating_sub(self.l1.o).clamp(16, 1024));
        crate::tick::lap("note net");
        let d: Vec<f32> = both.iter().step_by(2).map(|v| v + self.bias).collect();
        let theta = theta_weighted(&d, Some(&w), 0.1);
        // the chart's curve over ability: ln of the share of that 10 % a player FAR abilities higher still loses, from each bin's
        // second output (its difficulty for that player: the notes of a bin do not fade together)
        let (left, all) = both.iter().skip(1).step_by(2).zip(&w).fold((0., 0.), |a, (v, n)| (a.0 + f64::from(*n) / (1. + (theta + FAR - f64::from(v + self.bias2)).exp()), a.1 + f64::from(*n)));
        ((theta / 3.).exp(), (left / all / 0.1).max(1e-12).ln())
    }

    /// Whether its rating can move a chart-rate of this keymode with `notes` notes at the calculator's `overall`: `over` is 0
    /// without the keymode's curve, up to SHORT_NONE notes and past the curve's reach, and the net is not run for those.
    pub fn moves(&self, keys: usize, notes: f64, overall: f64) -> bool {
        notes > SHORT_NONE && overall > 0. && self.curves.iter().any(|l| l.0 == keys && l.1.reach(overall.ln()) > 0.)
    }

    /// The record column groups its summaries read (`notefeat::records_of` leaves the others out).
    pub fn need(&self) -> crate::notefeat::Need {
        crate::notefeat::Need::of(&[&self.mean[..], &self.min[..], &self.max[..]].concat())
    }

    /// Where the calculator's `overall` sits in the keymode's fitted levels, −1 … 1 (`Curve` u); None without a curve.
    pub fn place(&self, keys: usize, overall: f64) -> Option<f64> {
        let c = &self.curves.iter().find(|l| l.0 == keys)?.1;
        (overall > 0.).then(|| ((2. * overall.ln() - c.lo - c.hi) / (c.hi - c.lo)).clamp(-1., 1.))
    }

    /// ln of how far the net's `rating` sits over what the calculator's `overall` usually goes with in this keymode, in the
    /// calculator's own ln rating (the line's slope taken out), for a chart of `notes` notes (`SHORT_FULL`); 0 in a keymode the
    /// net has no curve for.
    pub fn over(&self, keys: usize, rating: f64, overall: f64, notes: f64) -> f64 {
        match self.curves.iter().find(|l| l.0 == keys) {
            Some((_, c)) if rating > 0. && overall > 0. => (rating.ln() - c.at(overall.ln())) / c.slope * ((notes - SHORT_NONE) / (SHORT_FULL - SHORT_NONE)).clamp(0., 1.) * c.reach(overall.ln()),
            _ => 0.,
        }
    }
}

/// Expected share of accuracy lost (lazer's weighted judgements, 305 = none lost) at ability θ.
pub fn expected_loss(d: &[f32], theta: f64) -> f64 {
    if d.is_empty() {
        return 0.;
    }
    d.iter().map(|&z| 1. / (1. + (theta - z as f64).exp())).sum::<f64>() / d.len() as f64
}

/// The θ at which `d` loses `loss`: what a played chart says the player's ability was then.
pub fn theta_of(d: &[f32], loss: f64) -> f64 {
    theta_weighted(d, None, loss)
}

/// The same with a weight for each difficulty (a bin's notes). The loss falls in θ: Newton steps inside a bracket (−20 … 20),
/// halving it when a step would leave (a bisection alone took 60 passes over the notes, a third of the note model's time;
/// this takes about six).
pub fn theta_weighted(d: &[f32], w: Option<&[f32]>, loss: f64) -> f64 {
    let (mut lo, mut hi, mut theta) = (-20f64, 20f64, 0f64);
    if d.is_empty() {
        return lo;
    }
    let all: f64 = w.map_or(d.len() as f64, |w| w.iter().map(|&x| f64::from(x)).sum());
    for _ in 0..80 {
        let (mut p, mut slope) = (0., 0.);
        for (i, &z) in d.iter().enumerate() {
            let (q, wi) = (1. / (1. + (theta - f64::from(z)).exp()), w.map_or(1., |w| f64::from(w[i])));
            p += wi * q;
            slope += wi * q * (1. - q);
        }
        let over = p / all - loss;
        if over > 0. { lo = theta } else { hi = theta }
        let step = theta + over * all / slope;
        let next = if step > lo && step < hi { step } else { 0.5 * (lo + hi) };
        if (next - theta).abs() < 1e-13 {
            return next;
        }
        theta = next;
    }
    theta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_matches_the_exact_curve_and_theta_inverts_the_loss() {
        // exact GELU at 1 and −1: 0.841345, −0.158655; and e^−v over the range the erf reads
        assert!((gelu(1.) - 0.841345).abs() < 1e-5 && (gelu(-1.) + 0.158655).abs() < 1e-5 && gelu(-12.).abs() < 1e-6);
        assert!((0..400).map(|i| i as f32 * 0.05).all(|v| (exp_neg(v) - (-v as f64).exp() as f32).abs() <= 3e-6 * (-v as f64).exp() as f32 + 1e-30));
        let d = [-2f32, 0., 1.5, 3.];
        let th = theta_of(&d, 0.1);
        assert!((expected_loss(&d, th) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn the_batched_net_matches_a_row_by_row_pass() {
        // a 3-input, width-5 net with arbitrary weights, 2,500 records (over two blocks), against plain loops per judgement
        let (k, h) = (3 + 1 + 2, 5);
        let w = |n: usize, seed: f32| (0..n).map(|i| (i as f32 * 0.37 + seed).sin() * 0.8).collect::<Vec<f32>>();
        let (w1, b1, w2, b2, w3) = (w(k * h, 0.1), w(h, 0.2), w(h * h, 0.3), w(h, 0.4), w(h, 0.5));
        let text = serde_json::json!({"inputs": ["size", "row_gap", "kept"], "mu": [0.5, 0.1, 1.0], "sd": [2.0, 0.3, 1.5], "session": true,
            "l1": {"in": k, "out": h, "w": w1, "b": b1}, "l2": {"in": h, "out": h, "w": w2, "b": b2}, "l3": {"in": h, "out": 1, "w": w3, "b": [0.25]}}).to_string();
        let net = Net::parse(&text).unwrap();
        let at = |n: &str| COLUMNS.iter().position(|c| *c == n).unwrap();
        let records: Vec<[f32; W]> = (0..2500).map(|i| {
            let mut r = [0f32; W];
            (r[at("size")], r[at("row_gap")], r[at("kept")], r[at("ln")]) = ((i % 4) as f32, (i as f32 * 0.01).cos(), (i % 3) as f32, (i % 5 == 0) as u8 as f32);
            r
        }).collect();
        let d = net.difficulties(&records, (12., 3.), true);
        let mut want = Vec::new();
        for r in &records {
            for tail in [false, true].into_iter().filter(|&t| !t || r[at("ln")] > 0.5) {
                let mut x: Vec<f32> = net.cols.iter().zip(net.mu.iter().zip(&net.sd)).map(|(&c, (m, s))| (r[c] - m) / s).collect();
                x.extend([if tail { 1. } else { 0. }, (12f64.ln_1p() / 3. - 1.) as f32, (3f64.ln_1p() / 2. - 1.) as f32]);
                let lay = |l: &Layer, x: &[f32]| (0..l.o).map(|o| l.b[o] + (0..l.i).map(|i| x[i] * l.w[i * l.o + o]).sum::<f32>()).collect::<Vec<f32>>();
                let a: Vec<f32> = lay(&net.l1, &x).into_iter().map(gelu).collect();
                let b: Vec<f32> = lay(&net.l2, &a).into_iter().map(gelu).zip(&a).map(|(v, a)| v + a).collect();
                want.push(lay(&net.l3, &b)[0]);
            }
        }
        assert_eq!(d.len(), want.len());
        assert!(d.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4), "batched and row-by-row disagree");
    }

    #[test]
    fn the_population_net_rates_a_chart_by_the_ability_that_loses_a_tenth_and_reads_its_keymodes_curve() {
        // 7K: ln net = 0.2 + ln calculator over ln 2 … ln 8 (u = (x − ln 4) / ln 2); 5K: a bend, 0.5 + 0.3u + 0.2u² − 0.1u³ over the same span
        let net = Pop::parse(r#"{"bins": {"shares": [], "mean": ["size"], "min": [], "max": [], "clip": 100, "bias": 0, "bias2": 0}, "mu": [0, 0, 0, 0, 0, 0, 0, 0],
            "sd": [1, 1, 1, 1, 1, 1, 1, 1], "level": true, "t0": true, "curves": {
            "5": {"lo": 0.6931471805599453, "hi": 2.0794415416798357, "c": [0.5, 0.3, 0.2, -0.1], "slope": 0.8, "bottom": 0, "top": 4.1588830833596715},
            "7": {"lo": 0.6931471805599453, "hi": 2.0794415416798357, "c": [1.5862943611198906, 0.6931471805599453, 0, 0], "slope": 1.0, "bottom": 0, "top": 9}},
            "l1": {"in": 8, "out": 2, "w": [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0], "b": [0, 0]}, "l2": {"in": 2, "out": 2, "w": [0, 0, 0, 0], "b": [-30, -30]},
            "l3": {"in": 2, "out": 2, "w": [0.1, 0.1, -0.1, -0.1], "b": [0, 1]}}"#).unwrap();
        let mut r = [0f32; W];
        r[COLUMNS.iter().position(|c| *c == "size").unwrap()] = 3.;
        // one bin of five notes: l1 passes 10 · its mean size and 10 · level through gelu, l2 adds nothing (gelu(−30) = 0), l3
        // takes a tenth of their difference
        let level = ((4f64.ln() - 1.5) / 0.7) as f32;
        let d = 0.1 * gelu(30.) - 0.1 * gelu(10. * level);
        let theta = f64::from(d) + 9f64.ln(); // σ(d − θ) = 0.1
        let (rating, far) = net.rating(&[r; 5], 4., 7, &[1.], &|_| &[], 0.5);
        assert!((rating - (theta / 3.).exp()).abs() < 1e-4, "{rating}");
        // the second output sits 1 over the first: 2 abilities higher the bin loses σ(d + 1 − θ − 2) = 1 / (1 + 9e), of the 10 %
        assert!((far - (1. / (1. + 9. * std::f64::consts::E) / 0.1).ln()).abs() < 1e-4, "{far}");
        assert!((net.over(7, rating, 4., 500.) - (rating.ln() - 0.2 - 4f64.ln())).abs() < 1e-12);
        // the middle of the span is u = 0, its top u = 1; past the top the line's slope carries on
        assert!((net.over(5, rating, 4., 500.) - (rating.ln() - 0.5) / 0.8).abs() < 1e-12);
        assert!((net.over(5, rating, 8., 500.) - (rating.ln() - 0.9) / 0.8).abs() < 1e-12);
        // … and the correction fades as the fitted chart-rates do: 5K's end at 64, so two thirds of it are left at 16 (ln 16 is
        // a third of the way from the span's top, ln 8, to ln 64), none from 64 on, where the net is not run at all
        assert!((net.over(5, rating, 16., 500.) - 2. / 3. * (rating.ln() - 0.9 - 0.8 * 2f64.ln()) / 0.8).abs() < 1e-12);
        assert!(net.over(5, rating, 64., 500.) == 0. && !net.moves(5, 500., 64.) && net.moves(5, 500., 16.) && !net.moves(5, 150., 16.) && !net.moves(2, 500., 4.));
        // a short chart: half the disagreement at 225 notes, none at 150
        assert!((net.over(7, rating, 4., 225.) - 0.5 * net.over(7, rating, 4., 500.)).abs() < 1e-12 && net.over(7, rating, 4., 150.) == 0.);
        assert_eq!(net.over(2, rating, 4., 500.), 0., "a keymode without a curve keeps the calculator's rating");
        assert!(net.place(7, 4.).unwrap().abs() < 1e-12 && net.place(7, 8.) == Some(1.) && net.place(7, 1.) == Some(-1.) && net.place(2, 4.).is_none());
        assert_eq!(net.rating(&[], 4., 7, &[1.], &|_| &[], 0.5).0, 0.);
    }

    #[test]
    fn the_population_net_summarises_each_bins_notes_and_weighs_the_bins_by_them() {
        // d = 4 · the bin's "jack" share + its mean size + its smallest column gap − its largest (l1 passes them, the bias takes
        // the 8 back): one note in bin 0, two in bin 3
        let file = |clip: f64| format!(r#"{{"bins": {{"shares": ["jack"], "mean": ["size"], "min": ["col_gap"], "max": ["col_gap"], "clip": {clip}, "bias": -8, "bias2": -8}},
            "mu": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "sd": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "level": true, "t0": true,
            "curves": {{"4": {{"lo": 0, "hi": 1, "c": [0, 0, 0, 0], "slope": 1, "bottom": -1, "top": 2}}}},
            "l1": {{"in": 11, "out": 1, "w": [0, 0, 0, 0, 0, 0, 0, 4, 1, 1, -1], "b": [8]}}, "l2": {{"in": 1, "out": 1, "w": [0], "b": [-30]}}, "l3": {{"in": 1, "out": 2, "w": [1, 1], "b": [0, 0]}}}}"#);
        let col = |n: &str| COLUMNS.iter().position(|c| *c == n).unwrap();
        let (t, size, gap) = (col("t"), col("size"), col("col_gap"));
        let note = |at: f32, s: f32, g: f32| { let mut r = [0f32; W]; (r[t], r[size], r[gap]) = (at, s, g); r };
        let records = [note(10., 1., 0.3), note(11.6, 1., 0.2), note(11.7, 3., 0.5)];
        let (total, jack) = ([2., 1., 1., 4.], [1., 0., 0., 1.]);
        let series = |n: &str| if n == "jack" { &jack[..] } else { &[][..] };
        // bin 0: 4 · 0.5 + 1 + 0.3 − 0.3 = 3; bin 3: 4 · 0.25 + 2 + 0.2 − 0.5 = 2.7 (gelu is the identity that far out)
        let want = |d: [f32; 2]| (theta_weighted(&d, Some(&[1., 2.]), 0.1) / 3.).exp();
        assert!((Pop::parse(&file(100.)).unwrap().rating(&records, 4., 4, &total, &series, 0.5).0 - want([3., 2.7])).abs() < 1e-4);
        // inputs held within half an sd: 4 · 0.5 + 0.5 + 0.3 − 0.3 and 4 · 0.25 + 0.5 + 0.2 − 0.5
        assert!((Pop::parse(&file(0.5)).unwrap().rating(&records, 4., 4, &total, &series, 0.5).0 - want([2.5, 1.2])).abs() < 1e-4);
        // two notes at 2 weigh as much as one at 2 and one more
        assert!((theta_weighted(&[2., 0.], Some(&[2., 1.]), 0.1) - theta_of(&[2., 2., 0.], 0.1)).abs() < 1e-9);
        assert!(Pop::parse(&file(100.).replace(r#""mean": ["size"]"#, r#""mean": ["size", "row_gap"]"#)).is_err(), "inputs and layers must agree");
        // per hand: 11 inputs become 14 (size's mean of the left hand's notes, of the right's, each hand's notes); the net reads the
        // left mean and the right hand's notes: bin 3 holds a left note of size 1 and a right one of size 3
        let hands = file(100.).replace(r#""clip""#, r#""hands": true, "clip""#).replace("[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]", "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]")
            .replace("[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]", "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]").replace(r#""in": 11, "out": 1, "w": [0, 0, 0, 0, 0, 0, 0, 4, 1, 1, -1]"#, r#""in": 14, "out": 1, "w": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]"#);
        let mut right = note(11.7, 3., 0.5);
        right[col("hand")] = 1.;
        let got = Pop::parse(&hands).unwrap().rating(&[records[0], records[1], right], 4., 4, &total, &series, 0.5).0;
        // bin 0: left mean 1, no right notes → 1; bin 3: left mean 1 + ln(1 + 1 right note)
        assert!((got - want([1., 1. + 2f32.ln()])).abs() < 1e-4, "{got}");
    }

    #[test]
    fn a_tiny_net_reads_its_named_columns_and_adds_a_release_for_each_hold() {
        let net = Net::parse(r#"{"inputs": ["size"], "mu": [0], "sd": [1], "session": false,
            "l1": {"in": 2, "out": 1, "w": [1, 0], "b": [0]}, "l2": {"in": 1, "out": 1, "w": [0], "b": [0]},
            "l3": {"in": 1, "out": 1, "w": [1], "b": [0]}}"#).unwrap();
        let mut a = [0f32; W];
        a[COLUMNS.iter().position(|c| *c == "size").unwrap()] = 2.;
        let mut b = a;
        b[COLUMNS.iter().position(|c| *c == "ln").unwrap()] = 1.;
        let d = net.difficulties(&[a, b], (0., 0.), true);
        // gelu(2) + gelu(0)·0 residual = gelu(2) for each judgement; the hold adds its release
        assert_eq!(d.len(), 3);
        assert!((d[0] - gelu(2.)).abs() < 1e-6 && (d[2] - gelu(2.)).abs() < 1e-6);
    }
}

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
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
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
        let want = cols.len() + 1 + if session { 2 } else { 0 };
        if mu.len() != cols.len() || sd.len() != cols.len() || l1.i != want || l2.i != l1.o || l2.o != l1.o || l3.i != l1.o || l3.o != 1 {
            return Err("notenet shape".into());
        }
        Ok(Net { cols, mu, sd, session, l1, l2, l3 })
    }

    /// d of every judgement: each record's head, then (records with `ln` set, when `releases`: lazer judges them, stable doesn't)
    /// its release. `place`: the play's place in its session, (minutes since the session began, plays before it in the session).
    pub fn difficulties(&self, records: &[[f32; W]], place: (f64, f64), releases: bool) -> Vec<f32> {
        let ln = COLUMNS.iter().position(|c| *c == "ln").unwrap_or(0);
        let (k, h) = (self.l1.i, self.l1.o);
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
            }
        }
        let n = x.len() / k;
        // a block of judgements at a time: its two hidden layers stay in cache (512 × 1024 × 4 B × 2 = 4 MB)
        const BLOCK: usize = 1024;
        let (mut h1, mut h2, mut out) = (vec![0f32; BLOCK * h], vec![0f32; BLOCK * h], vec![0f32; n]);
        for (b, xs) in x.chunks(BLOCK * k).enumerate() {
            let rows = xs.len() / k;
            let (h1, h2) = (&mut h1[..rows * h], &mut h2[..rows * h]);
            self.l1.forward(xs, rows, h1);
            h1.iter_mut().for_each(|v| *v = gelu(*v));
            self.l2.forward(h1, rows, h2);
            h2.iter_mut().zip(h1.iter()).for_each(|(v, a)| *v = gelu(*v) + a);
            self.l3.forward(h2, rows, &mut out[b * BLOCK..b * BLOCK + rows]);
        }
        out
    }
}

/// Expected share of accuracy lost (lazer's weighted judgements, 305 = none lost) at ability θ.
pub fn expected_loss(d: &[f32], theta: f64) -> f64 {
    if d.is_empty() {
        return 0.;
    }
    d.iter().map(|&z| 1. / (1. + (theta - z as f64).exp())).sum::<f64>() / d.len() as f64
}

/// The θ at which `d` loses `loss`: what a played chart says the player's ability was then (bisection; loss is falling in θ).
pub fn theta_of(d: &[f32], loss: f64) -> f64 {
    let (mut lo, mut hi) = (-20f64, 20f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if expected_loss(d, mid) > loss { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
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
        let w = |n: usize, seed: f32| (0..n).map(|i| ((i as f32 * 0.37 + seed).sin() * 0.8)).collect::<Vec<f32>>();
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

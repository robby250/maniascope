//! `popfit DIR [--rounds N] [--steps N] [--hid H] [--level] [--attr] [--seq [--pretrain DIR2 [--pre-steps N]]] [--pairs W [--pair-days D]] [--skill-theta note|chart|both [--skill-pen ROWS]] [--rows EVAL.tsv --out OUT.tsv] [--stars STARS.tsv] [--save NET.json]`, `popfit DIR --recurve NET.json --stars STARS.tsv`, `popfit DIR --bins TEACHER.json [--hid H] [--steps N] [--ctx] [--sub N] [--hands] [--drop a,b] [--mean a,b] [--min a,b] [--max a,b] [--skills SKILLS.tsv] --stars STARS.tsv --save NET.json`: the per-note model on the population's
//! scores (plan 2026-11-22, Stage 3a). DIR = `calib poprows`. Difficulty d_n = MLP(chart-only record of note n); a score's
//! accuracy lost is the mean of σ(d_n − θ) over the chart's notes, θ = the player's ability at that date + β·ln playcount.
//! Trained on TRAIN songs against ln(1 − accuracy), the gate's residual. Each chart's loss curve is taken on a grid of
//! abilities once and every score of the chart reads it by interpolation, so a step costs notes × grid, not notes × scores.
//! Abilities are not optimiser parameters (2 M sparse rows barely move under Adam: the first run ended at row RMSE .95):
//! after each round of net steps every TRAIN score's implied ability (the θ that would make its prediction exact, by the
//! curve's slope) is read off, and a score's θ = its player×keymode's level + that year's offset + the kernel mean
//! (e^-|days|/DRIFT) of the same player's other implied abilities around its date, as the yardstick's `MS_DRIFT`.
//! `--rows`: the held-out rows of `calib eval` (`MS_ROWS_OUT`) re-predicted, the same rows the yardstick is judged on (every
//! `--judge N` rounds and at the end; `--out` writes them with this model's residuals).
//! `--level`: the calculator's own rating of the chart-rate (ln) as one more input of every note: the model then starts from
//! the calculator's stars and learns what the notes add.
//! `--seq` (Stage 5, attempt 1): the notes see their neighbours (four dilated depthwise convolutions, ±15 notes) instead of
//! only the hand-built windows of their records; a long chart is trained on a contiguous window of `M` notes.
//! `--pairs W` (Stage 5, attempt 2: richer public targets): half of every step's charts come as two rates of one chart, and
//! each player's scores on both add W × the squared difference of their two residuals (ability error differenced out);
//! `--pair-days D`: only scores at most D days apart.
//! `--pretrain DIR2` (Stage 5, attempt 3: self-supervised encoder): the `--seq` encoder first learns to fill in masked notes
//! from their neighbours on unlabelled charts (`calib poprows OUT --extra FILES`) and the TRAIN charts, then trains as usual.
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::{gelu, relu, sigmoid};
use std::collections::HashMap;

// replay-derived or bookkeeping columns: never inputs (as notefit)
// "rate": the public scores hold three rates (0.75, 1, 1.5) and a net that reads the rate is untrained between them (`b4`
// halved a rating from 0.75× to 0.80×); the notes' gaps carry the speed
const SKIP: [&str; 12] = ["play", "j", "err", "signed", "recent_miss", "recent_lost", "fail", "row_miss", "block", "tail_j", "tail_err", "rate"];
const GRID: usize = 81; // abilities −10 … 10
const STEP: f32 = 0.25;
const DRIFT: f64 = 10.; // days of the ability kernel
const PRIOR: f64 = 1.5; // rows the kernel mean is shrunk by
const YEAR_PRIOR: f64 = 3.; // rows a player-year's offset is shrunk by
// A score that lost only a judgement or two says little about its ln loss: each row weighs 1 / (1 + COUNT_NOISE / the 300s'
// worth it lost), the inverse of its variance (counting noise 1 / lost on top of the play-to-play scatter).
const COUNT_NOISE: f64 = 10.;
// A keymode gets a line (and so the calculator its net) from this many trained chart-rates. Dev 2026-11-23 at 300: 8K (613)
// no gain at any weight, 9K (492) map error .272 → .308 at half weight; 10K (1,245), 5K (1,496), 6K (2,303) gain.
const LINE_CHARTS: usize = 1000;
// --zeros: perfect scores are fitted too, with this many 300s' worth added to every score's loss and to the prediction
// (ln(loss + c/n) against ln(p + c/n)): without them a short or easy chart is judged only on the plays that lost something.
const PSEUDO: f64 = 0.5;

// --skill-theta: the yardstick's families (`ms_tools::eval::FAMILIES`); a note's share of one = its bin's strongest member
// over the bin's strain
const FAMILIES: [&[&str]; NF] = [
    &["stream", "delay", "dump", "jumpstream", "handstream", "quadstream"],
    &["chordstream", "bracket"],
    &["chordjack"],
    &["jack", "jackspeed", "minijack", "longjack", "vibro", "anchor", "mash"],
    &["ln", "ln_general", "ln_release", "ln_inverse", "ln_tech", "shield"],
    &["jumptrill", "trill1h", "splittrill"],
    &["sv"],
    &["technical", "patterntech", "rhythmtech"],
];
const NF: usize = 8;

/// Solves the small symmetric system `a` x = `b` (n × n, row-major) by Gaussian elimination; a singular one gives zeros.
fn solve_small(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let p = (i..n).max_by(|&x, &y| a[x * n + i].abs().total_cmp(&a[y * n + i].abs())).unwrap();
        if a[p * n + i].abs() < 1e-12 { return vec![0.; n]; }
        for j in 0..n { a.swap(i * n + j, p * n + j); }
        b.swap(i, p);
        for r in i + 1..n {
            let f = a[r * n + i] / a[i * n + i];
            for j in i..n { a[r * n + j] -= f * a[i * n + j]; }
            b[r] -= f * b[i];
        }
    }
    let mut x = vec![0.; n];
    for i in (0..n).rev() { x[i] = (b[i] - (i + 1..n).map(|j| a[i * n + j] * x[j]).sum::<f64>()) / a[i * n + i]; }
    x
}

#[derive(Module, Debug)]
struct Net { l1: Linear, l2: Linear, l3: Linear, ctx: Vec<Conv1d>, rec: Option<Linear>, grind: Param<Tensor<2>>, kappa: Param<Tensor<2>> }

impl Net {
    fn difficulty(&self, x: Tensor<2>) -> Tensor<2> { self.seq(x, None) }
    /// `--seq` (plan Stage 5, attempt 1): dilated depthwise convolutions over the chart's notes in order between the first
    /// layer and the rest (receptive field ±15 notes); `shape` = (charts, notes, mask [charts, notes, 1]) of the padded rows
    fn seq(&self, x: Tensor<2>, shape: Option<(usize, usize, Tensor<3>)>) -> Tensor<2> {
        let h = self.encode(x, shape);
        self.l3.forward(gelu(self.l2.forward(h.clone())) + h)
    }
    /// the notes' hidden state after the first layer and the convolutions (what `--pretrain` trains)
    fn encode(&self, x: Tensor<2>, shape: Option<(usize, usize, Tensor<3>)>) -> Tensor<2> {
        let mut h = gelu(self.l1.forward(x));
        if let Some((b, m, mask)) = shape.filter(|_| !self.ctx.is_empty()) {
            let hid = h.dims()[1];
            let mask = mask.swap_dims(1, 2);
            let mut t = h.reshape([b, m, hid]).swap_dims(1, 2) * mask.clone();
            for c in &self.ctx { t = (t.clone() + gelu(c.forward(t))) * mask.clone(); }
            h = t.swap_dims(1, 2).reshape([b * m, hid]);
        }
        h
    }
}

/// A chart's curve over ability at its own θ: (−d ln loss / dθ, ln of the loss 2 abilities higher over the 10 %).
fn shape_of(d: &[f32], w: Option<&[f32]>, theta: f64) -> (f64, f64) {
    let (mut p, mut slope, mut far, mut all) = (0., 0., 0., 0.);
    for (i, &z) in d.iter().enumerate() {
        let (q, wi) = (1. / (1. + (theta - f64::from(z)).exp()), w.map_or(1., |w| f64::from(w[i])));
        p += wi * q;
        slope += wi * q * (1. - q);
        far += wi / (1. + (theta + 2. - f64::from(z)).exp());
        all += wi;
    }
    (slope / p.max(1e-12), (far / all.max(1e-12)).max(1e-12).ln() - 0.1f64.ln())
}

/// Days of a `yyyymmdd` date, as `calib eval` counts them.
/// Every note's standardised inputs as f16 (`x16-HASH.bin`, HASH of the inputs and standardisation) and its bin counted from
/// the chart's first note (`bin16-HASH.bin`), written once beside the f32 table: that table (34 GB) does not fit in RAM and a
/// fit read it from disk with the GPU two-thirds idle; these (15 GB) fit. `charts` = (first note, notes, bins), in note order.
#[allow(clippy::too_many_arguments)]
fn compact(dir: &str, rows: &[f32], w: usize, charts: &[(usize, usize, usize)], inputs: &[usize], t_col: usize, mu: &[f64], sd: &[f64], bin_s: f64) -> (memmap2::Mmap, memmap2::Mmap) {
    use std::hash::{Hash, Hasher};
    use std::io::Write;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (inputs, mu.iter().chain(sd).map(|v| v.to_bits()).collect::<Vec<u64>>(), bin_s.to_bits()).hash(&mut h);
    let (xn, bn) = (format!("x16-{:016x}.bin", h.finish()), format!("bin16-{:016x}.bin", h.finish()));
    if !std::path::Path::new(&format!("{dir}/{xn}")).exists() {
        let file = |n: &str| std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(format!("{dir}/{n}.part")).unwrap());
        let (mut fx, mut fb) = (file(&xn), file(&bn));
        let mut next = 0;
        for chunk in charts.chunks(1024) {
            let parts: Vec<(Vec<u16>, Vec<u16>)> = std::thread::scope(|s| chunk.chunks(64).map(|cs| s.spawn(move || {
                let (mut x, mut b) = (Vec::new(), Vec::new());
                for &(start, notes, bins) in cs {
                    let first = rows[start * w + t_col];
                    for i in start..start + notes {
                        let r = &rows[i * w..(i + 1) * w];
                        // f16's range: the calculator's `demand` reaches 1e12 on a few stacked notes (inf would turn the step NaN)
                        inputs.iter().for_each(|&j| x.push(half::f16::from_f64(((f64::from(r[j]) - if j == t_col { f64::from(first) } else { 0. } - mu[j]) / sd[j]).clamp(-65504., 65504.)).to_bits()));
                        b.push(((f64::from(r[t_col] - first) / bin_s).max(0.) as usize).min(bins.saturating_sub(1)) as u16);
                    }
                }
                (x, b)
            })).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect());
            for &(start, notes, _) in chunk { assert_eq!(start, next, "charts out of note order"); next += notes; }
            for (x, b) in parts {
                // SAFETY: u16 has no padding; read back on the same little-endian machine
                fx.write_all(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 2) }).unwrap();
                fb.write_all(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2) }).unwrap();
            }
        }
        (fx.flush().unwrap(), fb.flush().unwrap());
        // the bins first: the inputs' file is what says the pair is whole
        std::fs::rename(format!("{dir}/{bn}.part"), format!("{dir}/{bn}")).unwrap();
        std::fs::rename(format!("{dir}/{xn}.part"), format!("{dir}/{xn}")).unwrap();
    }
    (map_in(dir, &xn), map_in(dir, &bn))
}

fn u16s(m: &memmap2::Mmap) -> &[u16] {
    // SAFETY: a plain little-endian u16 table written by `compact` on this machine; a map is page-aligned
    unsafe { std::slice::from_raw_parts(m.as_ptr() as *const u16, m.len() / 2) }
}

fn map_in(dir: &str, name: &str) -> memmap2::Mmap {
    // SAFETY: the tables are written once by `calib poprows` and not changed while a fit reads them
    unsafe { memmap2::Mmap::map(&std::fs::File::open(format!("{dir}/{name}")).unwrap()).unwrap() }
}

fn day_of(date: i64) -> f64 {
    (date / 10000) as f64 * 365.25 + ((date / 100 % 100) as f64 - 1.) * 30.44 + (date % 100) as f64
}

struct Chart { beatmap: i64, rate: f64, keys: usize, part: u8, start: usize, notes: usize, overall: f32, bin_start: usize, bins: usize }

/// A step's charts: padded note inputs, their mask, and each sampled score's chart in the step and position among the TRAIN scores.
struct Batch { x: Vec<f32>, mask: Vec<f32>, m: usize, which: Vec<i32>, pos: Vec<u32>, pairs: Vec<(i32, i32)> }

/// A keymode's level curve for `ms_calc::notenet::Curve` from (ln calculator rating, ln net rating) points: the least-squares
/// cubic in the level's place within its 1st–99th percentile (u = −1 … 1), the line's slope over the same points, and where
/// the points end.
fn level_curve(points: &[(f64, f64)]) -> Option<serde_json::Value> {
    let mut xs: Vec<f64> = points.iter().map(|p| p.0).collect();
    xs.sort_by(f64::total_cmp);
    let at = |q: f64| xs[((xs.len().max(1) - 1) as f64 * q).round() as usize];
    let (lo, hi) = if xs.is_empty() { (0., 0.) } else { (at(0.01), at(0.99)) };
    let inside: Vec<(f64, f64)> = points.iter().map(|p| ((2. * p.0 - lo - hi) / (hi - lo), p.1)).filter(|p| p.0.abs() <= 1.).collect();
    if !(hi > lo) || inside.len() < 8 {
        return None;
    }
    let mut a = [[0f64; 5]; 4]; // normal equations of 1, u, u², u³
    for &(u, y) in &inside {
        let p = [1., u, u * u, u * u * u];
        for i in 0..4 {
            for j in 0..4 { a[i][j] += p[i] * p[j]; }
            a[i][4] += p[i] * y;
        }
    }
    for i in 0..4 {
        let pivot = (i..4).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        a.swap(i, pivot);
        if a[i][i].abs() < 1e-12 {
            return None;
        }
        for r in (0..4).filter(|&r| r != i) {
            let f = a[r][i] / a[i][i];
            for c in i..5 { a[r][c] -= f * a[i][c]; }
        }
    }
    let c: Vec<f64> = (0..4).map(|i| a[i][4] / a[i][i]).collect();
    let n = inside.len() as f64;
    let (mu, my) = (inside.iter().map(|p| p.0).sum::<f64>() / n, inside.iter().map(|p| p.1).sum::<f64>() / n);
    let slope = inside.iter().map(|p| (p.0 - mu) * (p.1 - my)).sum::<f64>() / inside.iter().map(|p| (p.0 - mu).powi(2)).sum::<f64>() * 2. / (hi - lo);
    // bottom, top: where the fitted chart-rates end (0.1st, 99.9th percentile; the last few are single charts far out)
    Some(serde_json::json!({"lo": lo, "hi": hi, "c": c, "slope": slope, "bottom": at(0.001), "top": at(0.999)}))
}

/// Each keymode's level curve from the net's stars file (`--stars`): keymodes of LINE_CHARTS rated chart-rates or more,
/// fitted on their charts of 300 notes or more (`ms_calc::notenet::SHORT_FULL`: shorter ones read as hard).
fn level_curves(charts: &[Chart], stars: &str) -> serde_json::Map<String, serde_json::Value> {
    let star: HashMap<(i64, u64), f64> = std::fs::read_to_string(stars).unwrap().lines().filter_map(|l| { let mut f = l.split('\t'); Some(((f.next()?.parse().ok()?, f.next()?.parse::<f64>().ok()?.to_bits()), f.next()?.parse().ok()?)) }).collect();
    let mut by: std::collections::BTreeMap<usize, (usize, Vec<(f64, f64)>)> = Default::default();
    for c in charts {
        if let Some(&v) = star.get(&(c.beatmap, c.rate.to_bits())).filter(|v| **v > 0.) {
            let e = by.entry(c.keys).or_default();
            e.0 += 1;
            if c.notes >= 300 { e.1.push((0.7 * f64::from(c.overall) + 1.5, v.ln())); }
        }
    }
    by.iter().filter(|(_, v)| v.0 >= LINE_CHARTS).filter_map(|(k, v)| Some((k.to_string(), level_curve(&v.1)?))).collect()
}

/// A note's loss at ability `th` under a teacher's outputs `z`: one head σ(d − θ), or (`--mix`) break σ(d₀ − θ) + timing
/// σ(d₁ − κθ) / 3.
fn note_loss(z: &[f32], th: f64, kappa: f64) -> f64 {
    let s = |x: f64| 1. / (1. + (-x).exp());
    if z.len() == 1 { s(f64::from(z[0]) - th) } else { s(f64::from(z[0]) - th) + s(f64::from(z[1]) - kappa * th) / 3. }
}

/// A two-head teacher's chart: (θ losing a tenth, −d ln loss / dθ there, ln of the loss 2 abilities higher over the 10 %), as
/// `theta_of` and `shape_of` for one head.
fn mix_curve(dn: &[f32], outs: usize, kappa: f64) -> (f64, (f64, f64)) {
    let mean = |th: f64| dn.chunks_exact(outs).map(|z| note_loss(z, th, kappa)).sum::<f64>() / (dn.len() / outs) as f64;
    let (mut lo, mut hi) = (-20f64, 20f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mean(mid) > 0.1 { lo = mid } else { hi = mid }
    }
    let theta = 0.5 * (lo + hi);
    (theta, (-(mean(theta + 0.01).ln() - mean(theta - 0.01).ln()) / 0.02, (mean(theta + 2.) / 0.1).max(1e-12).ln()))
}

/// The θ at which difficulties `d` (weights `w`, or all 1) lose a tenth: bracketed Newton, as `ms_calc::notenet::theta_of`.
fn theta_of(d: &[f32], w: Option<&[f32]>) -> f64 {
    let (mut lo, mut hi, mut theta) = (-20f64, 20f64, 0f64);
    let all: f64 = w.map_or(d.len() as f64, |w| w.iter().map(|&x| f64::from(x)).sum());
    for _ in 0..80 {
        let (mut p, mut slope) = (0., 0.);
        for (i, &z) in d.iter().enumerate() {
            let (q, wi) = (1. / (1. + (theta - f64::from(z)).exp()), w.map_or(1., |w| f64::from(w[i])));
            p += wi * q;
            slope += wi * q * (1. - q);
        }
        let over = p / all - 0.1;
        if over > 0. { lo = theta } else { hi = theta }
        let step = theta + over * all / slope;
        let next = if step > lo && step < hi { step } else { 0.5 * (lo + hi) };
        if (next - theta).abs() < 1e-11 { return next; }
        theta = next;
    }
    theta
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].as_str());
    let hid: usize = flag("--hid").map_or(256, |v| v.parse().unwrap());
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/charts.json")).unwrap()).unwrap();
    let cols: Vec<&str> = meta["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let w = cols.len();
    let charts: Vec<Chart> = meta["charts"].as_array().unwrap().iter().map(|c| Chart { beatmap: c["beatmap"].as_i64().unwrap(), rate: c["rate"].as_f64().unwrap(), keys: c["keys"].as_u64().unwrap() as usize,
        part: c["part"].as_u64().unwrap() as u8, start: c["start"].as_u64().unwrap() as usize, notes: c["notes"].as_u64().unwrap() as usize, overall: ((c["overall"].as_f64().unwrap_or(1.).max(0.1).ln() - 1.5) / 0.7) as f32, bin_start: c["bin_start"].as_u64().unwrap() as usize, bins: c["bins"].as_u64().unwrap() as usize }).collect();
    // --recurve NET.json (with --stars): a saved net's level curves fitted again, no training
    if let Some(path) = flag("--recurve") {
        let mut net: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let fitted = level_curves(&charts, flag("--stars").expect("--recurve reads --stars"));
        fitted.iter().for_each(|(k, c)| println!("{k}K {c}"));
        let o = net.as_object_mut().unwrap();
        o.remove("lines");
        o.insert("curves".into(), fitted.into());
        std::fs::write(path, serde_json::to_string(&net).unwrap()).unwrap();
        return;
    }
    let users: Vec<i64> = meta["players"].as_array().unwrap().iter().map(|p| p.as_i64().unwrap()).collect();
    let map = |name: &str| map_in(dir, name);
    let (mm, sm) = (map("rows.bin"), map("scores.bin"));
    // SAFETY: both files are plain little-endian f32 / u32 tables written by `calib poprows` on this machine
    let rows: &[f32] = unsafe { std::slice::from_raw_parts(mm.as_ptr() as *const f32, mm.len() / 4) };
    let scores: &[[u32; 12]] = unsafe { std::slice::from_raw_parts(sm.as_ptr() as *const [u32; 12], sm.len() / 48) };
    let inputs: Vec<usize> = (0..w).filter(|c| !SKIP.contains(&cols[*c])).collect();
    let level_in = args.iter().any(|a| a == "--level");
    // --attr: the calculator's skill series of the note's bin, each as a share of the bin's total strain, as more inputs
    let attr_names: Vec<&str> = meta["attr"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let (na, total_at, t_col) = (attr_names.len(), attr_names.iter().position(|n| *n == "total").unwrap(), cols.iter().position(|c| *c == "t").unwrap());
    let attr_in: Vec<usize> = if args.iter().any(|a| a == "--attr") { (0..na).filter(|&j| !["stamina", "overall", "total"].contains(&attr_names[j])).collect() } else { Vec::new() };
    // --skill-theta note|chart (Stage 3c): a player has an ability per family (`FAMILIES`) beside their level; a score's
    // ability moves by the player's family deviations δ times the chart's family exposure, centred on the player's usual
    // exposure. `note`: each family's share of the notes that cost at that ability (σ' of the note's difficulty there, the
    // note's bin's family share from the calculator's series), so δ reaches the notes of its family; `chart`: the chart's
    // plain family shares (the control: what the yardstick's skill mix already reads); `both`: no δ in training, then δ fitted
    // on each exposure after the judged rounds, the two compared on one net (run-to-run noise was as large as their gap).
    let skill_theta = flag("--skill-theta");
    let seq = args.iter().any(|a| a == "--seq");
    let pair_w: f32 = flag("--pairs").map_or(0., |v| v.parse().unwrap());
    let pair_days: f64 = flag("--pair-days").map_or(f64::INFINITY, |v| v.parse().unwrap());
    let am = (!attr_in.is_empty() || skill_theta.is_some()).then(|| map("attr.bin"));
    // SAFETY: a plain little-endian f32 table written by `calib poprows` on this machine
    let attr: &[f32] = am.as_ref().map_or(&[], |m| unsafe { std::slice::from_raw_parts(m.as_ptr() as *const f32, m.len() / 4) });
    let bin_s = meta["bin"].as_f64().unwrap();
    let k = inputs.len() + usize::from(level_in) + attr_in.len();
    // a score: its chart, ln(1 − accuracy) by the 300 weights, ln playcount, player×keymode, day
    let loss = |s: &[u32; 12]| { let n: u32 = s[5..11].iter().sum(); 1. - (300 * (s[5] + s[6]) + 200 * s[7] + 100 * s[8] + 50 * s[9]) as f64 / (300 * n.max(1)) as f64 };
    let mut pk_of: HashMap<(u32, usize), u32> = HashMap::new();
    let pk: Vec<u32> = scores.iter().map(|s| { let n = pk_of.len() as u32; *pk_of.entry((s[1], charts[s[0] as usize].keys)).or_insert(n) }).collect();
    let zeros = args.iter().any(|a| a == "--zeros");
    let usable = |i: usize| { let l = loss(&scores[i]); (l > 0. || zeros && l == 0.) && l < 0.5 };
    // TRAIN scores, grouped by chart for the steps and by player×keymode (in date order) for the abilities
    let fit: Vec<u32> = (0..scores.len()).filter(|&i| charts[scores[i][0] as usize].part == 0 && usable(i)).map(|i| i as u32).collect();
    let mut by_chart: Vec<Vec<u32>> = vec![Vec::new(); charts.len()]; // positions in `fit`
    fit.iter().enumerate().for_each(|(t, &i)| by_chart[scores[i as usize][0] as usize].push(t as u32));
    let pseudo = if zeros { PSEUDO } else { 0. };
    let fc: Vec<f32> = fit.iter().map(|&i| (pseudo / f64::from(scores[i as usize][5..11].iter().sum::<u32>().max(1))) as f32).collect();
    let fw: Vec<f32> = fit.iter().map(|&i| { let s = &scores[i as usize]; let lost = loss(s) * f64::from(s[5..11].iter().sum::<u32>()) + pseudo; (1. / (1. + COUNT_NOISE / lost.max(1e-3))) as f32 }).collect();
    let (fy, fday, flpc): (Vec<f32>, Vec<f64>, Vec<f32>) = (fit.iter().zip(&fc).map(|(&i, &c)| (loss(&scores[i as usize]) + f64::from(c)).ln() as f32).collect(), fit.iter().map(|&i| day_of(i64::from(scores[i as usize][3]))).collect(), fit.iter().map(|&i| (scores[i as usize][4].max(1) as f32).ln()).collect());
    // --mix: each TRAIN score's ln timing loss (200s) and ln break loss (100s, 50s, misses), no less than half a judgement,
    // weighed as the whole loss is (a score without a 200 or without a break says little about that kind)
    let mix = args.iter().any(|a| a == "--mix");
    let (mut kind_y, mut kind_w) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
    if mix {
        for &i in &fit {
            let s = &scores[i as usize];
            let n = f64::from(s[5..11].iter().sum::<u32>());
            let lost = [(f64::from(s[7]) / 3.).max(0.5 / 3.), (f64::from(s[8]) * 2. / 3. + f64::from(s[9]) * 5. / 6. + f64::from(s[10])).max(1. / 3.)];
            for j in 0..2 {
                kind_y[j].push((lost[j] / n).ln() as f32);
                kind_w[j].push((1. / (1. + COUNT_NOISE / lost[j])) as f32);
            }
        }
    }
    let mut by_pk: Vec<Vec<u32>> = vec![Vec::new(); pk_of.len()];
    fit.iter().enumerate().for_each(|(t, &i)| by_pk[pk[i as usize] as usize].push(t as u32));
    by_pk.iter_mut().for_each(|v| v.sort_by(|&a, &b| fday[a as usize].total_cmp(&fday[b as usize])));
    let train: Vec<usize> = (0..charts.len()).filter(|&c| by_chart[c].len() >= 5).collect();
    // --pairs: each TRAIN chart-rate's siblings (the same chart at its other rates)
    let sib: Vec<Vec<usize>> = {
        let mut of: HashMap<i64, Vec<usize>> = HashMap::new();
        train.iter().for_each(|&c| of.entry(charts[c].beatmap).or_default().push(c));
        (0..charts.len()).map(|c| of.get(&charts[c].beatmap).map_or_else(Vec::new, |v| v.iter().copied().filter(|&o| o != c).collect())).collect()
    };
    let paired: Vec<usize> = train.iter().copied().filter(|&c| !sib[c].is_empty()).collect();
    eprintln!("{} chart-rates ({} train), {} scores ({} train), {} player×keymodes, {k} inputs", charts.len(), train.len(), scores.len(), fit.len(), pk_of.len());
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    let row = |i: usize| &rows[i * w..(i + 1) * w];
    let (mut mu, mut sd) = (vec![0f64; w], vec![0f64; w]);
    for _ in 0..200_000 { let c = &charts[train[(rnd() % train.len() as u64) as usize]]; let r = row(c.start + (rnd() % c.notes as u64) as usize); let first = f64::from(row(c.start)[t_col]); for j in 0..w { let v = f64::from(r[j]) - if j == t_col { first } else { 0. }; mu[j] += v; sd[j] += v * v; } }
    for j in 0..w { mu[j] /= 2e5; sd[j] = (sd[j] / 2e5 - mu[j] * mu[j]).max(1e-9).sqrt(); }
    // a note's inputs; `c` = its chart. `t` counts from the chart's first note (as ms_calc::notenet::Pop): a lead-in is not
    // difficulty; the bin, counted from the chart's first record (as ms_calc::notenet::Pop does)
    let ni = inputs.len();
    let (xm, bm) = compact(dir, rows, w, &charts.iter().map(|c| (c.start, c.notes, c.bins)).collect::<Vec<_>>(), &inputs, t_col, &mu, &sd, bin_s);
    let (x16, bin16) = (u16s(&xm), u16s(&bm));
    let feats = |c: usize, i: usize, out: &mut Vec<f32>| {
        out.extend(x16[i * ni..(i + 1) * ni].iter().map(|&h| half::f16::from_bits(h).to_f32()));
        if level_in { out.push(charts[c].overall); }
        if !attr_in.is_empty() {
            let (ch, b) = (&charts[c], bin16[i] as usize);
            let a = &attr[(ch.bin_start + b) * na..(ch.bin_start + b + 1) * na];
            let all = a[total_at];
            attr_in.iter().for_each(|&j| out.push(if all > 0. { (a[j] / all).clamp(0., 2.) } else { 0. }));
        }
    };
    // a note's share of each family (--skill-theta)
    let fam_idx: Vec<Vec<usize>> = FAMILIES.iter().map(|m| m.iter().filter_map(|n| attr_names.iter().position(|a| a == n)).collect()).collect();
    let fam_of = |c: usize, i: usize, out: &mut Vec<f32>| {
        let (ch, b) = (&charts[c], bin16[i] as usize);
        let a = &attr[(ch.bin_start + b) * na..(ch.bin_start + b + 1) * na];
        let all = a[total_at];
        fam_idx.iter().for_each(|m| out.push(if all > 0. { m.iter().fold(0f32, |x, &j| x.max(a[j] / all)).clamp(0., 2.) } else { 0. }));
    };
    // --bins TEACHER.json: the per-bin student (R15: the per-note net cost 7× the calculator). Each half-second bin of a
    // chart-rate gets the teacher's effective difficulty there (the d whose σ(d − θ*) is the bin's mean note loss at the chart's
    // own 10 %-loss ability θ*); the student reads the bin's summaries of the note columns (mean / min / max), the
    // calculator's series of the bin and the chart's level, and its rating is the θ losing 10 % over the bins weighted by
    // their notes. The net then runs once a bin, and the calculator needs running sums instead of a record per note.
    if let Some(tpath) = flag("--bins") {
        let t: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tpath).unwrap()).unwrap();
        assert!(t["level"] == true && t["t0"] == true && t["attr"].as_array().is_none_or(|a| a.is_empty()), "teacher: a --level net with time from the first note and no --attr");
        let col_of = |n: &str| cols.iter().position(|c| *c == n).unwrap_or_else(|| panic!("no column {n}"));
        let names = |v: &serde_json::Value| -> Vec<String> { v.as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect() };
        let nums = |v: &serde_json::Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
        let (tin, tmu, tsd): (Vec<usize>, Vec<f32>, Vec<f32>) = (names(&t["inputs"]).iter().map(|n| col_of(n)).collect(), nums(&t["mu"]), nums(&t["sd"]));
        let d = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0)).autodiff();
        let dv = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0));
        let lin = |v: &serde_json::Value| -> Linear {
            let (i, o) = (v["in"].as_u64().unwrap() as usize, v["out"].as_u64().unwrap() as usize);
            Linear { weight: Param::from_tensor(Tensor::<2>::from_data(TensorData::new(nums(&v["w"]), [i, o]), &dv)), bias: Some(Param::from_tensor(Tensor::<1>::from_data(TensorData::new(nums(&v["b"]), [o]), &dv))) }
        };
        // a two-head teacher (`--mix`): each note's loss is its break and timing heads' (`note_loss`), the bins' targets as before
        let (outs, kappa) = (if t["mix"] == true { 2 } else { 1 }, t["kappa"].as_f64().unwrap_or(0.).exp());
        let teacher = Net { l1: lin(&t["l1"]), l2: lin(&t["l2"]), l3: lin(&t["l3"]), ctx: Vec::new(), rec: None, grind: Param::from_tensor(Tensor::<2>::zeros([1, 1], &dv)), kappa: Param::from_tensor(Tensor::<2>::zeros([1, 1], &dv)) };
        // the teacher's difficulty of every note
        let kt = tin.len() + 1;
        let mut d_all = vec![0f32; rows.len() / w * outs];
        let mut at = 0;
        while at < charts.len() {
            let mut end = at;
            let mut n = 0;
            while end < charts.len() && (n == 0 || n + charts[end].notes <= 200_000) { n += charts[end].notes; end += 1; }
            let mut x = Vec::with_capacity(n * kt);
            for c in &charts[at..end] {
                let first = row(c.start)[t_col];
                for i in c.start..c.start + c.notes {
                    let r = row(i);
                    tin.iter().enumerate().for_each(|(j, &cc)| x.push((r[cc] - if cc == t_col { first } else { 0. } - tmu[j]) / tsd[j]));
                    x.push(c.overall);
                }
            }
            if n > 0 {
                let out = teacher.difficulty(Tensor::<2>::from_data(TensorData::new(x, [n, kt]), &dv)).into_data().try_to_vec::<f32>().unwrap();
                let mut o = 0;
                for c in &charts[at..end] { d_all[c.start * outs..(c.start + c.notes) * outs].copy_from_slice(&out[o * outs..(o + c.notes) * outs]); o += c.notes; }
            }
            at = end;
        }
        eprintln!("teacher difficulties of {} notes", d_all.len());
        // the student's inputs of a bin: 7 of the chart and bin, the calculator's skill shares, the note columns' summaries, the neighbours
        let drop: Vec<&str> = flag("--drop").map_or_else(Vec::new, |v| v.split(',').collect());
        let list = |name: &str, default: Vec<String>| -> Vec<String> { flag(name).map_or(default, |v| v.split(',').map(String::from).collect()).into_iter().filter(|n| !drop.contains(&n.as_str())).collect() };
        let mean_cols = list("--mean", names(&t["inputs"]).into_iter().filter(|n| !["keys", "rate", "od", "t", "frac"].contains(&n.as_str())).collect());
        let min_cols = list("--min", ["row_gap", "hand_gap", "col_gap", "next_gap", "col_next"].map(String::from).to_vec());
        let max_cols = list("--max", ["size", "n_hand", "chord_run", "col_run", "jacked", "surp_row", "surp_hand", "load_2"].map(String::from).to_vec());
        let ctx = args.iter().any(|a| a == "--ctx");
        // --sub N: the student's bins are 1/N of the calculator's (its series are read at the bin they fall in)
        let split: usize = flag("--sub").map_or(1, |v| v.parse().unwrap());
        // --hands: the means of each hand's notes apart (and each hand's notes): a bin's mean over both hides which hand is in trouble
        let hands = usize::from(args.iter().any(|a| a == "--hands"));
        let hand_col = col_of("hand");
        let shares: Vec<usize> = (0..na).filter(|&j| !["stamina", "overall", "total"].contains(&attr_names[j])).collect();
        let (mc, nc, xc): (Vec<usize>, Vec<usize>, Vec<usize>) = (mean_cols.iter().map(|n| col_of(n)).collect(), min_cols.iter().map(|n| col_of(n)).collect(), max_cols.iter().map(|n| col_of(n)).collect());
        let od_col = col_of("od");
        let nm = mc.len() * (1 + hands); // the means' block: one hand after the other with --hands
        let kf = 7 + shares.len() + nm + 2 * hands + nc.len() + xc.len() + if ctx { 8 } else { 0 };
        let am = map("attr.bin");
        // SAFETY: as `rows`
        let attr: &[f32] = unsafe { std::slice::from_raw_parts(am.as_ptr() as *const f32, am.len() / 4) };
        let (mut feat, mut y, mut cnt, mut owner, mut place): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut y2: Vec<f32> = Vec::new();
        let mut teacher_theta = vec![f64::NAN; charts.len()];
        let mut teacher_shape = vec![(f64::NAN, f64::NAN); charts.len()];
        for (ci, c) in charts.iter().enumerate().filter(|x| x.1.notes >= 20 && x.1.bins >= 1) {
            let dn = &d_all[c.start * outs..(c.start + c.notes) * outs];
            let (theta, shape) = if outs == 1 { let th = theta_of(dn, None); (th, shape_of(dn, None, th)) } else { mix_curve(dn, outs, kappa) };
            teacher_theta[ci] = theta;
            teacher_shape[ci] = shape;
            let (first, nb, na_c) = (f64::from(row(c.start)[t_col]), c.bins * split, nm + nc.len() + xc.len());
            let (mut n, mut q, mut q2, mut agg, mut of_hand) = (vec![0f32; nb], vec![0f64; nb], vec![0f64; nb], vec![0f32; nb * na_c], vec![[0f32; 2]; nb]);
            for (j, i) in (c.start..c.start + c.notes).enumerate() {
                let r = row(i);
                // to the millisecond, as ms_calc::notenet::Pop: the f32 seconds put a note on a bin's edge on either side of it
                let b = ((((f64::from(r[t_col]) - first) * 1000.).round() / 1000. / (bin_s / split as f64)).max(0.) as usize).min(nb - 1);
                let a = &mut agg[b * na_c..(b + 1) * na_c];
                if n[b] == 0. {
                    (0..nc.len()).for_each(|k| a[nm + k] = f32::INFINITY);
                    (0..xc.len()).for_each(|k| a[nm + nc.len() + k] = f32::NEG_INFINITY);
                }
                let h = hands * usize::from(r[hand_col] > 0.5);
                mc.iter().enumerate().for_each(|(k, &cc)| a[h * mc.len() + k] += r[cc]);
                nc.iter().enumerate().for_each(|(k, &cc)| a[nm + k] = a[nm + k].min(r[cc]));
                xc.iter().enumerate().for_each(|(k, &cc)| a[nm + nc.len() + k] = a[nm + nc.len() + k].max(r[cc]));
                n[b] += 1.;
                of_hand[b][h] += 1.;
                let z = &dn[j * outs..(j + 1) * outs];
                q[b] += note_loss(z, theta, kappa);
                q2[b] += note_loss(z, theta + 2., kappa);
            }
            let total = |b: usize| attr[(c.bin_start + b / split) * na + total_at].max(0.);
            let r0 = row(c.start);
            for b in (0..nb).filter(|&b| n[b] > 0.) {
                let m = (q[b] / f64::from(n[b])).clamp(1e-7, 1. - 1e-7);
                y.push(((theta + (m / (1. - m)).ln()) as f32).clamp(-12., 12.));
                // the same for a player 2 abilities higher: what the bin's notes still cost there (the chart's curve over ability)
                let m2 = (q2[b] / f64::from(n[b])).clamp(1e-7, 1. - 1e-7);
                y2.push(((theta + 2. + (m2 / (1. - m2)).ln()) as f32).clamp(-12., 12.));
                cnt.push(n[b]);
                owner.push(ci as u32);
                place.push(b as u32);
                let a = &attr[(c.bin_start + b / split) * na..(c.bin_start + b / split + 1) * na];
                // no rate: the public scores hold three (0.75, 1, 1.5) and a net that reads it is untrained between them (the
                // teacher's rating fell by half from 0.75× to 0.80× on two of 84 charts); the notes' gaps carry the speed
                feat.extend([c.overall, c.keys as f32 / 10., r0[od_col] / 10., n[b].ln_1p(), (b as f64 * bin_s / split as f64 / 100.) as f32, b as f32 / nb as f32, total(b).ln_1p()]);
                feat.extend(shares.iter().map(|&j| if a[total_at] > 0. { (a[j] / a[total_at]).clamp(0., 2.) } else { 0. }));
                let g = &agg[b * na_c..(b + 1) * na_c];
                if hands == 0 {
                    feat.extend(g[..nm].iter().map(|v| v / n[b]));
                } else {
                    (0..2).for_each(|h| feat.extend(g[h * mc.len()..(h + 1) * mc.len()].iter().map(|v| v / of_hand[b][h].max(1.))));
                    feat.extend(of_hand[b].map(f32::ln_1p));
                }
                feat.extend(&g[nm..]);
                if ctx {
                    for o in [-1i64, 1, -2, 2] {
                        let nbr = b as i64 + o;
                        feat.extend(if nbr >= 0 && (nbr as usize) < nb { [total(nbr as usize).ln_1p(), n[nbr as usize].ln_1p()] } else { [0., 0.] });
                    }
                }
            }
        }
        let nbins = y.len();
        eprintln!("{nbins} bins, {kf} inputs ({} means, {} mins, {} maxes{})", mc.len(), nc.len(), xc.len(), if ctx { ", neighbours" } else { "" });
        // standardised on the TRAIN songs' bins, clipped at ± CLIP sd (a fit on the raw tails diverged: dev 4K .2020 against .1761)
        const CLIP: f32 = 4.;
        let train_bins: Vec<u32> = (0..nbins as u32).filter(|&i| charts[owner[i as usize] as usize].part == 0).collect();
        let (mut bmu, mut bsd) = (vec![0f64; kf], vec![0f64; kf]);
        let sample: Vec<u32> = train_bins.iter().copied().step_by(16).collect();
        for &i in &sample { let f = &feat[i as usize * kf..(i as usize + 1) * kf]; for j in 0..kf { bmu[j] += f64::from(f[j]); bsd[j] += f64::from(f[j]).powi(2); } }
        for j in 0..kf { bmu[j] /= sample.len() as f64; bsd[j] = (bsd[j] / sample.len() as f64 - bmu[j] * bmu[j]).max(1e-9).sqrt(); }
        for i in 0..nbins { for j in 0..kf { let v = &mut feat[i * kf + j]; *v = ((f64::from(*v) - bmu[j]) / bsd[j]).clamp(-f64::from(CLIP), f64::from(CLIP)) as f32; } }
        let mut net = Net { l1: LinearConfig::new(kf, hid).init(&d), l2: LinearConfig::new(hid, hid).init(&d), l3: LinearConfig::new(hid, 2).init(&d), ctx: Vec::new(), rec: None, grind: Param::from_tensor(Tensor::<2>::zeros([1, 1], &d)), kappa: Param::from_tensor(Tensor::<2>::zeros([1, 1], &d)) };
        let mut opt = AdamConfig::new().init();
        let steps: usize = flag("--steps").map_or(20_000, |v| v.parse().unwrap());
        const BATCH: usize = 16_384;
        let y2mean = train_bins.iter().map(|&i| f64::from(y2[i as usize]) * f64::from(cnt[i as usize])).sum::<f64>() / train_bins.iter().map(|&i| f64::from(cnt[i as usize])).sum::<f64>();
        let ymean = train_bins.iter().map(|&i| f64::from(y[i as usize]) * f64::from(cnt[i as usize])).sum::<f64>() / train_bins.iter().map(|&i| f64::from(cnt[i as usize])).sum::<f64>();
        for step in 0..steps {
            let lr = 2e-3 * (0.5 * (1. + (std::f64::consts::PI * step as f64 / steps as f64).cos())) + 1e-5;
            let pick: Vec<usize> = (0..BATCH).map(|_| train_bins[(rnd() % train_bins.len() as u64) as usize] as usize).collect();
            let x: Vec<f32> = pick.iter().flat_map(|&i| feat[i * kf..(i + 1) * kf].iter().copied()).collect();
            let col = |v: Vec<f32>| Tensor::<2>::from_data(TensorData::new(v, [BATCH, 1]), &d);
            let weight = col(pick.iter().map(|&i| cnt[i]).collect());
            let target = Tensor::<2>::from_data(TensorData::new(pick.iter().flat_map(|&i| [y[i] - ymean as f32, y2[i] - y2mean as f32]).collect::<Vec<f32>>(), [BATCH, 2]), &d);
            // Huber at 3: a bin the teacher puts far off (a lone note at the bound) pulls as a line, not a square
            let err = (net.difficulty(Tensor::<2>::from_data(TensorData::new(x, [BATCH, kf]), &d)) - target).abs();
            let near = err.clone().clamp_max(3.);
            let loss = ((near.clone().square() * 0.5 + (err - near) * 3.) * weight.clone()).sum() / (weight.sum() * 2.);
            if step % 1000 == 0 { eprintln!("step {step}/{steps} loss {:.5}", loss.clone().into_scalar::<f32>()); }
            let g = GradientsParams::from_grads(loss.backward(), &net);
            net = opt.step(lr, net, g);
        }
        // every bin's difficulty under the student, every chart-rate's rating
        let valid = net.valid();
        let (mut pred, mut pred2) = (vec![0f32; nbins], vec![0f32; nbins]);
        for ch in (0..nbins).collect::<Vec<_>>().chunks(200_000) {
            let x = feat[ch[0] * kf..(ch[ch.len() - 1] + 1) * kf].to_vec();
            let both = valid.difficulty(Tensor::<2>::from_data(TensorData::new(x, [ch.len(), kf]), &dv)).into_data().try_to_vec::<f32>().unwrap();
            for (j, &i) in ch.iter().enumerate() { (pred[i], pred2[i]) = (both[2 * j] + ymean as f32, both[2 * j + 1] + y2mean as f32); }
        }
        // --skills FILE: the same rating over each skill's part of the chart (a bin weighs its notes × the skill's share of its
        // strain; a skill under 1 % of the notes has none), beside the skill's share of the notes
        let mut skills = flag("--skills").map(|_| format!("beatmap\trate\tkeys\ttheta\t{}\t{}\n", shares.iter().map(|&j| format!("th.{}", attr_names[j])).collect::<Vec<_>>().join("\t"), shares.iter().map(|&j| format!("cov.{}", attr_names[j])).collect::<Vec<_>>().join("\t")).replace('\n', &format!("\tsteep\tfar\tsteep_t\tfar_t\tfar2\tle\t{}\n", shares.iter().map(|&j| format!("le.{}", attr_names[j])).collect::<Vec<_>>().join("\t"))));
        // a bin's local disagreement: its difficulty over what the calculator's strain there and the chart's level usually go
        // with in the keymode (a plane fitted on the TRAIN bins); a skill's = the mean over its part of the chart (`le.NAME`)
        let strain_of = |i: usize| attr[(charts[owner[i] as usize].bin_start + place[i] as usize / split) * na + total_at].max(0.).ln_1p();
        let mut planes: HashMap<usize, [f64; 3]> = HashMap::new();
        if skills.is_some() {
            let mut eq: HashMap<usize, [[f64; 4]; 3]> = HashMap::new();
            for &i in &train_bins {
                let (i, c) = (i as usize, &charts[owner[i as usize] as usize]);
                let (p, w) = ([1., f64::from(strain_of(i)), f64::from(c.overall)], f64::from(cnt[i]));
                let e = eq.entry(c.keys).or_insert([[0.; 4]; 3]);
                for r in 0..3 { for q in 0..3 { e[r][q] += w * p[r] * p[q]; } e[r][3] += w * p[r] * f64::from(pred[i]); }
            }
            for (k, mut a) in eq {
                for i in 0..3 {
                    let pivot = (i..3).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs())).unwrap();
                    a.swap(i, pivot);
                    if a[i][i].abs() < 1e-9 { continue; }
                    for r in (0..3).filter(|&r| r != i) { let f = a[r][i] / a[i][i]; for q in i..4 { a[r][q] -= f * a[i][q]; } }
                }
                planes.insert(k, std::array::from_fn(|i| if a[i][i].abs() < 1e-9 { 0. } else { a[i][3] / a[i][i] }));
            }
        }
        let (mut tsv, mut err, mut at) = (String::new(), HashMap::<(bool, usize), (f64, usize)>::new(), 0);
        while at < nbins {
            let end = at + owner[at..].iter().take_while(|&&o| o == owner[at]).count();
            let c = &charts[owner[at] as usize];
            let theta = theta_of(&pred[at..end], Some(&cnt[at..end]));
            tsv += &format!("{}\t{}\t{:.6}\n", c.beatmap, c.rate, (theta / 3.).exp());
            if let Some(out) = skills.as_mut() {
                let notes: f32 = cnt[at..end].iter().sum();
                let (mut th, mut cov) = (Vec::new(), Vec::new());
                for &j in &shares {
                    let w: Vec<f32> = (at..end).map(|i| { let a = &attr[(c.bin_start + place[i] as usize / split) * na..]; cnt[i] * if a[total_at] > 0. { (a[j] / a[total_at]).clamp(0., 2.) } else { 0. } }).collect();
                    let part = w.iter().sum::<f32>() / notes;
                    th.push(if part > 0.01 { format!("{:.4}", theta_of(&pred[at..end], Some(&w))) } else { "nan".into() });
                    cov.push(format!("{part:.4}"));
                }
                // the chart's curve over ability (`shape_of`), the student's and the teacher's
                let (own, of_teacher) = (shape_of(&pred[at..end], Some(&cnt[at..end]), theta), teacher_shape[owner[at] as usize]);
                let plane = planes.get(&c.keys).copied().unwrap_or([0.; 3]);
                let local: Vec<f64> = (at..end).map(|i| f64::from(pred[i]) - plane[0] - plane[1] * f64::from(strain_of(i)) - plane[2] * f64::from(c.overall)).collect();
                let le_all = local.iter().zip(&cnt[at..end]).map(|(e, n)| e * f64::from(*n)).sum::<f64>() / f64::from(notes);
                let le: Vec<String> = shares.iter().map(|&j| {
                    let w: Vec<f64> = (at..end).map(|i| { let a = &attr[(c.bin_start + place[i] as usize / split) * na..]; f64::from(cnt[i]) * if a[total_at] > 0. { f64::from((a[j] / a[total_at]).clamp(0., 2.)) } else { 0. } }).collect();
                    let all: f64 = w.iter().sum();
                    if all > 0.01 * f64::from(notes) { format!("{:.4}", local.iter().zip(&w).map(|(e, w)| e * w).sum::<f64>() / all) } else { "nan".into() }
                }).collect();
                // far2: the same as `far` from the student's second output (each bin's difficulty 2 abilities higher)
                let far2 = (at..end).map(|i| f64::from(cnt[i]) / (1. + (theta + 2. - f64::from(pred2[i])).exp())).sum::<f64>() / f64::from(notes);
                *out += &format!("{}\t{}\t{}\t{theta:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{le_all:.4}\t{}\n", c.beatmap, c.rate, c.keys, th.join("\t"), cov.join("\t"), own.0, own.1, of_teacher.0, of_teacher.1, far2.max(1e-12).ln() - 0.1f64.ln(), le.join("\t"));
            }
            if c.notes >= 300 { let e = err.entry((c.part == 0, c.keys)).or_default(); e.0 += ((theta - teacher_theta[owner[at] as usize]) / 3.).powi(2); e.1 += 1; }
            at = end;
        }
        let mut keys: Vec<_> = err.iter().filter(|x| x.1.1 >= 300).collect();
        keys.sort_by_key(|x| (x.0.1, x.0.0));
        keys.iter().for_each(|((train, k), (s, n))| println!("{k}K {} {n} chart-rates: ln rating RMSE against the teacher {:.4}", if *train { "TRAIN" } else { "held" }, (s / *n as f64).sqrt()));
        if let (Some(path), Some(text)) = (flag("--skills"), skills) {
            std::fs::write(path, text).unwrap();
        }
        if let Some(path) = flag("--stars") {
            std::fs::write(path, tsv).unwrap();
            if let Some(save) = flag("--save") {
                let layer = |l: &Linear| { let wv: Vec<f32> = l.weight.val().into_data().try_to_vec::<f32>().unwrap(); let b: Vec<f32> = l.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap(); let [i, o] = l.weight.val().dims(); serde_json::json!({"in": i, "out": o, "w": wv, "b": b}) };
                let j = serde_json::json!({"bins": {"shares": shares.iter().map(|&j| attr_names[j]).collect::<Vec<&str>>(), "mean": mean_cols, "min": min_cols, "max": max_cols, "ctx": ctx, "sub": split, "hands": hands == 1, "clip": CLIP, "bias": ymean, "bias2": y2mean},
                    "mu": bmu, "sd": bsd, "level": true, "t0": true, "calc": meta["calc"], "teacher": tpath, "curves": level_curves(&charts, path), "l1": layer(&valid.l1), "l2": layer(&valid.l2), "l3": layer(&valid.l3)});
                std::fs::write(save, serde_json::to_string(&j).unwrap()).unwrap();
            }
        }
        return;
    }
    // The abilities' start: ln loss = chart − player by alternating means (a curve's slope over ability is about −1).
    let mut implied: Vec<f32> = vec![0.; fit.len()]; // each TRAIN score's implied ability, without its playcount term
    {
        let (mut chart_mean, mut level) = (vec![0f64; charts.len()], vec![0f64; pk_of.len()]);
        for _ in 0..6 {
            let mean = |list: &[u32], f: &dyn Fn(usize) -> f64| { let (s, w) = list.iter().fold((0., 1e-9), |a, &t| (a.0 + f64::from(fw[t as usize]) * f(t as usize), a.1 + f64::from(fw[t as usize]))); s / w };
            for &c in &train { chart_mean[c] = mean(&by_chart[c], &|t| f64::from(fy[t]) + level[pk[fit[t] as usize] as usize]); }
            let now = chart_mean.clone();
            for (p, list) in by_pk.iter().enumerate().filter(|x| !x.1.is_empty()) { level[p] = mean(list, &|t| now[scores[fit[t] as usize][0] as usize] - f64::from(fy[t])); }
        }
        fit.iter().enumerate().for_each(|(t, &i)| implied[t] = (chart_mean[scores[i as usize][0] as usize] - f64::from(fy[t])) as f32);
    }
    // A player×keymode's level, each year's offset (shrunk), and the kernel mean of what the implied abilities leave around `day`
    // (`skip`: a TRAIN score's own position stays out of its mean).
    let year_of = |day: f64| (day / 365.25).floor() as i64;
    struct Ability { level: f64, years: Vec<(i64, f64)>, left: Vec<(f64, f64, u32, f64)> } // left: day, implied − level − year, position, weight
    let abilities = |implied: &[f32]| -> Vec<Ability> {
        by_pk.iter().map(|list| {
            let wt = |t: u32| f64::from(fw[t as usize]);
            let level = list.iter().map(|&t| wt(t) * f64::from(implied[t as usize])).sum::<f64>() / list.iter().map(|&t| wt(t)).sum::<f64>().max(1e-9);
            let mut years: Vec<(i64, f64, f64)> = Vec::new();
            for &t in list {
                let y = year_of(fday[t as usize]);
                if years.last().map(|x| x.0) != Some(y) { years.push((y, 0., 0.)); }
                let e = years.last_mut().unwrap();
                (e.1, e.2) = (e.1 + wt(t) * (f64::from(implied[t as usize]) - level), e.2 + wt(t));
            }
            let years: Vec<(i64, f64)> = years.into_iter().map(|(y, s, c)| (y, s / (c + YEAR_PRIOR))).collect();
            let off = |y: i64| years.iter().find(|x| x.0 == y).map_or(0., |x| x.1);
            let left = list.iter().map(|&t| (fday[t as usize], f64::from(implied[t as usize]) - level - off(year_of(fday[t as usize])), t, wt(t))).collect();
            Ability { level, years, left }
        }).collect()
    };
    let theta_at = |a: &Ability, day: f64, skip: Option<u32>| -> f32 {
        let (lo, hi) = (a.left.partition_point(|x| x.0 < day - 5. * DRIFT), a.left.partition_point(|x| x.0 <= day + 5. * DRIFT));
        let (mut sum, mut weight) = (0., PRIOR);
        for x in a.left[lo..hi].iter().filter(|x| skip != Some(x.2)) {
            let wt = x.3 * (-(x.0 - day).abs() / DRIFT).exp();
            sum += wt * x.1;
            weight += wt;
        }
        (a.level + a.years.iter().find(|x| x.0 == year_of(day)).map_or(0., |x| x.1) + sum / weight) as f32
    };
    let thetas = |ab: &[Ability]| -> Vec<f32> {
        let mut out = vec![0f32; fit.len()];
        let chunks: Vec<&[Ability]> = ab.chunks(ab.len().div_ceil(16).max(1)).collect();
        let parts: Vec<Vec<(u32, f32)>> = std::thread::scope(|s| chunks.iter().map(|c| s.spawn(|| c.iter().flat_map(|a| a.left.iter().map(|x| (x.2, theta_at(a, x.0, Some(x.2))))).collect::<Vec<_>>())).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap()).collect());
        parts.into_iter().flatten().for_each(|(t, v)| out[t as usize] = v);
        out
    };
    let d = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0)).autodiff();
    let dv = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0));
    let mut net = Net { l1: LinearConfig::new(k, hid).init(&d), l2: LinearConfig::new(hid, hid).init(&d), l3: LinearConfig::new(hid, if mix { 2 } else { 1 }).init(&d), ctx: if seq { [1, 2, 4, 8].iter().map(|&dl| Conv1dConfig::new(hid, hid, 3).with_dilation(dl).with_groups(hid).with_padding(PaddingConfig1d::Explicit(dl, dl)).init(&d)).collect() } else { Vec::new() }, rec: flag("--pretrain").map(|_| LinearConfig::new(hid, k).init(&d)), grind: Param::from_tensor(Tensor::<2>::zeros([1, 1], &d)), kappa: Param::from_tensor(Tensor::<2>::zeros([1, 1], &d)) };
    let grid = |d: &Device| Tensor::<1>::from_data(TensorData::new((0..GRID).map(|g| -10. + STEP * g as f32).collect::<Vec<f32>>(), [GRID]), d);
    // ln of each chart's mean σ(d − ability) on the grid: `x` = [charts × notes, k] padded, `mask` = [charts, notes, 1]
    // `--mix` (Stage 3b): two kinds of loss with a head each. Break loss (100s, 50s, misses: 2/3, 5/6, 1 of a note) falls
    // with ability as before; timing loss (200s: a third of a note) on its own slope κ = e^kappa. → ln of their sum, and
    // (ln timing, ln break) for the scores' judgement counts to be fitted on.
    let kinds = |net: &Net, x: Tensor<2>, mask: Tensor<3>, b: usize, m: usize, d: &Device| -> (Tensor<2>, Option<(Tensor<2>, Tensor<2>)>) {
        let out = net.seq(x, Some((b, m, mask.clone())));
        let mean = |z: Tensor<3>| ((sigmoid(z) * mask.clone()).sum_dim(1) / mask.clone().sum_dim(1)).reshape([b, GRID]);
        if !mix {
            return (mean(out.reshape([b, m, 1]) - grid(d).reshape([1, 1, GRID])).clamp_min(1e-7).log(), None);
        }
        let brk = mean(out.clone().narrow(1, 0, 1).reshape([b, m, 1]) - grid(d).reshape([1, 1, GRID]));
        let timing = mean(out.narrow(1, 1, 1).reshape([b, m, 1]) - grid(d).reshape([1, 1, GRID]) * net.kappa.val().exp().reshape([1, 1, 1])) / 3.;
        ((brk.clone() + timing.clone()).clamp_min(1e-7).log(), Some((timing.clamp_min(1e-7).log(), brk.clamp_min(1e-7).log())))
    };
    let curves = |net: &Net, x: Tensor<2>, mask: Tensor<3>, b: usize, m: usize, d: &Device| -> Tensor<2> { kinds(net, x, mask, b, m, d).0 };
    // every listed chart's whole curve, charts of a like length batched together
    let all_curves = |net: &Net, list: &[usize]| -> HashMap<usize, Vec<f32>> {
        let mut order = list.to_vec();
        order.sort_by_key(|&c| charts[c].notes);
        let mut out = HashMap::new();
        let mut at = 0;
        while at < order.len() {
            let mut end = at + 1;
            while end < order.len() && (end + 1 - at) * charts[order[end]].notes <= 160_000 { end += 1; }
            let (b, m) = (end - at, charts[order[end - 1]].notes);
            let (mut x, mut mask) = (Vec::with_capacity(b * m * k), vec![0f32; b * m]);
            for (bi, &c) in order[at..end].iter().enumerate() {
                (0..charts[c].notes).for_each(|j| { feats(c, charts[c].start + j, &mut x); mask[bi * m + j] = 1.; });
                x.extend(std::iter::repeat(0.).take((m - charts[c].notes) * k));
            }
            let lnc = curves(net, Tensor::<2>::from_data(TensorData::new(x, [b * m, k]), &dv), Tensor::<3>::from_data(TensorData::new(mask, [b, m, 1]), &dv), b, m, &dv).into_data().try_to_vec::<f32>().unwrap();
            order[at..end].iter().enumerate().for_each(|(bi, &c)| { out.insert(c, lnc[bi * GRID..(bi + 1) * GRID].to_vec()); });
            at = end;
        }
        out
    };
    // a curve read at an ability: (ln loss, its slope over ability)
    let read = |curve: &[f32], theta: f32| -> (f32, f32) {
        let pos = (theta.clamp(-9.9, 9.9) + 10.) / STEP;
        let (g, frac) = (pos.floor() as usize, pos.fract());
        (curve[g] * (1. - frac) + curve[g + 1] * frac, (curve[g + 1] - curve[g]) / STEP)
    };
    // --skill-theta: each listed chart's family exposure at every grid ability, [GRID × NF] row-major: `note` = the families'
    // shares of σ'(d − θ) over the chart's notes (the break head under --mix), `chart` = their plain shares at every θ
    let exposures = |net: &Net, list: &[usize], chart: bool| -> HashMap<usize, Vec<f32>> {
        let mut order = list.to_vec();
        order.sort_by_key(|&c| charts[c].notes);
        let mut out = HashMap::new();
        let mut at = 0;
        while at < order.len() {
            let mut end = at + 1;
            while end < order.len() && (end + 1 - at) * charts[order[end]].notes <= 160_000 { end += 1; }
            let (b, m) = (end - at, charts[order[end - 1]].notes);
            let (mut x, mut fam, mut mask) = (Vec::with_capacity(b * m * k), Vec::with_capacity(b * m * NF), vec![0f32; b * m]);
            for (bi, &c) in order[at..end].iter().enumerate() {
                (0..charts[c].notes).for_each(|j| { feats(c, charts[c].start + j, &mut x); fam_of(c, charts[c].start + j, &mut fam); mask[bi * m + j] = 1.; });
                x.extend(std::iter::repeat(0.).take((m - charts[c].notes) * k));
                fam.extend(std::iter::repeat(0.).take((m - charts[c].notes) * NF));
            }
            let mask = Tensor::<3>::from_data(TensorData::new(mask, [b, m, 1]), &dv);
            let wgt = if chart {
                mask.repeat_dim(2, GRID)
            } else {
                let z = net.seq(Tensor::<2>::from_data(TensorData::new(x, [b * m, k]), &dv), Some((b, m, mask.clone()))).narrow(1, 0, 1).reshape([b, m, 1]);
                let s = sigmoid(z - grid(&dv).reshape([1, 1, GRID]));
                s.clone() * (s.neg() + 1.) * mask
            };
            let num = wgt.clone().swap_dims(1, 2).matmul(Tensor::<3>::from_data(TensorData::new(fam, [b, m, NF]), &dv));
            let e = (num / wgt.sum_dim(1).reshape([b, GRID, 1]).clamp_min(1e-12)).into_data().try_to_vec::<f32>().unwrap();
            order[at..end].iter().enumerate().for_each(|(bi, &c)| { out.insert(c, e[bi * GRID * NF..(bi + 1) * GRID * NF].to_vec()); });
            at = end;
        }
        out
    };
    let expo_at = |e: &[f32], theta: f32| -> [f32; NF] {
        let pos = (theta.clamp(-9.9, 9.9) + 10.) / STEP;
        let (g, frac) = (pos.floor() as usize, pos.fract());
        std::array::from_fn(|f| e[g * NF + f] * (1. - frac) + e[(g + 1) * NF + f] * frac)
    };
    // a score's ability shift: the player's family deviations times the chart's exposure at that ability over their usual one
    let shift_of = |e: &[f32], theta: f32, delta: &[f64; NF], usual: &[f64; NF]| -> f32 {
        let x = expo_at(e, theta);
        (0..NF).map(|f| delta[f] * (f64::from(x[f]) - usual[f])).sum::<f64>() as f32
    };
    const B: usize = 64; // charts a step
    const M: usize = 2048; // notes of a chart a step (a longer chart: a random 2048)
    const S: usize = 2048; // scores of a chart a step
    let int = |v: Vec<i32>, d: &Device| { let n = v.len(); Tensor::<1, Int>::from_data(TensorData::new(v, [n]), d) };
    let col = |v: Vec<f32>, d: &Device| { let n = v.len(); Tensor::<2>::from_data(TensorData::new(v, [n, 1]), d) };
    let (rounds, per): (usize, usize) = (flag("--rounds").map_or(8, |v| v.parse().unwrap()), flag("--steps").map_or(800, |v| v.parse().unwrap()));
    // the gate's held-out rows: (line, chart, player×keymode, day, ln playcount, ln loss)
    let text = flag("--rows").map(|path| std::fs::read_to_string(path).unwrap()).unwrap_or_default();
    let header = text.lines().next().unwrap_or("");
    let held: Vec<(&str, usize, u32, f64, f32, f64)> = {
        let chart_at: HashMap<(i64, u64), usize> = charts.iter().enumerate().map(|(i, c)| ((c.beatmap, c.rate.to_bits()), i)).collect();
        let user_at: HashMap<i64, u32> = users.iter().enumerate().map(|(i, &u)| (u, i as u32)).collect();
        text.lines().skip(1).filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let c = *chart_at.get(&(f[0].parse().unwrap(), f[2].parse::<f64>().unwrap().to_bits()))?;
            let p = *pk_of.get(&(*user_at.get(&f[4].parse().unwrap())?, charts[c].keys))?;
            Some((l, c, p, day_of(f[8].parse().unwrap()), f[9].parse::<f32>().unwrap().max(1.).ln(), f[7].parse().unwrap()))
        }).collect()
    };
    let mut need: Vec<usize> = held.iter().map(|h| h.1).collect();
    need.sort_unstable();
    need.dedup();
    // each at its player's ability on its date (levels and kernel from TRAIN scores only) → per keymode rows, row RMSE, map-bias RMSE
    let judge = |net: &Net, ab: &[Ability], grind: f32, out: Option<&str>, skill: Option<(&[[f64; NF]], &[[f64; NF]], bool)>| {
        let curve = all_curves(net, &need);
        let ex = skill.map(|s| exposures(net, &need, s.2));
        let mut tsv = format!("{header}\n");
        let mut by_keys: std::collections::BTreeMap<usize, (f64, f64, HashMap<usize, (f64, f64)>)> = Default::default();
        for &(line, c, p, day, lpc, y) in &held {
            let th = theta_at(&ab[p as usize], day, None);
            let sh = skill.zip(ex.as_ref()).map_or(0., |((dl, us, _), ex)| shift_of(&ex[&c], th, &dl[p as usize], &us[p as usize]));
            let res = y - f64::from(read(&curve[&c], th + sh + grind * lpc).0);
            if out.is_some() {
                let f: Vec<&str> = line.split('\t').collect();
                tsv += &format!("{}\t{res:.4}\t{}\n", f[..6].join("\t"), f[7..].join("\t"));
            }
            let e = by_keys.entry(charts[c].keys).or_default();
            (e.0, e.1) = (e.0 + 1., e.1 + res * res);
            let m = e.2.entry(c).or_default();
            (m.0, m.1) = (m.0 + 1., m.1 + res);
        }
        for (keys, (n, sq, maps)) in by_keys.iter().filter(|x| x.1.0 >= 1000. || out.is_some()) {
            let bias: Vec<f64> = maps.values().filter(|m| m.0 >= 5.).map(|m| (m.1 / m.0).powi(2)).collect();
            println!("{keys}K  rows {n}  row RMSE {:.4}  maps {}  map-bias RMSE {:.4}", (sq / n).sqrt(), bias.len(), (bias.iter().sum::<f64>() / bias.len().max(1) as f64).sqrt());
        }
        if let Some(o) = out { std::fs::write(o, tsv).unwrap(); }
    };
    // --pretrain DIR2 (Stage 5, attempt 3): before any score is seen, the encoder (first layer + convolutions) learns to
    // fill in masked notes from their neighbours, on the unlabelled charts of DIR2 (`calib poprows OUT --extra FILES`) and
    // the TRAIN charts; `--pre-steps N` steps of 32 windows × 1,024 notes, 15 % of the real notes masked
    if let Some(dir2) = flag("--pretrain") {
        assert!(seq && attr_in.is_empty(), "--pretrain: a --seq net without --attr");
        let meta2: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir2}/charts.json")).unwrap()).unwrap();
        assert!(meta2["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).eq(cols.iter().copied()), "--pretrain: other columns");
        let extra: Vec<(usize, usize, f32)> = meta2["charts"].as_array().unwrap().iter().map(|c| (c["start"].as_u64().unwrap() as usize, c["notes"].as_u64().unwrap() as usize, ((c["overall"].as_f64().unwrap_or(1.).max(0.1).ln() - 1.5) / 0.7) as f32)).collect();
        let mm2 = map_in(dir2, "rows.bin");
        // SAFETY: a plain little-endian f32 table written by `calib poprows --extra` on this machine
        let rows2: &[f32] = unsafe { std::slice::from_raw_parts(mm2.as_ptr() as *const f32, mm2.len() / 4) };
        let (xm2, _) = compact(dir2, rows2, w, &extra.iter().map(|e| (e.0, e.1, 0)).collect::<Vec<_>>(), &inputs, t_col, &mu, &sd, bin_s);
        let x2 = u16s(&xm2);
        let feats2 = |(_, _, overall): (usize, usize, f32), i: usize, out: &mut Vec<f32>| {
            out.extend(x2[i * ni..(i + 1) * ni].iter().map(|&h| half::f16::from_bits(h).to_f32()));
            if level_in { out.push(overall); }
        };
        let pool: Vec<(bool, usize)> = extra.iter().enumerate().map(|(c, _)| (true, c)).chain(train.iter().map(|&c| (false, c))).collect();
        let pre_steps: usize = flag("--pre-steps").map_or(4000, |v| v.parse().unwrap());
        let (pb, pm) = (32usize, 1024usize);
        let mut popt = AdamConfig::new().init();
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for step in 0..pre_steps {
            let (mut x, mut tgt, mut mask, mut hide) = (Vec::with_capacity(pb * pm * k), Vec::with_capacity(pb * pm * k), vec![0f32; pb * pm], vec![0f32; pb * pm]);
            for b in 0..pb {
                let (un, c) = pool[(rnd() % pool.len() as u64) as usize];
                let (start, n) = if un { (extra[c].0, extra[c].1) } else { (charts[c].start, charts[c].notes) };
                let from = if n > pm { (rnd() % (n - pm + 1) as u64) as usize } else { 0 };
                for j in 0..pm {
                    if j < n.min(pm) {
                        let at = x.len();
                        if un { feats2(extra[c], start + from + j, &mut x) } else { feats(c, start + from + j, &mut x) }
                        tgt.extend_from_slice(&x[at..]);
                        mask[b * pm + j] = 1.;
                        if rnd() % 100 < 15 { hide[b * pm + j] = 1.; x[at..].iter_mut().for_each(|v| *v = 0.); }
                    } else {
                        x.extend(std::iter::repeat(0.).take(k));
                        tgt.extend(std::iter::repeat(0.).take(k));
                    }
                }
            }
            let lr = 1e-3 * (0.5 * (1. + (std::f64::consts::PI * step as f64 / pre_steps as f64).cos())) + 1e-5;
            let h = net.encode(Tensor::<2>::from_data(TensorData::new(x, [pb * pm, k]), &d), Some((pb, pm, Tensor::<3>::from_data(TensorData::new(mask, [pb, pm, 1]), &d))));
            let hide = Tensor::<2>::from_data(TensorData::new(hide, [pb * pm, 1]), &d);
            let err = (net.rec.as_ref().unwrap().forward(h) - Tensor::<2>::from_data(TensorData::new(tgt, [pb * pm, k]), &d)).square() * hide.clone();
            let loss = err.sum() / (hide.sum() * k as f32);
            if step % 200 == 0 { eprintln!("pretrain {step}/{pre_steps} masked-note MSE {:.4}", loss.clone().into_scalar::<f32>()); }
            let g = GradientsParams::from_grads(loss.backward(), &net);
            net = popt.step(lr, net, g);
        }
    }
    let every: usize = flag("--judge").map_or(usize::MAX, |v| v.parse().unwrap());
    let mut opt = AdamConfig::new().init();
    let mut ab = abilities(&implied);
    let mut theta = thetas(&ab);
    // --skill-theta state: each player×keymode's family deviations δ and usual exposure, every TRAIN score's θ + shift
    let (mut delta, mut usual) = (vec![[0f64; NF]; pk_of.len()], vec![[0f64; NF]; pk_of.len()]);
    let mut eff = theta.clone();
    let penalty: f64 = flag("--skill-pen").map_or(15., |v| v.parse().unwrap());
    // δ: per player×keymode a ridge fit (`--skill-pen` rows) of what each score implies over its level on its exposure, both
    // centred on the player's means
    let fit_delta = |ex: &HashMap<usize, Vec<f32>>, implied: &[f32], theta: &[f32]| -> (Vec<[f64; NF]>, Vec<[f64; NF]>) {
        let (mut delta, mut usual) = (vec![[0f64; NF]; by_pk.len()], vec![[0f64; NF]; by_pk.len()]);
        for (p, list) in by_pk.iter().enumerate() {
            let xs: Vec<([f32; NF], f64, f64)> = list.iter().filter_map(|&t| { let t = t as usize; let e = ex.get(&(scores[fit[t] as usize][0] as usize))?; Some((expo_at(e, theta[t]), f64::from(fw[t]), f64::from(implied[t] - theta[t]))) }).collect();
            let sw: f64 = xs.iter().map(|x| x.1).sum();
            if sw <= 0. { continue; }
            let mean: [f64; NF] = std::array::from_fn(|f| xs.iter().map(|x| x.1 * f64::from(x.0[f])).sum::<f64>() / sw);
            let rm = xs.iter().map(|x| x.1 * x.2).sum::<f64>() / sw;
            let (mut a, mut b) = (vec![0f64; NF * NF], vec![0f64; NF]);
            (0..NF).for_each(|f| a[f * NF + f] = penalty);
            for (x, w, r) in &xs {
                let c: [f64; NF] = std::array::from_fn(|f| f64::from(x[f]) - mean[f]);
                for i in 0..NF {
                    b[i] += w * c[i] * (r - rm);
                    for j in 0..NF { a[i * NF + j] += w * c[i] * c[j]; }
                }
            }
            (delta[p], usual[p]) = (solve_small(a, b).try_into().unwrap(), mean);
        }
        (delta, usual)
    };
    let steps = rounds * per;
    let (tx, rx) = std::sync::mpsc::sync_channel::<Batch>(24);
    std::thread::scope(|sc| {
        // batches are put together beside the steps (one core did them: 0.35 s a step with the GPU waiting)
        for worker in 0..10u64 {
            let (tx, train, charts, by_chart, feats, sib, paired, pk, fit, fday) = (tx.clone(), &train, &charts, &by_chart, &feats, &sib, &paired, &pk, &fit, &fday);
            sc.spawn(move || {
                let mut seed = 0x9e3779b97f4a7c15u64 ^ (worker + 1).wrapping_mul(0xd1b54a32d192ed03);
                let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
                loop {
                    // --pairs: the second half of the charts come as (chart-rate, a sibling rate) pairs
                    let half = if pair_w > 0. && !paired.is_empty() { B / 2 } else { B };
                    let mut picked: Vec<usize> = (0..half).map(|_| train[(rnd() % train.len() as u64) as usize]).collect();
                    while picked.len() < B { let a = paired[(rnd() % paired.len() as u64) as usize]; picked.extend([a, sib[a][(rnd() % sib[a].len() as u64) as usize]]); }
                    let mut from_to = Vec::with_capacity(B);
                    let m = picked.iter().map(|&c| charts[c].notes.min(M)).max().unwrap();
                    let (mut x, mut mask) = (Vec::with_capacity(B * m * k), vec![0f32; B * m]);
                    let (mut which, mut pos) = (Vec::new(), Vec::new());
                    for (b, &c) in picked.iter().enumerate() {
                        let ch = &charts[c];
                        let from = if seq && ch.notes > M { (rnd() % (ch.notes - M + 1) as u64) as usize } else { 0 };
                        for j in 0..m {
                            if j < ch.notes.min(M) {
                                feats(c, ch.start + if ch.notes <= M || seq { from + j } else { (rnd() % ch.notes as u64) as usize }, &mut x);
                                mask[b * m + j] = 1.;
                            } else {
                                x.extend(std::iter::repeat(0.).take(k));
                            }
                        }
                        let list = &by_chart[c];
                        from_to.push(pos.len());
                        for j in 0..list.len().min(S) {
                            which.push(b as i32);
                            pos.push(if list.len() <= S { list[j] } else { list[(rnd() % list.len() as u64) as usize] });
                        }
                    }
                    from_to.push(pos.len());
                    // the same player's scores on both rates of a pair: (row, row) in the batch's scores
                    let mut pairs = Vec::new();
                    for b in (half..B).step_by(2) {
                        let player = |r: usize| pk[fit[pos[r] as usize] as usize];
                        let other: HashMap<u32, usize> = (from_to[b + 1]..from_to[b + 2]).map(|r| (player(r), r)).collect();
                        (from_to[b]..from_to[b + 1]).for_each(|r| if let Some(&o) = other.get(&player(r)).filter(|&&o| (fday[pos[r] as usize] - fday[pos[o] as usize]).abs() <= pair_days) { pairs.push((r as i32, o as i32)); });
                    }
                    if tx.send(Batch { x, mask, m, which, pos, pairs }).is_err() { return; }
                }
            });
        }
        drop(tx);
        for step in 0..steps {
            let lr = 2e-3 * (0.5 * (1. + (std::f64::consts::PI * step as f64 / steps as f64).cos())) + 1e-5;
            let Batch { x, mask, m, which, pos, pairs } = rx.recv().unwrap();
            let at = |v: &[f32]| pos.iter().map(|&t| v[t as usize]).collect::<Vec<f32>>();
            let (lnc, parts) = kinds(&net, Tensor::<2>::from_data(TensorData::new(x, [B * m, k]), &d), Tensor::<3>::from_data(TensorData::new(mask, [B, m, 1]), &d), B, m, &d);
            let ability = col(at(&eff), &d) + col(at(&flpc), &d) * net.grind.val();
            // linear interpolation on the grid: hat weights of every grid point around each score's ability
            let hat = relu((((ability.clamp(-9.9, 9.9) + 10.) / STEP) - Tensor::<1, Int>::arange(0..GRID as i64, &d).float().reshape([1, GRID])).abs().neg() + 1.);
            let which = int(which, &d);
            let fitted = |curve: Tensor<2>, y: &[f32], w: &[f32]| { let weight = col(at(w), &d); (((curve.select(0, which.clone()) * hat.clone()).sum_dim(1) - col(at(y), &d)).square() * weight.clone()).sum() / weight.sum() };
            let paired_lnc = (pair_w > 0.).then(|| lnc.clone());
            let mut mse = if zeros {
                let weight = col(at(&fw), &d);
                let lnp = (lnc.select(0, which.clone()) * hat.clone()).sum_dim(1);
                (((lnp.exp() + col(at(&fc), &d)).log() - col(at(&fy), &d)).square() * weight.clone()).sum() / weight.sum()
            } else {
                fitted(lnc, &fy, &fw)
            };
            if let Some((timing, brk)) = parts {
                mse = mse + fitted(timing, &kind_y[0], &kind_w[0]) + fitted(brk, &kind_y[1], &kind_w[1]);
            }
            if let Some(lnc) = paired_lnc.filter(|_| !pairs.is_empty()) {
                // a player's two rates of one chart: the difference of their residuals, the player's ability error taken out
                let r = (lnc.select(0, which.clone()) * hat.clone()).sum_dim(1) - col(at(&fy), &d);
                let wv = at(&fw);
                let wt = col(pairs.iter().map(|&(a, b)| (wv[a as usize] * wv[b as usize]).sqrt()).collect(), &d);
                let (ia, ib): (Vec<i32>, Vec<i32>) = pairs.into_iter().unzip();
                let diff = r.clone().select(0, int(ia, &d)) - r.select(0, int(ib, &d));
                mse = mse + (diff.square() * wt.clone()).sum() / wt.sum() * pair_w;
            }
            if step % 100 == 0 { eprintln!("step {step}/{steps} mse {:.5}", mse.clone().into_scalar::<f32>()); }
            let g = GradientsParams::from_grads(mse.backward(), &net);
            net = opt.step(lr, net, g);
            if (step + 1) % per == 0 {
                // the abilities again, from what every TRAIN score implies under the net as it is now
                let (valid, grind) = (net.valid(), net.grind.val().into_data().try_to_vec::<f32>().unwrap()[0]);
                let curve = all_curves(&valid, &train);
                let mut sq = 0f64;
                for (t, &i) in fit.iter().enumerate() {
                    let Some(cv) = curve.get(&(scores[i as usize][0] as usize)) else { continue };
                    let (lnp, slope) = read(cv, eff[t] + grind * flpc[t]);
                    // with the pseudo-count the curve flattens where p is small next to it
                    let (p, c) = (lnp.exp(), fc[t]);
                    let (pred, slope) = ((p + c).ln(), slope * p / (p + c));
                    sq += f64::from(fw[t]) * f64::from(fy[t] - pred).powi(2);
                    implied[t] = eff[t] + ((fy[t] - pred) / slope.min(-0.2)).clamp(-3., 3.);
                }
                // `both`: no δ while training; after the last round δ is fitted on each exposure and judged (one net, two exposures)
                let ex = skill_theta.filter(|m| *m != "both").map(|m| exposures(&valid, &train, m == "chart"));
                if let Some(ex) = &ex {
                    (delta, usual) = fit_delta(ex, &implied, &theta);
                    for (t, &i) in fit.iter().enumerate() {
                        if let Some(e) = ex.get(&(scores[i as usize][0] as usize)) { implied[t] -= shift_of(e, theta[t], &delta[pk[i as usize] as usize], &usual[pk[i as usize] as usize]); }
                    }
                    let spread: f64 = delta.iter().map(|d| d.iter().map(|v| v * v).sum::<f64>()).sum::<f64>() / delta.len().max(1) as f64;
                    eprintln!("skill θ: rms δ {:.4}", (spread / NF as f64).sqrt());
                }
                ab = abilities(&implied);
                theta = thetas(&ab);
                eff = theta.clone();
                if let Some(ex) = &ex {
                    for (t, &i) in fit.iter().enumerate() {
                        if let Some(e) = ex.get(&(scores[i as usize][0] as usize)) { eff[t] += shift_of(e, theta[t], &delta[pk[i as usize] as usize], &usual[pk[i as usize] as usize]); }
                    }
                }
                let round = (step + 1) / per;
                if mix { eprintln!("κ {:.4}", valid.kappa.val().into_data().try_to_vec::<f32>().unwrap()[0].exp()); }
                eprintln!("round {round}: train row RMSE (weighted) {:.4}, β {grind:.4}", (sq / fw.iter().map(|&w| f64::from(w)).sum::<f64>()).sqrt());
                // --stars FILE: every chart-rate's rating for `calib eval BASE stars=FILE:PACK` (the gate's own yardstick and
                // rows): e^(θ/3) of the ability θ at which the chart's curve loses 10 % (the calculator's ACC_TARGET).
                if let Some(path) = flag("--stars").filter(|_| round == rounds) {
                    let mut order: Vec<usize> = (0..charts.len()).filter(|&c| charts[c].notes > 0).collect();
                    order.sort_by_key(|&c| charts[c].notes);
                    let (mut tsv, mut at) = (String::new(), 0);
                    let (outs, kap) = (if mix { 2 } else { 1 }, f64::from(valid.kappa.val().into_data().try_to_vec::<f32>().unwrap()[0]).exp());
                    while at < order.len() {
                        let mut end = at + 1;
                        while end < order.len() && (end + 1 - at) * charts[order[end]].notes <= 160_000 { end += 1; }
                        let m = charts[order[end - 1]].notes;
                        let (mut x, mut mask) = (Vec::with_capacity((end - at) * m * k), vec![0f32; (end - at) * m]);
                        for (bi, &c) in order[at..end].iter().enumerate() {
                            (0..charts[c].notes).for_each(|j| { feats(c, charts[c].start + j, &mut x); mask[bi * m + j] = 1.; });
                            x.extend(std::iter::repeat(0.).take((m - charts[c].notes) * k));
                        }
                        let mask = Tensor::<3>::from_data(TensorData::new(mask, [end - at, m, 1]), &dv);
                        let dn = valid.seq(Tensor::<2>::from_data(TensorData::new(x, [(end - at) * m, k]), &dv), Some((end - at, m, mask))).into_data().try_to_vec::<f32>().unwrap();
                        for (bi, &c) in order[at..end].iter().enumerate() {
                            // as ms_calc::notenet::theta_of: the ability at which the notes lose 10 % on average
                            let dc = &dn[bi * m * outs..(bi * m + charts[c].notes) * outs];
                            let lost = |z: &[f32], th: f64| if mix { 1. / (1. + (th - f64::from(z[0])).exp()) + 1. / (1. + (kap * th - f64::from(z[1])).exp()) / 3. } else { 1. / (1. + (th - f64::from(z[0])).exp()) };
                            let (mut lo, mut hi) = (-20f64, 20f64);
                            for _ in 0..60 {
                                let mid = 0.5 * (lo + hi);
                                if dc.chunks_exact(outs).map(|z| lost(z, mid)).sum::<f64>() / charts[c].notes as f64 > 0.1 { lo = mid } else { hi = mid }
                            }
                            tsv += &format!("{}\t{}\t{:.5}\n", charts[c].beatmap, charts[c].rate, (0.5 * (lo + hi) / 3.).exp());
                        }
                        at = end;
                    }
                    std::fs::write(path, tsv).unwrap();
                }
                // --save FILE (after --stars FILE, whose ratings its lines are fitted on): the difficulty net for ms_calc::notenet::Pop (inputs by column name, standardisation, layers as [in][out] rows)
                if let Some(path) = flag("--save").filter(|_| round == rounds) {
                    assert!(!seq, "--save: ms_calc::notenet reads no --seq net");
                    let lin = |l: &Linear| { let w: Vec<f32> = l.weight.val().into_data().try_to_vec::<f32>().unwrap(); let b: Vec<f32> = l.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap();
                        let [i, o] = l.weight.val().dims(); serde_json::json!({"in": i, "out": o, "w": w, "b": b}) };
                    let names: Vec<&str> = inputs.iter().map(|&c| cols[c]).collect();
                    let curves = level_curves(&charts, flag("--stars").expect("--save reads --stars"));
                    let j = serde_json::json!({"inputs": names, "mu": inputs.iter().map(|&c| mu[c]).collect::<Vec<f64>>(), "sd": inputs.iter().map(|&c| sd[c]).collect::<Vec<f64>>(),
                        "level": level_in, "attr": attr_in.iter().map(|&j| attr_names[j]).collect::<Vec<&str>>(), "calc": meta["calc"], "t0": true, "mix": mix, "kappa": valid.kappa.val().into_data().try_to_vec::<f32>().unwrap()[0], "curves": curves, "l1": lin(&valid.l1), "l2": lin(&valid.l2), "l3": lin(&valid.l3)});
                    std::fs::write(path, serde_json::to_string(&j).unwrap()).unwrap();
                }
                if !held.is_empty() && (round % every == 0 || round == rounds) {
                    println!("round {round}");
                    judge(&valid, &ab, grind, (round == rounds).then(|| flag("--out")).flatten(), None);
                    if skill_theta == Some("both") {
                        for (name, chart) in [("note", false), ("chart", true)] {
                            let (dl, us) = fit_delta(&exposures(&valid, &train, chart), &implied, &theta);
                            println!("with the players' family deviations ({name} exposure, fitted after the round)");
                            judge(&valid, &ab, grind, None, Some((&dl, &us, chart)));
                        }
                    } else if let Some(m) = skill_theta {
                        println!("with the players' family deviations (--skill-theta {m})");
                        judge(&valid, &ab, grind, None, Some((&delta, &usual, m == "chart")));
                    }
                }
            }
        }
        drop(rx);
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_compact_table_holds_the_standardised_inputs_and_bins() {
        let dir = std::env::temp_dir().join(format!("popfit-compact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // two charts, columns (t, a); t counts from each chart's first note, bins of 0.5 s capped at the chart's count
        let rows = [10., 1., 11.2, 3., 5., 1e12];
        let (xm, bm) = super::compact(dir.to_str().unwrap(), &rows, 2, &[(0, 2, 3), (2, 1, 1)], &[0, 1], 0, &[0., 2.], &[1., 0.5], 0.5);
        let x: Vec<f32> = super::u16s(&xm).iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
        let want = [0., -2., 1.2, 2., 0., 65504.];
        assert!(x.iter().zip(want).all(|(a, b)| (a - b).abs() < 2e-3), "{x:?}");
        assert_eq!(super::u16s(&bm), [0, 2, 0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_small_solver_inverts_a_ridge_system() {
        let x = super::solve_small(vec![4., 1., 0., 1., 3., 1., 0., 1., 2.], vec![5., 7., 7.]);
        [1., 1., 3.].iter().zip(&x).for_each(|(want, got)| assert!((want - got).abs() < 1e-9, "{x:?}"));
    }

    #[test]
    fn the_level_curve_finds_a_cubic_and_its_lines_slope() {
        // y = 0.5 + 0.3u + 0.2u² − 0.1u³ over x = 1 … 3 (u = x − 2); the 1st–99th percentile of 201 even levels is 1.02 … 2.98
        let pts: Vec<(f64, f64)> = (0..=200).map(|i| { let x = 1. + i as f64 / 100.; let u = x - 2.; (x, 0.5 + 0.3 * u + 0.2 * u * u - 0.1 * u * u * u) }).collect();
        let c = super::level_curve(&pts).unwrap();
        let (lo, hi) = (c["lo"].as_f64().unwrap(), c["hi"].as_f64().unwrap());
        assert!((lo - 1.02).abs() < 1e-9 && (hi - 2.98).abs() < 1e-9);
        let at = |x: f64| { let u = (2. * x - lo - hi) / (hi - lo); (0..4).rev().fold(0., |s, i| s * u + c["c"][i].as_f64().unwrap()) };
        assert!([1.1, 2., 2.7].iter().all(|&x| { let u: f64 = x - 2.; (at(x) - (0.5 + 0.3 * u + 0.2 * u * u - 0.1 * u * u * u)).abs() < 1e-9 }));
        // the line through the same points: slope 0.3 − 0.1 · (3/5 of the span's u², 0.98²) by least squares on even levels
        assert!((c["slope"].as_f64().unwrap() - (0.3 - 0.1 * 0.6 * 0.98 * 0.98)).abs() < 2e-3, "{}", c["slope"]);
        assert!(super::level_curve(&pts[..3]).is_none());
    }
}

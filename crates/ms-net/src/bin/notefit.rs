//! S7 per-note probe: difficulty d_i = MLP(chart-only features of note i), ability θ_p per play,
//! P(note's accuracy lost) = σ(d_i − θ_p) against the head judgement's lazer loss. Train = song half 0; judged on half 1:
//! a play's θ = mean θ of train plays in the DAYS before it (walk-forward: no later play, none of its own notes). `--baseline`: d = MLP(ln old calc
//! rating of the play, keymode, rate, OD) through the same MLP and ability model, so the two compare chart difficulty only.
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::{gelu, log_sigmoid};

const DAYS: f64 = 7.;
// replay-derived or bookkeeping columns: never inputs
const SKIP: [&str; 11] = ["play", "j", "err", "signed", "recent_miss", "recent_lost", "fail", "row_miss", "block", "tail_j", "tail_err"];
const TAIL: u64 = 1 << 63; // a sample id with this bit is the row's LN release, not its head
const WEIGHT: [f32; 6] = [305., 300., 200., 100., 50., 0.];

#[derive(Module, Debug)]
struct Net { l1: Linear, l2: Linear, l3: Linear, theta: Param<Tensor<2>> }

impl Net {
    fn new(inputs: usize, hid: usize, plays: usize, d: &Device) -> Self {
        Net { l1: LinearConfig::new(inputs, hid).init(d), l2: LinearConfig::new(hid, hid).init(d), l3: LinearConfig::new(hid, 1).init(d),
              theta: Param::from_tensor(Tensor::<2>::zeros([plays, 1], d)) }
    }
    fn difficulty(&self, x: Tensor<2>) -> Tensor<2> {
        let h = gelu(self.l1.forward(x));
        self.l3.forward(gelu(self.l2.forward(h.clone())) + h)
    }
}

fn days(s: &str) -> f64 {
    let p = |a: usize, b: usize| s.get(a..b).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
    let (y, m, d) = (p(0, 4), p(5, 7), p(8, 10));
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + d - 1;
    let frac = p(11, 13) as f64 / 24. + p(14, 16) as f64 / 1440.;
    (era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy) as f64 + frac
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let baseline = args.iter().any(|a| a == "--baseline");
    let epochs: usize = args.iter().position(|a| a == "--epochs").map_or(4, |i| args[i + 1].parse().unwrap());
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/rows.json")).unwrap()).unwrap();
    let cols: Vec<&str> = meta["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let w = cols.len();
    let plays = meta["plays"].as_array().unwrap();
    let half: Vec<u8> = plays.iter().map(|p| p["half"].as_u64().unwrap() as u8).collect();
    let when: Vec<f64> = plays.iter().map(|p| days(p["played"].as_str().unwrap_or(""))).collect();
    // rowsmerge's player of each play: every time-local part (sessions, previous play, θ walk, θ forecasts) stays within one
    let player: Vec<usize> = plays.iter().map(|p| p["player"].as_u64().unwrap_or(0) as usize).collect();
    let by_player_time = |a: &usize, b: &usize| player[*a].cmp(&player[*b]).then(when[*a].total_cmp(&when[*b]));
    let rating: Vec<f32> = plays.iter().map(|p| p["rating"].as_f64().unwrap_or(1.).max(0.1).ln() as f32).collect();
    let file = std::fs::File::open(format!("{dir}/rows.bin")).unwrap();
    let mm = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let rows: &[f32] = unsafe { std::slice::from_raw_parts(mm.as_ptr() as *const f32, mm.len() / 4) };
    let n = rows.len() / w;
    let row = |i: usize| &rows[i * w..(i + 1) * w];
    let (ci, cj) = (cols.iter().position(|c| *c == "play").unwrap(), cols.iter().position(|c| *c == "j").unwrap());
    let drop: Vec<&str> = args.iter().position(|a| a == "--drop").map_or(vec![], |i| args[i + 1].split(',').collect());
    let inputs: Vec<usize> = (0..w).filter(|c| !SKIP.contains(&cols[*c]) && !drop.contains(&cols[*c])).collect();
    let ct = cols.iter().position(|c| *c == "tail_j");
    // --session: each play's place in its session (ln 1+minutes since the session began, ln 1+plays before; a gap over
    // 50 min (the app's SESSION_GAP) starts one) as two more inputs to every note's difficulty, so warm-up and fatigue can act per note type
    let session = args.iter().any(|a| a == "--session");
    let sess_in: Vec<[f32; 2]> = {
        let mut ord: Vec<usize> = (0..plays.len()).collect();
        ord.sort_by(by_player_time);
        let mut out = vec![[0f32; 2]; plays.len()];
        let (mut st, mut k) = (0f64, 0usize);
        for (i, &p) in ord.iter().enumerate() {
            if i == 0 || player[p] != player[ord[i - 1]] || when[p] - when[ord[i - 1]] > 3000. / 86400. { st = when[p]; k = 0; }
            out[p] = [(((when[p] - st) * 1440.).ln_1p()) as f32, ((k as f64).ln_1p()) as f32];
            k += 1;
        }
        out
    };
    // --context: hour of day (sin, cos of the UTC hour), ln 1+days since the previous play, and the previous play in the session
    // (its accuracy lost, its ln rating; 0, 0 and a flag when none): all known before the play starts
    let context = args.iter().any(|a| a == "--context");
    let ctx_in: Vec<[f32; 5]> = {
        let mut ord: Vec<usize> = (0..plays.len()).collect();
        ord.sort_by(by_player_time);
        let lost = |p: usize| { let st = plays[p]["stats"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.)).collect::<Vec<f64>>()).unwrap_or_default();
            let n: f64 = st.iter().sum(); if n > 0. { 1. - st.iter().zip(WEIGHT).map(|(c, w)| c * w as f64).sum::<f64>() / (305. * n) } else { 0. } };
        let mut out = vec![[0f32; 5]; plays.len()];
        for (i, &p) in ord.iter().enumerate() {
            let h = (when[p].fract()) * std::f64::consts::TAU;
            let prev = (i > 0 && player[ord[i - 1]] == player[p]).then(|| ord[i - 1]);
            let gap = prev.map_or(30., |q| when[p] - when[q]);
            let in_sess = prev.filter(|_| gap <= 3000. / 86400.);
            out[p] = [h.sin() as f32, h.cos() as f32, (gap.ln_1p() as f32) - 0.5, in_sess.map_or(0., |q| (lost(q) * 10. - 0.5) as f32), in_sess.map_or(0., |q| rating[q] - 1.6)];
        }
        out
    };
    // --player: one input per merged player after the first (rowsmerge's `player`), so the shared chart model keeps each
    // player's own difficulty apart; the app always asks as player 0 (all zeros)
    let others = if args.iter().any(|a| a == "--player") { player.iter().copied().max().unwrap_or(0) } else { 0 };
    let k = if baseline { 4 } else { inputs.len() } + 1 + if session { 2 } else { 0 } + if context { 5 } else { 0 } + others; // + is_tail
    // play ranges (rows are written play by play)
    let mut start = vec![0usize; plays.len() + 1];
    for i in 0..n { start[row(i)[ci] as usize + 1] = i + 1; }
    for p in 1..start.len() { start[p] = start[p].max(start[p - 1]); }
    eprintln!("{n} notes, {} plays, {k} inputs", plays.len());
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    // samples of a play: every head, and every LN release (lazer judges both)
    let samples = |p: usize| (start[p]..start[p + 1]).flat_map(move |i| std::iter::once(i as u64).chain(ct.filter(|&c| row(i)[c] < 5.5).map(|_| i as u64 | TAIL)));
    // --all-songs: the shipped model trains on every play (no held-out half)
    let all_songs = args.iter().any(|a| a == "--all-songs");
    let train: Vec<u64> = (0..plays.len()).filter(|&p| all_songs || half[p] == 0).flat_map(&samples).collect();
    let train_n = train.len();
    let pick = |r: u64| train[(r % train_n as u64) as usize];
    eprintln!("{train_n} train judgements ({} releases)", train.iter().filter(|s| **s & TAIL != 0).count());
    // standardisation from a sample
    let (mut mu, mut sd) = (vec![0f64; w], vec![0f64; w]);
    for _ in 0..200_000 { let r = row((pick(rnd()) & !TAIL) as usize); for c in 0..w { mu[c] += r[c] as f64; sd[c] += (r[c] as f64).powi(2); } }
    for c in 0..w { mu[c] /= 2e5; sd[c] = (sd[c] / 2e5 - mu[c] * mu[c]).max(1e-9).sqrt(); }
    let feats = |s: u64, out: &mut Vec<f32>| {
        let r = row((s & !TAIL) as usize);
        if baseline {
            // chart-level only: old calc rating at the played rate, keymode, rate, OD through the same MLP
            out.push(rating[r[ci] as usize]);
            for c in ["keys", "rate", "od"] { let c = cols.iter().position(|x| *x == c).unwrap(); out.push(((r[c] as f64 - mu[c]) / sd[c]) as f32); }
        } else { inputs.iter().for_each(|&c| out.push(((r[c] as f64 - mu[c]) / sd[c]) as f32)); }
        out.push(if s & TAIL != 0 { 1. } else { 0. });
        if session { let x = sess_in[r[ci] as usize]; out.push(x[0] / 3. - 1.); out.push(x[1] / 2. - 1.); }
        if context { out.extend_from_slice(&ctx_in[r[ci] as usize]); }
        (1..=others).for_each(|o| out.push(if player[r[ci] as usize] == o { 1. } else { 0. }));
    };
    let loss_of = |s: u64| { let r = row((s & !TAIL) as usize); 1. - WEIGHT[(if s & TAIL != 0 { r[ct.unwrap()] } else { r[cj] } as usize).min(5)] / 305. };
    let d = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0)).autodiff();
    let hid: usize = args.iter().position(|a| a == "--hid").map_or(128, |i| args[i + 1].parse().unwrap());
    let mut net = Net::new(k, hid, plays.len(), &d);
    let walk: f64 = args.iter().position(|a| a == "--walk").map_or(0., |i| args[i + 1].parse().unwrap());
    // consecutive same-player, same-keymode train plays: (play, previous, 1 / (0.05 + days between))
    let pairs: (Vec<i32>, Vec<i32>, Vec<f32>) = {
        let mut ord: Vec<usize> = (0..plays.len()).filter(|&p| half[p] == 0).collect();
        ord.sort_by(by_player_time);
        let mut last: std::collections::HashMap<(usize, u64), usize> = Default::default();
        let (mut x, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for p in ord {
            let kk = (player[p], plays[p]["keys"].as_u64().unwrap_or(0));
            if let Some(&q) = last.get(&kk) { x.push(p as i32); y.push(q as i32); w.push((1. / (0.05 + (when[p] - when[q]).max(0.))) as f32); }
            last.insert(kk, p);
        }
        (x, y, w)
    };
    let mut opt = AdamConfig::new().init();
    let bs = 65536;
    let steps = epochs * train_n / bs;
    for step in 0..steps {
        let lr = 2e-3 * (0.5 * (1. + (std::f64::consts::PI * step as f64 / steps as f64).cos())) + 1e-5;
        let idx: Vec<u64> = (0..bs).map(|_| pick(rnd())).collect();
        let mut x = Vec::with_capacity(bs * k);
        idx.iter().for_each(|&i| feats(i, &mut x));
        let y: Vec<f32> = idx.iter().map(|&i| loss_of(i)).collect();
        let pid: Vec<i32> = idx.iter().map(|&s| row((s & !TAIL) as usize)[ci] as i32).collect();
        let x = Tensor::<2>::from_data(TensorData::new(x, [bs, k]), &d);
        let y = Tensor::<2>::from_data(TensorData::new(y, [bs, 1]), &d);
        let th = net.theta.val().select(0, Tensor::<1, Int>::from_data(TensorData::new(pid, [bs]), &d));
        let z = net.difficulty(x) - th;
        // soft-target cross-entropy: −[y log σ(z) + (1−y) log σ(−z)]
        let ce = -(y.clone() * log_sigmoid(z.clone()) + (y.neg() + 1.) * log_sigmoid(z.neg())).mean();
        let mut reg = net.theta.val().powf_scalar(2.).mean() * 1e-3;
        if walk > 0. {
            // random walk: each play's θ near the previous same-keymode play's, by the days between
            let th = net.theta.val();
            let a = th.clone().select(0, Tensor::<1, Int>::from_data(TensorData::new(pairs.0.clone(), [pairs.0.len()]), &d));
            let b = th.select(0, Tensor::<1, Int>::from_data(TensorData::new(pairs.1.clone(), [pairs.1.len()]), &d));
            let w = Tensor::<2>::from_data(TensorData::new(pairs.2.clone(), [pairs.2.len(), 1]), &d);
            reg = reg + ((a - b).powf_scalar(2.) * w).mean() * walk;
        }
        let loss = ce.clone() + reg;
        if step % 200 == 0 { eprintln!("step {step}/{steps} ce {:.5}", ce.into_scalar::<f32>()); }
        let g = GradientsParams::from_grads(loss.backward(), &net);
        net = opt.step(lr, net, g);
    }
    // judge: per play, mean predicted loss with θ from train plays within ±DAYS (excluding the play itself)
    let net = net.valid();
    // --save FILE: the difficulty net for ms-calc::notenet (inputs by column name, standardisation, layers as [in][out] rows)
    if let Some(i) = args.iter().position(|a| a == "--save") {
        let lin = |l: &Linear| { let w: Vec<f32> = l.weight.val().into_data().try_to_vec::<f32>().unwrap(); let b: Vec<f32> = l.bias.as_ref().unwrap().val().into_data().try_to_vec::<f32>().unwrap();
            let [i, o] = l.weight.val().dims(); serde_json::json!({"in": i, "out": o, "w": w, "b": b}) };
        let names: Vec<&str> = inputs.iter().map(|&c| cols[c]).collect();
        let j = serde_json::json!({"inputs": names, "mu": inputs.iter().map(|&c| mu[c]).collect::<Vec<f64>>(), "sd": inputs.iter().map(|&c| sd[c]).collect::<Vec<f64>>(),
            "tail": true, "session": session, "others": others, "l1": lin(&net.l1), "l2": lin(&net.l2), "l3": lin(&net.l3)});
        std::fs::write(&args[i + 1], serde_json::to_string(&j).unwrap()).unwrap();
    }
    let theta: Vec<f32> = net.theta.val().into_data().try_to_vec::<f32>().unwrap();
    let dv = Device::wgpu(burn::tensor::DeviceKind::DiscreteGpu(0));
    let mut by_time: Vec<usize> = (0..plays.len()).filter(|&p| half[p] == 0 && start[p + 1] > start[p]).collect();
    by_time.sort_by(by_player_time);
    let keys_of = |p: usize| plays[p]["keys"].as_u64().unwrap_or(0);
    // θ forecasts (round 4): (name, same keymode, decay days (0 = flat 7-day mean), same-session boost)
    let variants: [(&str, bool, f64, f64); 6] = [("flat7-all", false, 0., 1.), ("flat7-keys", true, 0., 1.), ("ewma1-keys", true, 1., 1.),
        ("ewma3-keys", true, 3., 1.), ("ewma3-keys-sess4", true, 3., 4.), ("ewma1-keys-sess4", true, 1., 4.)];
    let forecast = |p: usize, v: &(&str, bool, f64, f64)| -> Option<f32> {
        let span = if v.2 == 0. { DAYS } else { 5. * v.2 };
        let pick = |same: bool| -> Option<f32> {
            let (mut sw, mut s, mut n) = (0f64, 0f64, 0usize);
            for &q in by_time.iter().filter(|&&q| player[q] == player[p] && when[q] < when[p] && when[p] - when[q] <= span && (!same || keys_of(q) == keys_of(p))) {
                let dt = when[p] - when[q];
                let w = if v.2 == 0. { 1. } else { (-dt / v.2).exp() } * if dt < 0.125 { v.3 } else { 1. };
                sw += w; s += w * theta[q] as f64; n += 1;
            }
            (n >= 3).then(|| (s / sw) as f32)
        };
        if v.1 { pick(true).or_else(|| pick(false)) } else { pick(false) }
    };
    let mut outs: Vec<Vec<(u8, usize, f64, f64)>> = vec![Vec::new(); variants.len()];
    // --dump DIR: every judged play's note difficulties and actual loss (dd.bin) and every play's fitted θ (theta.f32), so θ
    // forecasts can be tried offline
    let dump_dir = args.iter().position(|a| a == "--dump").map(|i| args[i + 1].clone());
    let mut dump = dump_dir.as_ref().map(|d| { std::fs::create_dir_all(d).unwrap();
        let t: Vec<u8> = theta.iter().flat_map(|x| x.to_le_bytes()).collect(); std::fs::write(format!("{d}/theta.f32"), t).unwrap();
        std::io::BufWriter::new(std::fs::File::create(format!("{d}/dd.bin")).unwrap()) });
    for p in 0..plays.len() {
        if start[p + 1] - start[p] < 50 { continue; }
        // only plays every variant can forecast: the same plays judged
        let ths: Vec<Option<f32>> = variants.iter().map(|v| forecast(p, v)).collect();
        if ths.iter().any(|t| t.is_none()) && dump.is_none() { continue; }
        let ss: Vec<u64> = samples(p).collect();
        let mut x = Vec::with_capacity(ss.len() * k);
        ss.iter().for_each(|&s| feats(s, &mut x));
        let dd: Vec<f32> = net.difficulty(Tensor::<2>::from_data(TensorData::new(x, [ss.len(), k]), &dv)).into_data().try_to_vec::<f32>().unwrap();
        let act = ss.iter().map(|&s| loss_of(s) as f64).sum::<f64>() / ss.len() as f64;
        if let Some(w) = dump.as_mut() {
            use std::io::Write;
            w.write_all(&(p as u32).to_le_bytes()).unwrap(); w.write_all(&(dd.len() as u32).to_le_bytes()).unwrap(); w.write_all(&act.to_le_bytes()).unwrap();
            dd.iter().for_each(|z| w.write_all(&z.to_le_bytes()).unwrap());
        }
        if ths.iter().any(|t| t.is_none()) { continue; }
        for (vi, th) in ths.iter().enumerate() {
            let th = th.unwrap();
            let pred = dd.iter().map(|z| 1. / (1. + (-(z - th) as f64).exp())).sum::<f64>() / ss.len() as f64;
            outs[vi].push((half[p], plays[p]["keys"].as_u64().unwrap_or(0) as usize, 100. * (1. - pred), 100. * (1. - act)));
        }
    }
    for (vi, out) in outs.iter().enumerate() {
    println!("-- theta {}", variants[vi].0);
    for (h, name) in [(0u8, "train songs"), (1, "held-out songs")] {
        for keys in [0usize, 4, 7, 10] {
            let e: Vec<f64> = out.iter().filter(|o| o.0 == h && (keys == 0 || o.1 == keys)).map(|o| o.2 - o.3).collect();
            if e.is_empty() { continue; }
            let mae = e.iter().map(|x| x.abs()).sum::<f64>() / e.len() as f64;
            let bias = e.iter().sum::<f64>() / e.len() as f64;
            println!("{} {name:15} {:>3} plays {:5}  MAE {mae:.3} pp  bias {bias:+.3}  >3pp {:.1} %", if baseline { "baseline" } else { "per-note" }, if keys == 0 { "all".into() } else { format!("{keys}K") }, e.len(), 100. * e.iter().filter(|x| x.abs() > 3.).count() as f64 / e.len() as f64);
        }
    }
    }
}

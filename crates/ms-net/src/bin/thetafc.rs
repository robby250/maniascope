//! S7 offline θ forecasts: reads notefit --dump (theta.f32, dd.bin) and rows.json; forecasts each judged play's ability from
//! earlier train-half plays only, scores held-out-song plays (MAE pp of accuracy over heads and releases).
use std::collections::HashMap;

fn days(s: &str) -> f64 {
    let p = |a: usize, b: usize| s.get(a..b).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
    let (y, m, d) = (p(0, 4), p(5, 7), p(8, 10));
    let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + d - 1;
    (era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy) as f64 + p(11, 13) as f64 / 24. + p(14, 16) as f64 / 1440. + p(17, 19) as f64 / 86400.
}

fn solve(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        a[i][i] += 1e-6;
        let piv = (i..n).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs())).unwrap();
        a.swap(i, piv);
        b.swap(i, piv);
        for r in 0..n {
            if r != i {
                let f = a[r][i] / a[i][i];
                for c in i..n { a[r][c] -= f * a[i][c]; }
                b[r] -= f * b[i];
            }
        }
    }
    (0..n).map(|i| b[i] / a[i][i]).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (rows, dump) = (&args[1], &args[2]);
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{rows}/rows.json")).unwrap()).unwrap();
    let plays = meta["plays"].as_array().unwrap();
    let np = plays.len();
    let half: Vec<u8> = plays.iter().map(|p| p["half"].as_u64().unwrap() as u8).collect();
    let when: Vec<f64> = plays.iter().map(|p| days(p["played"].as_str().unwrap_or(""))).collect();
    // rowsmerge's player of each play: sessions and every history a forecast reads stay within one
    let player: Vec<usize> = plays.iter().map(|p| p["player"].as_u64().unwrap_or(0) as usize).collect();
    let keys: Vec<u64> = plays.iter().map(|p| p["keys"].as_u64().unwrap_or(0)).collect();
    let lr: Vec<f64> = plays.iter().map(|p| (p["rating"].as_f64().unwrap_or(1.).max(0.1) * p["rate"].as_f64().unwrap_or(1.).max(0.5).powf(0.)).ln()).collect();
    let sha: Vec<String> = plays.iter().map(|p| p["sha"].as_str().unwrap_or("").to_string()).collect();
    let theta: Vec<f32> = std::fs::read(format!("{dump}/theta.f32")).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let raw = std::fs::read(format!("{dump}/dd.bin")).unwrap();
    let mut judged: HashMap<usize, (f64, Vec<f32>)> = HashMap::new();
    let mut at = 0;
    while at < raw.len() {
        let p = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(raw[at + 4..at + 8].try_into().unwrap()) as usize;
        let act = f64::from_le_bytes(raw[at + 8..at + 16].try_into().unwrap());
        let dd = raw[at + 16..at + 16 + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        judged.insert(p, (act, dd));
        at += 16 + 4 * n;
    }
    let all_hist = args.iter().any(|a| a == "--all-history");
    let mut theta = theta;
    if all_hist {
        // a held-out play's θ once it has happened: the θ whose mean predicted loss equals its actual loss, the chart model fixed
        for (p, (act, dd)) in &judged {
            if half[*p] == 0 { continue; }
            let m = |th: f64| dd.iter().map(|z| 1. / (1. + (-(*z as f64 - th)).exp())).sum::<f64>() / dd.len() as f64;
            let (mut lo, mut hi) = (-20f64, 20f64);
            for _ in 0..60 { let mid = 0.5 * (lo + hi); if m(mid) > *act { lo = mid } else { hi = mid } }
            theta[*p] = (0.5 * (lo + hi)) as f32;
        }
    }
    let mut order: Vec<usize> = (0..np).collect();
    order.sort_by(|a, b| player[*a].cmp(&player[*b]).then(when[*a].total_cmp(&when[*b])));
    // session: a gap over 50 min (SESSION_GAP) starts a new one (all plays, either half)
    let mut sess = vec![0usize; np];
    let mut sess_start = vec![0f64; np];
    let mut idx_in = vec![0usize; np];
    let (mut s, mut st, mut k) = (0usize, 0f64, 0usize);
    for (i, &p) in order.iter().enumerate() {
        if i == 0 || player[p] != player[order[i - 1]] || when[p] - when[order[i - 1]] > 3000. / 86400. { s += 1; st = when[p]; k = 0; }
        sess[p] = s; sess_start[p] = st; idx_in[p] = k; k += 1;
    }
    // history the forecasts read: train plays, or with --all-history every judged play (outcomes known once played)
    let train: Vec<usize> = order.iter().copied().filter(|&p| (all_hist || half[p] == 0) && judged.contains_key(&p)).collect();
    // --player N: fit and score one player of a rowsmerge set (default 0); the others' plays still trained the chart model
    let who = args.iter().position(|a| a == "--player").map_or(0, |i| args[i + 1].parse::<usize>().unwrap());
    let fit_on: Vec<usize> = order.iter().copied().filter(|&p| half[p] == 0 && judged.contains_key(&p) && player[p] == who).collect();
    // --clip K: each history play's θ enters the forecasts at most K from its keymode's EWMA just before it (both sides: one
    // disaster or career-best play moves the next forecast by at most K)
    let theta: Vec<f32> = match args.iter().position(|a| a == "--clip").map(|i| args[i + 1].parse::<f64>().unwrap()) {
        None => theta,
        Some(k) => {
            let mut out = theta.clone();
            for &q in &train {
                let (mut sw, mut sm, mut n) = (0f64, 0f64, 0usize);
                for &r in train.iter().filter(|&&r| player[r] == player[q] && when[r] < when[q] && when[q] - when[r] <= 15. && keys[r] == keys[q]) {
                    let dt = when[q] - when[r];
                    let w = (-dt / 3.).exp() * if dt < 0.125 { 4. } else { 1. };
                    sw += w; sm += w * theta[r] as f64; n += 1;
                }
                if n >= 3 { let e = sm / sw; out[q] = (e + (theta[q] as f64 - e).clamp(-k, k)) as f32; }
            }
            out
        }
    };
    // features of play p from earlier train plays (and earlier plays of the same chart, either half)
    let ewma_at = |p: usize, k: u64| -> Option<f64> {
        let (mut sw, mut sm, mut n) = (0f64, 0f64, 0usize);
        for &q in train.iter().filter(|&&q| player[q] == player[p] && when[q] < when[p] && when[p] - when[q] <= 15. && keys[q] == k) {
            let dt = when[p] - when[q];
            let w = (-dt / 3.).exp() * if dt < 0.125 { 4. } else { 1. };
            sw += w; sm += w * theta[q] as f64; n += 1;
        }
        (n >= 3).then(|| sm / sw)
    };
    let feats = |p: usize| -> Option<Vec<f64>> {
        let e = ewma_at(p, keys[p])?;
        let mut last_sess: Option<f64> = None;
        let mut last_any: Option<f64> = None;
        let mut form: Vec<f64> = Vec::new(); // this session's earlier train plays, any keymode: θ − that keymode's ewma now
        let mut retry: Option<f64> = None;
        for &q in train.iter().filter(|&&q| player[q] == player[p] && when[q] < when[p] && when[p] - when[q] <= 15.) {
            if keys[q] == keys[p] { last_any = Some(when[p] - when[q]); if sess[q] == sess[p] { last_sess = Some(theta[q] as f64); } }
            if sess[q] == sess[p] { if let Some(eq) = ewma_at(p, keys[q]) { form.push(theta[q] as f64 - eq); } }
            if sess[q] == sess[p] && sha[q] == sha[p] { retry = Some(theta[q] as f64 - e); }
        }
        let recent: Vec<f64> = form.iter().rev().take(3).copied().collect();
        // similar-difficulty plays (same keymode, 15 days): θ − e weighted by closeness in ln rating; and how far this chart sits
        // over the recent plays' usual rating
        let (mut sw2, mut sm2, mut sl, mut n2) = (0f64, 0f64, 0f64, 0f64);
        for &q in train.iter().filter(|&&q| player[q] == player[p] && when[q] < when[p] && when[p] - when[q] <= 15. && keys[q] == keys[p]) {
            let w = (-(when[p] - when[q]) / 3.).exp();
            let near = (-(lr[q] - lr[p]).abs() / 0.1).exp();
            sw2 += w * near; sm2 += w * near * (theta[q] as f64 - e);
            sl += w * lr[q]; n2 += w;
        }
        let similar = if sw2 > 0.5 { sm2 / sw2 } else { 0. };
        let above = lr[p] - sl / n2.max(1e-9);
        let seen = order.iter().filter(|&&q| player[q] == player[p] && when[q] < when[p] && sha[q] == sha[p]).count() as f64;
        Some(vec![e, last_sess.map_or(0., |l| l - e), last_sess.map_or(0., |_| 1.), ((when[p] - sess_start[p]) * 1440.).ln_1p(),
                  (idx_in[p] as f64).ln_1p(), last_any.unwrap_or(15.).ln_1p(), seen.ln_1p(),
                  if recent.is_empty() { 0. } else { recent.iter().sum::<f64>() / recent.len() as f64 }, retry.unwrap_or(0.), retry.map_or(0., |_| 1.), similar, above, above * above,
                  last_sess.map_or(0., |l| l - e) * ((when[p] - sess_start[p]) * 1440.).ln_1p(), e * e, (e - lr[p]), last_sess.map_or(0., |l| (l - e).abs())])
    };
    let names: [&str; 17] = ["ewma", "last_sess-ewma", "has_last", "ln(1+min in session)", "ln(1+plays before)", "ln(1+days since keymode)", "ln(1+times seen chart)", "session form (any keymode)", "retry: same chart this session", "has retry", "similar-rating θ", "rating over usual", "(rating over usual)²", "last×minutes", "ewma²", "ewma−ln rating", "|last−ewma|"];
    // fit θ − ewma on train plays (walk-forward features), each feature set
    let sets: [(&str, &[usize]); 12] = [("ship: last+retry", &[1, 2, 8, 9]), ("ship: last+retry+form", &[1, 2, 7, 8, 9]), ("ewma only", &[]), ("+last in session", &[1, 2]), ("+session time/count", &[1, 2, 3, 4]),
        ("+days since keymode", &[1, 2, 3, 4, 5]), ("+chart seen", &[1, 2, 3, 4, 5, 6]), ("+session form", &[1, 2, 3, 4, 5, 6, 7]), ("+retry", &[1, 2, 3, 4, 5, 6, 7, 8, 9]), ("+similar rating", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), ("+over usual", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), ("+interactions", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])];
    let tf: Vec<(usize, Vec<f64>)> = fit_on.iter().filter_map(|&p| feats(p).map(|f| (p, f))).collect();
    // --since YYYY-MM-DD: score only plays from then on (the app's walk-forward audit window)
    let since = args.iter().position(|a| a == "--since").map_or(0., |i| days(&args[i + 1]));
    let held: Vec<(usize, Vec<f64>)> = order.iter().copied().filter(|&p| half[p] == 1 && judged.contains_key(&p) && when[p] >= since && player[p] == who).filter_map(|p| feats(p).map(|f| (p, f))).collect();
    eprintln!("{} train plays fit, {} held-out plays judged", tf.len(), held.len());
    // oracle: train-song plays scored with their own fitted θ (what a perfect ability forecast leaves to the chart model)
    let mut oe: Vec<f64> = Vec::new();
    for (p, f) in &tf {
        let (act, dd) = &judged[p];
        let pred = |th: f64| dd.iter().map(|z| 1. / (1. + (-(*z as f64 - th)).exp())).sum::<f64>() / dd.len() as f64;
        oe.push((100. * (act - pred(theta[*p] as f64))).abs());
        let _ = f;
    }
    println!("oracle own θ (train songs) MAE {:.3}  >1pp {:.1} %", oe.iter().sum::<f64>() / oe.len() as f64, 100. * oe.iter().filter(|x| **x > 1.).count() as f64 / oe.len() as f64);
    let per_keys = args.iter().any(|a| a == "--per-keys");
    // --write OUT.tsv: every scored play per feature set (set, sha, played, keys, predicted accuracy, actual)
    let out_tsv = args.iter().position(|a| a == "--write").map(|i| std::cell::RefCell::new(std::io::BufWriter::new(std::fs::File::create(&args[i + 1]).unwrap())));
    for (name, cols) in sets {
        let groups: Vec<Option<u64>> = if per_keys { vec![Some(4), Some(7), Some(10), None] } else { vec![None] };
        let mut err: Vec<(u64, f64)> = Vec::new();
        let mut w = Vec::new();
        for g in &groups {
        let mine = |p: &usize| match g { Some(k) => keys[*p] == *k, None => !per_keys || ![4, 7, 10].contains(&keys[*p]) };
        let m = cols.len() + 1;
        let (mut a, mut b) = (vec![vec![0f64; m]; m], vec![0f64; m]);
        for (p, f) in tf.iter().filter(|(p, _)| mine(p) || g.is_none()) {
            let x: Vec<f64> = std::iter::once(1.).chain(cols.iter().map(|&c| f[c])).collect();
            let y = theta[*p] as f64 - f[0];
            for i in 0..m { b[i] += x[i] * y; for j in 0..m { a[i][j] += x[i] * x[j]; } }
        }
        w = solve(&mut a, &mut b);
        for (p, f) in held.iter().filter(|(p, _)| mine(p)) {
            let x: Vec<f64> = std::iter::once(1.).chain(cols.iter().map(|&c| f[c])).collect();
            let th = f[0] + x.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
            let (act, dd) = &judged[p];
            let pred = dd.iter().map(|z| 1. / (1. + (-(*z as f64 - th)).exp())).sum::<f64>() / dd.len() as f64;
            err.push((keys[*p], 100. * (act - pred)));
            if let Some(w) = out_tsv.as_ref() { use std::io::Write; let mut w = w.borrow_mut(); writeln!(w, "{name}\t{}\t{}\t{}\t{:.5}\t{:.5}", sha[*p], plays[*p]["played"].as_str().unwrap_or(""), keys[*p], 1. - pred, 1. - act).unwrap(); }
        }
        }
        // --coef OUT.json: the shipped set's coefficients, in pernote::Personal's "forecast" names
        if name == "ship: last+retry+form" && !per_keys {
            if let Some(i) = args.iter().position(|a| a == "--coef") {
                std::fs::write(&args[i + 1], serde_json::json!({"clip": args.iter().position(|a| a == "--clip").map(|i| args[i + 1].parse::<f64>().unwrap()), "const": w[0], "last": w[1], "has_last": w[2], "form": w[3], "retry": w[4], "has_retry": w[5]}).to_string()).unwrap();
            }
        }
        let line = |k: u64| { let e: Vec<f64> = err.iter().filter(|x| k == 0 || x.0 == k).map(|x| x.1).collect();
            format!("{:.3} ({})", e.iter().map(|x| x.abs()).sum::<f64>() / e.len().max(1) as f64, e.len()) };
        let coef: Vec<String> = cols.iter().zip(&w[1..]).map(|(&c, v)| format!("{} {v:+.3}", names[c])).collect();
        println!("{name:22} held-out MAE all {}  4K {}  7K {}  10K {}  | const {:+.3} {}", line(0), line(4), line(7), line(10), w[0], coef.join(", "));
    }
}

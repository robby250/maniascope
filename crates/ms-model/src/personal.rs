//! recommend.Personal's fit: the warm-up models (completed-score cold-to-warm fit, the constant-response control and
//! the rate-response model), then level curves, shape and recent-form ridges, session context, chart and rate
//! affinities and the retry effect.
use crate::fit::{ridge_alternation, solve, warmup_timeline};
use crate::predict::{interp, Consts, Feats, HashMap, Mode, Slopes};
use ms_calc::py::{round, round_digits};
use sha2::{Digest, Sha256};
use std::hash::Hash;

/// chart_blocks' `r.get('key', i)` for rows without a session.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Best {
    Key(String),
    None,
    Index(usize),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Block {
    Session(i64, i64),
    Best(Best),
}

/// One evidence row. `f` indexes the features of its chart-rate.
#[derive(Clone)]
pub struct Row {
    pub keys: i64,
    pub y: f64,
    pub base: f64,
    pub t: f64,
    pub ts: f64, // `r.get('ts') or 0`
    pub public: bool,
    pub local: bool, // src realm/live
    pub rd: bool,
    pub completed: bool,
    pub chart: String,
    pub rate: f64,
    pub rate3: f64, // round(rate, 3)
    pub best: Best,
    pub family: String, // f['nps']['family'] or the chart (fit_shape's groups)
    pub f: usize,
}

/// Group ids in first-appearance order, and the distinct keys in that order.
fn index_of<K: Hash + Eq + Clone>(keys: impl IntoIterator<Item = K>) -> (Vec<usize>, Vec<K>) {
    let mut at: HashMap<K, usize> = HashMap::default();
    let mut order = Vec::new();
    let ids = keys.into_iter().map(|k| {
        *at.entry(k.clone()).or_insert_with(|| {
            order.push(k);
            order.len() - 1
        })
    }).collect();
    (ids, order)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// np.var(v, ddof)
fn var(v: &[f64], ddof: usize) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - ddof) as f64
}

fn average(v: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut s, mut w) = (0., 0.);
    for (x, wi) in v {
        s += x * wi;
        w += wi;
    }
    s / w
}

/// Weighted column means and scales max(.05, weighted sd) of an n×p row-major matrix.
fn standardise(z: &[f64], p: usize, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let total: f64 = w.iter().sum();
    let mut m = vec![0.; p];
    for (row, wi) in z.chunks_exact(p).zip(w) {
        m.iter_mut().zip(row).for_each(|(a, x)| *a += wi * x);
    }
    m.iter_mut().for_each(|a| *a /= total);
    let mut s = vec![0.; p];
    for (row, wi) in z.chunks_exact(p).zip(w) {
        s.iter_mut().zip(row).zip(&m).for_each(|((a, x), mu)| *a += wi * (x - mu) * (x - mu));
    }
    (m, s.iter().map(|a| (a / total).sqrt().max(0.05)).collect())
}

const GRID: [f64; 4] = [0.02, 0.05, 0.2, 0.8];

struct Shape {
    beta: Vec<f64>,
    ridge: Option<f64>,
    families: usize,
    cv: Vec<f64>, // risk per GRID value; empty when there was too little to validate
}

/// personal_support.fit_shape: ridge with chart-family-blocked 4-fold CV and the one-SE choice; the last
/// coefficient (near-limit fatigue) is never negative.
fn fit_shape(z: &[f64], p: usize, target: &[f64], w: &[f64], groups: &[&str]) -> Shape {
    let n = target.len();
    let (ids, order) = index_of(groups.iter().copied());
    let fold_of: Vec<usize> = order.iter().map(|g| {
        let h = Sha256::digest(g.as_bytes());
        (u32::from_be_bytes([h[0], h[1], h[2], h[3]]) % 4) as usize
    }).collect();
    let fold: Vec<usize> = ids.iter().map(|&g| fold_of[g]).collect();
    // Zᵀ·W·Z and Zᵀ·W·y per fold (upper triangle), summed for the whole.
    let mut fa = vec![vec![0.; p * p]; 4];
    let mut fb = vec![vec![0.; p]; 4];
    for i in 0..n {
        let row = &z[i * p..(i + 1) * p];
        let (a, b) = (&mut fa[fold[i]], &mut fb[fold[i]]);
        for j in 0..p {
            let v = w[i] * row[j];
            b[j] += v * target[i];
            let line = &mut a[j * p..(j + 1) * p];
            for k in j..p {
                line[k] += v * row[k];
            }
        }
    }
    let mirror = |a: &mut Vec<f64>| {
        for j in 0..p {
            for k in 0..j {
                a[j * p + k] = a[k * p + j];
            }
        }
    };
    fa.iter_mut().for_each(mirror);
    let all_a: Vec<f64> = (0..p * p).map(|x| fa.iter().map(|a| a[x]).sum()).collect();
    let all_b: Vec<f64> = (0..p).map(|x| fb.iter().map(|b| b[x]).sum()).collect();
    let fit = |a: &[f64], b: &[f64], penalty: f64| -> Vec<f64> {
        let mut m = a.to_vec();
        (0..p).for_each(|j| m[j * p + j] += penalty);
        let mut beta = solve(&m, b);
        if beta[p - 1] < 0. {
            let q = p - 1;
            let sub: Vec<f64> = (0..q).flat_map(|j| m[j * p..j * p + q].to_vec()).collect();
            beta = solve(&sub, &b[..q]);
            beta.push(0.);
        }
        beta
    };
    let mut folds = fold_of;
    folds.sort_unstable();
    folds.dedup();
    if order.len() < 8 || folds.len() < 2 {
        // One heavily retried chart cannot validate a structural specialization.
        return Shape { beta: vec![0.; p], ridge: None, families: order.len(), cv: Vec::new() };
    }
    let mut errors = vec![[0f64; 4]; n];
    for f in 0..4 {
        if !fold.contains(&f) || fold.iter().all(|&x| x == f) {
            continue;
        }
        let a: Vec<f64> = all_a.iter().zip(&fa[f]).map(|(x, y)| x - y).collect();
        let b: Vec<f64> = all_b.iter().zip(&fb[f]).map(|(x, y)| x - y).collect();
        let mass: f64 = (0..n).filter(|&i| fold[i] != f).map(|i| w[i]).sum();
        for (j, lam) in GRID.iter().enumerate() {
            let beta = fit(&a, &b, lam * mass);
            for i in (0..n).filter(|&i| fold[i] == f) {
                let fitted: f64 = z[i * p..(i + 1) * p].iter().zip(&beta).map(|(x, c)| x * c).sum();
                errors[i][j] = (target[i] - fitted).powi(2);
            }
        }
    }
    let g = order.len();
    let mut mass = vec![0.; g];
    let mut grouped = vec![[0f64; 4]; g];
    for i in 0..n {
        mass[ids[i]] += w[i];
        for j in 0..4 {
            grouped[ids[i]][j] += w[i] * errors[i][j];
        }
    }
    grouped.iter_mut().zip(&mass).for_each(|(r, m)| r.iter_mut().for_each(|x| *x /= m));
    let risk: Vec<f64> = (0..4).map(|j| grouped.iter().map(|r| r[j]).sum::<f64>() / g as f64).collect();
    // The ridge with the least cross-validated risk. The one-SE rule (the largest within a standard error) shrank the shape
    // further: walk-forward audit .8397 -> .8384 pp, RMSE 1.278 -> 1.276 without it (2026-11-15).
    let chosen = (0..4).min_by(|&a, &b| risk[a].total_cmp(&risk[b])).unwrap();
    let total: f64 = w.iter().sum();
    Shape { beta: fit(&all_a, &all_b, GRID[chosen] * total), ridge: Some(GRID[chosen]), families: g, cv: risk }
}

struct ByKey {
    global_variance: f64,
    skill_variances: Vec<f64>,
    family_sessions: Vec<usize>,
    sessions: usize,
}

struct Context {
    sessions: usize,
    supported: usize,
    between: f64,
    within: f64,
    skill_variance: f64,
    by_key: Vec<(i64, ByKey)>,
    shrink: f64,
}

/// personal_support.session_context: leave-one-chart-out condition from the other charts of each historical
/// session → (adjustment per row, (session, keys) per row, validation).
fn session_context(c: &Consts, rows: &[Row], feats: &[Feats], u: &[f64]) -> (Vec<f64>, Vec<Option<(i64, i64)>>, Context) {
    let n = rows.len();
    let mut order: Vec<usize> = (0..n).filter(|&i| {
        let r = &rows[i];
        r.local && r.ts != 0. && r.completed && !r.rd
    }).collect();
    order.sort_by(|&a, &b| rows[a].ts.total_cmp(&rows[b].ts));
    let mut ids = vec![None; n];
    let (mut last, mut session) = (None::<f64>, -1i64);
    for &i in &order {
        let r = &rows[i];
        let seconds = (feats[r.f].length / 1000. / r.rate.max(0.01)).min(1800.).max(0.);
        if last.is_none_or(|l| r.ts - seconds - l > 3000.) {
            session += 1;
        }
        ids[i] = Some((session, r.keys));
        last = Some(r.ts);
    }
    // Per session (first-row order), per chart: (chart, rows, mean residual, mean time).
    let mut gat: HashMap<(i64, i64), usize> = HashMap::default();
    let mut groups: Vec<((i64, i64), Vec<(String, Vec<usize>)>, HashMap<String, usize>)> = Vec::new();
    for (i, g) in ids.iter().enumerate() {
        let Some(g) = *g else { continue };
        let gi = *gat.entry(g).or_insert_with(|| {
            groups.push((g, Vec::new(), HashMap::default()));
            groups.len() - 1
        });
        let (_, charts, at) = &mut groups[gi];
        let ci = *at.entry(rows[i].chart.clone()).or_insert_with(|| {
            charts.push((rows[i].chart.clone(), Vec::new()));
            charts.len() - 1
        });
        charts[ci].1.push(i);
    }
    struct B {
        ix: Vec<usize>,
        e: f64,
        t: f64,
    }
    let blocks: Vec<((i64, i64), Vec<B>)> = groups.into_iter().map(|(g, charts, _)| {
        (g, charts.into_iter().map(|(_, ix)| {
            let k = ix.len() as f64;
            let e = ix.iter().map(|&i| u[i]).sum::<f64>() / k;
            let t = ix.iter().map(|&i| rows[i].ts).sum::<f64>() / k;
            B { ix, e, t }
        }).collect())
    }).collect();
    let values = |v: &[B]| v.iter().map(|b| b.e).collect::<Vec<f64>>();
    let multiple: Vec<&Vec<B>> = blocks.iter().filter(|(_, v)| v.len() >= 3).map(|(_, v)| v).collect();
    let (shrink, within, between) = if multiple.len() >= 10 {
        let means: Vec<f64> = multiple.iter().map(|v| mean(&values(v))).collect();
        let within = mean(&multiple.iter().map(|v| var(&values(v), 1)).collect::<Vec<_>>());
        let between = (var(&means, 0) - mean(&multiple.iter().map(|v| within / v.len() as f64).collect::<Vec<_>>())).max(0.);
        (within / between.max(1e-6), within, between)
    } else {
        (f64::INFINITY, 0., 0.)
    };
    let mut adjustment = vec![0.; n];
    for (_, v) in &blocks {
        for (a, b) in v.iter().enumerate() {
            if v.len() < 2 {
                continue;
            }
            let (mut s, mut ws) = (0., 0.);
            for (o, other) in v.iter().enumerate() {
                if o != a {
                    let w = (-(b.t - other.t).abs() / 1800.).exp();
                    s += other.e * w;
                    ws += w;
                }
            }
            let offset = s / (shrink + ws);
            b.ix.iter().for_each(|&i| adjustment[i] = offset);
        }
    }
    let nf = c.families.len();
    // Chart means per dominant family of a session's charts, in first-seen order.
    let by_family = |v: &[B]| -> Vec<(usize, Vec<f64>)> {
        let mut out: Vec<(usize, Vec<f64>)> = Vec::new();
        for b in v {
            let fam = c.dominant(&feats[rows[b.ix[0]].f]);
            match out.iter_mut().find(|(f, _)| *f == fam) {
                Some((_, o)) => o.push(b.e),
                None => out.push((fam, vec![b.e])),
            }
        }
        out
    };
    let mut local = Vec::new();
    for v in &multiple {
        let pooled = mean(&values(v));
        for (_, obs) in by_family(v) {
            if obs.len() >= 2 {
                let noise = within * (1. / obs.len() as f64 - 1. / v.len() as f64).max(0.);
                local.push((mean(&obs) - pooled).powi(2) - noise);
            }
        }
    }
    let skill_variance = if local.is_empty() { 0. } else { mean(&local).max(0.) };
    let mut keys: Vec<i64> = rows.iter().map(|r| r.keys).collect();
    keys.sort_unstable();
    keys.dedup();
    let by_key = keys.into_iter().map(|k| {
        let sessions: Vec<&Vec<B>> = blocks.iter().filter(|(g, v)| g.1 == k && v.len() >= 3).map(|(_, v)| v).collect();
        let mut samples = vec![Vec::new(); nf];
        for v in &sessions {
            let all = values(v);
            let (pooled, noise_all) = (mean(&all), var(&all, 1));
            for (fam, obs) in by_family(v) {
                if obs.len() >= 2 {
                    let noise = noise_all * (1. / obs.len() as f64 - 1. / v.len() as f64).max(0.);
                    samples[fam].push((mean(&obs) - pooled).powi(2) - noise);
                }
            }
        }
        let global_variance = if sessions.is_empty() {
            between
        } else {
            let means: Vec<f64> = sessions.iter().map(|v| mean(&values(v))).collect();
            let error = mean(&sessions.iter().map(|v| var(&values(v), 1) / v.len() as f64).collect::<Vec<_>>());
            let own = (var(&means, 0) - error).max(0.);
            let s = sessions.len() as f64;
            (s * own + 10. * between) / (s + 10.)
        };
        // Unobserved families borrow map uncertainty, not near-zero variance from well-modelled rice.
        let prior = skill_variance.max(within);
        let skill_variances = samples.iter().map(|s| {
            let m = s.len() as f64;
            let estimate = if s.is_empty() { 0. } else { mean(s).max(0.) };
            (m * estimate + 5. * prior) / (m + 5.)
        }).collect();
        (k, ByKey { global_variance, skill_variances, family_sessions: samples.iter().map(|s| s.len()).collect(), sessions: sessions.len() })
    }).collect();
    let ctx = Context { sessions: blocks.len(), supported: multiple.len(), between, within, skill_variance, by_key, shrink };
    (adjustment, ids, ctx)
}

/// personal_support.chart_blocks: one weighted observation per chart/rate/session → per chart id
/// [(rate, mean residual, weight, count)].
fn chart_blocks(rows: &[Row], chart: &[usize], charts: usize, u: &[f64], ids: &[Option<(i64, i64)>], now: f64) -> Vec<Vec<(f64, f64, f64, f64)>> {
    let (gid, order) = index_of(rows.iter().enumerate().map(|(i, r)| {
        let block = match ids[i] {
            Some((s, k)) => Block::Session(s, k),
            None => Block::Best(r.best.clone()),
        };
        (chart[i], r.rate3.to_bits(), block)
    }));
    let m = order.len();
    let (mut count, mut recent, mut public, mut rd, mut sum) = (vec![0usize; m], vec![f64::NEG_INFINITY; m], vec![0usize; m], vec![false; m], vec![0.; m]);
    for (i, &g) in gid.iter().enumerate() {
        count[g] += 1;
        recent[g] = recent[g].max(rows[i].t);
        public[g] += rows[i].public as usize;
        rd[g] |= rows[i].rd;
        sum[g] += u[i];
    }
    let mut out = vec![Vec::new(); charts];
    for (g, (c, rate, _)) in order.iter().enumerate() {
        let weight = 0.5f64.powf((now - recent[g]).max(0.) / 12.)
            * if public[g] == count[g] { 0.5 } else { 1. } * if rd[g] { 0.35 } else { 1. };
        out[*c].push((f64::from_bits(*rate), sum[g] / count[g] as f64, weight, count[g] as f64));
    }
    out
}

/// Independent blocks for variance estimation, distinct from recency precision: three old sessions still
/// estimate a between-chart variance from three observations (their age weakens only the prediction).
fn effective_count(v: &[(f64, f64, f64, f64)]) -> f64 {
    let s: f64 = v.iter().map(|x| x.2).sum();
    s * s / v.iter().map(|x| x.2 * x.2).sum::<f64>().max(1e-12)
}

/// Bounded familiarity effect: the 2nd attempt of a normal (non-RD) chart within a session vs the 1st, on the
/// residual `u` before shape and session context.
fn retry(rows: &[Row], u: &[f64], session_gap: f64) -> f64 {
    let mut seq: Vec<usize> = (0..rows.len()).collect();
    seq.sort_by(|&a, &b| rows[a].ts.total_cmp(&rows[b].ts));
    let (mut last, mut diffs): (HashMap<&str, usize>, Vec<f64>) = (HashMap::default(), Vec::new());
    for &i in &seq {
        let r = &rows[i];
        if r.public || r.ts == 0. || r.rd {
            continue;
        }
        if let Some(&p) = last.get(r.chart.as_str()) {
            if 0. < r.ts - rows[p].ts && r.ts - rows[p].ts < session_gap {
                diffs.push(u[p] - u[i]);
            }
        }
        last.insert(r.chart.as_str(), i);
    }
    if diffs.len() < 20 { 0. } else { (mean(&diffs) * diffs.len() as f64 / (diffs.len() as f64 + 50.)).clamp(0., 0.15) }
}

/// The chart's SV rating: its SV share × overall.
pub fn sv_rating(f: &Feats, c: &Consts) -> f64 {
    c.names.iter().position(|n| n == "sv").map_or(0., |j| f.sk[j] * f.overall)
}

/// SV reading's population start (plan 2026-11-20 S1: public best scores against play counts, passes of the top 10,000 players,
/// a lower bound on the loss): (p, L, s), a first read losing L·SV^p ln that falls as 1/(1 + s·reads). `calib sv_reading pop`.
pub fn read_pop(keys: i64) -> (f64, f64, f64) {
    match keys {
        4 => (1.25, 0.0230, 0.10),
        7 => (2.0, 0.0092, 0.02),
        _ => (2.0, 0.0056, 0.10),
    }
}

/// How a player reads SV (plan 2026-11-20 C1). The deficit of a play = (a + loss·L·SV^p) / (1 + learn·s·reads), with (p, L, s)
/// the keymode's population start and reads = Σ over the chart's earlier local plays of e^(−forget·years since). The three
/// personal values start at the population (1, 1, 0) and move by the player's own SV plays (`fit_reading`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub a: f64,      // ln on any unread chart, SV or not
    pub loss: f64,   // × the population's first-read loss
    pub learn: f64,  // × the population's learning speed
    pub forget: f64, // per year away, how much of an earlier read is lost (0: never)
}

impl Default for Reading {
    fn default() -> Self {
        Reading { a: 0., loss: 1., learn: 1., forget: 0. }
    }
}

impl Reading {
    /// An earlier read's weight `months` later.
    pub fn kept(&self, months: f64) -> f64 {
        (-self.forget * months.max(0.) / 12.).exp()
    }

    /// The deficit (ln) of a play of a chart with SV rating `sv` after `reads` (`kept`-weighted earlier plays).
    pub fn deficit(&self, keys: i64, sv: f64, reads: f64) -> f64 {
        let (p, l, s) = read_pop(keys);
        (self.a + self.loss * l * sv.max(0.).powf(p)) / (1. + self.learn * s * reads)
    }
}

/// A play as `fit_reading` reads it: its residual `y` (ln, + = more misses), keymode, SV rating, the months since each earlier
/// read of its chart, and whether no earlier read is known (`chart` groups a chart's plays).
pub struct ReadObs {
    pub y: f64,
    pub keys: i64,
    pub sv: f64,
    pub before: Vec<f64>,
    pub chart: usize,
}

/// Ridge strengths of `fit_reading`, each in plays' worth of evidence, and the evidence a first known play needs to count as
/// already read (likelihood ratio). `calib sv_reading kappa` searches them (chart-grouped cross-validation on own plays).
#[derive(Clone, Copy, Debug)]
pub struct ReadPriors {
    pub a: f64,
    pub loss: f64,
    pub learn: f64,
    pub forget: f64,
    pub read_before: f64,
}

/// `calib sv_reading own` 2026-11-20: CV SSE flat for ridge 0.3–3 (109.1–109.8; 10: 113.6, none: 148.7), read-before 20–1000.
pub const READ_PRIORS: ReadPriors = ReadPriors { a: 0.3, loss: 0.3, learn: 0.3, forget: 0.3, read_before: 1000. };

const LEARN_GRID: [f64; 17] = [0.25, 0.5, 1., 2., 3.5, 5., 7.5, 10., 15., 25., 35., 50., 75., 100., 150., 250., 400.];
const FORGET_GRID: [f64; 6] = [0., 0.05, 0.1, 0.2, 0.5, 1.];

/// The player's Reading and the charts read before their first known play. A chart whose first known play fits "already read"
/// (deficit 0) better than "first read" by `read_before` (likelihood ratio, at the residuals' spread) counts as read: no deficit.
/// Then (learn, forget) on a grid and (a, loss) by ridge least squares toward the population, the whole penalised error least.
pub fn fit_reading(obs: &[ReadObs], pr: &ReadPriors) -> (Reading, Vec<usize>) {
    if obs.is_empty() {
        return (Reading::default(), Vec::new());
    }
    let reads = |r: &Reading, o: &ReadObs| o.before.iter().map(|&m| r.kept(m)).sum::<f64>();
    let first = fit_unread(obs, &[], pr, obs.iter().map(|o| o.y * o.y).sum::<f64>() / obs.len() as f64);
    // the residuals' spread under that fit: what a first play's distance from each forecast is measured in
    let var = (obs.iter().map(|o| (o.y - first.deficit(o.keys, o.sv, reads(&first, o))).powi(2)).sum::<f64>() / obs.len() as f64).max(1e-4);
    let mut known: Vec<usize> = obs.iter().filter(|o| o.before.is_empty()).filter(|o| {
        let d = first.deficit(o.keys, o.sv, 0.);
        ((o.y - d).powi(2) - o.y * o.y) / (2. * var) > pr.read_before.ln()
    }).map(|o| o.chart).collect();
    known.sort_unstable();
    known.dedup();
    (fit_unread(obs, &known, pr, var), known)
}

/// `fit_reading` on the plays of charts not in `known`, penalties scaled by the residual variance `var`.
fn fit_unread(obs: &[ReadObs], known: &[usize], pr: &ReadPriors, var: f64) -> Reading {
    let pop = Reading::default();
    let unread: Vec<&ReadObs> = obs.iter().filter(|o| known.binary_search(&o.chart).is_err()).collect();
    let mut best = (f64::INFINITY, pop);
    for &learn in &LEARN_GRID {
        for &forget in &FORGET_GRID {
            let r = Reading { learn, forget, ..pop };
            // y ≈ (a + loss·x)·w: two unknowns, ridge to (0, 1)
            let (mut saa, mut sal, mut sll, mut say, mut sly, mut syy) = (0., 0., 0., 0., 0., 0.);
            for o in &unread {
                let reads: f64 = o.before.iter().map(|&m| r.kept(m)).sum();
                let (p, l, s) = read_pop(o.keys);
                let w = 1. / (1. + learn * s * reads);
                let (ua, ul) = (w, w * l * o.sv.max(0.).powf(p));
                saa += ua * ua;
                sal += ua * ul;
                sll += ul * ul;
                say += ua * o.y;
                sly += ul * o.y;
                syy += o.y * o.y;
            }
            let m = unread.len().max(1) as f64;
            let (ka, kl) = (pr.a * saa / m, pr.loss * sll / m);
            // (saa + ka)·a + sal·loss = say; sal·a + (sll + kl)·loss = sly + kl·1
            let (p11, p12, p22, q1, q2) = (saa + ka, sal, sll + kl, say, sly + kl);
            let det = p11 * p22 - p12 * p12;
            if det <= 0. {
                continue;
            }
            let a = ((q1 * p22 - p12 * q2) / det).clamp(0., 0.3);
            // both sources bound the loss from below (public passes after failed tries, an above-average reader who knew most
            // charts; plan 2026-11-20): never under the population's
            let loss = ((q2 - p12 * a) / p22).max(1.); // given the clamped a
            let sse = syy - 2. * (a * say + loss * sly) + a * a * saa + 2. * a * loss * sal + loss * loss * sll;
            let pen = ka * a * a + kl * (loss - 1.).powi(2) + var * (pr.learn * learn.ln().powi(2) + pr.forget * forget * forget);
            if sse + pen < best.0 {
                best = (sse + pen, Reading { a, loss, learn, forget });
            }
        }
    }
    best.1
}

/// A chart's local plays at the fit: their `kept` weight (∞: read before its first known play), count and the last one (months).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Read {
    pub weight: f64,
    pub count: usize,
    pub last: f64,
}

/// The player's Reading from their own plays, the deficit taken out of the residual `u` (so an unread play doesn't pull the level
/// down), and each chart's reads at `now`.
fn reading_of(rows: &[Row], feats: &[Feats], c: &Consts, u: &mut [f64], reads: &Reads, now: f64) -> (Reading, HashMap<String, Read>) {
    let (mut charts, mut at): (Vec<&str>, HashMap<&str, usize>) = (Vec::new(), HashMap::default());
    let own: Vec<usize> = (0..rows.len()).filter(|&i| !rows[i].public && rows[i].ts > 0.).collect();
    let obs: Vec<ReadObs> = own.iter().map(|&i| {
        let r = &rows[i];
        let chart = *at.entry(r.chart.as_str()).or_insert_with(|| { charts.push(r.chart.as_str()); charts.len() - 1 });
        let before = reads.get(&r.chart).map_or_else(Vec::new, |v| v.iter().filter(|x| x.1 < r.ts).map(|x| r.t - x.0).collect());
        ReadObs { y: u[i], keys: r.keys, sv: feats.get(r.f).map_or(0., |f| sv_rating(f, c)), before, chart }
    }).collect();
    let (reading, known) = if obs.len() < 50 { (Reading::default(), Vec::new()) } else { fit_reading(&obs, &READ_PRIORS) };
    let known: std::collections::HashSet<&str> = known.iter().map(|&k| charts[k]).collect();
    for (o, &i) in obs.iter().zip(&own) {
        if !known.contains(charts[o.chart]) {
            u[i] -= reading.deficit(o.keys, o.sv, o.before.iter().map(|&m| reading.kept(m)).sum());
        }
    }
    let plays = reads.iter().map(|(chart, v)| {
        let weight = if known.contains(chart.as_str()) { f64::INFINITY } else { v.iter().map(|x| reading.kept(now - x.0)).sum() };
        (chart.clone(), Read { weight, count: v.len(), last: v.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max) })
    }).collect();
    (reading, plays)
}

/// Personal._rate_anchors
fn rate_anchors(stats: &[(f64, f64, f64)], bandwidth: f64, shrinkage: f64) -> Vec<(f64, f64, f64)> {
    let mut anchors: Vec<(f64, f64, f64)> = stats.iter().map(|&(rate, total, count)| {
        let (mut prior_sum, mut prior_n) = (0., 0.);
        for &(other, value, n) in stats {
            if other != rate {
                let transfer = (-0.5 * ((other / rate).ln() / bandwidth).powi(2)).exp();
                prior_sum += transfer * value;
                prior_n += transfer * n;
            }
        }
        let prior = prior_sum / (prior_n + shrinkage).max(1e-9);
        (rate, (total + shrinkage * prior) / (count + shrinkage).max(1e-9), count + shrinkage.min(prior_n))
    }).collect();
    anchors.sort_by(|a, b| a.0.total_cmp(&b.0));
    anchors
}

const BANDS: [f64; 8] = [0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 1., f64::INFINITY];

/// Per (keys or 0): (bandwidth, folds, Some((train mse, full transfer mse)) or None = too few, pooled flag).
type Transfer = Vec<(i64, f64, usize, Option<(f64, f64)>, bool)>;

/// Personal._rate_transfer: extrapolation bandwidth from leave-one-chart-rate-out history per keymode.
fn rate_transfer(rows: &[Row], u: &[f64], by: &[Vec<usize>], rates_of: &[Vec<(f64, Vec<usize>)>], shrinkage: f64) -> Transfer {
    let mut folds: Vec<(i64, Vec<[f64; 4]>)> = Vec::new();
    for (indices, rates) in by.iter().zip(rates_of) {
        if rates.len() < 2 {
            continue;
        }
        let total: f64 = indices.iter().map(|&i| u[i]).sum();
        let count = indices.len() as f64;
        for (rate, ix) in rates {
            let others = rates.iter().filter(|(r, _)| r != rate).map(|(r, _)| *r);
            let lo = others.clone().fold(f64::INFINITY, f64::min);
            let hi = others.fold(f64::NEG_INFINITY, f64::max);
            let mut distance = 0f64.max((lo / rate).ln()).max((rate / hi).ln());
            if distance < 0.02 {
                continue;
            }
            let own: f64 = ix.iter().map(|&i| u[i]).sum();
            let estimate = (total - own) / (count - ix.len() as f64 + shrinkage);
            if estimate > 0. && *rate > hi {
                distance = 0.;
            }
            let k = rows[ix[0]].keys;
            let example = [distance, estimate, own / ix.len() as f64, ix.len().min(3) as f64 / rates.len() as f64];
            match folds.iter_mut().find(|(key, _)| *key == k) {
                Some((_, v)) => v.push(example),
                None => folds.push((k, vec![example])),
            }
        }
    }
    let select = |examples: &[[f64; 4]]| -> (f64, (f64, f64)) {
        let error: Vec<f64> = BANDS.iter().map(|b| {
            average(examples.iter().map(|x| ((x[1] * (-0.5 * (x[0] / b).powi(2)).exp() - x[2]).powi(2), x[3])))
        }).collect();
        let mut at = 0;
        for j in 1..BANDS.len() {
            if error[j] < error[at] {
                at = j;
            }
        }
        (BANDS[at], (error[at], error[BANDS.len() - 1]))
    };
    let pooled: Vec<[f64; 4]> = folds.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let (default, meta) = if pooled.len() >= 25 {
        let (b, m) = select(&pooled);
        (b, Some(m))
    } else {
        (0.27, None)
    };
    let mut out = vec![(0, default, pooled.len(), meta, false)];
    for (k, examples) in &folds {
        let entry = if examples.len() >= 40 {
            let (b, m) = select(examples);
            (*k, b, examples.len(), Some(m), false)
        } else {
            (*k, default, examples.len(), None, true)
        };
        match out.iter_mut().find(|x| x.0 == *k) {
            Some(x) => *x = entry,
            None => out.push(entry),
        }
    }
    out
}

/// warmup.py's constants and the population's rate response.
pub struct WarmupArgs {
    pub transfer: Vec<Vec<f64>>, // [target][played] between FAMILIES
    pub slopes: Slopes,
}

pub const GAP: f64 = 3000.;
pub const COOLING: f64 = 600.;
pub const TAUS: [f64; 6] = [0.5, 1., 2., 4., 8., 16.];
/// −ln .90
pub const CAPACITY_PRIOR: f64 = 0.10536051565782628;
pub const SESSION_TAU: f64 = 8.; // minutes of clock time since the session's first map
pub const FATIGUE_ONSET: f64 = 120.; // minutes; the user feels it at 2–3 h and the fit needs only its slope

/// warmup.transfer between FAMILIES names.
pub fn transfer(a: &str, b: &str) -> f64 {
    let within = |set: &[&str]| set.contains(&a) && set.contains(&b);
    if a == b {
        1.
    } else if within(&["rice", "chords", "trill"]) {
        0.55
    } else if within(&["chordjack", "jack"]) {
        0.65
    } else {
        0.2
    }
}

/// One completion in time order: row, session, readiness minutes, exposure minutes, seconds, family, minutes
/// since the session's first map began (any keymode).
pub struct Step {
    pub row: usize,
    pub session: i64,
    pub minutes: f64,
    pub d: f64,
    pub seconds: f64,
    pub family: usize,
    pub clock: f64,
}

/// warmup.timeline: exact local completions (public bests cannot establish session order; imported and live
/// copies of one completion are one map) through the readiness state machine.
pub fn steps(c: &Consts, rows: &[Row], feats: &[Feats], transfer: &[Vec<f64>], gap: f64, cooling: f64) -> Vec<Step> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| rows[a].ts.total_cmp(&rows[b].ts));
    let mut seen = std::collections::HashSet::new();
    let done: Vec<(usize, f64, usize)> = order.into_iter().filter(|&i| {
        let r = &rows[i];
        r.local && r.ts != 0. && !r.rd && r.completed && seen.insert((r.chart.clone(), round(r.ts) as i64, r.rate3.to_bits()))
    }).map(|i| {
        let r = &rows[i];
        (i, (feats[r.f].length / 1000. / r.rate.max(0.01)).min(1800.).max(0.), c.dominant(&feats[r.f]))
    }).collect();
    let pick = |f: fn(&(usize, f64, usize), &[Row]) -> f64| done.iter().map(|x| f(x, rows)).collect::<Vec<f64>>();
    let (sessions, minutes, ds) = warmup_timeline(&pick(|x, rows| rows[x.0].ts), &pick(|x, _| x.1),
        &done.iter().map(|x| rows[x.0].keys).collect::<Vec<_>>(), &done.iter().map(|x| x.2).collect::<Vec<_>>(), transfer, gap, cooling);
    let mut first: HashMap<i64, f64> = HashMap::default();
    done.iter().enumerate().map(|(j, &(row, seconds, family))| {
        let start = rows[row].ts - seconds;
        let clock = (start - *first.entry(sessions[j]).or_insert(start)).max(0.) / 60.;
        Step { row, session: sessions[j], minutes: minutes[j], d: ds[j], seconds, family, clock }
    }).collect()
}

/// Mean cold exposure over a completion: the score averages the cold and warming portions of the map.
fn exposure(minutes: f64, d: f64, tau: f64) -> f64 {
    if d > 0.001 { (-minutes / tau).exp() * -(-d / tau).exp_m1() * tau / d } else { (-minutes / tau).exp() }
}

/// The completions of one warm-up fit as columns (timeline order).
struct Design {
    nz: usize, // warmup.nuisance columns: 1, lr, lr², ln, stam, load, the strongest member per family
    g: usize,  // families
    keys: Vec<i64>,
    session: Vec<i64>,
    activation: Vec<f64>,
    t: Vec<f64>,
    month: Vec<i64>,
    chart: Vec<usize>, // (chart, round(rate, 2)) ids
    z: Vec<f64>,       // n × nz
    family: Vec<usize>,
    y: Vec<f64>,
    exposure: Vec<Vec<f64>>, // per tau
    session_cold: Vec<f64>,  // exposure to the session-long warm-up
    fatigue: Vec<f64>,       // hours past the fatigue onset
}

impl Design {
    fn new(c: &Consts, rows: &[Row], feats: &[Feats], steps: &[Step], taus: &[f64], session_tau: f64, onset: f64) -> Design {
        let members: Vec<Vec<usize>> = c.families.iter().map(|fam| {
            c.members.iter().find(|(k, _)| k == fam).map_or(Vec::new(), |(_, m)| m.iter().filter_map(|x| c.idx.get(x).copied()).collect())
        }).collect();
        let (chart, _) = index_of(steps.iter().map(|s| (rows[s.row].chart.clone(), round_digits(rows[s.row].rate, 2).to_bits())));
        let mut z = Vec::with_capacity(steps.len() * (6 + members.len()));
        for s in steps {
            // Chart/rate, month, pattern mix and session intercepts absorb familiarity, drift and good/bad days.
            let f = &feats[rows[s.row].f];
            let lr = f.overall.max(0.1).ln();
            z.extend_from_slice(&[1., lr, lr * lr, f.z_tail[0], f.z_tail[1], f.z_tail[2]]);
            z.extend(members.iter().map(|m| m.iter().map(|&x| f.sk[x]).fold(if m.is_empty() { 0. } else { f64::NEG_INFINITY }, f64::max)));
        }
        Design {
            nz: 6 + members.len(),
            g: members.len(),
            keys: steps.iter().map(|s| rows[s.row].keys).collect(),
            session: steps.iter().map(|s| s.session).collect(),
            activation: steps.iter().map(|s| s.minutes).collect(),
            t: steps.iter().map(|s| rows[s.row].t).collect(),
            month: steps.iter().map(|s| rows[s.row].t as i64).collect(),
            chart,
            z,
            family: steps.iter().map(|s| s.family).collect(),
            y: steps.iter().map(|s| rows[s.row].y - rows[s.row].base).collect(),
            exposure: taus.iter().map(|&tau| steps.iter().map(|s| exposure(s.minutes, s.d, tau)).collect()).collect(),
            session_cold: steps.iter().map(|s| exposure(s.clock, s.d, session_tau)).collect(),
            fatigue: steps.iter().map(|s| (s.clock - onset).max(0.) / 60.).collect(),
        }
    }
}

/// Sorted distinct values and each value's position among them (np.unique(..., return_inverse=True)).
fn unique<T: Ord + Copy>(v: &[T]) -> (Vec<T>, Vec<usize>) {
    let mut values = v.to_vec();
    values.sort_unstable();
    values.dedup();
    let at = v.iter().map(|x| values.binary_search(x).unwrap()).collect();
    (values, at)
}

struct WarmFit {
    amplitudes: Vec<f64>,
    session: f64,
    fatigue: f64,
    theta: Vec<f64>,
    months: Vec<f64>,
    month_effect: Vec<f64>,
    charts: Vec<usize>,
    chart_effect: Vec<f64>,
}

/// warmup._fit: small mixed ridge model of completions `ix`; nuisance effects are never exposed as ability.
fn warm_fit(d: &Design, slope: &[f64], speed: bool, ix: &[usize], tau: usize, use_cold: bool, capacity_prior: f64) -> WarmFit {
    let (n, g, nz) = (ix.len(), d.g, d.nz);
    // The last column is the session-long warm-up: same-chart replays stay worse for about six plays, far past
    // the readiness curve, and it is clock time so 7K warms up a 4K map played later in the session.
    let p = nz + g + 2;
    let (months, month_at) = unique(&ix.iter().map(|&i| d.month[i]).collect::<Vec<_>>());
    let (charts, chart_at) = unique(&ix.iter().map(|&i| d.chart[i]).collect::<Vec<_>>());
    let (_, session_at) = unique(&ix.iter().map(|&i| d.session[i]).collect::<Vec<_>>());
    let sessions = session_at.iter().max().map_or(0, |m| m + 1);
    let mut x = vec![0.; n * p];
    for (r, &i) in ix.iter().enumerate() {
        x[r * p..r * p + nz].copy_from_slice(&d.z[i * nz..(i + 1) * nz]);
        if use_cold {
            x[r * p + nz + d.family[i]] = d.exposure[tau][i] * slope[i];
            x[r * p + nz + g] = d.session_cold[i] * slope[i];
            x[r * p + nz + g + 1] = d.fatigue[i] * slope[i];
        }
    }
    let y: Vec<f64> = ix.iter().map(|&i| d.y[i]).collect();
    // Repeat/favourite maps cannot dominate; old sessions remain useful.
    let mut count = vec![0usize; charts.len()];
    chart_at.iter().for_each(|&c| count[c] += 1);
    let t_max = ix.iter().map(|&i| d.t[i]).fold(f64::NEG_INFINITY, f64::max);
    let mut w: Vec<f64> = ix.iter().zip(&chart_at).map(|(&i, &c)| 0.5f64.powf((t_max - d.t[i]).max(0.) / 18.) / (count[c] as f64).sqrt()).collect();
    let total: f64 = w.iter().sum();
    w.iter_mut().for_each(|v| *v *= n as f64 / total);
    let total: f64 = w.iter().sum();
    let mut a = vec![0.; p * p];
    for r in 0..n {
        let row = &x[r * p..(r + 1) * p];
        for j in 0..p {
            let v = w[r] * row[j];
            if v != 0. {
                a[j * p..(j + 1) * p].iter_mut().zip(row).for_each(|(q, xk)| *q += v * xk);
            }
        }
    }
    // Shrink rare families toward their keymode's curve, not toward perfect cold performance.
    let prior_precision = if speed { 12. } else { 2. };
    for j in 0..p {
        a[j * p + j] += if j == 0 { 1e-6 } else if j < nz { (0.005 * total).max(1.) } else if j >= nz + g { prior_precision } else { 0. };
    }
    for j in nz..nz + g {
        for k in nz..nz + g {
            a[j * p + k] += 12. * (if j == k { 1. } else { 0. } - 1. / g as f64) + if j == k { prior_precision } else { 0. };
        }
    }
    let lower: Vec<f64> = (0..p).map(|j| if j < nz { f64::NEG_INFINITY } else { 0. }).collect();
    let upper: Vec<f64> = (0..p).map(|j| if j < nz { f64::INFINITY } else if speed { 0.4 } else { 2. }).collect();
    let prior: Vec<f64> = (0..p).map(|j| if j >= nz && j < nz + g && speed && use_cold { prior_precision * capacity_prior } else { 0. }).collect();
    // Coordinate projection (non-negative cold losses) alternating with month/chart/session effects.
    let groups = [(month_at, months.len(), 4.), (chart_at, charts.len(), 5.), (session_at, sessions, 8.)];
    let (coeff, effects) = ridge_alternation(&x, &y, &w, &a, &lower, &upper, &prior, &groups, 16);
    WarmFit { amplitudes: coeff[nz..nz + g].to_vec(), session: coeff[nz + g], fatigue: coeff[nz + g + 1], theta: coeff[..nz].to_vec(), months: months.iter().map(|&m| m as f64).collect(),
              month_effect: effects[0].clone(), charts, chart_effect: effects[1].clone() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub rmse: f64,
    pub cold_rmse: Option<f64>,
    pub rows: usize,
    pub cold_rows: usize,
}

impl Stats {
    /// rmse + (cold_rmse or rmse)
    fn objective(&self) -> f64 {
        self.rmse + self.cold_rmse.filter(|&c| c != 0.).unwrap_or(self.rmse)
    }
}

/// warmup._evaluate
fn evaluate(fit: &WarmFit, d: &Design, slope: &[f64], ix: &[usize], tau: usize) -> Stats {
    let (mut sq, mut cold, mut cold_n) = (0., 0., 0);
    for &i in ix {
        let month = interp(d.t[i], &fit.months, &fit.month_effect);
        let chart = fit.charts.binary_search(&d.chart[i]).map_or(0., |at| fit.chart_effect[at]); // unseen: no effect
        let theta: f64 = d.z[i * d.nz..(i + 1) * d.nz].iter().zip(&fit.theta).map(|(a, b)| a * b).sum();
        let error = d.y[i] - (theta + month + chart + (fit.amplitudes[d.family[i]] * d.exposure[tau][i] + fit.session * d.session_cold[i] + fit.fatigue * d.fatigue[i]) * slope[i]);
        sq += error * error;
        if d.activation[i] < 1. {
            cold += error * error;
            cold_n += 1;
        }
    }
    Stats { rmse: (sq / ix.len() as f64).sqrt(), cold_rmse: (cold_n > 0).then(|| (cold / cold_n as f64).sqrt()), rows: ix.len(), cold_rows: cold_n }
}

struct ModeFit {
    keys: i64,
    tau: f64,
    amplitudes: Vec<f64>,
    session: f64,
    fatigue: f64,
    sessions: usize,
    obs: Vec<usize>,
    cold_rows: usize,
    dev: Stats,
    no_cold: Stats,
    adjust: Vec<f64>, // per obs
}

/// warmup.fit's per-keymode loop: decay chosen on earlier sessions only (a late-session holdout is diagnostic,
/// not a fitting target), then the final fit on every completion.
fn warm_modes(d: &Design, slope: &[f64], speed: bool, taus: &[f64], capacity_prior: f64) -> Vec<ModeFit> {
    let mut keys = d.keys.clone();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter().filter_map(|k| {
        let obs: Vec<usize> = (0..d.keys.len()).filter(|&i| d.keys[i] == k).collect();
        let (sessions, _) = unique(&obs.iter().map(|&i| d.session[i]).collect::<Vec<_>>());
        if obs.len() < 100 || sessions.len() < 25 {
            return None;
        }
        let cut = sessions[1usize.max((sessions.len() as f64 * 0.7) as usize)];
        let train: Vec<usize> = obs.iter().copied().filter(|&i| d.session[i] < cut).collect();
        let dev: Vec<usize> = obs.iter().copied().filter(|&i| d.session[i] >= cut).collect();
        let mut best: Option<(f64, usize, Stats)> = None;
        for t in 0..taus.len() {
            let stats = evaluate(&warm_fit(d, slope, speed, &train, t, true, capacity_prior), d, slope, &dev, t);
            if best.is_none_or(|b| stats.objective() < b.0) {
                best = Some((stats.objective(), t, stats));
            }
        }
        let (_, t, dev_stats) = best.unwrap();
        let no_cold = evaluate(&warm_fit(d, slope, speed, &train, t, false, capacity_prior), d, slope, &dev, t);
        let last = warm_fit(d, slope, speed, &obs, t, true, capacity_prior);
        let adjust = obs.iter().map(|&i| (last.amplitudes[d.family[i]] * d.exposure[t][i] + last.session * d.session_cold[i] + last.fatigue * d.fatigue[i]) * slope[i]).collect();
        Some(ModeFit { keys: k, tau: taus[t], sessions: sessions.len(), cold_rows: obs.iter().filter(|&&i| d.activation[i] < 1.).count(),
                       amplitudes: last.amplitudes, session: last.session, fatigue: last.fatigue, obs, dev: dev_stats, no_cold, adjust })
    }).collect()
}

/// warmup.Model's parts of one keymode's report.
#[derive(Clone, Debug, PartialEq)]
pub struct WarmReport {
    pub sessions: usize,
    pub completed_rows: usize,
    pub cold_rows: usize,
    pub selected_tau: f64,
    pub session_amplitude: f64,
    pub fatigue_per_hour: f64,
    pub dev: Stats,
    pub no_cold_dev: Stats,
    pub amplitudes: Vec<f64>, // per FAMILIES
    pub log_rate: bool,
    pub cold_capacity: Vec<f64>, // per FAMILIES; rate-response models only
    pub rate_response_accepted: Option<bool>,
    pub constant_response_dev: Option<Stats>,
}

/// A compact recent completion: seeds a live session after a restart.
#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    pub end: f64,
    pub t: f64,
    pub keys: i64,
    pub family: usize,
    pub played_seconds: f64,
    pub key: Option<String>,
    pub chart: String,
}

#[derive(Clone, Default)]
pub struct WarmModels {
    pub control: HashMap<i64, Mode>,
    pub control_report: HashMap<i64, WarmReport>,
    pub modes: HashMap<i64, Mode>,
    pub report: HashMap<i64, WarmReport>,
    pub history: Vec<Completion>,
    pub rust: Rust,
}

/// Rust from too little recent practice in a keymode, in log-error units (positive = worse). Practice = minutes
/// played in the keymode, each fading over `memory_days`; readiness = e^(−practice/`minutes`) runs 1 (none) → 0.
/// Shift = amplitude · max(0, readiness − the keymode's usual readiness over its plays): the level already holds the
/// usual amount. The user's 4K came back after 5 months at its May level (G835LX 2026-11-09: 96 % offered, 90 % played).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rust {
    pub amplitude: f64,
    pub memory_days: f64,
    pub minutes: f64,
    /// Per keymode at fit time: (last own completion's end, practice minutes then, usual readiness).
    pub state: HashMap<i64, (f64, f64, f64)>,
}

impl Rust {
    /// The keymode's rust at `now` after the fit's state and later completions (end, played minutes).
    pub fn at(&self, keys: i64, later: &[(f64, f64)], now: f64) -> f64 {
        let Some(&(mut last, mut practice, usual)) = self.state.get(&keys).filter(|_| self.amplitude > 0.) else { return 0. };
        let mut later: Vec<(f64, f64)> = later.iter().copied().filter(|x| x.0 > last).collect();
        later.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (end, played) in later {
            practice = practice * self.fade(end - last) + played;
            last = end;
        }
        practice *= self.fade((now - last).max(0.));
        self.amplitude * ((-practice / self.minutes).exp() - usual).max(0.)
    }

    /// Rust over a subset of the keymode's plays (end, played minutes), e.g. one skill's charts: the same curve, practice
    /// counted on those plays alone, against the keymode's usual readiness.
    pub fn of_plays(&self, keys: i64, plays: &[(f64, f64)], now: f64) -> f64 {
        let Some(&(_, _, usual)) = self.state.get(&keys).filter(|_| self.amplitude > 0.) else { return 0. };
        let practice: f64 = plays.iter().map(|&(end, played)| played * self.fade((now - end).max(0.))).sum();
        self.amplitude * ((-practice / self.minutes).exp() - usual).max(0.)
    }

    fn fade(&self, seconds: f64) -> f64 {
        (-seconds / 86400. / self.memory_days).exp()
    }
}

/// Rust's curve on own completions' residuals from the level: one pooled curve over keymodes, memory and minutes from
/// a grid, amplitude by ridge least squares. No positive amplitude: no rust.
// ponytail: the level curve is fitted before rust, so rusty months still pull it down a little; refit it on
// rust-corrected residuals if the two start to trade off.
fn fit_rust(rows: &[Row], feats: &[Feats], resid: &[f64], w: &[f64]) -> Rust {
    let mut order: Vec<usize> = (0..rows.len()).filter(|&i| {
        let r = &rows[i];
        r.local && !r.public && r.ts != 0. && !r.rd && r.completed
    }).collect();
    order.sort_by(|&a, &b| rows[a].ts.total_cmp(&rows[b].ts));
    let mut seen = std::collections::HashSet::new();
    order.retain(|&i| seen.insert((rows[i].chart.clone(), round(rows[i].ts) as i64, rows[i].rate3.to_bits())));
    let played: Vec<f64> = order.iter().map(|&i| (feats[rows[i].f].length / 1000. / rows[i].rate.max(0.01) / 60.).clamp(0., 30.)).collect();
    let mut best = Rust::default();
    let mut gain = 0.;
    for memory_days in [7., 14., 30., 60.] {
        for minutes in [10., 20., 30., 60., 120., 240.] {
            let mut probe = Rust { memory_days, minutes, ..Rust::default() };
            // Practice before each play, then each keymode's usual readiness.
            let mut last: HashMap<i64, (f64, f64)> = HashMap::default();
            let x: Vec<f64> = order.iter().zip(&played).map(|(&i, &p)| {
                let r = &rows[i];
                let start = r.ts - p * 60.;
                let s = last.entry(r.keys).or_insert((start, 0.));
                let before = s.1 * probe.fade((start - s.0).max(0.));
                *s = (r.ts, before + p);
                (-before / minutes).exp()
            }).collect();
            let mut usual: HashMap<i64, (f64, f64)> = HashMap::default();
            order.iter().zip(&x).for_each(|(&i, &v)| {
                let u = usual.entry(rows[i].keys).or_default();
                u.0 += w[i] * v;
                u.1 += w[i];
            });
            let usual: HashMap<i64, f64> = usual.into_iter().map(|(k, (s, m))| (k, s / m.max(1e-9))).collect();
            let (mut xy, mut xx) = (0., 1.); // the 1: a ridge toward no rust
            for (&i, &v) in order.iter().zip(&x) {
                let v = (v - usual[&rows[i].keys]).max(0.);
                xy += w[i] * v * resid[i];
                xx += w[i] * v * v;
            }
            if xy > 0. && xy * xy / xx > gain {
                gain = xy * xy / xx;
                probe.amplitude = xy / xx;
                probe.state = last.into_iter().map(|(k, (t, p))| (k, (t, p, usual[&k]))).collect();
                best = probe;
            }
        }
    }
    best
}

/// Both warm-up fits and the historical cold adjustment per row key. The rate response replaces the constant one in
/// a keymode only when its own earlier-session DEV check wins: sparse modes must not inherit a large rate deficit.
fn warmup_models(c: &Consts, rows: &[Row], feats: &[Feats], args: &WarmupArgs) -> (WarmModels, HashMap<Best, f64>) {
    let steps = steps(c, rows, feats, &args.transfer, GAP, COOLING);
    let d = Design::new(c, rows, feats, &steps, &TAUS, SESSION_TAU, FATIGUE_ONSET);
    let speed: Vec<f64> = steps.iter().map(|s| {
        let f = &feats[rows[s.row].f];
        args.slopes.physical(f, &c.family_weights(f))
    }).collect();
    let ones = vec![1.; steps.len()];
    let baseline = warm_modes(&d, &ones, false, &TAUS, CAPACITY_PRIOR);
    let fitted = warm_modes(&d, &speed, true, &TAUS, CAPACITY_PRIOR);
    let mode = |m: &ModeFit, log_rate: bool| Mode { tau: m.tau, log_rate, amplitudes: m.amplitudes.clone(), session: m.session, fatigue: m.fatigue };
    let report = |m: &ModeFit, log_rate: bool| WarmReport {
        sessions: m.sessions, completed_rows: m.obs.len(), cold_rows: m.cold_rows, selected_tau: m.tau, session_amplitude: m.session,
        fatigue_per_hour: m.fatigue, dev: m.dev, no_cold_dev: m.no_cold, amplitudes: m.amplitudes.clone(), log_rate,
        cold_capacity: Vec::new(), rate_response_accepted: None, constant_response_dev: None,
    };
    let mut out = WarmModels::default();
    for m in &baseline {
        out.control.insert(m.keys, mode(m, false));
        out.control_report.insert(m.keys, report(m, false));
    }
    let mut adjust: HashMap<Best, f64> = HashMap::default();
    for m in &fitted {
        let mut r = report(m, true);
        r.cold_capacity = m.amplitudes.iter().map(|a| (-a).exp()).collect();
        let mut values = &m.adjust;
        out.modes.insert(m.keys, mode(m, true));
        if let Some(control) = baseline.iter().find(|b| b.keys == m.keys) {
            let within = |a: Option<f64>, b: Option<f64>| b.is_none_or(|b| a.is_some_and(|a| a <= 1.01 * b));
            let accepted = m.dev.objective() < control.dev.objective() && m.dev.rmse <= 1.01 * control.dev.rmse
                && within(m.dev.cold_rmse, control.dev.cold_rmse);
            r.rate_response_accepted = Some(accepted);
            r.constant_response_dev = Some(control.dev);
            if !accepted {
                out.modes.insert(m.keys, mode(control, false));
                values = &control.adjust;
            }
        }
        out.report.insert(m.keys, r);
        for (&i, v) in m.obs.iter().zip(values) {
            adjust.insert(rows[steps[i].row].best.clone(), *v);
        }
    }
    out.history = steps[steps.len().saturating_sub(64)..].iter().map(|s| {
        let r = &rows[s.row];
        Completion { end: r.ts, t: r.ts - s.seconds, keys: r.keys, family: s.family, played_seconds: s.seconds,
                     key: match &r.best { Best::Key(k) => Some(k.clone()), _ => None }, chart: r.chart.clone() }
    }).collect();
    (out, adjust)
}

/// Personal's knobs.
#[derive(Clone, Copy)]
pub struct Knobs {
    pub half_life: f64,
    pub level_bw: f64,
    pub history_zone: f64,
    pub recent_months: f64,
    pub att_sd: f64,
    pub session_gap: f64,
}

/// The training support of a shape ridge (calib recommendation_audit shape trace).
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeReport {
    pub rows: usize,
    pub z_min: Vec<f64>,
    pub z_max: Vec<f64>,
    pub ridge: Option<f64>,
    pub chart_families: usize,
    pub cv_mse: Vec<f64>, // per GRID value
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeySession {
    pub global_variance: f64,
    pub skill_variances: Vec<f64>,
    pub family_sessions: Vec<usize>,
    pub sessions: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionReport {
    pub sessions: usize,
    pub supported_sessions: usize,
    pub condition_variance: f64,
    pub within_variance: f64,
    pub skill_condition_variance: f64,
    pub by_key: Vec<(i64, KeySession)>,
    pub prior_observations: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RateReport {
    pub folds: usize,
    pub train_mse: Option<(f64, f64)>, // (train, full transfer)
    pub pooled: bool,
}

/// What recommend.Personal holds after its fit.
#[derive(Default)]
pub struct Fit {
    pub keys: Vec<i64>,
    pub level: HashMap<i64, f64>,
    pub level_var: HashMap<i64, f64>,
    pub curve: HashMap<i64, (i64, Vec<f64>)>,
    pub sd: HashMap<i64, f64>,
    pub monthly: HashMap<i64, (i64, Vec<f64>, Vec<f64>, Vec<f64>)>, // first month; own plays' weighted sum, weight, count
    pub beta: HashMap<i64, Vec<f64>>,
    pub zmean: HashMap<i64, Vec<f64>>,
    pub zscale: HashMap<i64, Vec<f64>>,
    pub shape_validation: HashMap<i64, ShapeReport>,
    pub steep: HashMap<i64, (f64, f64)>,
    pub far: HashMap<i64, (f64, f64)>, // keys → (centre, slope) on the chart's curve over ability (`note_far`)
    pub centre: HashMap<i64, f64>, // the own plays' usual ln loss on the model scale (recency-weighted): `PredModel::blend` stretches about it
    pub recent: HashMap<i64, (Vec<f64>, Vec<f64>, Vec<f64>)>,
    pub session_validation: SessionReport,
    pub aff: HashMap<String, (f64, f64)>,
    pub rate_aff: HashMap<String, Vec<(f64, f64, f64)>>,
    pub rate_support: HashMap<String, (f64, f64)>,
    pub rate_bandwidth: HashMap<i64, f64>,
    pub rate_validation: HashMap<i64, RateReport>,
    pub sd_att: HashMap<i64, f64>,
    pub best_gap: f64,
    pub glob: f64,
    pub tau2_c: f64,
    pub kap_c: f64,
    pub sd_all: f64,
    pub retry: f64,
    pub reading: Reading,               // how the player reads SV (`fit_reading`)
    pub plays: HashMap<String, Read>,   // every chart with a local play: its reads at the fit
    pub n_rows: usize,
    pub warmup: WarmModels,
}

/// The steepness ridge, in ln-rating variance per unit weight: a keymode whose ratings spread less than this keeps under half its slope.
pub const STEEP_RIDGE: f64 = 0.005; // walk-forward audit 2026-11-14: MAE .9836 (none) → .9728 (0.005), .9773 (0.02), .9756 (0.08)

/// The same ridge for the slope on the chart's curve over ability, in its variance per unit weight. Own walk-forward audit
/// 2026-11-23 (1,446 plays, mean |error| against none): 0.01 −.0025 ± .0029, 0.05 −.0023 ± .0020, 0.25 −.0009 ± .0008: flat
/// from 0.01 to 0.05, the steadier end taken.
pub const FAR_RIDGE: f64 = 0.05;

/// (centre, slope) of a ridge of `u` on ln rating `lr` around its weighted mean; None without weight or with ridge = ∞.
fn steepness(lr: &[f64], u: &[f64], w: &[f64], ridge: f64) -> Option<(f64, f64)> {
    let sw: f64 = w.iter().sum();
    if !(ridge.is_finite() && sw > 0.) {
        return None;
    }
    let centre = lr.iter().zip(w).map(|(l, x)| x * l).sum::<f64>() / sw;
    let (sxy, sxx) = lr.iter().zip(u).zip(w).fold((0., 0.), |(a, b), ((l, u), x)| (a + x * (l - centre) * u, b + x * (l - centre).powi(2)));
    Some((centre, sxy / (sxx + ridge * sw)))
}

/// recommend.Personal.__init__ on the kept evidence rows.
/// Local plays of each chart (months, unix seconds): every local score of any client, rate and outcome; Constant Speed left out
/// (it reads no SV). Keyed like `Row::chart`.
pub type Reads = HashMap<String, Vec<(f64, f64)>>;

pub fn personal_fit(c: &Consts, mut rows: Vec<Row>, feats: &[Feats], now: f64, levels: &HashMap<i64, f64>, knobs: Knobs, warmup: &WarmupArgs, reads: &Reads) -> Fit {
    let Knobs { half_life, level_bw, history_zone, recent_months, att_sd, session_gap } = knobs;
    // A row whose chart has no population prediction (no features) would spread NaN through its keymode's level curve.
    rows.retain(|r| r.y.is_finite() && r.base.is_finite());
    let (models, cold) = warmup_models(c, &rows, feats, warmup);
    // A typical warmed baseline, not a mixture that counts the historical cold deficit here and again at
    // next-session startup.
    rows.iter_mut().for_each(|r| r.y -= cold.get(&r.best).copied().unwrap_or(0.));
    let n = rows.len();
    let fe = |i: usize| &feats[rows[i].f];
    let (chart, charts) = index_of(rows.iter().map(|r| if r.rd { format!("{}|RD", r.chart) } else { r.chart.clone() }));
    let mut by: Vec<Vec<usize>> = vec![Vec::new(); charts.len()];
    chart.iter().enumerate().for_each(|(i, &c)| by[c].push(i));
    let mut e: Vec<f64> = rows.iter().map(|r| r.y - r.base).collect();
    let lr: Vec<f64> = (0..n).map(|i| fe(i).overall.max(0.05).ln()).collect();
    // Stable high scores are bests: shift them by the best-of-several gap measured on lazer attempts.
    let gaps: Vec<f64> = by.iter().filter_map(|ix| {
        let v: Vec<f64> = ix.iter().filter(|&&i| !rows[i].public).map(|&i| e[i]).collect();
        (v.len() >= 3).then(|| v.iter().copied().fold(f64::INFINITY, f64::min) - mean(&v))
    }).collect();
    let best_gap = if gaps.is_empty() { -0.15 } else { mean(&gaps) };
    (0..n).filter(|&i| rows[i].public).for_each(|i| e[i] -= best_gap);
    let w_rec: Vec<f64> = rows.iter().map(|r| 0.5f64.powf((now - r.t).max(0.) / half_life)).collect();
    // One heavily retried chart can't dominate. RD permutes columns, hence hand assignments and pattern
    // character: its broad ability evidence keeps a reduced weight, never exact-chart familiarity or retry.
    let w: Vec<f64> = (0..n).map(|i| {
        let r = &rows[i];
        (if r.public { 0.5 } else { 1. }) / (by[chart[i]].len() as f64).sqrt() * if r.rd { 0.35 } else { 1. }
    }).collect();
    let mut keys: Vec<i64> = rows.iter().map(|r| r.keys).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.retain(|&k| rows.iter().filter(|r| r.keys == k).count() >= 30);
    let glob = if n > 0 { average((0..n).map(|i| (e[i], w[i] * w_rec[i]))) } else { 0. };
    let of_keys = |k: i64| -> Vec<usize> { (0..n).filter(|&i| rows[i].keys == k).collect() };

    let mut out = Fit { keys: keys.clone(), best_gap, glob, n_rows: n, ..Fit::default() };
    let mut l_at = vec![0.; n];
    for &k in &keys {
        let s = of_keys(k);
        let mb: Vec<i64> = s.iter().map(|&i| rows[i].t.floor() as i64).collect();
        let g0 = *mb.iter().min().unwrap();
        let g1 = (*mb.iter().max().unwrap()).max(now as i64);
        let len = (g1 - g0 + 1) as usize;
        let (mut sw, mut se, mut own_e, mut own_w, mut own_n) = (vec![0.; len], vec![0.; len], vec![0.; len], vec![0.; len], vec![0.; len]);
        for (&i, &m) in s.iter().zip(&mb) {
            let at = (m - g0) as usize;
            sw[at] += w[i];
            se[at] += w[i] * e[i];
            // Stats history: own plays only, and only maps near the player's limit say where the limit is.
            let own = if rows[i].public || levels.get(&k).is_some_and(|l| lr[i] < l - history_zone) { 0. } else { w[i] };
            own_e[at] += own * e[i];
            own_w[at] += own;
            own_n[at] += f64::from(own > 0.);
        }
        let kernel = |a: usize, b: usize| (-0.5 * ((a as f64 - b as f64) / level_bw).powi(2)).exp();
        let num: Vec<f64> = (0..len).map(|a| (0..len).map(|b| kernel(a, b) * se[b]).sum()).collect();
        let den: Vec<f64> = (0..len).map(|a| (0..len).map(|b| kernel(a, b) * sw[b]).sum()).collect();
        let prior = average(s.iter().map(|&i| (e[i], w[i])));
        let curve: Vec<f64> = (0..len).map(|a| (num[a] + 3. * prior) / (den[a] + 3.)).collect();
        for (&i, &m) in s.iter().zip(&mb) {
            l_at[i] = curve[(m - g0) as usize];
        }
        let sdk = average(s.iter().map(|&i| ((e[i] - l_at[i]).powi(2), w[i]))).sqrt();
        out.level.insert(k, curve[len - 1]);
        if s.iter().any(|&i| !rows[i].public) {
            out.centre.insert(k, average(s.iter().filter(|&&i| !rows[i].public).map(|&i| (e[i] + rows[i].base, w[i] * w_rec[i]))));
        }
        out.sd.insert(k, sdk);
        out.level_var.insert(k, sdk * sdk / den[len - 1].max(1.));
        out.curve.insert(k, (g0, curve));
        out.monthly.insert(k, (g0, own_e, own_w, own_n));
    }

    // Personal steepness (user 2026-11-14, 4K offers at 9★+ landing 2–6 pp under while lower ones hit): how much faster or
    // slower than the population curve the player's accuracy falls with the chart's rating, per keymode, by a ridge on what
    // the level leaves, shrunk to the population's slope (0).
    let mut u: Vec<f64> = (0..n).map(|i| e[i] - l_at[i]).collect();
    let mut steep_at = vec![0.; n];
    let ridge = std::env::var("MS_STEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(STEEP_RIDGE);
    for &k in &keys {
        let s = of_keys(k);
        let pick = |v: &[f64]| s.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let ww: Vec<f64> = s.iter().map(|&i| w[i] * w_rec[i]).collect();
        let Some((centre, beta)) = steepness(&pick(&lr), &pick(&u), &ww, ridge) else { continue };
        for &i in &s {
            steep_at[i] = beta * (lr[i] - centre);
            u[i] -= steep_at[i];
        }
        out.steep.insert(k, (centre, beta));
    }
    // Personal response to the chart's curve over ability (the note model's `note_far`; plan 2026-11-22: as a per-player
    // slope it takes 0.3 % (4K) / 0.8 % (7K) off the public held-out error): the same ridge on what the steepness leaves.
    let mut far_at = vec![0.; n];
    let far_ridge = std::env::var("MS_FAR").ok().and_then(|v| v.parse().ok()).unwrap_or(FAR_RIDGE);
    for &k in &keys {
        let s: Vec<usize> = of_keys(k).into_iter().filter(|&i| fe(i).shape.is_finite()).collect();
        let pick = |v: &[f64]| s.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let ww: Vec<f64> = s.iter().map(|&i| w[i] * w_rec[i]).collect();
        let far: Vec<f64> = s.iter().map(|&i| fe(i).shape).collect();
        let Some((centre, beta)) = steepness(&far, &pick(&u), &ww, far_ridge) else { continue };
        for &i in &s {
            far_at[i] = beta * (fe(i).shape - centre);
            u[i] -= far_at[i];
        }
        out.far.insert(k, (centre, beta));
    }
    let rust = fit_rust(&rows, feats, &u, &w);
    let shape_report = |s: &Shape, zc: &[f64], p: usize| -> ShapeReport {
        let column = |j: usize| zc.iter().skip(j).step_by(p).copied();
        ShapeReport {
            rows: zc.len() / p,
            z_min: (0..p).map(|j| column(j).fold(f64::INFINITY, f64::min)).collect(),
            z_max: (0..p).map(|j| column(j).fold(f64::NEG_INFINITY, f64::max)).collect(),
            ridge: s.ridge,
            chart_families: s.families,
            cv_mse: s.cv.clone(),
        }
    };
    for &k in &keys {
        let s = of_keys(k);
        // Lists aim every chart at one accuracy and 10K has one player, so the ridge read "denser = easier" and cancelled the stars:
        // a 7★ chordjack came out level with a 3.8★ map (user 2026-11-01: 97 % shown, 84 % played). Later plays: .420 → .358.
        if s.len() < 200 || k == 10 {
            continue;
        }
        let z: Vec<f64> = s.iter().flat_map(|&i| c.zvec(fe(i), rows[i].base + l_at[i] + steep_at[i] + far_at[i])).collect();
        let p = z.len() / s.len();
        let ww: Vec<f64> = s.iter().map(|&i| w[i] * w_rec[i]).collect();
        let (zm, scale) = standardise(&z, p, &ww);
        let zc: Vec<f64> = z.chunks_exact(p).flat_map(|row| row.iter().zip(&zm).zip(&scale).map(|((x, m), sc)| (x - m) / sc)).collect();
        let target: Vec<f64> = s.iter().map(|&i| u[i]).collect();
        let groups: Vec<&str> = s.iter().map(|&i| rows[i].family.as_str()).collect();
        let fit = fit_shape(&zc, p, &target, &ww, &groups);
        for (row, &i) in zc.chunks_exact(p).zip(&s) {
            u[i] -= row.iter().zip(&fit.beta).map(|(x, b)| x * b).sum::<f64>();
        }
        out.shape_validation.insert(k, shape_report(&fit, &zc, p));
        out.beta.insert(k, fit.beta);
        out.zmean.insert(k, zm);
        out.zscale.insert(k, scale);
    }
    // What the year-long shape leaves on the last RECENT_MONTHS of own plays is current per-family form.
    for &k in &keys {
        let idx = of_keys(k);
        let sel: Vec<bool> = idx.iter().map(|&i| !rows[i].public && now - rows[i].t <= recent_months).collect();
        let s: Vec<usize> = idx.iter().zip(&sel).filter(|x| *x.1).map(|x| *x.0).collect();
        if s.len() < 100 {
            continue;
        }
        let r: Vec<Vec<f64>> = idx.iter().map(|&i| c.recent_vec(fe(i))).collect();
        let p = r[0].len();
        let rsel: Vec<f64> = r.iter().zip(&sel).filter(|x| *x.1).flat_map(|x| x.0.iter().copied()).collect();
        let ws: Vec<f64> = s.iter().map(|&i| w[i]).collect();
        let (rm, rs) = standardise(&rsel, p, &ws);
        let rc: Vec<Vec<f64>> = r.iter().map(|v| v.iter().zip(&rm).zip(&rs).map(|((x, m), sc)| (x - m) / sc).collect()).collect();
        let zc: Vec<f64> = rc.iter().zip(&sel).filter(|x| *x.1).flat_map(|x| x.0.iter().copied()).collect();
        let target: Vec<f64> = s.iter().map(|&i| u[i]).collect();
        let groups: Vec<&str> = s.iter().map(|&i| rows[i].family.as_str()).collect();
        let fit = fit_shape(&zc, p, &target, &ws, &groups);
        for (row, &i) in rc.iter().zip(&idx) {
            u[i] -= row.iter().zip(&fit.beta).map(|(x, b)| x * b).sum::<f64>();
        }
        out.recent.insert(k, (rm, rs, fit.beta));
    }

    // Historical session condition is not a permanent property of a chart: the target chart is left out of
    // its own context, and retries in one session are one block of affinity evidence.
    let (context, session_ids, ctx) = session_context(c, &rows, feats, &u);
    u.iter_mut().zip(&context).for_each(|(x, a)| *x -= a);
    (out.reading, out.plays) = reading_of(&rows, feats, c, &mut u, reads, now);
    let blocks = chart_blocks(&rows, &chart, charts.len(), &u, &session_ids, now);
    let within = |ix: &Vec<usize>| -> Vec<f64> {
        let m = mean(&ix.iter().map(|&i| u[i]).collect::<Vec<_>>());
        ix.iter().map(|&i| u[i] - m).collect()
    };
    let spread = |charts: &mut dyn Iterator<Item = &Vec<usize>>| -> Option<f64> {
        let all: Vec<f64> = charts.filter(|ix| ix.len() >= 2).flat_map(within).collect();
        (!all.is_empty()).then(|| (mean(&all.iter().map(|x| x * x).collect::<Vec<_>>()) * att_sd).sqrt())
    };
    let sd_all = spread(&mut by.iter()).unwrap_or(0.35);
    let tau2_c = if blocks.is_empty() {
        0.05
    } else {
        let terms: Vec<f64> = blocks.iter().map(|v| average(v.iter().map(|x| (x.1, x.2))).powi(2) - sd_all * sd_all / effective_count(v)).collect();
        mean(&terms).max(0.005)
    };
    let kap_c = sd_all * sd_all / tau2_c;
    for (name, v) in charts.iter().zip(&blocks) {
        let s: f64 = v.iter().map(|x| x.1 * x.2).sum();
        let m: f64 = v.iter().map(|x| x.2).sum();
        out.aff.insert(name.clone(), (s / (m + kap_c), m));
    }
    let rates_of: Vec<Vec<(f64, Vec<usize>)>> = by.iter().map(|ix| {
        let mut rates: Vec<(f64, Vec<usize>)> = Vec::new();
        for &i in ix {
            match rates.iter_mut().find(|(r, _)| *r == rows[i].rate3) {
                Some((_, v)) => v.push(i),
                None => rates.push((rows[i].rate3, vec![i])),
            }
        }
        rates
    }).collect();
    let transfer = rate_transfer(&rows, &u, &by, &rates_of, kap_c);
    for (k, b, folds, meta, pooled) in &transfer {
        out.rate_bandwidth.insert(*k, *b);
        out.rate_validation.insert(*k, RateReport { folds: *folds, train_mse: *meta, pooled: *pooled });
    }
    let bandwidth_of = |k: i64| transfer.iter().find(|x| x.0 == k).unwrap_or(&transfer[0]).1;
    for ((name, ix), (rates, v)) in charts.iter().zip(&by).zip(rates_of.iter().zip(&blocks)) {
        let lo = rates.iter().map(|x| x.0).fold(f64::INFINITY, f64::min);
        let hi = rates.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
        out.rate_support.insert(name.clone(), (lo, hi));
        if rates.len() > 1 {
            // A directly observed DT must not inherit pseudo-scores from comfortable NM: borrow only
            // rate-nearby evidence, with the measured between-chart shrinkage as the prior precision.
            let stats: Vec<(f64, f64, f64)> = rates.iter().map(|(rate, _)| {
                let at = v.iter().filter(|x| x.0 == *rate);
                (*rate, at.clone().map(|x| x.1 * x.2).sum(), at.map(|x| x.2).sum())
            }).collect();
            out.rate_aff.insert(name.clone(), rate_anchors(&stats, bandwidth_of(rows[ix[0]].keys), kap_c));
        }
    }
    for &k in &keys {
        let v = spread(&mut by.iter().filter(|ix| rows[ix[0]].keys == k)).unwrap_or(0.);
        out.sd_att.insert(k, if v > 0.05 { v } else { sd_all });
    }

    let retry = retry(&rows, &(0..n).map(|i| e[i] - l_at[i]).collect::<Vec<_>>(), session_gap);

    out.session_validation = SessionReport {
        sessions: ctx.sessions, supported_sessions: ctx.supported, condition_variance: ctx.between, within_variance: ctx.within,
        skill_condition_variance: ctx.skill_variance,
        by_key: ctx.by_key.into_iter().map(|(k, b)| (k, KeySession { global_variance: b.global_variance, skill_variances: b.skill_variances,
                                                                     family_sessions: b.family_sessions, sessions: b.sessions })).collect(),
        prior_observations: ctx.shrink.is_finite().then_some(ctx.shrink),
    };
    out.tau2_c = tau2_c;
    out.kap_c = kap_c;
    out.sd_all = sd_all;
    out.retry = retry;
    out.warmup = WarmModels { rust, ..models };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(chart: &str, rate: f64, ts: f64, rd: bool) -> Row {
        Row { keys: 7, y: 0., base: 0., t: 300., ts, public: false, local: true, rd, completed: true, chart: chart.into(),
              rate, rate3: round_digits(rate, 3), best: Best::None, family: chart.into(), f: 0 }
    }

    #[test]
    fn steepness_follows_a_player_who_falls_off_faster_than_the_population() {
        let lr: Vec<f64> = (0..40).map(|i| (5. + i as f64 * 0.125).ln()).collect();
        let steep: Vec<f64> = lr.iter().map(|l| 0.8 * (l - 2.)).collect();
        let (centre, beta) = steepness(&lr, &steep, &vec![1.; 40], 0.).unwrap();
        assert!((beta - 0.8).abs() < 1e-9 && centre > 1.6 && centre < 2.3);
        let shrunk = steepness(&lr, &steep, &vec![1.; 40], STEEP_RIDGE).unwrap().1;
        assert!(shrunk > 0.3 && shrunk < 0.8, "{shrunk}");
        assert!(steepness(&lr, &vec![0.; 40], &vec![1.; 40], STEEP_RIDGE).unwrap().1.abs() < 1e-12);
        assert!(steepness(&lr, &steep, &vec![1.; 40], f64::INFINITY).is_none());
    }

    #[test]
    fn rust_after_a_layoff_is_found_and_fades_with_play() {
        let c = Consts::standard();
        let mut f = Feats::bare(7, 5., 8., &c);
        f.length = 180_000.;
        let day = 86400.;
        // Daily 10-map sessions for 40 days, 90 days off, then 10 days more: the first maps back play worse.
        let ts: Vec<f64> = (0..50).filter(|d| *d < 40 || *d >= 45).flat_map(|d| {
            let start = if d < 40 { d as f64 * day } else { (d as f64 + 85.) * day };
            (0..10).map(move |m| start + (m as f64 + 1.) * 200.)
        }).collect();
        let rows: Vec<Row> = ts.iter().enumerate().map(|(i, &t)| row(&i.to_string(), 1., t, false)).collect();
        let back = 40. * day + 1000.;
        let resid: Vec<f64> = ts.iter().map(|&t| if t > back { 0.6 * (-(rows.iter().filter(|r| r.ts > back && r.ts < t).count() as f64) * 3. / 30.).exp() } else { 0. }).collect();
        let rust = fit_rust(&rows, &[f], &resid, &vec![1.; rows.len()]);
        assert!(rust.amplitude > 0.2, "{rust:?}");
        let last = *ts.last().unwrap();
        assert!(rust.at(7, &[], last + 120. * day) > 0.2);
        assert!(rust.at(7, &[], last + 600.) < 0.05);
        // Live plays after the fit wear it off again.
        let later: Vec<(f64, f64)> = (0..20).map(|m| (last + 120. * day + m as f64 * 200., 3.)).collect();
        assert!(rust.at(7, &later, last + 120. * day + 4000.) < 0.05);
    }

    #[test]
    fn rate_validation_can_choose_finite_or_full_transfer() {
        let (mut rows, mut by, mut rates, mut stable, mut different) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for chart in 0..30 {
            let i = rows.len();
            for (rate, sign) in [(0.75, 1.), (1.5, -1.)] {
                rows.push(row(&chart.to_string(), rate, 0., false));
                stable.push(0.4);
                different.push(0.4 * sign);
            }
            by.push(vec![i, i + 1]);
            rates.push(vec![(0.75, vec![i]), (1.5, vec![i + 1])]);
        }
        let full = rate_transfer(&rows, &stable, &by, &rates, 0.);
        let partial = rate_transfer(&rows, &different, &by, &rates, 0.);
        let seven = |t: &Transfer| t.iter().find(|x| x.0 == 7).cloned().unwrap();
        assert!(seven(&full).1.is_infinite() && seven(&full).2 == 60);
        let (_, band, _, meta, _) = seven(&partial);
        let (train, all) = meta.unwrap();
        assert!(band < 0.2 && train < all);
    }

    #[test]
    fn direct_high_rate_evidence_is_not_three_pseudo_scores_from_nomod() {
        let anchors = rate_anchors(&[(1., -1.8, 3.), (1.5, -0.07, 1.)], 0.08, 0.35);
        assert!((anchors[1].1 - (-0.07 / 1.35)).abs() < 1e-5 && anchors[1].2 < 1.001 && anchors[0].1 < -0.5);
        // Nearby rates still share real support.
        let nearby = rate_anchors(&[(1.45, -1.8, 3.), (1.5, -0.07, 1.)], 0.08, 0.35);
        assert!(nearby[1].1 < anchors[1].1 && nearby[1].2 > anchors[1].2);
    }

    #[test]
    fn one_session_of_retries_is_one_affinity_block() {
        let rows: Vec<Row> = (0..3).map(|i| row("a", 1., 1000. + i as f64 * 130., false)).collect();
        let u = [0.2, 0.3, 0.4];
        let one = chart_blocks(&rows, &[0, 0, 0], 1, &u, &[Some((0, 7)); 3], 300.);
        assert_eq!(one[0].len(), 1);
        let (_, m, w, c) = one[0][0];
        assert!((m - 0.3).abs() < 1e-12 && w == 1. && c == 3.);
        let ids: Vec<_> = (0..3).map(|i| Some((i, 7))).collect();
        assert_eq!(chart_blocks(&rows, &[0, 0, 0], 1, &u, &ids, 300.)[0].len(), 3);
        assert!((effective_count(&[(1., 0.2, 0.1, 1.), (1., 0.3, 0.1, 1.), (1., 0.4, 0.1, 1.)]) - 3.).abs() < 1e-12);
    }

    fn retried(pattern: &[(bool, f64)]) -> f64 {
        let mut rows = Vec::new();
        let mut u = Vec::new();
        for i in 0..20 {
            for (j, &(rd, value)) in pattern.iter().enumerate() {
                rows.push(row(&format!("chart-{i}"), 1., 1000. + i as f64 * 1000. + j as f64 * 120., rd));
                u.push(value);
            }
        }
        retry(&rows, &u, 3000.)
    }

    #[test]
    fn retry_uses_only_normal_chart_attempt_pairs() {
        let expected = 0.4 * 20. / 70.;
        assert!((retried(&[(false, 0.4), (false, 0.)]) - expected).abs() < 1e-12);
        for rd in [(true, true), (false, true), (true, false)] {
            assert_eq!(retried(&[(rd.0, 0.4), (rd.1, 0.)]), 0.);
        }
        // An intervening RD attempt keeps the previous normal attempt as the anchor.
        assert!((retried(&[(false, 0.6), (true, 3.), (false, 0.2)]) - expected).abs() < 1e-12);
    }

    /// Plays of `charts` 7K charts at SV `sv`, each played `n` times `gap` months apart, residuals from `truth`.
    fn read_plays(truth: Reading, charts: usize, n: usize, gap: f64, sv: f64) -> Vec<ReadObs> {
        (0..charts).flat_map(|c| (0..n).map(move |i| {
            let sv = sv * (0.5 + (c % 5) as f64 / 4.); // SV 0.5–1.5× `sv`: the loss apart from `a`
            let before: Vec<f64> = (0..i).map(|j| (i - j) as f64 * gap).collect();
            let y = truth.deficit(7, sv, before.iter().map(|&m| truth.kept(m)).sum());
            ReadObs { y, keys: 7, sv, before, chart: c }
        })).collect()
    }

    #[test]
    fn a_player_without_sv_plays_reads_as_the_population() {
        let (r, known) = fit_reading(&[], &READ_PRIORS);
        assert_eq!((r, known.len()), (Reading::default(), 0));
        // plays without SV say nothing of the SV loss, learning or forgetting: those stay at the population's
        let (r, _) = fit_reading(&read_plays(Reading { a: 0.05, ..Reading::default() }, 60, 3, 1., 0.), &READ_PRIORS);
        assert_eq!((r.loss, r.learn, r.forget), (1., 1., 0.), "{r:?}");
        let mut f = Feats::bare(7, 10., 8., &Consts::standard());
        f.sk[Consts::standard().names.iter().position(|n| n == "sv").unwrap()] = 0.5;
        assert!((sv_rating(&f, &Consts::standard()) - 5.).abs() < 1e-12);
    }

    #[test]
    fn a_player_who_loses_more_on_first_reads_gets_a_larger_loss() {
        let (r, _) = fit_reading(&read_plays(Reading { loss: 4., ..Reading::default() }, 60, 3, 0.1, 6.), &READ_PRIORS);
        assert!(r.loss > 2.5, "{r:?}");
    }

    #[test]
    fn a_player_who_learns_faster_gets_a_faster_learning_speed() {
        let (r, _) = fit_reading(&read_plays(Reading { loss: 4., learn: 10., ..Reading::default() }, 60, 4, 0.1, 6.), &READ_PRIORS);
        assert!(r.learn >= 5., "{r:?}");
    }

    #[test]
    fn a_player_who_forgets_gets_a_forgetting_value() {
        // three plays a few days apart, then one two years later: forgetting brings the loss back on the last
        let plays = |truth: Reading| -> Vec<ReadObs> {
            let at = [0., 0.1, 0.2, 24.2];
            (0..60).flat_map(|c| (0..4).map(move |i| {
                let sv = 6. * (0.5 + (c % 5) as f64 / 4.);
                let before: Vec<f64> = (0..i).map(|j| at[i] - at[j]).collect();
                ReadObs { y: truth.deficit(7, sv, before.iter().map(|&m| truth.kept(m)).sum()), keys: 7, sv, before, chart: c }
            })).collect()
        };
        let (r, _) = fit_reading(&plays(Reading { a: 0., loss: 4., learn: 25., forget: 1. }), &READ_PRIORS);
        assert!(r.forget >= 0.5, "{r:?}");
        let (r, _) = fit_reading(&plays(Reading { a: 0., loss: 4., learn: 25., forget: 0. }), &READ_PRIORS);
        assert_eq!(r.forget, 0., "{r:?}");
    }

    #[test]
    fn a_first_play_that_reads_as_learned_counts_as_read_before() {
        let mut obs = read_plays(Reading { loss: 4., ..Reading::default() }, 60, 3, 0.1, 6.);
        obs.push(ReadObs { y: 0., keys: 7, sv: 8., before: Vec::new(), chart: 99 }); // SV 8, no loss at all
        let (_, known) = fit_reading(&obs, &READ_PRIORS);
        assert_eq!(known, vec![99]);
    }

    #[test]
    fn one_retried_family_cannot_validate_a_shape_effect() {
        let fit = fit_shape(&[1.; 60], 3, &[1.; 20], &[1.; 20], &["same"; 20]);
        assert!(fit.beta.iter().all(|&b| b == 0.) && fit.ridge.is_none());
    }
}

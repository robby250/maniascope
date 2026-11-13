//! Player ability in chart-difficulty units, using the existing personal model. A skill estimate is the difficulty giving
//! 96% displayed accuracy on a diverse set of that player's representative chart shapes. It is not a raw regression
//! coefficient, an average of personal bests, or a claimed population percentile. No chart analyses or model fits are
//! started by this page.
use crate::dans;
use crate::engine::{Engine, Now};
use crate::evidence::{personal_row, Evidence, FeatureSource};
use crate::history::play_history;
use crate::catalog::SUPPORTED_KEYS;
use ms_calc::py::round_digits;
use ms_calc::val::Dict;
use ms_data::store::{Db, Result};
use ms_model::personal::{self, Fit, FATIGUE_ONSET};
use ms_model::session::{WARMUP_MAX, WARMUP_MINUTES};
use ms_model::predict::{ChartSession, Feats, HashMap, KeyState, PredModel, PredSession};
use ms_model::skills;
use ms_model::warmup::Model;
use std::collections::BTreeMap;
use std::sync::Arc;

pub const TARGET: f64 = 0.96; // user 2026-11-10: "we should put this at like 96%" (was .94)
const FORM_SESSIONS: usize = 12;
const BINS: usize = 10;                 // rating deciles of the keymode's charts a ★ level reads
const HISTORY_BW: f64 = 1.;            // months: Gaussian smoothing of the monthly history (a peak month stays visible)
const VIBRO: &str = "jackspeed/vibro";

#[derive(Clone, Default)]
pub struct Group {
    pub value: Option<f64>,
    pub low: f64,
    pub high: f64,
    pub current: Option<f64>,
    pub slope: f64,
    pub maps: usize,
    pub plays: usize,
    pub newest: f64,
    pub confidence: &'static str,
    pub references: usize,
    pub tested_to: f64,
    pub extrapolated: bool,
    pub activation: f64,
    pub dan: Option<String>, // Vibro: its dan from the vibro dan songs (vibro tiers are in vibro speed, not ★)
    pub rusty: bool,         // the skill's charts unplayed long enough that rust shifts results past the noise (the level stays)
}

#[derive(Clone, Default)]
pub struct FormSession {
    pub t: f64,
    pub hours: f64,
    pub plays: Vec<(f64, f64, &'static str)>, // clock, value, mark
}

#[derive(Clone, Default)]
pub struct FormHistory {
    pub sessions: Vec<FormSession>,
    pub warmup: f64,
    pub fatigue: f64,
    pub warm_form: f64, // the shown sessions' warmed-up form, log miss against the model (+ = worse), shrunk by the fitted FormPrior
}

#[derive(Clone, Default)]
pub struct Page {
    pub keys: i64,
    pub groups: BTreeMap<String, Group>,
    pub order: Vec<String>,
    pub history: Vec<(i64, f64, f64)>, // month, level, plays that month
    pub now: f64,
    pub form: Option<FormHistory>,
    pub evidence: (usize, usize, f64),
    pub peak: Option<(i64, f64)>,
    pub maps: usize,
    pub plays: usize,
    pub mash_maps: usize,
    pub diagnostics: Vec<(String, f64, usize)>, // skill, delta, maps
    pub session: bool,
}

pub struct Profile {
    pub pages: BTreeMap<i64, Page>,
}

pub struct Current {
    pub target: f64,
    pub pages: BTreeMap<i64, Page>,
    pub phase: String,
}

/// Each tier's (log miss, songs) made non-decreasing up the ladder by pooling violators weighted by their songs, so every
/// dan song has a say and one song predicted easy can't carry a tier past harder ones (user 2026-11-11: "check expected acc
/// on all dan songs").
fn pooled(y: &[Option<(f64, f64)>]) -> Vec<Option<f64>> {
    let mut blocks: Vec<(f64, f64, Vec<usize>)> = Vec::new(); // (mean, weight, tiers)
    for (j, &(v, w)) in y.iter().enumerate().filter_map(|(j, v)| v.as_ref().map(|v| (j, v))) {
        blocks.push((v, w, vec![j]));
        while blocks.len() > 1 && blocks[blocks.len() - 2].0 > blocks[blocks.len() - 1].0 {
            let (v2, w2, j2) = blocks.pop().unwrap();
            let (v1, w1, j1) = blocks.last_mut().unwrap();
            *v1 = (*v1 * *w1 + v2 * w2) / (*w1 + w2);
            *w1 += w2;
            j1.extend(j2);
        }
    }
    let mut out = vec![None; y.len()];
    for (v, _, js) in blocks {
        js.into_iter().for_each(|j| out[j] = Some(v));
    }
    out
}

/// The fractional tier where the tiers' log(1 − acc) crosses `goal`: a tier below the easiest or above the hardest when
/// every known one is on the same side.
fn crossing(y: &[Option<f64>], goal: f64) -> Option<f64> {
    let known: Vec<(usize, f64)> = y.iter().enumerate().filter_map(|(j, v)| v.map(|v| (j, v))).collect();
    let Some(j) = known.iter().filter(|x| x.1 <= goal).map(|x| x.0).max() else { // none at or above the goal
        return known.first().map(|k| k.0 as f64 - 1.);
    };
    let Some(&(i, v)) = known.iter().find(|k| k.0 > j) else { return Some(j as f64 + 1.) };
    let at = y[j].unwrap();
    Some(j as f64 + (i - j) as f64 * (goal - at) / (v - at))
}

/// Fractional tier where per-tier (log miss vs mark, songs) crosses the mark. Past the group's own songs inside the ladder
/// the songs measure nothing: LN General has songs to 4th only (hard LN songs all read a specialty), so a Gamma player
/// read 6th from the low tiers' slope. Off the ladder's ends it stays (below 0th, Stellium+).
fn dan_crossing(y: &[Option<(f64, f64)>]) -> Option<f64> {
    (y.iter().flatten().count() >= 2).then_some(())?;
    let at = crossing(&pooled(y), 0.)?;
    let (first, last) = (y.iter().position(Option::is_some)?, y.iter().rposition(Option::is_some)?);
    (!(at > last as f64 && last + 1 < y.len()) && !(at < first as f64 && first > 0)).then_some(at)
}

/// One reading for every Stats row (plan 2026-11-11 P5): bins of charts in difficulty order, each bin's (log of its mean
/// predicted miss against its mark, charts, ln rating); where the pooled curve crosses the mark. → (ln rating there, the
/// curve's slope there in log miss per ln rating, the bins' spread around the curve in log miss). None off the ends.
fn reading(bins: &[Option<(f64, f64, f64)>]) -> Option<(f64, f64, f64)> {
    let pooled = pooled(&bins.iter().map(|b| b.map(|b| (b.0, b.1))).collect::<Vec<_>>());
    let at = crossing(&pooled, 0.)?;
    let known: Vec<(f64, f64, f64)> = bins.iter().zip(&pooled).filter_map(|(b, p)| Some((b.as_ref()?.2, (*p)?, b.as_ref()?.1))).collect();
    (known.len() >= 2 && at >= 0. && at <= (bins.len() - 1) as f64).then_some(())?;
    // the bracketing bins: the last at or under the mark and the next one over it
    let j = bins.iter().zip(&pooled).enumerate().filter(|(i, (b, p))| b.is_some() && p.is_some_and(|p| p <= 0.) && (*i as f64) <= at).map(|x| x.0).max()?;
    let i = (j + 1..bins.len()).find(|&i| bins[i].is_some()).unwrap_or(j);
    let (xj, pj) = (bins[j]?.2, pooled[j]?);
    let (x, slope) = if i == j { (xj, 0.) } else {
        let (xi, pi) = (bins[i]?.2, pooled[i]?);
        let t = if pi > pj { (0. - pj) / (pi - pj) } else { 0. };
        (xj + t * (xi - xj), if xi > xj { (pi - pj) / (xi - xj) } else { 0. })
    };
    let w: f64 = known.iter().map(|k| k.2).sum();
    let spread = (bins.iter().zip(&pooled).filter_map(|(b, p)| Some(b.as_ref()?.1 * (b.as_ref()?.0 - (*p)?).powi(2))).sum::<f64>() / w).sqrt();
    Some((x, slope, spread))
}

/// ★ bins: charts (ln rating, predicted miss) in BINS equal-count rating quantiles, each against TARGET.
fn rating_bins(mut charts: Vec<(f64, f64)>) -> Vec<Option<(f64, f64, f64)>> {
    charts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = charts.len();
    let bins = BINS.min(n);
    (0..bins).map(|b| {
        let part = &charts[b * n / bins..(b + 1) * n / bins];
        let m = part.len() as f64;
        (m > 0.).then(|| ((part.iter().map(|c| c.1).sum::<f64>() / m).max(1e-6).ln() - (1. - TARGET).ln(), m, part.iter().map(|c| c.0).sum::<f64>() / m))
    }).collect()
}

struct Ctx<'a> {
    model: &'a PredModel,
    fit: &'a Fit,
    warmup: &'a Arc<Model>,
    rec: &'a Engine,
    dicts: HashMap<(String, u64), Arc<Dict>>,
    shown: (f64, f64), // displayed log miss = a + b · the model's
}

impl Ctx<'_> {
    fn feats(&self, r: &Evidence) -> &Feats {
        &self.rec.r.fitted.rows.feats[r.f]
    }

    fn dict(&self, r: &Evidence) -> Option<&Arc<Dict>> {
        self.dicts.get(&(r.chart.clone(), r.rate.to_bits()))
    }

    /// Predicted lazer miss of a chart as the playlist predicts an unseen one: rested, warm, neutral form, no rust, practised
    /// once (the model's retry gain). `id` never matches a played chart, so own scores on it get no say.
    fn miss(&self, f: &Feats, id: &str, rate: f64) -> Option<f64> {
        let families = self.rec.c.families.len();
        let neutral = KeyState { form: vec![0.; families + 1], penalty: vec![0.; families], session: 0., session_tau: 1., fatigue: 0., activation: vec![10.; families], warmth: 10., rust: 0. };
        let mut session = PredSession { warm: true, by_keys: HashMap::default(), charts: HashMap::default() };
        session.by_keys.insert(f.keys, neutral);
        session.charts.insert(id.to_string(), ChartSession { retried: true, ..ChartSession::default() });
        let miss = (1. - self.model.predict(f, id, None, rate, &session).ok()?.acc_mid).max(1e-6);
        Some((self.shown.0 + self.shown.1 * miss.ln()).exp())
    }

    /// Rating on the series' dan ladder where the player's predicted accuracy on the group's dan songs crosses the series'
    /// pass mark (user 2026-11-11: "estimated acc on each dan course/map", "each dan's acc target"). Overall reads every
    /// song of a tier (the course's accuracy), a skill only the songs demanding it. Predicted as the playlist predicts an
    /// unseen chart, rested, warm and in neutral form, practised once (the model's retry gain): own scores on dan charts
    /// get no say, old ones would drag it (user 2026-11-11). None below two tiers with such songs.
    fn dan_value(&self, series: &str, group: &str, miss_ratio: f64) -> Option<f64> {
        let at = self.dan_position(series, group, miss_ratio)?;
        // Off the ladder the songs measure nothing: the own-charts reading stays, and the dan reads "below 0th" or "Stellium+".
        (-0.5..=dans::tiers(series)?.len() as f64 - 0.5).contains(&at).then_some(())?;
        dans::rating_at(series, at)
    }

    /// The fractional tier where the player's predicted accuracy on the series' songs for `group` crosses each tier's pass mark.
    fn dan_position(&self, series: &str, group: &str, miss_ratio: f64) -> Option<f64> {
        let tiers = dans::tiers(series)?;
        // Each tier's own pass mark, y = log miss against it (crossing at 0). A stable mark judges holds as stable does: each
        // song's lazer miss × miss_ratio × e^(STABLE_LN_Y × LN share), as the playlist shows a stable client.
        let pass = |tier: &str| dans::pass(series, tier);
        let songs = dans::songs(series, &self.rec.c);
        let mut y = Vec::new();
        for (tier, _) in tiers {
            let picked: Vec<(String, &Feats)> = songs.get(tier).into_iter().flatten().enumerate()
                .filter(|(_, (_, skills))| group == "overall" || skills.contains(&group))
                .map(|(i, (f, _))| (format!("dan:{series}:{tier}:{i}"), f)).collect();
            let Some((mark, stable)) = pass(tier) else { y.push(None); continue };
            let misses: Vec<f64> = picked.iter().filter_map(|(id, f)| {
                let lazer = self.miss(f, id, 1.)?;
                Some(if stable { lazer * miss_ratio * (crate::evidence::STABLE_LN_Y * f.z_tail[0]).exp() } else { lazer })
            }).collect();
            y.push((!misses.is_empty()).then(|| ((misses.iter().sum::<f64>() / misses.len() as f64).max(1e-6).ln() - (1. - mark).ln(), misses.len() as f64)));
        }
        dan_crossing(&y)
    }

    /// [(month, ability)]: today's overall ability moved by how much better or worse each month's recorded plays went
    /// against their predictions. Lazer keeps every play; website/stable top-100s keep only bests and are left out, so a
    /// new install shows little or no history. Months with less than 20 (smoothed) plays are left out.
    fn history(&self, keys: i64, overall: Option<&Group>) -> Vec<(i64, f64, f64)> {
        const SUPPORT: f64 = 20.;
        let (Some(g), Some((start, se, sw, count))) = (overall.filter(|g| g.value.is_some()), self.fit.monthly.get(&keys)) else { return Vec::new() };
        let n = sw.len();
        let smooth = |i: usize, v: &[f64]| (0..n).map(|j| (-0.5 * ((i as f64 - j as f64) / HISTORY_BW).powi(2)).exp() * v[j]).sum::<f64>();
        let months: Vec<usize> = (0..n).filter(|&i| smooth(i, count) >= SUPPORT && sw[i] > 0.).collect(); // plays, not their weights
        let Some(&last) = months.last() else { return Vec::new() };
        let level = |i: usize| smooth(i, se) / smooth(i, sw);
        let now = level(last); // the latest played month is today's ability
        months.into_iter().map(|i| (start + i as i64, g.value.unwrap() * ((now - level(i)) / g.slope).exp(), count[i])).collect()
    }

    /// The last sessions play by play: accuracy points against the model's rested expectation (at 95 %), each play marked
    /// warm-up, peak, bad or fading; the fitted warm-up cost and fatigue per hour beside them.
    fn form_history(&self, keys: i64, local: &[&Evidence]) -> Option<FormHistory> {
        if !self.fit.keys.contains(&keys) {
            return None;
        }
        let c = &self.rec.c;
        let rows: Vec<personal::Row> = local.iter().map(|r| personal_row(r, r.y, 0.)).collect();
        let transfer = crate::recommender::family_transfer(c);
        let steps = personal::steps(c, &rows, &self.rec.r.fitted.rows.feats, &transfer, personal::GAP, personal::COOLING);
        let mut ids: Vec<i64> = steps.iter().map(|s| s.session).collect();
        ids.sort_unstable();
        ids.dedup();
        let keep: std::collections::HashSet<i64> = ids[ids.len().saturating_sub(FORM_SESSIONS)..].iter().copied().collect();
        let points = |shift: f64| 100. * 0.05 * (1. - shift.clamp(-3., 3.).exp());
        let p = self.warmup.parameters(keys);
        let unit = if p.log_rate { 3. } else { 1. };
        let mut sessions: Vec<(i64, Vec<&personal::Step>)> = Vec::new();
        for s in steps.iter().filter(|s| keep.contains(&s.session)) {
            match sessions.iter_mut().find(|x| x.0 == s.session) {
                Some(x) => x.1.push(s),
                None => sessions.push((s.session, vec![s])),
            }
        }
        if sessions.is_empty() {
            return None;
        }
        let mut warm: Vec<f64> = Vec::new();
        let out = sessions.into_iter().map(|(_, plays)| {
            let at = |s: &personal::Step| local[s.row];
            let first = at(plays[0]).ts.unwrap_or(0.) - plays[0].seconds;
            // Warm-up = the live phase's rule. The fitted readiness loss can't mark it: with 10-minute cooling it stays
            // above any threshold all session, so every play read as warm-up (user 2026-11-02).
            let cold = warm_up_marks(plays.iter().map(|s| s.seconds));
            let values: Vec<(f64, f64, bool)> = plays.iter().zip(cold).map(|(s, cold)| {
                let r = at(s);
                let m = self.model.personal(self.feats(r), &r.chart, r.b, round_digits(r.rate, 3)).0;
                let clock = (r.ts.unwrap_or(0.) - s.seconds - first) / 60.;
                if !cold {
                    warm.push((r.y_lazer - m).clamp(-3., 3.));
                }
                (clock, points(r.y_lazer - m), cold)
            }).collect();
            let n = values.len() as f64;
            let centre = values.iter().map(|v| v.1).sum::<f64>() / n;
            let spread = 0.3f64.max((values.iter().map(|v| (v.1 - centre).powi(2)).sum::<f64>() / n).sqrt());
            let marked: Vec<(f64, f64, &'static str)> = values.iter().map(|&(clock, value, cold)| {
                let mark = if cold { "warm-up" } else if value < centre - spread { "bad" } else if clock >= FATIGUE_ONSET && value < centre { "fading" } else if value > centre + spread { "peak" } else { "" };
                (round_digits(clock, 1), round_digits(value, 2), mark)
            }).collect();
            FormSession { t: first, hours: marked[marked.len() - 1].0 / 60., plays: marked }
        }).collect();
        let mean = p.amplitudes.iter().sum::<f64>() / p.amplitudes.len() as f64;
        // Stats shows the warmed-up level you played at recently (user 2026-11-11): the warm plays of the sessions the Form chart
        // shows, shrunk as the session model shrinks form (FormPrior: the keymode's form variance against a play's noise).
        let prior = &self.warmup.prior;
        let tau = prior.by_key.get(&keys).map_or(prior.global_variance, |b| b.0);
        let n = warm.len() as f64;
        let warm_form = if n == 0. { 0. } else { warm.iter().sum::<f64>() / (n + prior.noise_variance / tau.max(1e-9)) };
        Some(FormHistory { sessions: out, warmup: -points(unit * (p.session + mean)), fatigue: -points(unit * p.fatigue), warm_form })
    }
}

pub fn build(rec: &Engine, db: &Db) -> Result<Profile> {
    let f = &rec.r.fitted;
    let target = f.display.as_ref().unwrap_or(&f.pp);
    // Without a displayed-accuracy model the pp model's log miss maps to the displayed one by the fitted line.
    let shown = if f.display.is_some() { (0., 1.) } else { f.shown };
    let usable: Vec<&Evidence> = f.rows.rows.iter().filter(|r| SUPPORTED_KEYS.contains(&r.keys) && !r.rd && f.rows.feats[r.f].overall > 0.1).collect();
    // The representative charts' features, once: Feats keep no display fields.
    let mut source = FeatureSource { public: rec.public.clone(), calc: rec.source.calc.clone(), cache: HashMap::default() };
    let mut dicts: HashMap<(String, u64), Arc<Dict>> = HashMap::default();
    source.prefetch(db, usable.iter().filter_map(|r| Some(source_key(r.sha.as_deref()?, r.rate))))?;
    for r in &usable {
        let id = (r.chart.clone(), r.rate.to_bits());
        if !dicts.contains_key(&id) {
            let md5 = (r.chart.len() == 32).then_some(r.chart.as_str());
            if let (Some(d), _) = source.of(db, r.sha.as_deref(), md5, r.b, Some(r.rate))? {
                dicts.insert(id, d);
            }
        }
    }
    let ctx = Ctx { model: &target.model, fit: &target.fit, warmup: &target.warmup, rec, dicts, shown };
    let miss_ratio = ms_data::store::stable_miss_ratio(db)?;
    let month = crate::evidence::now_months();
    let mut pages = BTreeMap::new();
    // Vibro charts are mashed: only the Vibro skill reads them (Consts::vibro_chart).
    let vibro = |r: &Evidence| rec.c.vibro_chart(&f.rows.feats[r.f]);
    let of_keys = |keys: i64| usable.iter().filter(|r| r.keys == keys && !vibro(r)).copied().collect::<Vec<&Evidence>>();
    let vibro_of = |keys: i64| usable.iter().filter(|r| r.keys == keys && vibro(r)).copied().collect::<Vec<&Evidence>>();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0., |d| d.as_secs_f64());
    for keys in SUPPORTED_KEYS {
        let rows = of_keys(keys);
        let known = ctx.fit.keys.contains(&keys);
        // The keymode's analysed charts, once per chart and rate: predicted miss and rating, whatever was scored on them.
        let mut charts: HashMap<(&str, u64), &Evidence> = HashMap::default();
        let vibro_rows = vibro_of(keys);
        for &row in rows.iter().chain(&vibro_rows) {
            charts.entry((row.chart.as_str(), ms_calc::py::round_digits(row.rate, 2).to_bits())).or_insert(row);
        }
        let predicted: HashMap<(&str, u64), (f64, f64)> = if known {
            charts.iter().filter_map(|(k, r)| Some((*k, (ctx.feats(r).overall.ln(), ctx.miss(ctx.feats(r), &format!("stats:{}", r.chart), ms_calc::py::round_digits(r.rate, 2))?)))).collect()
        } else { HashMap::default() };
        let mut names = vec!["overall".to_string()];
        names.extend(skills::dimensions(keys).iter().map(|n| n.key.clone()));
        let mut groups: BTreeMap<String, Group> = BTreeMap::new();
        let shown = |r: &Evidence| ctx.dict(r).map_or(0., |d| d.f("overall"));
        for group in &names {
            // Vibro charts are mashed: only Vibro reads them; every other row reads the charts counted for it (P3's rank).
            let member = |r: &Evidence| if group == VIBRO { rec.c.vibro_chart(ctx.feats(r)) } else {
                !rec.c.vibro_chart(ctx.feats(r)) && (group == "overall" || ctx.dict(r).is_some_and(|d| skills::demanded(d).contains(&group.as_str())))
            };
            let relevant: Vec<(&(&str, u64), &&Evidence)> = charts.iter().filter(|(_, r)| member(r)).collect();
            let played: Vec<&&Evidence> = rows.iter().chain(&vibro_rows).filter(|r| r.src != "public" && member(r)).collect();
            let newest = played.iter().map(|r| r.ts.unwrap_or(0.)).fold(0., f64::max);
            let count = relevant.len();
            let read = reading(&rating_bins(relevant.iter().filter_map(|(k, _)| predicted.get(*k).copied()).collect()));
            let rust = ctx.warmup.rust.of_plays(keys, &played.iter().filter(|r| r.completed).filter_map(|r| Some((r.ts?, ctx.feats(r).length / 1000. / r.rate.max(0.01) / 60.))).collect::<Vec<_>>(), now);
            let mut g = Group { maps: count, plays: played.len(), newest, references: count.min(BINS), rusty: rust > ms_model::session::FORM_NOTICE,
                                tested_to: relevant.iter().map(|(_, r)| shown(r)).fold(0., f64::max), ..Group::default() };
            match read {
                Some((x, slope, spread)) if slope > 0. => {
                    let value = x.exp();
                    (g.value, g.slope, g.low, g.high) = (Some(value), slope, value * (-spread / slope).exp(), value * (spread / slope).exp());
                    g.confidence = if played.len() >= 40 { "Strong history" } else if played.len() >= 10 { "Moderate history" } else { "Limited history" };
                }
                _ => {
                    // Off the charts' range: every chart predicted under (or over) the mark measures no limit.
                    g.extrapolated = count > 0 && known;
                    g.confidence = if !known || count == 0 { "Not enough evidence" } else { "Limit not measured" };
                }
            }
            groups.insert(group.clone(), g);
        }
        // One source for a skill's ★ and its dan: the dan songs' reading replaces the charts' where it exists (the same
        // estimator on the series' tiers and pass marks).
        let overall_slope = groups.get("overall").map_or(1., |g| if g.slope > 0. { g.slope } else { 1. });
        for (name, g) in groups.iter_mut().filter(|(n, _)| known && !n.ends_with("vibro")) {
            let Some(series) = dans::series(keys, dans::ability_kind(name)) else { continue };
            let Some(v) = ctx.dan_value(series, name, miss_ratio) else { continue };
            (g.low, g.high) = match g.value { Some(old) => (g.low * v / old, g.high * v / old), None => (v, v) };
            if g.value.is_none() {
                (g.confidence, g.slope, g.extrapolated) = ("From the dan songs", overall_slope, false);
            }
            g.value = Some(v);
        }
        if let Some(g) = groups.get_mut(VIBRO).filter(|g| g.value.is_some()) {
            g.dan = ctx.dan_position("4K Vibro", "overall", miss_ratio).and_then(|at| dans::named("4K Vibro", at)).filter(|_| keys == 4);
        }
        let local: Vec<&Evidence> = rows.iter().filter(|r| r.src != "public").copied().collect();
        let form = ctx.form_history(keys, &local);
        if let Some(f) = form.as_ref().filter(|f| f.warm_form != 0.) {
            for (_, g) in groups.iter_mut().filter(|(n, g)| g.value.is_some() && g.slope > 0. && n.as_str() != VIBRO) {
                let k = (-f.warm_form / g.slope).exp();
                (g.value, g.low, g.high) = (g.value.map(|v| v * k), g.low * k, g.high * k);
            }
        }
        let history = ctx.history(keys, groups.get("overall"));
        pages.insert(keys, Page { keys, order: names, history: history.clone(), now: month, form, evidence: (local.len(), rows.len() - local.len(), rows.iter().map(|r| r.ts.unwrap_or(0.)).fold(0., f64::max)),
                                  peak: history.iter().map(|&(m, v, _)| (m, v)).max_by(|a, b| a.1.total_cmp(&b.1)), maps: charts.len(), plays: rows.len(),
                                  mash_maps: charts.values().filter(|r| ctx.dict(r).is_some_and(|d| skills::relative(d, d.dict("sk").map_or(0., |sk| sk.fget("mash", 0.))) >= skills::cut())).count(), groups, ..Page::default() });
    }
    // Only frozen pre-play predictions can support a model-error diagnosis. Collapse retries by exact chart, and require
    // several charts before a label.
    let mut residuals: HashMap<(i64, String), HashMap<String, Vec<f64>>> = HashMap::default();
    for row in play_history(db, 300, false, None)? {
        let (Some(expected), Some(keys), Some(features)) = (row.expected, row.keys, row.prediction.get("features").filter(|f| f.is_object())) else { continue };
        if !pages.contains_key(&keys) {
            continue;
        }
        let ms_calc::val::Val::Dict(f) = ms_data::codec::from_json(features) else { continue };
        for group in skills::matched_families(&f) {
            residuals.entry((keys, group.to_string())).or_default().entry(row.sha256.clone().unwrap_or_default()).or_default().push(100. * (row.actual - expected));
        }
    }
    for (keys, page) in &mut pages {
        let mut found: Vec<(String, f64, usize)> = Vec::new();
        for ((k, group), charts) in residuals.iter().filter(|x| x.0 .0 == *keys && x.1.len() >= 4) {
            let _ = k;
            let errors: Vec<f64> = charts.values().map(|e| e.iter().sum::<f64>() / e.len() as f64).collect();
            let n = errors.len() as f64;
            let delta = errors.iter().sum::<f64>() / n;
            let se = (errors.iter().map(|e| (e - delta).powi(2)).sum::<f64>() / n).sqrt() / n.sqrt();
            if delta.abs() > 0.75f64.max(1.5 * se) {
                found.push((group.clone(), delta, charts.len()));
            }
        }
        found.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        found.truncate(3);
        page.diagnostics = found;
    }
    Ok(Profile { pages })
}

fn source_key(sha: &str, rate: f64) -> String {
    format!("{sha}@{:.3}|{}", round_digits(rate, 3), crate::engine::CALC_ID)
}

/// Cheap session overlay. This never alters the persistent skill estimate.
pub fn current(profile: &Profile, n: &Now, rec: &Engine) -> Current {
    let f = &rec.r.fitted;
    let warmup = &f.display.as_ref().unwrap_or(&f.pp).warmup;
    let view = &n.view;
    let mut pages = BTreeMap::new();
    for (keys, old) in &profile.pages {
        let keys = *keys;
        let mut page = old.clone();
        let shown_form = page.form.as_ref().map_or(0., |f| f.warm_form); // already in the level: the session moves it from there
        for (group, row) in &mut page.groups {
            let skill = if group == "overall" { None } else { skills::session_skill(group, keys) };
            let family = skill.map(|s| rec.c.family(s));
            row.activation = view.activation(keys, family);
            if let Some(value) = row.value {
                let cold = if view.warm_flag { 0. } else { warmup.penalty(&rec.c, keys, skill, row.activation, 0., None, view.session_clock()) };
                let correction = view.correction(keys, skill) + cold - if group == VIBRO { 0. } else { shown_form };
                row.current = Some(value * (-correction / row.slope).exp());
            }
        }
        let shown = |v: f64| v;
        page.history = old.history.iter().map(|&(m, v, n)| (m, shown(v), n)).collect();
        page.peak = old.peak.map(|(m, v)| (m, shown(v)));
        // Before a session is warm the session marker only shows the cold start, not the player's form.
        page.session = view.activation(keys, None) >= 1.5;
        pages.insert(keys, page);
    }
    Current { target: TARGET, pages, phase: n.session.phase().to_string() }
}

/// Per play of a session (map lengths in seconds): still warming up, i.e. fewer than WARMUP_MAX maps and WARMUP_MINUTES
/// played before it.
fn warm_up_marks(seconds: impl Iterator<Item = f64>) -> Vec<bool> {
    let mut played = 0.;
    seconds.enumerate().map(|(i, s)| {
        let cold = i < WARMUP_MAX && played < WARMUP_MINUTES;
        played += s / 60.;
        cold
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::{rating_bins, reading, warm_up_marks, TARGET};

    #[test]
    fn a_session_warms_after_three_maps_or_eight_minutes() {
        assert_eq!(warm_up_marks([120., 120., 120., 120., 120.].into_iter()), [true, true, true, false, false]);
        assert_eq!(warm_up_marks([300., 300., 120.].into_iter()), [true, true, false]);
    }
    use super::{crossing, dan_crossing, pooled};

    #[test]
    fn a_level_is_where_the_pooled_rating_bins_cross_the_target() {
        // 200 charts from ★ 4 to 12, predicted log miss rising 2.5 per ln ★ through the target at ★ 8.
        let charts: Vec<(f64, f64)> = (0..200).map(|i| {
            let x = (4. + 8. * i as f64 / 199.).ln();
            (x, (1. - TARGET) * (2.5 * (x - 8f64.ln())).exp())
        }).collect();
        let (x, slope, spread) = reading(&rating_bins(charts.clone())).unwrap();
        assert!((x.exp() - 8.).abs() < 0.15 && (slope - 2.5).abs() < 0.05 && spread < 0.05, "{} {slope} {spread}", x.exp());
        // One easy chart predicted hard pools with its neighbours instead of moving the level much.
        let mut odd = charts.clone();
        odd[10].1 = 0.5;
        assert!((reading(&rating_bins(odd)).unwrap().0.exp() - 8.).abs() < 0.4);
        // Every chart under the mark: no limit measured.
        assert!(reading(&rating_bins(charts.iter().map(|c| (c.0, 0.001)).collect())).is_none());
    }

    #[test]
    fn a_dan_reading_needs_songs_on_both_sides_of_the_mark() {
        let under = |v: f64| Some((v, 2.));
        // songs to 4th of 15 tiers, all passed: unmeasured, not "one tier past the last song"
        let mut y = vec![under(-1.9), under(-1.8), under(-1.5), under(-1.3), under(-1.1)];
        y.resize(15, None);
        assert_eq!(dan_crossing(&y), None);
        // every tier passed: Stellium+ stays readable (off the top)
        let all: Vec<_> = (0..15).map(|i| under(-2. + 0.1 * i as f64)).collect();
        assert_eq!(dan_crossing(&all), Some(15.));
        // a bracketed crossing reads as before
        let mut mid = vec![under(-1.), under(-0.5), under(0.5)];
        mid.resize(15, None);
        assert!((dan_crossing(&mid).unwrap() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn the_dan_is_where_predicted_accuracy_crosses_the_target() {
        let y = |x: f64| x.ln();
        // 94 % at log(1 − acc) = ln .06: tier 1 is easier, tier 2 harder, linear between them.
        let goal = y(0.06);
        let at = crossing(&[Some(y(0.02)), Some(y(0.04)), Some(y(0.08))], goal).unwrap();
        assert!((at - (1. + (goal - y(0.04)) / (y(0.08) - y(0.04)))).abs() < 1e-12);
        assert_eq!(crossing(&[Some(y(0.01)), Some(y(0.02)), Some(y(0.03))], goal), Some(3.)); // beyond the top
        assert_eq!(crossing(&[Some(y(0.1)), Some(y(0.2))], goal), Some(-1.)); // below the first
        assert_eq!(crossing(&[None, Some(y(0.02)), None, Some(y(0.08))], goal), Some(1. + 2. * (goal - y(0.02)) / (y(0.08) - y(0.02))));
        assert_eq!(crossing(&[None, None], goal), None);
        // A tier predicted easier than the one below pools with it by song count; a gap stays a gap.
        assert_eq!(pooled(&[Some((-3., 2.)), None, Some((-4.5, 1.)), Some((-1., 1.))]), vec![Some(-3.5), None, Some(-3.5), Some(-1.)]);
    }
}

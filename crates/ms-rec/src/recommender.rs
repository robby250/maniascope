//! recommend.Recommender's fit: both accuracy targets on the historical rows.
use crate::evidence::{evidence_rows, now_months, personal_row, play_levels, Evidence, FeatureSource, Rows};
use ms_data::events::load_events;
use ms_data::public::Public;
use ms_data::store::{acc320, acc_lazer, kv_get, now, parse_stats, Db, Result};
use ms_model::personal::{personal_fit, transfer, Fit, Knobs, WarmupArgs};
use ms_model::predict::{Consts, HashMap, PredModel};
use ms_model::public::{population, slopes};
use ms_model::session::{Session, HISTORY_S, SESSION_GAP};
use ms_model::warmup::{FormPrior, Model};
use std::sync::Arc;

pub const KNOBS: Knobs = Knobs { half_life: 12., level_bw: 4., history_zone: 0.15, recent_months: 2., att_sd: 1., session_gap: SESSION_GAP };

pub struct Target {
    pub fit: Fit,
    pub warmup: Arc<Model>,
    pub model: Arc<PredModel>,
}

pub struct Fitted {
    pub rows: Rows,
    pub levels: HashMap<i64, f64>,
    pub pp: Target,
    pub display: Option<Target>,
    pub shown: (f64, f64),
}

/// {target: {played: share}} between warm-up families.
pub fn family_transfer(c: &Consts) -> Vec<Vec<f64>> {
    c.families.iter().map(|a| c.families.iter().map(|b| transfer(a, b)).collect()).collect()
}

/// log(1 − shown) = a + b·log(1 − acc320), fit on the player's own judgement mix: the model works in acc320, the
/// player reads lazer's accuracy.
pub fn fit_shown(db: &Db, until: f64) -> Result<(f64, f64)> {
    let mut query = "SELECT stats FROM scores WHERE src IN ('realm', 'live') AND stats IS NOT NULL".to_string();
    if until.is_finite() {
        query += &format!(" AND julianday(played)<julianday({until:?},'unixepoch')");
    }
    let mut statement = db.prepare(&query)?;
    let (mut xs, mut zs) = (Vec::new(), Vec::new());
    for s in statement.query_map([], |r| r.get::<_, String>(0))? {
        let Some(s) = parse_stats(&s?) else { continue };
        let ((a, n), al) = (acc320(&s), acc_lazer(&s));
        if n != 0 && 0. < a && a < 1. && 0. < al && al < 1. {
            xs.push((1. - a).ln());
            zs.push((1. - al).ln());
        }
    }
    if xs.len() < 20 {
        return Ok((0.75f64.ln(), 1.));
    }
    let n = xs.len() as f64;
    let (mx, mz) = (xs.iter().sum::<f64>() / n, zs.iter().sum::<f64>() / n);
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let sxz: f64 = xs.iter().zip(&zs).map(|(x, z)| (x - mx) * (z - mz)).sum();
    let b = sxz / sxx;
    Ok((mz - b * mx, b))
}

/// Vibro plays needed for their own fit: fewer leave vibro charts on the population's prediction.
const VIBRO_ROWS: usize = 10;

#[allow(clippy::too_many_arguments)]
pub fn fit_target(c: &Arc<Consts>, public: &Public, pop: &ms_data::public::Population, rows: &[&Evidence], feats: &[ms_model::predict::Feats],
              levels: &HashMap<i64, f64>, display: bool, month: f64, shown: (f64, f64), judged_ln: f64, reads: &ms_model::personal::Reads) -> Target {
    let native = Arc::new(population(pop, c));
    let base_model = PredModel::population(c.clone(), native.clone());
    let personal = |rows: &[&&Evidence]| -> Vec<_> { rows.iter().map(|r| {
        let (base, _) = base_model.base_of(&feats[r.f], r.b, ms_calc::py::round_digits(r.rate, 3), levels.get(&r.keys).copied());
        personal_row(r, if display { r.y_lazer } else { r.y }, base)
    }).collect() };
    // Vibro charts are mashed: kept out of the ability fit, read by a fit of their own (PredModel::vibro).
    let (vibro, ability): (Vec<&&Evidence>, Vec<&&Evidence>) = rows.iter().partition(|r| c.vibro_chart(&feats[r.f]));
    let args = WarmupArgs { transfer: family_transfer(c), slopes: slopes(pop, &public.meta, c) };
    let fit = personal_fit(c, personal(&ability), feats, month, levels, KNOBS, &args, reads);
    let own = (vibro.len() >= VIBRO_ROWS).then(|| {
        let mut own = personal_fit(c, personal(&vibro), feats, month, levels, KNOBS, &args, reads);
        borrow_levels(&mut own, &fit);
        own
    });
    let context = &fit.session_validation;
    let prior = FormPrior {
        global_variance: context.condition_variance.max(1e-4),
        skill_variance: context.skill_condition_variance.max(1e-4),
        by_key: context.by_key.iter().map(|(k, b)| (*k, (b.global_variance, b.skill_variances.clone()))).collect(),
        noise_variance: (fit.sd_all * fit.sd_all).max(1e-4),
    };
    let warmup = Arc::new(Model::new(&fit.warmup, args.slopes, prior, c.families.len()));
    let mut model = PredModel::fitted(c.clone(), native.clone(), levels.clone(), &fit, &warmup, shown, judged_ln);
    if let Some(own) = own {
        let own = PredModel::fitted(c.clone(), native, levels.clone(), &own, &warmup, shown, judged_ln);
        // the rate response and rate loss read a vibro chart's memory by its id
        model.aff.extend(own.aff.clone());
        model.rate_aff.extend(own.rate_aff.clone());
        model.rate_support.extend(own.rate_support.clone());
        model.vibro = Some(Arc::new(own));
    }
    model.vibro_pace = ms_model::predict::vibro_pace(&vibro.iter().map(|r| (r.ts.unwrap_or(0.), feats[r.f].vibro_speed)).collect::<Vec<_>>());
    Target { fit, warmup, model: Arc::new(model) }
}

/// A keymode with too few vibro plays for its own level reads the ability fit's level there, moved by the vibro fit's mean
/// gap on the keymodes that have both: the fallback `glob` mixes every keymode's vibro charts, whose bases differ by keymode
/// (user 2026-11-14: a 10K vibro convert at 1.29× predicted 99.99 % from 4K/7K vibro plays; 72 % played).
fn borrow_levels(own: &mut Fit, ability: &Fit) {
    let shared: Vec<i64> = own.level.keys().filter(|k| ability.level.contains_key(k)).copied().collect();
    let weight = |k: &i64| 1. / own.level_var[k].max(1e-6);
    let mass: f64 = shared.iter().map(weight).sum();
    let gap = if mass > 0. { shared.iter().map(|k| weight(k) * (own.level[k] - ability.level[k])).sum::<f64>() / mass } else { 0. };
    for (&k, &l) in &ability.level {
        if let std::collections::hash_map::Entry::Vacant(e) = own.level.entry(k) {
            e.insert(l + gap);
            own.level_var.insert(k, ability.level_var[&k] + if mass > 0. { 1. / mass } else { 0.25 });
            if let Some(&sd) = ability.sd_att.get(&k) {
                own.sd_att.insert(k, sd);
            }
        }
    }
}

/// Recommender._fit's models. The current session's plays are left out: form reads them live.
pub fn fit(db: &Db, public: &Arc<Public>, calc: &str, c: &Arc<Consts>) -> Result<Fitted> {
    let mut source = FeatureSource { public: public.clone(), calc: calc.to_string(), cache: HashMap::default() };
    let rows = evidence_rows(db, &mut source, c)?;
    let current = now();
    let session = Session::new(load_events(db, ((current - HISTORY_S) / 60.).floor() * 60.)?, current, None, c.clone(), Arc::new(family_transfer(c)));
    let begin = session.events.iter().filter(|e| e.kind == "start").map(|e| e.t).fold(f64::INFINITY, f64::min);
    let month = now_months();
    let historical: Vec<&Evidence> = rows.rows.iter().filter(|r| r.ts.is_none_or(|t| t == 0. || t < begin)).collect();
    let levels = play_levels(&historical.iter().filter(|r| !c.vibro_chart(&rows.feats[r.f])).copied().collect::<Vec<_>>(), month);
    let shown = fit_shown(db, begin)?;
    // reads before this session: the session's own attempts are read live (`ChartSession::reads`)
    let reads: ms_model::personal::Reads = rows.reads.iter().map(|(k, v)| (k.clone(), v.iter().copied().filter(|x| x.1 < begin).collect())).collect();
    // Evidence is in lazer units; a stable player's forecasts are what stable judges (more lost per hold).
    let stable = kv_get(db, "client").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| crate::history::default_client().to_string()) == "stable";
    let judged = if stable { crate::evidence::STABLE_LN_Y } else { 0. };
    let pp = fit_target(c, public, &public.meta.pop, &historical, &rows.feats, &levels, false, month, shown, judged, &reads);
    let display = public.meta.pop_lazer.as_ref().map(|pop| fit_target(c, public, pop, &historical, &rows.feats, &levels, true, month, (0., 1.), judged, &reads));
    Ok(Fitted { rows, levels, pp, display, shown })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_keymode_without_vibro_plays_reads_its_own_level_moved_by_the_vibro_gap() {
        let fit = |levels: &[(i64, f64)]| {
            let mut f = Fit::default();
            for &(k, l) in levels {
                f.level.insert(k, l);
                f.level_var.insert(k, 0.01);
            }
            f
        };
        let ability = fit(&[(4, -3.), (7, -3.5), (10, -2.9)]);
        let mut vibro = fit(&[(4, -9.), (7, -9.6)]); // 4K/7K vibro bases sit far above their charts' play
        borrow_levels(&mut vibro, &ability);
        assert!((vibro.level[&10] - (-2.9 - 6.05)).abs() < 1e-9, "10K = its ability level + the mean gap, not another keymode's");
        assert_eq!(vibro.level[&4], -9.);
    }
}

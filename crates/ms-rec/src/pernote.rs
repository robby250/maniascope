//! The per-note forecast in the app (plan 2026-11-13b S7). The net (`calib/notenet.json`: `notefit --all-songs --save` on the
//! developer's replays, plus the ability forecast's coefficients) rates each note; a `notenet.json` in the data folder
//! replaces it. Each own play's ability θ is solved from its result and kept in `play_theta`, so the net's difficulties meet
//! every player at their own level; the next play's θ is forecast from them, and the card's expectation averages the
//! population model's accuracy with the per-note one.
use crate::evidence::evidence_mods;
use crate::pp::chart_path;
use ms_calc::notefeat::chart_records;
use ms_calc::notenet::{expected_loss, theta_of, Net};
use ms_data::store::{acc_lazer, parse_stats, Db};
use rusqlite::Result;
use serde_json::Value;

const SHIPPED: &str = include_str!("../../../calib/notenet.json");
pub const WINDOW: f64 = 15.; // days of history the forecast reads
const GAP: f64 = ms_model::session::SESSION_GAP / 86_400.; // the app's session rule, in days

pub struct PerNote {
    net: Net,
    coef: [f64; 6], // θ − ewma: constant, last same-keymode θ in the session − ewma, has one, session form, retry θ − ewma, has one
    clip: f64,      // a history play's θ enters the forecast at most this far from its keymode's EWMA before it
    id: String,     // the weights' hash: new weights solve the history again
}

/// An own play of the last WINDOW days; θ None when not solvable (chart gone, mods, short).
struct Past {
    t: f64, // days
    keys: i64,
    sha: String,
    theta: Option<f64>,
}

fn days(played: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(played).ok().map(|d| d.timestamp_millis() as f64 / 86_400_000.)
}

/// Days as `scores.played` writes them, for a coarse SQL bound (callers keep a day's margin and filter exactly after).
fn iso(days: f64) -> String {
    chrono::DateTime::from_timestamp_millis((days.max(0.) * 86_400_000.) as i64).map_or_else(String::new, |d| d.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string())
}

/// Now in days.
pub fn today() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 86_400_000.
}

impl PerNote {
    pub fn load() -> Option<PerNote> {
        let own = std::fs::read_to_string(ms_data::paths::data().join("notenet.json")).ok();
        PerNote::parse(own.as_deref().unwrap_or(SHIPPED)).map_err(|e| eprintln!("notenet.json: {e}")).ok()
    }

    fn parse(text: &str) -> std::result::Result<PerNote, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let f = &v["forecast"];
        let c = |k: &str| f[k].as_f64().ok_or(format!("forecast {k}"));
        let coef = [c("const")?, c("last")?, c("has_last")?, c("form")?, c("retry")?, c("has_retry")?];
        let clip = f["clip"].as_f64().unwrap_or(f64::INFINITY);
        Ok(PerNote { net: Net::parse(text)?, coef, clip, id: crate::engine::hashes(text.as_bytes()).0[..12].to_string() })
    }

    /// Every own play from `from` to `to` (days), by time, its θ when solved under these weights.
    fn history(&self, db: &Db, from: f64, to: f64) -> Result<Vec<Past>> {
        let mut q = db.prepare_cached("SELECT s.played, s.sha256, p.keys, p.theta FROM scores s LEFT JOIN play_theta p ON p.key=s.key AND p.net=?
                                       WHERE s.src!='public' AND s.played>=?")?;
        let mut out: Vec<Past> = q.query_map(rusqlite::params![self.id, iso(from - 1.)], |r| {
            Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<i64>>(2)?, r.get::<_, Option<f64>>(3)?))
        })?.flatten().filter_map(|(played, sha, keys, theta)| Some(Past { t: days(played.as_deref()?)?, keys: keys.unwrap_or(0), sha: sha.unwrap_or_default(), theta }))
            .filter(|p| p.t >= from && p.t <= to).collect();
        out.sort_by(|a, b| a.t.total_cmp(&b.t));
        Ok(out)
    }

    /// Solves θ for the own plays from `from` (days) on that lack one (background: a chart parse and a net pass per play). A
    /// play's θ reads only its own result and its place in its session, so solving later plays never leaks into earlier ones.
    pub fn fill(&self, db: &Db, from: f64) -> Result<usize> {
        db.execute_batch("CREATE TABLE IF NOT EXISTS play_theta(key TEXT, net TEXT, keys INTEGER, theta REAL, PRIMARY KEY(key, net))")?;
        type Row = (String, Option<String>, Option<String>, Option<f64>, Option<String>, Option<String>, Option<String>, Option<i64>);
        let todo: Vec<Row> = db.prepare("SELECT s.key,s.played,s.sha256,s.rate,s.stats,s.mods,s.client,s.n FROM scores s
                                         WHERE s.src!='public' AND s.played>=?
                                         AND NOT EXISTS (SELECT 1 FROM play_theta p WHERE p.key=s.key AND p.net=?)")?
            .query_map(rusqlite::params![iso(from - 1.), self.id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?)))?
            .collect::<Result<_>>()?;
        if todo.is_empty() {
            return Ok(0);
        }
        let past = self.history(db, from - 1., f64::INFINITY)?;
        for (key, played, sha, rate, stats, mods, client, n) in &todo {
            let solved = (|| {
                let (t, rate, stats) = (days(played.as_deref()?)?, (*rate)?, parse_stats(stats.as_deref()?)?);
                let acc = acc_lazer(&stats);
                // Random moves the columns the records read; short and failed plays aren't evidence (evidence_rows' rule)
                if evidence_mods(mods.as_deref()) != Some(false) || acc < 0.5 || n.unwrap_or(0) < 100 {
                    return None;
                }
                let c = ms_calc::chart::parse(&std::fs::read(chart_path(sha.as_deref())?).ok()?).ok()?;
                let d = self.net.difficulties(&chart_records(&c, rate).ok()?, place(&past, t), !client.as_deref().unwrap_or("").starts_with("stable"));
                Some((c.keys as i64, theta_of(&d, 1. - acc)))
            })();
            db.execute("INSERT OR REPLACE INTO play_theta VALUES (?,?,?,?)", rusqlite::params![key, self.id, solved.map(|s| s.0), solved.map(|s| s.1)])?;
        }
        Ok(todo.len())
    }

    /// The per-note forecast's accuracy (lazer units) for chart `sha` at `rate`, played at `now` (days); None without enough
    /// history (three solved plays of the keymode in the window) or without the chart file.
    pub fn accuracy(&self, db: &Db, sha: &str, rate: f64, stable: bool, now: f64) -> Option<f64> {
        let c = ms_calc::chart::parse(&std::fs::read(chart_path(Some(sha))?).ok()?).ok()?;
        let past = self.history(db, now - WINDOW - 1., now).ok()?;
        let theta = self.forecast(&past, now, c.keys as i64, sha)?;
        let d = self.net.difficulties(&chart_records(&c, rate).ok()?, place(&past, now), !stable);
        Some(1. - expected_loss(&d, theta))
    }

    /// thetafc's "ship: last+retry+form": a same-keymode ewma (3-day decay, same session ×4), moved by the session's last
    /// same-keymode play, the session's form over every keymode (last three), and a retry of this chart.
    fn forecast(&self, past: &[Past], now: f64, keys: i64, sha: &str) -> Option<f64> {
        // one disaster or career-best play moves the forecast by at most `clip` (both sides; held-out 0.868 → 0.861 at 0.35)
        let clipped: Vec<Past> = past.iter().map(|q| Past { theta: q.theta.map(|th| ewma(past, q.t, q.keys).map_or(th, |e| e + (th - e).clamp(-self.clip, self.clip))), sha: q.sha.clone(), ..*q }).collect();
        let past = &clipped[..];
        let ewma = |k: i64| ewma(past, now, k);
        let e = ewma(keys)?;
        let start = session_start(past, now);
        let (mut last, mut retry, mut form) = (None, None, Vec::new());
        for (p, th) in past.iter().filter(|p| p.t < now && p.t >= start).filter_map(|p| Some((p, p.theta?))) {
            if p.keys == keys {
                last = Some(th - e);
            }
            if p.sha == sha {
                retry = Some(th - e);
            }
            if let Some(eq) = ewma(p.keys) {
                form.push(th - eq);
            }
        }
        let recent: Vec<f64> = form.iter().rev().take(3).copied().collect();
        let form = if recent.is_empty() { 0. } else { recent.iter().sum::<f64>() / recent.len() as f64 };
        let x = [1., last.unwrap_or(0.), f64::from(u8::from(last.is_some())), form, retry.unwrap_or(0.), f64::from(u8::from(retry.is_some()))];
        Some(e + x.iter().zip(&self.coef).map(|(a, b)| a * b).sum::<f64>())
    }
}

/// A keymode's ability just before `now`: EWMA over its solved plays of the window (3-day decay, same session ×4); None under
/// three plays.
fn ewma(past: &[Past], now: f64, k: i64) -> Option<f64> {
    let (mut sw, mut sm, mut n) = (0., 0., 0);
    for (p, th) in past.iter().filter_map(|p| Some((p, p.theta?))).filter(|(p, _)| p.keys == k && p.t < now && now - p.t <= WINDOW) {
        let dt = now - p.t;
        let w = (-dt / 3.).exp() * if dt < 0.125 { 4. } else { 1. };
        (sw, sm, n) = (sw + w, sm + w * th, n + 1);
    }
    (n >= 3).then(|| sm / sw)
}

/// When the session holding time `t` began: the earliest play reached back from `t` without a gap over GAP (`t` itself when
/// the last play is longer ago).
fn session_start(past: &[Past], t: f64) -> f64 {
    let mut start = t;
    for p in past.iter().rev().filter(|p| p.t < t) {
        if start - p.t > GAP {
            break;
        }
        start = p.t;
    }
    start
}

/// A play's place in its session: (minutes since the session began, plays before it).
fn place(past: &[Past], t: f64) -> (f64, f64) {
    let start = session_start(past, t);
    ((t - start) * 1440., past.iter().filter(|p| p.t >= start && p.t < t).count() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_net_reads_the_current_calc() {
        // its `demand`/`cj` inputs are the calc's per-note strain, set by the rule knobs (a families refit leaves them): new
        // knobs retrain it (s7net notefit on fresh `calib rows`)
        let v: Value = serde_json::from_str(SHIPPED).unwrap();
        assert_eq!(v["rules"].as_str(), Some(ms_calc::features::rules_stamp(&ms_calc::features::standard_params()).as_str()), "calib/notenet.json was trained under other calc rules");
        assert!(PerNote::parse(SHIPPED).is_ok());
    }

    fn p(min: f64, keys: i64, sha: &str, theta: f64) -> Past {
        Past { t: 100. + min / 1440., keys, sha: sha.into(), theta: Some(theta) }
    }

    #[test]
    fn sessions_break_on_the_app_gap_and_the_forecast_reads_this_one() {
        // three days-old 7K plays at θ 1 set the ewma; tonight's session: 4K, then 7K at θ 2 on chart "a" 20 min later
        let mut past: Vec<Past> = (0..3).map(|i| Past { t: 98. + f64::from(i) * 0.01, keys: 7, sha: "x".into(), theta: Some(1.) }).collect();
        past.extend([p(0., 4, "y", 0.), p(20., 7, "a", 2.)]);
        let now = 100. + 25. / 1440.;
        let (minutes, before) = place(&past, now);
        assert!((minutes - 25.).abs() < 1e-6 && before == 2.);
        assert_eq!(place(&past, 100. + 80. / 1440.).1, 0.); // 60 min after the last play: a new session
        let pn = PerNote { net: Net::parse(r#"{"inputs": [], "mu": [], "sd": [], "l1": {"in": 1, "out": 1, "w": [0], "b": [0]},
            "l2": {"in": 1, "out": 1, "w": [0], "b": [0]}, "l3": {"in": 1, "out": 1, "w": [0], "b": [0]}}"#).unwrap(),
                            coef: [0., 0.5, 0., 0., 0.25, 0.], clip: f64::INFINITY, id: String::new() };
        let e = { let w2 = 4. * (-(5. / 1440.) / 3f64).exp(); let w1: f64 = (0..3).map(|i| (-(now - 98. - f64::from(i) * 0.01) / 3.).exp()).sum(); (w1 + 2. * w2) / (w1 + w2) };
        // last same-keymode play and a retry of "a" both read 2 − e
        let th = pn.forecast(&past, now, 7, "a").unwrap();
        assert!((th - (e + 0.75 * (2. - e))).abs() < 1e-9, "{th} vs {e}");
        assert!(pn.forecast(&past, now, 4, "a").is_none()); // one 4K play: no ewma
        // clipped at 0.1: the 7K play at θ 2 enters 0.1 over the ewma of 1 before it
        let pn = PerNote { clip: 0.1, ..pn };
        let e = { let w2 = 4. * (-(5. / 1440.) / 3f64).exp(); let w1: f64 = (0..3).map(|i| (-(now - 98. - f64::from(i) * 0.01) / 3.).exp()).sum(); (w1 + 1.1 * w2) / (w1 + w2) };
        let th = pn.forecast(&past, now, 7, "a").unwrap();
        assert!((th - (e + 0.75 * (1.1 - e))).abs() < 1e-9, "{th} vs {e}");
    }
}



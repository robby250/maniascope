//! The Recommender: one fit's models and everything derived from it, the card of a selected chart, the PP pool.
use crate::catalog::{installed_by_md5, local_shas, rate_label_pp, Inst, Pick};
use crate::evidence::{now_months, FeatureSource};
use crate::pp::{self, build_ledger, candidates, estimates, home_keys, newer_ranked, prepare, reference, score, status_of, typical, warmups, Ledger, PpSettings, Reference, Scoring, Variant};
use crate::recommender::{family_transfer, fit, Fitted};
use ms_calc::features::standard_params;
use ms_calc::params::Params;
use ms_calc::py::round_digits;
use ms_calc::val::{Dict, Val};
use ms_data::codec;
use ms_data::events::load_events;
use ms_data::public::Public;
use ms_data::store::{connect, now, Db, Result, RANKED_STATUS};
use ms_model::choices::preview_order;
use ms_model::predict::{Consts, Feats, HashMap, PredSession, Prediction, BAND_Z};
use ms_model::session::{Session, HISTORY_S};
use rusqlite::OptionalExtension;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// recommend.predictor_id of the model the Rust port reproduces: forecasts stay comparable across the port. A change to
/// the prediction model takes a new id, so old forecasts are not read as this model's.
pub const PREDICTOR_ID: &str = "532ce129f6fa"; // predictor families from the skill table (2026-11-11); was a6d61e058f5d
pub const CALC_ID: &str = "cc36b2f14526";
/// Differ from CALC_ID in calibration only (calib/carry.json), newest first: a crawl still running holds the website rows back a whole
/// calculator, so those can start older than the library's. Each carry keeps its source row.
// Every shipped calculator, newest last: a ship that leaves its predecessor out drops the scores whose chart file is gone
// (1.10.0: 7K evidence 8718 → 3959 plays on a copy without the files).
pub const CARRIED_CALCS: [&str; 34] = ["2f01eca5e738", "7cee1dfdba7f", "a88aea03e865", "f9622a2b09c8", "65c0a4cb63d6", "9148bd02bce2", "7a35f25f603a", "faff7ab286c4", "a4e17d09c2b6", "3c695ee7a23f",
                                       "4ccede2d49cf", "09639faa6673", "d3a86f7b014f", "f92680b7b136", "0f9f7da275f1", "a386c60fbb06", "a990d0ea0ada", "2789b337ae54", "bc901ee5b05a", "4c42635f416e", "5f0c6b2fdd16",
                                       "29a02d72f4a7", "eea12425de4c", "6f3c526261aa", "83c4b154d906", "8b4eb56bb201", "c4681b3a3229", "da01b4c516f3", "0c5d5c8a61e7", "fc49ee7e2b64", "635154fd6f71", "009e2b28af08", "ea60729870d6", "14be0f41537b"];
const SURE_PLAYS: usize = 50; // of the player's own, on a keymode: fewer and the model is mostly the population's guess
pub const ACTIVE_KEYS: std::ops::RangeInclusive<i64> = 4..=10;

/// One event log state: the session, its display view, the frozen tables of both.
pub struct Now {
    pub session: Arc<Session>,
    pub view: Arc<Session>,
    pub tables: PredSession,
    pub view_tables: PredSession,
    pub target: f64,
}

pub struct Refit {
    pub fitted: Arc<Fitted>,
    pub installed: Arc<HashMap<String, Arc<Inst>>>,
    pub local_shas: Arc<HashSet<String>>,
    pub ledger: Ledger,
    pub missing: usize,
    pub ranks: HashMap<i64, i64>,
    pub references: HashMap<i64, Reference>,
}

pub fn refit(db: &Db, public: &Arc<Public>, c: &Arc<Consts>) -> Result<Refit> {
    let installed = Arc::new(installed_by_md5(db)?);
    let (ledger, missing) = build_ledger(db, public, &installed)?;
    let (ranks, references) = reference(db, &ledger)?;
    Ok(Refit { fitted: Arc::new(fit(db, public, CALC_ID, c)?), local_shas: Arc::new(local_shas(db)?), installed, ledger, missing, ranks, references })
}

pub struct Engine {
    pub public: Arc<Public>,
    pub c: Arc<Consts>,
    pub prm: Arc<Params>,
    pub r: Refit,
    pub source: FeatureSource,
    pub settings: PpSettings,
    pub observing: Arc<Mutex<Option<Vec<String>>>>, // results that land while a refit runs
    variants: Option<(Vec<i64>, Arc<Vec<Variant>>)>,
    transfer: Arc<Vec<Vec<f64>>>,
    pub pernote: Option<Arc<crate::pernote::PerNote>>, // the per-note forecast (pernote)
}

/// Both targets of one prediction: `pp` (acc320) and what the player reads.
pub struct Expected {
    pub pp: Prediction,
    pub shown: Prediction,
    pub direct: bool, // `shown` comes from a directly fitted display target
}

impl Expected {
    /// recommend.Predictor.predict's forecast: the display state, chart-bound when `chart` is a hash.
    pub fn forecast(&self, chart: &str, rate: f64) -> Option<Value> {
        self.direct.then(|| {
            let p = &self.shown;
            let (sha, md5) = (chart.len() == 64, chart.len() == 32);
            json!({"schema": 1, "target": "native_displayed_lazer", "kind": "usual_range", "generated_at": now(), "calculator": CALC_ID, "predictor": PREDICTOR_ID,
                   "sha256": if sha { Some(chart) } else { None }, "md5": if md5 { Some(chart) } else { None }, "rate": rate, "band_z": BAND_Z,
                   "mid": p.acc_mid, "lo": p.acc_lo, "hi": p.acc_hi, "mu": p.mu, "sdm": p.sdm, "sda": p.sda, "sd": p.sd})
        })
    }

    /// The expectation's fields of a start event.
    pub fn fields(&self) -> Map<String, Value> {
        let (p, s) = (&self.pp, &self.shown);
        let mut out = Map::new();
        for (k, v) in [("mu", p.mu), ("base_mu", p.base_mu), ("cold_penalty", p.cold), ("activation_minutes", s.activation), ("opening_acc", s.opening_acc), ("rate_slope", s.slope), ("sd", p.sd), ("sdm", p.sdm), ("ln", p.ln)] {
            out.insert(k.to_string(), json!(v));
        }
        if self.direct {
            for (k, v) in [("mu", s.mu), ("base_mu", s.base_mu), ("cold_penalty", s.cold), ("sdm", s.sdm), ("sda", s.sda), ("sd", s.sd)] {
                out.insert(format!("display_{k}"), json!(v));
            }
        }
        out
    }
}

/// accuracy_targets.forecast_chart: an exact chart identity, without changing prediction provenance.
pub fn forecast_chart(state: Option<Value>, sha: Option<&str>, md5: Option<&str>) -> Option<Value> {
    let mut state = state?;
    for (k, v) in [("sha256", sha), ("md5", md5)] {
        match (state[k].as_str(), v) {
            (Some(a), Some(b)) if a != b => return None,
            (None, Some(b)) => state[k] = json!(b),
            _ => {}
        }
    }
    Some(state)
}

/// The top own score on a chart and the chance a play now passes it.
#[derive(Clone)]
pub struct Beat {
    pub stats: ms_data::store::Stats,
    pub score: Option<i64>,
    pub rate: f64,
    pub chance: f64,
    pub multipliers: (f64, f64), // osu!mania's score multiplier of this play's mods and of the top score's
}

/// osu!mania's score multiplier of a mod list played at `rate` (the mods' product). lazer: `ManiaScoreMultiplierCalculator`,
/// checked on the user's 9,053 lazer scores against unmodded ones at the same accuracy (2026-11-22): HT 0.9× 0.50, 0.8× 0.40,
/// 0.7× 0.30, 0.6× 0.20, 0.5× 0.10; NF 0.50; AS 0.49; CS·HT 0.8× 0.36; DT 1.0–1.5× 1.00 (lazer's source now gives a sped-up
/// play more: not in these scores, so not here); MR, RD 1. stable: Easy, No Fail and Half Time halve.
pub fn score_multiplier(mods: &[Value], rate: f64, stable: bool) -> f64 {
    mods.iter().filter_map(|m| m["acronym"].as_str()).map(|a| match a {
        "EZ" | "NF" => 0.5,
        "HT" | "DC" if stable => 0.5,
        "HT" | "DC" if rate < 1. => 0.6 + (rate * 10. + 1e-9).floor() / 10. - 1.,
        _ if stable => 1.,
        "DA" | "WU" | "WD" | "AS" => 0.5,
        "NR" | "CS" | "HO" => 0.9,
        k if k.strip_suffix('K').is_some_and(|n| n.parse::<u8>().is_ok()) => 0.9, // the key mods, since lazer 2025.718
        _ => 1.,
    }).product()
}

#[derive(Clone, Default)]
pub struct Card {
    pub bid: Option<i64>,
    pub md5: String,
    pub keys: i64,
    pub title: String,
    pub var: String,
    pub rate: f64,
    pub ranked: bool,
    pub artist: Option<String>,
    pub song: Option<String>,
    pub version: Option<String>,
    pub acc_mid: f64,
    pub acc_lo: f64,
    pub acc_hi: f64,
    /// Mods the numbers include (`Mods::labels`) and mods they leave out: the card names both.
    pub mods: Vec<String>,
    pub without: Vec<String>,
    pub sure: bool, // enough of the player's own plays on this keymode to show acc_mid unasked
    pub form: (Option<&'static str>, f64), // tooltip only (user 2026-11-21)
    pub length: f64,
    pub skill: Option<String>,
    pub nps: Option<f64>,
    pub pp: Option<(f64, f64, Option<f64>, f64, f64)>, // pp_lo, pp_hi, best, gain, p_up
    pub pp_expected: Option<f64>, // shown on every map with stars, unranked too
    /// The player's top own score on this chart (its judgements) and the chance to beat its accuracy now: for the
    /// maps `pp` does not cover (user 2026-11-22: "on every selected map not just ranked pp maps").
    pub beat: Option<Beat>,
    pub plays_like: Option<(f64, i64)>,
    pub unread: Option<String>, // `Prediction::read_note`, tooltip only: acc_mid/lo/hi carry the deficit
    pub random: bool, // Random before its seed: the mean over three shuffles (worker::mean_card)
}

/// recommend.top_skill
pub fn top_skill(f: &Dict) -> Option<String> {
    let sk = f.dict("sk")?;
    let mut best: Option<(&str, f64)> = None;
    for (k, v) in &sk.0 {
        if let ms_calc::val::Key::S(k) = k {
            if k != "stamina" && best.is_none_or(|b| v.f() > b.1) {
                best = Some((k, v.f()));
            }
        }
    }
    best.map(|b| b.0.to_string())
}

/// Hex of a digest.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn hashes(data: &[u8]) -> (String, String) {
    (hex(&Sha256::digest(data)), format!("{:x}", md5::compute(data)))
}

impl Engine {
    pub fn new(public: Arc<Public>, c: Arc<Consts>, r: Refit, settings: PpSettings) -> Engine {
        let source = FeatureSource { public: public.clone(), calc: CALC_ID.to_string(), cache: HashMap::default() };
        Engine { transfer: Arc::new(family_transfer(&c)), public, c, prm: Arc::new(standard_params()), r, source, settings, observing: Arc::default(), variants: None, pernote: None }
    }

    /// Stored features changed under the PP candidates (the background carry landed): they are read again on the next request.
    pub fn forget_variants(&mut self) {
        self.variants = None;
    }

    /// A finished refit replaces the models; results that landed meanwhile are observed again.
    pub fn apply(&mut self, db: &Db, r: Refit, observed: Vec<String>) -> Result<()> {
        self.r = r;
        self.variants = None; // ~77k variants: built on the first PP-tab request, not on every refit
        for key in observed {
            self.observe(db, &key)?;
        }
        Ok(())
    }

    pub fn model(&self) -> &ms_model::predict::PredModel {
        self.r.fitted.display.as_ref().map_or(&self.r.fitted.pp.model, |d| &d.model)
    }

    pub fn now_state(&self, db: &Db, target: f64) -> Result<Now> {
        self.state_at(db, target, now())
    }

    pub fn state_at(&self, db: &Db, target: f64, current: f64) -> Result<Now> {
        self.state_with(load_events(db, Self::events_since(current))?, target, current)
    }

    /// The start of the event window a state at `current` reads (a whole minute: callers cache the events per minute).
    pub fn events_since(current: f64) -> f64 {
        ((current - HISTORY_S) / 60.).floor() * 60.
    }

    /// A state from already loaded events (`load_events(db, events_since(current))`).
    pub fn state_with(&self, events: Vec<Arc<ms_data::store::Event>>, target: f64, current: f64) -> Result<Now> {
        let session = Session::new(events, current, Some(self.r.fitted.pp.warmup.clone()), self.c.clone(), self.transfer.clone());
        let view = self.r.fitted.display.as_ref().map_or_else(|| session.display_view(self.r.fitted.pp.warmup.clone()), |d| session.display_view(d.warmup.clone()));
        let all: Vec<i64> = ACTIVE_KEYS.collect();
        let (tables, view_tables) = (session.tables(&all), view.tables(&all));
        Ok(Now { session: Arc::new(session), view: Arc::new(view), tables, view_tables, target })
    }

    /// recommend.Predictor.predict: PP means and spreads stay; playlist quality and intervals use the display target.
    pub fn predict(&self, f: &Feats, chart: &str, bid: Option<i64>, rate: f64, n: &Now) -> Option<Expected> {
        let pp = self.r.fitted.pp.model.predict(f, chart, bid, rate, &n.tables).ok()?;
        let Some(d) = self.r.fitted.display.as_ref() else {
            return Some(Expected { shown: pp.clone(), pp, direct: false });
        };
        let shown = d.model.predict(f, chart, bid, rate, &n.view_tables).ok()?;
        Some(Expected { pp, shown, direct: true })
    }

    /// Features of a chart-rate: the local table, then the public one when the md5 matches.
    pub fn features(&mut self, db: &Db, sha: Option<&str>, md5: Option<&str>, bid: Option<i64>, rate: f64) -> Result<(Option<Arc<Dict>>, Option<i64>)> {
        self.source.of(db, sha, md5, bid, Some(rate))
    }

    /// FeatureSource._store
    pub fn store(&mut self, db: &Db, sha: &str, features: &[(f64, Dict)]) -> Result<()> {
        let tx = db.unchecked_transaction()?;
        for (rate, f) in features {
            let key = self.source.key(sha, *rate);
            tx.execute("INSERT OR REPLACE INTO feats VALUES (?, ?)", rusqlite::params![key, codec::encode(&Val::Dict(f.clone()))])?;
            self.source.cache.insert(key, Arc::new(f.clone()));
        }
        tx.commit()
    }

    /// A result updates the ledger and session, not the frozen baseline model: rebuilding it on every play blocked Next.
    /// Session correction already consumes this result; refits remain for startup, imports and old scores.
    pub fn observe(&mut self, db: &Db, key: &str) -> Result<()> {
        if let Some(list) = self.observing.lock().unwrap().as_mut() {
            list.push(key.to_string());
        }
        type Row = (Option<i64>, Option<i64>, Option<String>, String, Option<i64>, Option<String>, Option<f64>, Option<String>, Option<String>, Option<String>, Option<String>, Option<i64>);
        let row: Option<Row> = db.query_row("SELECT ranked,beatmap_id,md5,src,online_id,client,pp,sha256,stats,mods,played,status FROM scores WHERE key=?", [key],
                                            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?, r.get(10)?, r.get(11)?))).optional()?;
        let Some((ranked, bid, md5, src, online, client, stored, sha, stats, mods, played, status)) = row else { return Ok(()) };
        let Some(bid) = bid.filter(|&b| b > 0 && ranked.unwrap_or(0) != 0) else { return Ok(()) };
        let counted = status_of(status, md5.as_deref(), Some(bid), &self.public, &self.r.installed).is_some_and(|s| RANKED_STATUS.contains(&s));
        if !counted || !(src != "realm" || online.unwrap_or(0) != 0 || client.as_deref() == Some("lazer-live")) {
            return Ok(());
        }
        let mods: Value = mods.as_deref().and_then(|m| serde_json::from_str(m).ok()).unwrap_or(json!([]));
        let lazer = !client.as_deref().unwrap_or("").starts_with("stable");
        let pp = stored.or_else(|| pp::score_pp(&pp::scored_chart(sha.as_deref(), Some(bid), md5.as_deref())?, &ms_data::store::parse_stats(stats.as_deref()?)?, &mods, lazer));
        let Some(pp) = pp else { return Ok(()) };
        db.execute("UPDATE scores SET pp=? WHERE key=?", rusqlite::params![pp, key])?;
        if pp > self.r.ledger.best_of(bid) {
            let mut best = self.r.ledger.best.clone();
            best.insert(bid, (pp, true, played));
            self.r.ledger = Ledger::new(best);
            (self.r.ranks, self.r.references) = reference(db, &self.r.ledger)?;
        }
        Ok(())
    }

    fn variants(&mut self, db: &Db, keys: &[i64]) -> Result<Arc<Vec<Variant>>> {
        if let Some((k, v)) = &self.variants {
            if k == keys {
                return Ok(v.clone());
            }
        }
        let newer = newer_ranked(db, &mut self.source, &self.r.installed, keys, &self.c)?;
        let all = Arc::new(candidates(&self.public, &self.r.installed, keys, &self.c, newer));
        self.variants = Some((keys.to_vec(), all.clone()));
        Ok(all)
    }

    /// All PP candidates under this session, the keymodes' best first.
    pub fn scored(&mut self, db: &Db, keys: &[i64], n: &Now) -> Result<Vec<Pick>> {
        let variants = self.variants(db, keys)?;
        let month = now_months();
        let typ = typical(&self.r.fitted.rows, month);
        let threads = std::thread::available_parallelism().map_or(1, |x| x.get()).saturating_sub(2).max(1);
        let f = &self.r.fitted;
        let s = Scoring { pp_model: &f.pp.model, pp_tables: &n.tables, display: f.display.as_ref().map(|d| (&*d.model, &n.view_tables)), shown: f.shown, session: &n.session, ledger: &self.r.ledger,
                          ranks: &self.r.ranks, references: &self.r.references, settings: &self.settings, typical: &typ, installed: &self.r.installed, public: &self.public, now: n.session.now };
        Ok(score(&variants, &s, threads))
    }

    /// The PP list's rows, best first in the session's weighted shuffle: (phase, the eligible maps, note). The worker's list
    /// gives them their places, as it does on every tab.
    pub fn pool(&mut self, db: &Db, keys: &[i64], n: &Now) -> Result<(&'static str, Vec<Pick>, String)> {
        let list = self.scored(db, keys, n)?;
        let phase = n.session.phase();
        let month = now_months();
        let typ = typical(&self.r.fitted.rows, month);
        let f = self.r.fitted.clone();
        let home = home_keys(&self.r.ledger, &self.public, &f.rows, month);
        let (mut rows, relaxed) = prepare(&list, &n.session, &self.r.ledger, &self.r.references, &self.settings, home);
        if rows.is_empty() && phase == "warmup" {
            let k0 = n.session.current_keys().filter(|k| keys.contains(k)).or(Some(home).filter(|k| keys.contains(k))).or(keys.first().copied());
            if let Some(k0) = k0 {
                let by_chart = |chart: &str| self.r.installed.get(chart).cloned();
                rows = warmups(&f, &list, &n.session, k0, &n.tables, &n.view_tables, &by_chart, &typ);
            }
        }
        let ids: Vec<(String, f64)> = rows.iter().map(|c| (c.md5.clone().or_else(|| c.sha.clone()).or_else(|| c.bid.map(|b| b.to_string())).unwrap_or_else(|| c.title.clone()), c.weight)).collect();
        let order = preview_order(&ids, &n.session.shuffle_seed().to_string(), 0.);
        let shown: Vec<Pick> = order.into_iter().map(|i| rows[i].clone()).collect();
        let top = rows.iter().filter(|c| c.best_rank <= 100).count();
        let mut note = if rows.is_empty() { "No likely PP gains at this readiness: try Skills or an easier map".to_string() } else { format!("{} eligible · {top} top-100", commas(rows.len())) };
        if relaxed {
            note += " · all cycled, repeats allowed";
        }
        Ok((phase, shown, note))
    }

    /// Pre-play expectation for a chart about to be played (frozen into the start event). None: no features yet.
    pub fn expectation(&mut self, db: &Db, sha: &str, md5: &str, rate: f64, n: &Now) -> Result<Option<(Expected, Arc<Dict>, Feats, Option<i64>)>> {
        let rate = round_digits(rate, 3);
        let (f, b) = self.features(db, Some(sha), Some(md5), None, rate)?;
        let Some(f) = f else { return Ok(None) };
        let Ok(feats) = Feats::of(&f, &self.c) else { return Ok(None) };
        let e = self.expected(db, &feats, sha, md5, b, rate, n, None);
        Ok(e.map(|e| (e, f, feats, b)))
    }

    /// The shown Expected: the prediction blended with the personal per-note forecast when the chart's file is found. The
    /// card and the playlist's aimed offer both read it (user 2026-11-14: offer ~96.2 % while the card showed 95.7 %).
    #[allow(clippy::too_many_arguments)]
    pub fn expected(&self, db: &Db, feats: &Feats, sha: &str, chart: &str, bid: Option<i64>, rate: f64, n: &Now, data: Option<&[u8]>) -> Option<Expected> {
        let mut e = self.predict(feats, chart, bid, rate, n)?;
        if let Some(p) = self.pernote.as_ref() {
            let stable = ms_data::store::kv_get(db, "client").is_some_and(|c| c == "stable");
            if let Some(acc) = p.accuracy(db, sha, data, round_digits(rate, 3), stable, crate::pernote::today()) {
                self.model().blend(&mut e.shown, feats, acc);
            }
        }
        Some(e)
    }

    fn own_plays(&self, keys: i64) -> usize {
        self.r.fitted.rows.rows.iter().filter(|r| r.keys == keys && r.src != "public" && r.completed).count()
    }

    /// Stats for the chart the game has selected, at its rate: the same numbers a list entry shows.
    /// `multiplier`: `score_multiplier` of the mods this play would have.
    pub fn card(&mut self, db: &Db, data: &[u8], rate: f64, multiplier: f64, n: &Now) -> Result<Option<Card>> {
        let (sha, md5) = hashes(data);
        // a modded chart goes by its sha, as its plays do (evidence_rows): its affinity and reads are found under it
        let chart = if crate::modcharts::source_of(db, &sha).is_some() { sha.clone() } else { md5.clone() };
        let Some((e, f, feats, b)) = self.expectation(db, &sha, &chart, rate, n)? else { return Ok(None) };
        let inst = self.r.installed.get(&md5).cloned();
        let bid = b.or(inst.as_ref().and_then(|i| i.beatmap_id).filter(|&b| b != 0));
        let approved = b.and_then(|b| self.public.map_info(b)).is_some_and(|m| RANKED_STATUS.contains(&m.approved));
        let stars = f.get("stars").map(Val::f).or_else(|| b.and_then(|b| self.public.feature(b, round_digits(rate, 3)))
            .and_then(|v| if let Val::Dict(d) = v { d.get("stars").map(Val::f) } else { None }));
        // Off 0.75×/1×/1.5× the analysis has no stars: computed here for the card's pp (user 2026-11-08: 1.10× showed none).
        let at = if stars.is_none() { ms_calc::features::stars_at(data, rate) } else { None };
        let stars = stars.or(at.map(|a| a.0));
        let hits = f.get("hits").map(Val::f).or(at.map(|a| f64::from(a.1))).unwrap_or_else(|| f.fget("notes", 0.));
        let ranked = (approved || inst.as_ref().is_some_and(|i| i.status.is_some_and(|s| RANKED_STATUS.contains(&s)))) && [0.75, 1., 1.5].contains(&round_digits(rate, 2)) && stars.is_some();
        let meta = if inst.is_none() { ms_data::lazer_index::metadata(data) } else { None };
        let get = |k: &str| meta.as_ref().and_then(|m| m.get(k).cloned());
        let (artist, song, version) = inst.as_ref().map_or_else(|| (get("Artist"), get("Title"), get("Version")), |i| (Some(i.artist.clone()), Some(i.title.clone()), Some(i.version.clone())));
        let title = if inst.is_some() || meta.is_some() {
            format!("{} - {} [{}]", artist.as_deref().unwrap_or(""), song.as_deref().unwrap_or(""), version.as_deref().unwrap_or(""))
        } else {
            sha.clone() // a store file is named by its sha
        };
        let pp = ranked.then(|| estimates(&e.pp, stars.unwrap(), hits, bid, &self.r.ledger));
        // The score on top of osu!'s local list, whatever its rate and mods, and the chance this play's score passes it
        // (user 2026-11-22). osu!mania scores a play by its mods (`score_multiplier`: 0.92× at 94.7 % = 408k where 1× gives
        // ~830k), so a play that scores less against a top that scored more has to do that much better: the accuracy
        // needed comes from the top's own score-for-accuracy, score ∝ multiplier · accuracy^p.
        let best: Option<(String, f64, Option<f64>, Option<i64>, Option<String>)> = db.query_row("SELECT stats,acc,rate,total,mods FROM scores WHERE src!='public' AND (sha256=?1 OR md5=?2) AND stats IS NOT NULL ORDER BY total DESC, acc DESC LIMIT 1",
            rusqlite::params![sha, md5], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).ok();
        let beat = best.and_then(|(stats, acc, top_rate, total, mods)| {
            // a stable score keeps its mods as osu!'s bit mask
            let mods: Value = mods.and_then(|m| serde_json::from_str(&m).ok()).unwrap_or(Value::Null);
            let list = mods.as_i64().map(ms_data::share::stable_mods).unwrap_or_else(|| mods.clone());
            let (now, then) = (multiplier, score_multiplier(list.as_array().map_or(&[][..], Vec::as_slice), top_rate.unwrap_or(1.), mods.is_i64()));
            let share = total.map_or(0., |t| t as f64 / 1e6 / then); // of a full score at its multiplier
            let needed = if now == then || !(0.01..0.999).contains(&share) || !(0.01..0.9999).contains(&acc) { if now >= then { acc } else { 1. } } else { (share * then / now).powf(acc.ln() / share.ln()).min(1.) };
            Some(Beat { stats: ms_data::store::parse_stats(&stats)?, score: total, rate: top_rate.unwrap_or(1.), chance: crate::pp::beat_chance(&e.pp, needed), multipliers: (now, then) })
        });
        let plays_like = self.plays_like(&f, b, rate);
        let nps = f.dict("nps").and_then(|d| d.get("nps")).filter(|v| !v.is_none()).map(Val::f); // None under two notes (structure.rs)
        Ok(Some(Card { mods: Vec::new(), without: Vec::new(), bid, md5, keys: feats.keys, title, var: rate_label_pp(rate), rate, ranked, artist, song, version, acc_mid: e.shown.acc_mid, acc_lo: e.shown.acc_lo, acc_hi: e.shown.acc_hi, unread: e.shown.read_note(crate::evidence::now_months()), random: false, sure: self.own_plays(feats.keys) >= SURE_PLAYS,
                       form: n.session.form_state(feats.keys), length: f.f("length") / 1000. / rate, skill: top_skill(&f), nps, pp, beat, pp_expected: stars.map(|s| crate::pp::expected(&e.pp, s, hits)), plays_like }))
    }

    /// (stars, scores) the public scores say a chart plays like at this rate, its stars moved by its map effect; None
    /// with few scores or a small shift (user 2026-10-01: Aim Burst 12.5★ has 98 % top scores; the same players score
    /// 92 % on a 12.5★ Camellia map).
    pub fn plays_like(&self, f: &Dict, b: Option<i64>, rate: f64) -> Option<(f64, i64)> {
        let pop = self.r.fitted.display.as_ref().map_or(&self.r.fitted.pp.model.pop, |d| &d.model.pop);
        let (ratio, n) = plays_like_ratio(pop, f.f("keys") as i64, f.f("overall"), b?, rate)?;
        Some((f.f("overall") * ratio, n))
    }

    /// Selected-chart analysis: a panic in one chart loses that chart, not the app.
    pub fn analyse(data: &[u8], rate: f64, prm: &Params) -> std::result::Result<Vec<(f64, Dict)>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ms_calc::features::selected_features(data, rate, prm))).map_err(|_| "the analysis crashed on this chart".to_string())?
    }
}

/// "12,345"
pub fn commas(n: usize) -> String {
    let text = n.to_string();
    let bytes = text.as_bytes();
    bytes.iter().enumerate().flat_map(|(i, b)| {
        let sep = i > 0 && (bytes.len() - i).is_multiple_of(3);
        sep.then_some(',').into_iter().chain(std::iter::once(*b as char))
    }).collect()
}

/// A connection of its own for a background job.
pub fn second_connection(db: &Db) -> Result<Db> {
    connect(Path::new(db.path().unwrap_or("")))
}

/// How much harder (> 1) or easier a chart plays than its stars by the public map effect, with its score count. None
/// with few scores or a small shift.
pub fn plays_like_ratio(pop: &ms_model::predict::Pop, k: i64, overall: f64, b: i64, rate: f64) -> Option<(f64, i64)> {
    const MIN_SCORES: f64 = 30.;
    const MIN_SHIFT: f64 = 0.05;
    let lr_rate = round_digits(rate, 3).ln();
    let (mb, n) = pop.mbr.get(&b).and_then(|v| v.iter().find(|x| (x.0 - lr_rate).abs() < 1e-9)).map(|x| (x.1, x.2)).or_else(|| pop.mb.get(&b).copied())?;
    if n < MIN_SCORES {
        return None;
    }
    // Invert the whole link (slope·x + curve·(x − knee)²), not the slope alone: above the knee the link is ~3× steeper
    // at 9★, and slope-only turned a displayed 9.2★ (mb −.17) into "plays like 7.4★" where the full link gives 8.9★
    // (user 2026-10-02).
    let (s, c, knee) = (pop.slope(k), pop.curve(k), pop.knee.ln());
    let g = |x: f64| s * x + c * (x - knee).max(0.).powi(2);
    let lr = overall.max(0.05).ln();
    let target = g(lr) + mb; // more misses = harder
    let x = if target <= g(knee) || c <= 0. { target / s } else { knee + (-s + (s * s + 4. * c * (target - s * knee)).sqrt()) / (2. * c) };
    let ratio = (x - lr).exp();
    ((ratio - 1.).abs() >= MIN_SHIFT).then_some((ratio, n as i64))
}

/// Stored ratings of the past calculators moved to this one, without their charts. Rows already there stay.
/// `only`: the `sha@rate` names to carry (`played_names`); None: every stored row.
pub fn carry_rows(db: &ms_data::store::Db, only: Option<&std::collections::HashSet<String>>) -> std::result::Result<(), String> {
    let charts = Charts::new(db).map_err(|e| e.to_string())?;
    for from in CARRIED_CALCS.iter().rev() { // newest first: the first source carried wins
        ms_data::store::carry_feats(db, from, CALC_ID, &ms_calc::features::standard_params(), &|sha| charts.counted(sha), only).map_err(|e| e.to_string())?;
    }
    let missing = charts.missing.load(std::sync::atomic::Ordering::Relaxed);
    if missing > 0 {
        eprintln!("{missing} charts without a file kept their rating");
    }
    Ok(())
}

/// The names the app carries at a calculator change: every played row and each chart's row nearest 1×. A chart's other rates
/// were refinements of past playlists (half of G835LX's 204,906 rows, 2026-11-19) and each was analysed again before the first
/// list; the builder analyses the few it asks for (`refined`).
pub fn app_names(db: &ms_data::store::Db) -> rusqlite::Result<std::collections::HashSet<String>> {
    let mut near: HashMap<String, f64> = HashMap::default();
    for key in db.prepare("SELECT key FROM feats")?.query_map([], |r| r.get::<_, String>(0))? {
        let key = key?;
        let Some((sha, rate)) = key.rsplit_once('|').and_then(|n| n.0.rsplit_once('@')) else { continue };
        let Ok(rate) = rate.parse::<f64>() else { continue };
        near.entry(sha.to_string()).and_modify(|r| if (rate - 1.).abs() < (*r - 1.).abs() { *r = rate }).or_insert(rate);
    }
    let mut names = played_names(db)?;
    names.extend(near.into_iter().map(|(sha, rate)| format!("{sha}@{rate:.3}")));
    Ok(names)
}

/// The `sha@rate` feature names of every score in the database (what a fit reads), as `evidence` keys them.
pub fn played_names(db: &ms_data::store::Db) -> rusqlite::Result<std::collections::HashSet<String>> {
    db.prepare("SELECT sha256, rate FROM scores WHERE sha256 IS NOT NULL")?.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<f64>>(1)?.unwrap_or(1.))))?
        .map(|x| x.map(|(sha, rate)| format!("{sha}@{:.3}", ms_calc::py::round_digits(rate.max(0.) + f64::from(u8::from(rate == 0.)), 3)))).collect()
}

/// Stored website features moved the same way.
pub fn carry_web(db: &ms_data::store::Db) -> std::result::Result<(), String> {
    ms_data::webmaps::forget_in_client()?;
    let charts = Charts::new(db).map_err(|e| e.to_string())?;
    CARRIED_CALCS.iter().rev().try_for_each(|from| ms_data::webmaps::carry(from, CALC_ID, &ms_calc::features::standard_params(), &|sha| charts.counted(sha)))?;
    ms_data::webmaps::prune(CALC_ID)
}

/// A stored chart by its sha: lazer's store, the website cache, or a convert remade from its source.
pub struct Charts {
    converts: HashMap<String, Inst>,
    modded: HashMap<String, (String, String)>, // modcharts: sha → (source sha, tag)
    root: std::path::PathBuf,
    last: std::sync::Mutex<Option<(String, Option<ms_calc::chart::Chart>)>>, // rows of one chart's rates come in a run
    pub missing: std::sync::atomic::AtomicUsize, // charts without a file: their rows keep the old rating
}

impl Charts {
    pub fn new(db: &Db) -> Result<Charts> {
        let modded = db.prepare("SELECT sha, source, tag FROM mod_charts")?.query_map([], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?))))?.collect::<Result<_>>()?;
        Ok(Charts { modded, converts: crate::converts::rows(db)?.into_iter().map(|i| (i.sha256.clone(), i)).collect(), root: ms_data::paths::lazer_data(), last: Default::default(), missing: Default::default() })
    }

    pub fn chart(&self, sha: &str) -> Option<ms_calc::chart::Chart> {
        if let Some((source, tag)) = self.modded.get(sha) {
            return ms_calc::chart::parse(&ms_calc::mods::tagged(&self.data(source)?, tag)).ok(); // a chart as played with mods: its source + the tag
        }
        ms_calc::chart::parse(&self.data(sha)?).ok()
    }

    fn data(&self, sha: &str) -> Option<Vec<u8>> {
        let web = ms_data::webmaps::web_dir().join("osu").join(format!("{sha}.osu"));
        // Stable's store too: a stable player's charts all read as missing and kept the old calculator's analysis (a player
        // 2026-11-05: every LN skill at 0 charts, v1.4.0).
        crate::pp::chart_path(Some(sha)).and_then(|p| std::fs::read(p).ok()).or_else(|| std::fs::read(web).ok()).or_else(|| crate::converts::bytes(self.converts.get(sha)?, &self.root))
    }

    /// `chart`, counting the ones missing.
    pub fn counted(&self, sha: &str) -> Option<ms_calc::chart::Chart> {
        let found = self.chart(sha);
        self.missing.fetch_add(found.is_none() as usize, std::sync::atomic::Ordering::Relaxed);
        found
    }

    pub fn flow(&self, sha: &str, rate: f64, k: &ms_calc::params::Knobs) -> Option<ms_calc::flow::Flow> {
        let mut last = self.last.lock().unwrap();
        if last.as_ref().is_none_or(|l| l.0 != sha) {
            *last = Some((sha.to_string(), self.counted(sha)));
        }
        last.as_ref()?.1.as_ref().map(|c| ms_calc::flow::flow(c, rate, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_model::predict::{ByKeys, Pop};

    #[test]
    fn osu_mania_pays_a_play_by_its_mods() {
        // the user's own lazer scores against unmodded ones at the same accuracy (2026-11-22): the medians
        let lazer = |mods: &[&str], rate: f64| score_multiplier(&mods.iter().map(|m| json!({"acronym": m})).collect::<Vec<_>>(), rate, false);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        for (rate, seen) in [(0.92, 0.5), (0.9, 0.5), (0.85, 0.4), (0.8, 0.4), (0.75, 0.3), (0.7, 0.3), (0.6, 0.2), (0.5, 0.1)] {
            assert!(close(lazer(&["HT"], rate), seen), "{rate}: {}", lazer(&["HT"], rate));
        }
        assert!(close(lazer(&["DT"], 1.3), 1.) && close(lazer(&["NF"], 1.), 0.5) && close(lazer(&["HT", "NF"], 0.8), 0.2) && close(lazer(&["CS", "HT"], 0.8), 0.36));
        assert!(close(lazer(&[], 1.), 1.) && close(lazer(&["MR", "RD", "IN", "HD"], 1.), 1.) && close(lazer(&["10K"], 1.), 0.9));
        // stable halves under Half Time whatever the rate, and knows no key-mod cut on a keymode's own map
        let stable = |mods: &[&str]| score_multiplier(&mods.iter().map(|m| json!({"acronym": m})).collect::<Vec<_>>(), 0.75, true);
        assert!(close(stable(&["HT"]), 0.5) && close(stable(&["HT", "EZ"]), 0.25) && close(stable(&["DT"]), 1.));
    }

    fn pop() -> Pop {
        let mut p = Pop { slope: ByKeys { map: [(0, 2.), (4, 2.)].into_iter().collect() }, knee: 4., ..Pop::default() };
        p.mb = [(1, (-0.5, 300.)), (2, (0.02, 300.)), (3, (-0.5, 5.))].into_iter().collect();
        p.mbr = [(1, vec![(1.5f64.ln(), 0.4, 80.)])].into_iter().collect();
        p
    }

    #[test]
    fn a_calculator_change_carries_played_rows_and_each_charts_row_nearest_1x() {
        let db = rusqlite::Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE feats(key TEXT PRIMARY KEY, data TEXT); CREATE TABLE scores(sha256 TEXT, rate REAL);
            INSERT INTO feats VALUES ('a@1.000|old',''),('a@0.970|old',''),('a@1.200|old',''),('b@0.900|old',''),('b@1.300|older','');
            INSERT INTO scores VALUES ('a', 1.2), ('c', NULL);").unwrap();
        let mut names: Vec<String> = app_names(&db).unwrap().into_iter().collect();
        names.sort();
        assert_eq!(names, ["a@1.000", "a@1.200", "b@0.900", "c@1.000"], "a@0.970 was a refinement of a past list");
    }

    #[test]
    fn plays_like_moves_stars_by_the_public_map_effect() {
        let p = pop();
        let (ratio, n) = plays_like_ratio(&p, 4, 10., 1, 1.).unwrap();
        assert!(n == 300 && (ratio - (-0.25f64).exp()).abs() < 1e-9, "fewer misses than its stars: easier");
        assert!(plays_like_ratio(&p, 4, 10., 1, 1.5).unwrap().0 > 1., "its own 1.5× evidence wins");
        assert!(plays_like_ratio(&p, 4, 10., 2, 1.).is_none() && plays_like_ratio(&p, 4, 10., 3, 1.).is_none());
        // Above the knee the link bends: the shift inverts slope and curve together (Chinmoku 1.5×: slope-only said
        // 7.4★ for a 9.2★ chart the full link puts at 8.9★).
        let mut curved = pop();
        curved.curve = ByKeys { map: [(0, 1.6), (4, 1.6)].into_iter().collect() };
        let (ratio, _) = plays_like_ratio(&curved, 4, 10., 1, 1.).unwrap();
        let g = |x: f64| 2. * x + 1.6 * (x - 4f64.ln()).max(0.).powi(2);
        assert!((g((10. * ratio).ln()) - (g(10f64.ln()) - 0.5)).abs() < 1e-9 && ratio > (-0.25f64).exp());
    }
}

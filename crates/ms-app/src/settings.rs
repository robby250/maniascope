//! ui.json: same file and keys as ever; observed rates are never persisted.
use ms_data::paths::config;
use ms_model::skills;
use ms_rec::builder::{normalize_focus, normalize_rates, normalize_target, DEFAULT_RATES, DEFAULT_TARGET};
use ms_rec::catalog::SUPPORTED_KEYS;
use ms_rec::converts::{default_sources, Setting};
use ms_rec::pp::PpSettings;
use ms_rec::worker::PpMix;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

pub const MODES: [&str; 3] = ["pp", "nps", "skills"];

/// A tab of the strip: PP, NPS or a custom playlist (its index).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tab {
    Mode(&'static str),
    List(usize),
}

pub fn mode_name(m: &str) -> &'static str {
    match m { "pp" => "PP", "nps" => "NPS", _ => "Skills" }
}

#[derive(Clone, PartialEq, Debug)]
/// One switch per main-window row, top to bottom; the head row's rating, skills and NPS each keep their own
/// (user 2026-11-02: "group them by row").
pub struct Display {
    pub title: bool,
    pub version: bool,
    pub info: bool, // keys, dan, rate, length, LN %
    pub prediction: bool, // expected accuracy, form, PP gain
    pub rating: bool,
    pub skills: usize,
    pub nps: bool,
    pub timeline: bool,
    pub status: bool,
}

pub struct Ui {
    pub mode: String,
    // Custom playlists (user 2026-11-06, in place of the Skills tab): each a preset of the "skills" fields below, which
    // hold the open playlist live. Switching stores the live fields and loads the other preset.
    pub playlists: Vec<(String, Value)>,
    pub playlist: usize,
    pub strip: Vec<&'static str>, // the tab order: "pp", "nps" and one "list" per playlist in its order (user 2026-11-08: PP and NPS move too)
    pub skills_focus: f64, // variety (0) ↔ high NPS (1); below 0: off, skills + accuracy only
    pub skills_mix: PpMix,
    pub keys: HashMap<String, Vec<i64>>,
    pub keys_auto: BTreeSet<String>, // never chosen: follow the keymodes actually played
    pub practice: Vec<String>,
    pub skill_weights: HashMap<String, f64>,
    pub skill_avoid: HashMap<String, f64>,
    pub skill_other: f64,
    pub nps_focus: f64,
    pub web_bias: HashMap<String, f64>,
    pub web_weighted: HashMap<String, bool>,
    pub by_count: HashMap<String, bool>, // tab → keymode shares follow map counts (else even)
    pub key_weights: HashMap<String, BTreeMap<i64, f64>>, // tab → keymode → how much of the list (several keymodes)
    pub converts: HashMap<String, BTreeMap<i64, Setting>>, // tab → target keymode
    pub targets: HashMap<String, f64>,
    pub rates: HashMap<String, (f64, f64)>, // NPS and Skills HT/DT range
    pub pp: PpSettings,
    pub mix: PpMix,
    pub display: Display,
    pub next_action: String,
    pub auto_next: bool,
    pub keep_above: bool,
    pub overlay: bool,
    pub details: bool,
    pub win: (f32, f32),
    pub recs_win: (f32, f32),
    pub win_pos: Option<(f32, f32)>,
    pub recs_pos: Option<(f32, f32)>,
    pub recs_open: bool,
    pub stats_keys: Option<i64>, // the Stats page last viewed; None: all keymodes
    pub expanded: BTreeSet<(i64, String)>, // Stats skill families left open
    pub chart_hints: u32, // chart hovers that showed "Scroll to zoom…": it stops after a few
}

pub fn normalize_keys(v: &Value, default: &[i64]) -> Vec<i64> {
    match v {
        Value::Null => default.to_vec(),
        Value::String(s) if s == "all" => SUPPORTED_KEYS.collect(),
        Value::Array(a) => {
            let mut out: Vec<i64> = a.iter().filter_map(Value::as_i64).filter(|k| SUPPORTED_KEYS.contains(k)).collect();
            out.sort_unstable();
            out.dedup();
            out
        }
        _ => Vec::new(),
    }
}

fn file() -> PathBuf {
    config().join("ui.json")
}

fn pair(v: &Value, a: &str, b: &str, default: (f32, f32)) -> (f32, f32) {
    (v[a].as_f64().map_or(default.0, |x| x as f32), v[b].as_f64().map_or(default.1, |x| x as f32))
}

fn point(v: &Value) -> Option<(f32, f32)> {
    Some((v[0].as_f64()? as f32, v[1].as_f64()? as f32))
}

/// {skill key: value}, older trees' keys moved to where those skills are now.
fn per_skill(v: &Value) -> HashMap<String, f64> {
    v.as_object().map_or_else(HashMap::new, |o| o.iter().filter_map(|(k, x)| Some((k, x.as_f64()?))).flat_map(|(k, x)| skills::migrate(k).into_iter().map(move |k| (k.to_string(), x))).collect())
}

fn pp_settings(v: &Value) -> PpSettings {
    let d = PpSettings::default();
    let (f, b) = (|k: &str, x: f64| v[k].as_f64().unwrap_or(x), |k: &str, x: bool| v[k].as_bool().unwrap_or(x));
    PpSettings { min_gain: f("min_gain", d.min_gain), min_chance: f("min_chance", d.min_chance), ht: b("ht", d.ht), ht_chance: f("ht_chance", d.ht_chance), ht_gain: f("ht_gain", d.ht_gain), dt: b("dt", d.dt), push: f("push", d.push), cooldown_days: f("cooldown_days", d.cooldown_days), top100: b("top100", d.top100) }
}

// ui.json keys a playlist owns, top-level and per tab.
const PRESET: [&str; 9] = ["skills_keys", "practice_skills", "skill_weights", "skill_avoid", "skill_other", "skills_focus", "skills_acc_target", "skills_rates", "skills_mix"];
const PRESET_TABS: [&str; 5] = ["web_bias", "web_weighted", "converts", "key_weights", "by_count"];
pub const FOCUS_OFF: f64 = -0.05; // the slider's leftmost notch

fn mix_json(m: &PpMix) -> Value {
    json!({"on": m.on, "every": m.every, "min_gain": m.min_gain, "chance": m.chance, "ht_chance": m.ht_chance})
}

fn pp_mix(v: &Value) -> PpMix {
    let d = PpMix::default();
    let f = |k: &str, x: f64| v[k].as_f64().unwrap_or(x);
    PpMix { on: v["on"].as_bool().unwrap_or(d.on), every: v["every"].as_f64().map_or(d.every, |x| x as usize), min_gain: f("min_gain", d.min_gain), chance: f("chance", d.chance), ht_chance: f("ht_chance", d.ht_chance) }
}

impl Ui {
    pub fn skill_tuning(&self) -> ms_model::skills::Tuning {
        ms_model::skills::Tuning { avoid: self.skill_avoid.clone(), other: self.skill_other }
    }

    pub fn load() -> Ui {
        let v: Value = std::fs::read(file()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        Ui::from_json(&v)
    }

    /// The open playlist's settings as a preset: the "skills" keys of ui.json.
    pub fn preset(&self) -> Value {
        let v = self.to_json();
        let mut p: serde_json::Map<String, Value> = PRESET.iter().map(|k| (k.to_string(), v[*k].clone())).collect();
        for k in PRESET_TABS {
            p.insert(k.to_string(), json!({"skills": v[k]["skills"]}));
        }
        Value::Object(p)
    }

    /// Makes a preset the live "skills" settings; everything else stays.
    pub fn load_preset(&mut self, p: &Value) {
        let mut v = self.to_json();
        for k in PRESET {
            v[k] = p[k].clone();
        }
        for k in PRESET_TABS {
            v[k]["skills"] = p[k]["skills"].clone();
        }
        *self = Ui::from_json(&v);
    }

    /// The live settings back into the open playlist's preset (before a switch or a save).
    pub fn store_live(&mut self) {
        let p = self.preset();
        if let Some(slot) = self.playlists.get_mut(self.playlist) {
            slot.1 = p;
        }
    }

    /// A new playlist after the open one, with its settings; it opens.
    pub fn add_playlist(&mut self, name: String) {
        self.store_live();
        let at = (self.playlist + 1).min(self.playlists.len());
        let slot = self.list_slot(self.playlist).map_or(self.strip.len(), |i| i + 1);
        self.strip.insert(slot, "list");
        self.playlists.insert(at, (name, self.preset()));
        self.playlist = at; // same settings: nothing to load
    }

    /// The strip position of playlist `i`.
    fn list_slot(&self, i: usize) -> Option<usize> {
        self.strip.iter().enumerate().filter(|t| *t.1 == "list").nth(i).map(|t| t.0)
    }

    /// The tabs in strip order.
    pub fn tabs(&self) -> Vec<Tab> {
        let mut list = 0..;
        self.strip.iter().map(|&t| if t == "list" { Tab::List(list.next().unwrap()) } else { Tab::Mode(t) }).collect()
    }

    /// Strip tab `from` moves to place `to`; the playlists take the new order and the open one stays open.
    pub fn move_tab(&mut self, from: usize, to: usize) {
        let mut order = self.tabs();
        if from >= order.len() || to >= order.len() || from == to {
            return;
        }
        let t = order.remove(from);
        order.insert(to, t);
        let seq: Vec<usize> = order.iter().filter_map(|t| if let Tab::List(i) = t { Some(*i) } else { None }).collect();
        let mut old: Vec<Option<(String, Value)>> = std::mem::take(&mut self.playlists).into_iter().map(Some).collect();
        self.playlists = seq.iter().map(|&i| old[i].take().expect("each playlist once")).collect();
        self.playlist = seq.iter().position(|&i| i == self.playlist).unwrap_or(0);
        self.strip = order.iter().map(|t| if let Tab::Mode(m) = t { *m } else { "list" }).collect();
    }

    /// The last playlist stays: an empty list would bring back a default one on the next start.
    pub fn remove_playlist(&mut self, i: usize) {
        if i >= self.playlists.len() || self.playlists.len() == 1 {
            return;
        }
        self.store_live();
        if let Some(slot) = self.list_slot(i) {
            self.strip.remove(slot);
        }
        self.playlists.remove(i);
        if i < self.playlist || self.playlist == self.playlists.len() {
            self.playlist -= 1;
        }
        if i <= self.playlist + 1 {
            let p = self.playlists[self.playlist].1.clone();
            self.load_preset(&p); // the open one may have been the one removed
        }
    }

    /// Opens playlist `i`: the live settings are stored first.
    pub fn open_playlist(&mut self, i: usize) {
        if i < self.playlists.len() && i != self.playlist {
            self.store_live();
            self.playlist = i;
            let p = self.playlists[i].1.clone();
            self.load_preset(&p);
        }
    }

    pub fn from_json(v: &Value) -> Ui {
        let mode = v["recommendation_mode"].as_str().filter(|m| MODES.contains(m)).unwrap_or("pp").to_string();
        let all: Vec<i64> = SUPPORTED_KEYS.collect();
        let keys = [("pp".to_string(), normalize_keys(&v["pp_keys"], &all)), ("nps".to_string(), normalize_keys(&v["nps_keys"], &[7])), ("skills".to_string(), normalize_keys(&v["skills_keys"], &[7]))].into_iter().collect();
        let keys_auto = ["nps", "skills"].into_iter().filter(|m| v[format!("{m}_keys")].is_null()).map(str::to_string).collect();
        let d = &v["display"];
        let flag = |k: &str| d[k].as_bool().unwrap_or(true);
        let selected: Vec<String> = v["practice_skills"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect());
        let mut ui = Ui {
            mode, keys, keys_auto, practice: skills::normalize(&selected), skill_weights: per_skill(&v["skill_weights"]),
            skill_avoid: per_skill(&v["skill_avoid"]), skill_other: v["skill_other"].as_f64().unwrap_or(0.),
            nps_focus: normalize_focus(v["nps_focus"].as_f64().unwrap_or(0.5)), web_bias: ["nps", "skills"].into_iter().map(|m| (m.to_string(), normalize_focus(v["web_bias"][m].as_f64().unwrap_or(0.5)))).collect(),
            web_weighted: ["nps", "skills"].into_iter().map(|m| (m.to_string(), v["web_weighted"][m].as_bool().unwrap_or(true))).collect(),
            by_count: ["nps", "skills"].into_iter().map(|m| (m.to_string(), v["by_count"][m].as_bool().unwrap_or(false))).collect(),
            key_weights: ["nps", "skills"].into_iter().map(|m| (m.to_string(), v["key_weights"][m].as_object().map_or_else(BTreeMap::new, |o| o.iter().filter_map(|(k, w)| Some((k.parse().ok().filter(|k| SUPPORTED_KEYS.contains(k))?, w.as_f64()?.clamp(0.1, 10.)))).collect()))).collect(),
            converts: ["nps", "skills"].into_iter().map(|m| (m.to_string(), v["converts"][m].as_object().map_or_else(BTreeMap::new, |o| o.iter().filter_map(|(k, s)| {
                let target: i64 = k.parse().ok().filter(|k| SUPPORTED_KEYS.contains(k))?;
                let sources = s["sources"].as_array().map_or_else(|| default_sources(target), |a| a.iter().filter_map(|x| x.as_i64()).filter(|x| SUPPORTED_KEYS.contains(x) && *x != target).collect());
                Some((target, Setting { on: s["on"].as_bool().unwrap_or(false), sources, often: normalize_focus(s["often"].as_f64().unwrap_or(0.5)), weighted: s["weighted"].as_bool().unwrap_or(true) }))
            }).collect()))).collect(),
            targets: ["nps", "skills"].into_iter().map(|m| (m.to_string(), normalize_target(v[format!("{m}_acc_target")].as_f64().unwrap_or(DEFAULT_TARGET)))).collect(),
            rates: ["nps", "skills"].into_iter().map(|m| (m.to_string(), v[format!("{m}_rates")].as_array().and_then(|a| Some(normalize_rates((a.first()?.as_f64()?, a.get(1)?.as_f64()?)))).unwrap_or(DEFAULT_RATES))).collect(),
            pp: pp_settings(&v["pp_settings"]), mix: pp_mix(&v["pp_mix"]),
            playlists: v["playlists"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|p| Some((p["name"].as_str()?.to_string(), p["preset"].clone()))).collect()),
            playlist: v["playlist"].as_u64().unwrap_or(0) as usize,
            strip: v["tab_strip"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|t| ["pp", "nps", "list"].into_iter().find(|m| t.as_str() == Some(*m))).collect()),
            skills_focus: v["skills_focus"].as_f64().filter(|f| f.is_finite()).map_or(0., |f| f.clamp(FOCUS_OFF, 1.)),
            skills_mix: v.get("skills_mix").map_or(PpMix { on: false, ..PpMix::default() }, pp_mix),
            display: Display { title: flag("title"), version: flag("version"), info: flag("info"), prediction: flag("prediction"), rating: flag("rating"), nps: flag("nps"), timeline: flag("timeline"), status: flag("status"), skills: d["skills"].as_u64().map_or(3, |x| x as usize) },
            next_action: "auto".to_string(), auto_next: v["auto_next"].as_bool().unwrap_or(true), keep_above: v["keep_above"].as_bool().unwrap_or(false), overlay: cfg!(windows) && v["overlay"].as_bool().unwrap_or(false),
            details: v["details"].as_bool().unwrap_or(false),
            // Defaults: the user's own sizes on G835LX (2026-10-30; main window measured on screen 2026-11-02).
            win: pair(v, "w", "h", (442., 238.)), recs_win: pair(v, "recs_w", "recs_h", (442., 420.)),
            win_pos: point(&v["pos"]), recs_pos: point(&v["recs_pos"]), recs_open: v["recs_open"].as_bool().unwrap_or(false),
            stats_keys: if v["stats_keys"] == "all" { None } else { Some(v["stats_keys"].as_i64().filter(|k| SUPPORTED_KEYS.contains(k)).unwrap_or(7)) },
            expanded: v["stats_expanded"].as_array().map_or_else(BTreeSet::new, |a| a.iter().filter_map(|x| { let (k, g) = x.as_str()?.split_once(':')?; Some((k.parse().ok()?, g.to_string())) }).collect()),
            chart_hints: v["chart_hints"].as_u64().map_or(0, |n| n as u32),
        };
        if ui.playlists.is_empty() {
            // Before custom playlists: the Skills tab's settings become the first one.
            let p = ui.preset();
            ui.playlists.push(("Skills".to_string(), p));
        }
        ui.playlist = ui.playlist.min(ui.playlists.len() - 1);
        // One PP, one NPS and a "list" per playlist, whatever was saved: missing ones go where they were before tabs moved.
        let mut seen = Vec::new();
        ui.strip.retain(|t| *t == "list" || { let new = !seen.contains(t); seen.push(*t); new });
        for (at, m) in [(0, "pp"), (1, "nps")] {
            if !ui.strip.contains(&m) {
                ui.strip.insert(at.min(ui.strip.len()), m);
            }
        }
        let lists = ui.strip.iter().filter(|t| **t == "list").count();
        for _ in lists..ui.playlists.len() {
            ui.strip.push("list");
        }
        for _ in ui.playlists.len()..lists {
            let last = ui.strip.iter().rposition(|t| *t == "list").expect("a list token");
            ui.strip.remove(last);
        }
        ui
    }

    pub fn to_json(&self) -> Value {
        let keys = |m: &str| if self.keys_auto.contains(m) { Value::Null } else { json!(self.keys[m]) };
        let d = &self.display;
        let (p, m) = (&self.pp, &self.mix);
        let mut v = json!({"mode": "auto", "recommendation_mode": self.mode, "next_action": self.next_action, "auto_next": self.auto_next, "keep_above": self.keep_above, "overlay": self.overlay,
               "pp_keys": self.keys["pp"], "nps_keys": keys("nps"), "skills_keys": keys("skills"), "practice_skills": self.practice, "skill_weights": self.skill_weights, "skill_avoid": self.skill_avoid, "skill_other": self.skill_other, "nps_focus": self.nps_focus, "web_bias": self.web_bias,
               "web_weighted": self.web_weighted, "key_weights": self.key_weights.iter().map(|(m, t)| (m.clone(), Value::Object(t.iter().map(|(k, w)| (k.to_string(), json!(w))).collect()))).collect::<serde_json::Map<_, _>>(), "converts": self.converts.iter().map(|(m, t)| (m.clone(), Value::Object(t.iter().map(|(k, s)| (k.to_string(), json!({"on": s.on, "sources": s.sources, "often": s.often, "weighted": s.weighted}))).collect()))).collect::<serde_json::Map<_, _>>(),
               "nps_acc_target": self.targets["nps"], "skills_acc_target": self.targets["skills"], "nps_rates": [self.rates["nps"].0, self.rates["nps"].1], "skills_rates": [self.rates["skills"].0, self.rates["skills"].1],
               "pp_settings": {"min_gain": p.min_gain, "min_chance": p.min_chance, "ht": p.ht, "ht_chance": p.ht_chance, "ht_gain": p.ht_gain, "dt": p.dt, "push": p.push, "cooldown_days": p.cooldown_days, "top100": p.top100},
               "pp_mix": {"on": m.on, "every": m.every, "min_gain": m.min_gain, "chance": m.chance, "ht_chance": m.ht_chance},
               "skills_focus": self.skills_focus, "skills_mix": mix_json(&self.skills_mix), "playlist": self.playlist,
               "playlists": self.playlists.iter().map(|(n, p)| json!({"name": n, "preset": p})).collect::<Vec<_>>(), "tab_strip": self.strip,
               "display": {"title": d.title, "version": d.version, "info": d.info, "prediction": d.prediction, "rating": d.rating, "nps": d.nps, "timeline": d.timeline, "status": d.status, "skills": d.skills}, "details": self.details,
               "w": self.win.0, "h": self.win.1, "recs_w": self.recs_win.0, "recs_h": self.recs_win.1,
               "recs_pos": self.recs_pos.map(|p| [p.0, p.1]), "pos": self.win_pos.map(|p| [p.0, p.1]), "recs_open": self.recs_open,
               "stats_keys": self.stats_keys.map_or(json!("all"), |k| json!(k)), "stats_expanded": self.expanded.iter().map(|(k, g)| format!("{k}:{g}")).collect::<Vec<_>>(), "chart_hints": self.chart_hints});
        v["by_count"] = json!(self.by_count); // past json!'s recursion limit
        v
    }

    pub fn save(&mut self) {
        self.store_live();
        let path = file();
        let tmp = path.with_extension("json.tmp");
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if std::fs::write(&tmp, self.to_json().to_string()).is_ok() {
            let _ = std::fs::rename(tmp, path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_python_ui_file_keeps_its_choices_through_a_round_trip() {
        let old = json!({"mode": "manual", "recommendation_mode": "skills", "pp_keys": [4, 7], "nps_keys": [7], "skills_keys": null, "practice_skills": ["!ln/release", "chordstream"], "skill_weights": {"chordstream": 2.0},
                         "nps_focus": 0.25, "web_bias": {"nps": 0.1, "skills": 0.9}, "nps_acc_target": 0.93, "pp_settings": {"min_gain": 2.0, "ht": false}, "pp_mix": {"on": true, "every": 3}, "display": {"nps": false, "skills": 5}, "rate": 1.05, "w": 500, "h": 300, "pos": [10.0, 20.0], "recs_open": true});
        let ui = Ui::from_json(&old);
        assert_eq!(ui.keys["pp"], [4, 7]);
        assert!(ui.keys_auto.contains("skills") && !ui.keys_auto.contains("nps"));
        assert!(!ui.pp.ht && ui.mix.on && !ui.display.nps);
        let back = ui.to_json();
        assert_eq!((back["mode"].as_str(), back["next_action"].as_str()), (Some("auto"), Some("auto")));
        for k in ["recommendation_mode", "pp_keys", "skills_keys", "practice_skills", "skill_weights", "nps_focus", "recs_open", "pos"] {
            assert_eq!(back[k], old[k], "{k}");
        }
        assert_eq!(Ui::from_json(&back).to_json(), back);
    }

    #[test]
    fn playlists_keep_their_own_presets_and_the_skills_tab_becomes_the_first() {
        let mut ui = Ui::from_json(&json!({"skills_keys": [10], "practice_skills": ["chordjack"], "skills_acc_target": 0.9, "web_bias": {"nps": 0.2, "skills": 0.7}}));
        assert_eq!(ui.playlists.len(), 1);
        assert_eq!(ui.playlists[0].0, "Skills");
        assert_eq!(ui.playlists[0].1["practice_skills"], json!(["chordjack"]));
        ui.playlists.push(("Jacks".into(), ui.preset()));
        ui.open_playlist(1);
        ui.practice = vec!["ln".into()];
        ui.keys.insert("skills".into(), vec![7]);
        ui.skills_focus = FOCUS_OFF;
        ui.open_playlist(0);
        assert_eq!((ui.practice.clone(), ui.keys["skills"].clone(), ui.targets["skills"]), (vec!["chordjack".to_string()], vec![10], 0.9));
        assert_eq!(ui.web_bias["nps"], 0.2, "the NPS tab is not part of a preset");
        ui.open_playlist(1);
        assert_eq!((ui.practice.clone(), ui.keys["skills"].clone(), ui.skills_focus), (vec!["ln".to_string()], vec![7], FOCUS_OFF));
        let back = Ui::from_json(&ui.to_json());
        assert_eq!((back.playlist, back.playlists.len(), back.practice.clone()), (1, 2, vec!["ln".to_string()]));
        ui.add_playlist("Copy".into());
        assert_eq!((ui.playlist, ui.playlists[2].0.as_str(), ui.practice.clone()), (2, "Copy", vec!["ln".to_string()]));
        ui.move_tab(4, 2);
        assert_eq!((ui.playlist, ui.playlists[0].0.as_str()), (0, "Copy"));
        ui.move_tab(3, 4);
        assert_eq!(ui.playlists.iter().map(|p| p.0.as_str()).collect::<Vec<_>>(), ["Copy", "Jacks", "Skills"]);
        ui.move_tab(0, 4);
        assert_eq!(ui.tabs(), [Tab::Mode("nps"), Tab::List(0), Tab::List(1), Tab::List(2), Tab::Mode("pp")], "PP moves like a playlist");
        assert_eq!(Ui::from_json(&ui.to_json()).tabs(), ui.tabs());
        ui.remove_playlist(0);
        assert_eq!((ui.playlist, ui.practice.clone()), (0, vec!["ln".to_string()]), "the open one removed: its neighbour opens");
        ui.remove_playlist(1);
        ui.remove_playlist(0);
        assert_eq!(ui.playlists.len(), 1, "the last one stays");
    }
}

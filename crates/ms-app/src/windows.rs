//! The small windows: Settings (General and Website charts tabs), profile, history. Each is
//! a real window the compositor places, so it stays on screen wherever the small main window sits (a popup inside the main
//! window would be cut off by it).
use crate::app::{App, Mail, Msg};
use crate::fmt::{self, keys_label};
use crate::theme::{self, DIM};
use eframe::egui::{self, RichText, ScrollArea, Ui, ViewportBuilder, ViewportCommand, ViewportId};
use ms_data::paths::cache;
use ms_data::store::{kv_get, now};
use ms_data::webmaps::{self, Estimate, Want};
use ms_rec::catalog::SUPPORTED_KEYS;
use ms_rec::worker::Task;
use serde_json::Value;
use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub const DEFAULT_SKILLS: usize = 3;
pub const MAX_SKILLS: usize = ms_calc::card::CARD_MAX;

/// − value + : the buttons step 0.5 ★, the box drags or takes a typed value (two sliders didn't fit, user 2026-11-11).
fn stepper(ui: &mut Ui, v: &mut f64, range: std::ops::RangeInclusive<f64>, tip: &str) -> bool {
    let before = *v;
    if ui.small_button("−").clicked() {
        *v -= 0.5;
    }
    ui.add(egui::DragValue::new(v).range(range.clone()).speed(0.05).fixed_decimals(1)).on_hover_text(tip);
    if ui.small_button("+").clicked() {
        *v += 0.5;
    }
    *v = v.clamp(*range.start(), *range.end());
    *v != before
}

fn window(ctx: &egui::Context, id: &str, title: &str, size: [f32; 2], above: bool, mut add: impl FnMut(&mut Ui)) -> bool {
    let mut builder = ViewportBuilder::default().with_title(title).with_inner_size(theme::window_in(ctx, size)).with_app_id("maniascope");
    if above {
        builder = builder.with_always_on_top();
    }
    ctx.show_viewport_immediate(ViewportId::from_hash_of(id), builder, |ui, _| {
        crate::theme::dark_frame(ui.ctx());
        theme::scaled(ui, 1.); // settings and dialogs keep their size
        theme::backdrop(ui, &[], false); // the fixed one (user 2026-11-21: settings don't follow the map)
        let used = egui::CentralPanel::default().frame(egui::Frame::new().inner_margin(10)).show(ui, |ui| {
            add(ui);
            ui.min_rect().max + egui::vec2(10., 10.) // + the panel margin
        }).inner;
        // Grows to what it shows, up to the screen: a fixed size cut off the buttons (user 2026-11-13).
        // content_rect, not viewport().inner_rect: Wayland gives windows no position, so inner_rect is always None there.
        let (r, monitor) = ui.ctx().input(|i| (i.content_rect(), i.viewport().monitor_size));
        let want = used.to_vec2().max(r.size()).min(monitor.unwrap_or(egui::vec2(f32::MAX, f32::MAX)) * 0.9);
        if want.x > r.width() + 1. || want.y > r.height() + 1. {
            crate::resize::inner(ui.ctx(), title, want);
        }
        ui.ctx().input(|i| i.viewport().close_requested())
    })
}

#[derive(Default)]
pub struct WebInfo {
    pub progress: Value,
    pub running: bool,
    pub estimate: Option<Estimate>,
    pub left: f64,
    pub done: f64,
    pub error: Option<String>,
    pub reopen: Option<(String, Vec<Want>)>, // the batch closing the app paused: command, choice
    opened: Option<(Vec<Want>, HashMap<i64, (f64, f64)>)>, // for the tab's first show: the saved choice, the suggested ★ ranges
}

#[derive(Default)]
pub struct WebView {
    polling: bool,
    pub continued: bool,
    next: Option<Instant>,
    child: Option<Child>,
    pub info: Option<WebInfo>,
    ready: Option<i64>,
    wants: Vec<Want>, // one per keymode, on or off
    suggest: HashMap<i64, (f64, f64)>, // ★ range the player's plays suggest, per keymode (read when the window opens)
    saved: bool, // the tab has asked for its saved choice
    loaded: bool, // … and has it
    pp_due: Option<Instant>, // next new-ranked-PP-maps check (None: at start)
    pp: bool, // the child is the PP fetch, not the player's batch: closing the app doesn't mark it for the next launch
    estimated: Option<Vec<Want>>, // the choice the shown counts are for: recounted only when it or the lists change
    delete_asked: bool,
    deleted: Option<String>,
}

impl WebView {
    #[cfg(test)]
    pub fn loaded(&self) -> bool {
        self.loaded
    }

    fn alive(&mut self) -> bool {
        self.child.as_mut().is_some_and(|c| matches!(c.try_wait(), Ok(None)))
    }
}

/// Both runs append: one run's error survives the next.
fn log() -> std::io::Result<std::fs::File> {
    std::fs::create_dir_all(cache())?;
    std::fs::OpenOptions::new().create(true).append(true).open(cache().join("webmaps.log"))
}

/// osu! ★ range (1.0×) covering the player's recent plays at 0.9–1.2×, for each keymode with 30 of them. One pass
/// over the rows for all keymodes, off the UI thread: once per keymode on it, the tab froze for seconds when it
/// opened (user 2026-11-22; 135 ms a keymode on the user's database, 260 ms more where the fallback ran).
fn suggested_stars(db: &rusqlite::Connection) -> HashMap<i64, (f64, f64)> {
    let mut stars: HashMap<i64, Vec<f64>> = HashMap::new();
    if let Ok(mut q) = db.prepare("SELECT info FROM events WHERE kind='start' ORDER BY id DESC LIMIT 400") {
        for info in q.query_map([], |r| r.get::<_, Option<String>>(0)).into_iter().flatten().flatten().flatten() {
            let Ok(v) = serde_json::from_str::<Value>(&info) else { continue };
            let f = &v["features"];
            if let (Some(s), Some(k)) = (f["stars"].as_f64().filter(|s| *s != 0.), f["keys"].as_i64()) {
                stars.entry(k).or_default().push(s);
            }
        }
    }
    if SUPPORTED_KEYS.into_iter().any(|k| stars.get(&k).map_or(0, Vec::len) < 30) {
        // A new install has no tracked starts yet: its imported lazer plays say the same thing.
        let mut imported: HashMap<i64, Vec<f64>> = HashMap::new();
        if let Ok(mut q) = db.prepare("SELECT f.data FROM scores s JOIN feats f ON f.key = s.sha256 || '@' || printf('%.3f', s.rate) || '|' || ? WHERE s.src != 'public' AND s.rate BETWEEN .9 AND 1.2 ORDER BY s.played DESC LIMIT 400") {
            for data in q.query_map([ms_rec::engine::CALC_ID], |r| r.get::<_, Vec<u8>>(0)).into_iter().flatten().flatten() {
                if let Ok(ms_calc::val::Val::Dict(f)) = ms_data::codec::decode(&data) {
                    if f.fget("stars", 0.) != 0. {
                        imported.entry(f.fget("keys", 0.) as i64).or_default().push(f.f("stars"));
                    }
                }
            }
        }
        for (k, v) in imported {
            if stars.get(&k).map_or(0, Vec::len) < 30 {
                stars.insert(k, v);
            }
        }
    }
    stars.into_iter().filter(|(_, v)| v.len() >= 30).map(|(k, mut v)| {
        v.sort_by(f64::total_cmp);
        (k, ((v[v.len() / 10] / 1.2 * 2.).round() / 2., (v[v.len() * 9 / 10] / 0.9 * 2.).round() / 2.))
    }).collect()
}

/// `opening`: the tab is being shown for the first time since it was closed; the player's database, for what it
/// shows first (the saved choice and the suggested ★ ranges).
fn poll(mail: Mail, view: Option<Vec<Want>>, opening: Option<std::path::PathBuf>) {
    std::thread::spawn(move || {
        let read = || -> rusqlite::Result<WebInfo> {
            let db = webmaps::state_db()?;
            let opened = opening.as_ref().map(|path| (webmaps::wants(&db), ms_data::store::connect(path).map(|rec| suggested_stars(&rec)).unwrap_or_default()));
            let running = webmaps::running();
            let state = webmaps::state(&db);
            let p = state.get("progress").cloned().unwrap_or(Value::Null);
            let estimate = match view.as_ref().filter(|_| !running) {
                Some(wants) => Some(webmaps::estimate(&db, wants, ms_rec::engine::CALC_ID)?),
                None => None,
            };
            let num = |k: &str| p[k].as_i64().unwrap_or(0);
            let (mut left, mut done) = (0., 0.);
            if running && !p["fetch_total"].is_null() {
                left = webmaps::seconds(&db, num("fetch_total") - num("fetched"), num("analyse_total") - num("analysed"));
                done = 1. - left / webmaps::seconds(&db, num("fetch_total"), num("analyse_total")).max(1.);
            }
            let reopen = state.get("reopen").map(|_| (state.get("command").and_then(Value::as_str).unwrap_or("crawl").to_string(), webmaps::wants(&db)));
            Ok(WebInfo { running, estimate, left, done, progress: p, error: None, reopen, opened })
        };
        mail.send(Msg::Web(Box::new(read().unwrap_or_else(|e| WebInfo { error: Some(e.to_string()), ..WebInfo::default() }))));
    });
}

impl App {
    pub fn web_label(&self) -> String {
        let Some(info) = &self.web.info else { return "Website charts…".into() };
        let phase = info.progress["phase"].as_str().unwrap_or("");
        if info.running && phase != "check" {
            format!("Website charts… {:.0}%", 100. * info.done.max(0.))
        } else if matches!(phase, "paused" | "failed") {
            "Website charts… paused".into()
        } else {
            "Website charts…".into()
        }
    }

    fn web_poll(&mut self) {
        if self.web.polling {
            return;
        }
        self.web.polling = true;
        if !self.open.web {
            self.web.estimated = None; // recount when the window opens again
        }
        let view = (self.open.web && self.web.loaded && self.web.estimated.as_ref() != Some(&self.web.wants)).then(|| self.web.wants.clone());
        if view.is_some() {
            self.web.estimated = view.clone();
        }
        let opening = (self.web.saved && !self.web.loaded).then(|| self.db.path().map(std::path::PathBuf::from)).flatten();
        poll(self.mail.clone(), view, opening);
    }

    pub fn web_show(&mut self, mut info: WebInfo) {
        self.web.polling = false;
        if let Some((wants, suggest)) = info.opened.take().filter(|_| self.web.saved && !self.web.loaded) {
            (self.web.wants, self.web.suggest, self.web.loaded) = (wants, suggest, true);
            if !self.web.wants.iter().any(|w| w.on) {
                // first open: the playlists' keymodes
                let chosen: Vec<i64> = self.s.keys["nps"].iter().chain(&self.s.keys["skills"]).copied().collect();
                for k in chosen {
                    self.web_want(k).on = true;
                }
            }
        }
        let ready = info.progress["ready"].as_i64().unwrap_or(0);
        if self.web.ready.is_some_and(|r| r != ready) {
            self.q.put(Task::WebUpdated); // new website charts join the open playlist without a restart
        }
        let was = self.web.info.as_mut().map(|i| (i.running, i.estimate.take()));
        if self.web.ready.is_some_and(|r| r != ready) || info.error.is_some() || was.as_ref().is_some_and(|w| w.0 && !info.running) {
            self.web.estimated = None; // a crawl or check finished: the counts changed
        }
        if info.estimate.is_none() && info.error.is_none() {
            info.estimate = was.and_then(|w| w.1); // shown until the recount arrives
        }
        self.web.ready = Some(ready);
        // Once per launch, after the PP fetch spawned at start has ended; the child clears the flag a moment after it starts.
        if let Some((command, wants)) = info.reopen.clone().filter(|_| !info.running && !self.web.continued && !self.web.alive()) {
            self.web_run(&command, &wants);
            self.web.continued = true;
        }
        let child = self.web.alive();
        // at once: the counts for a choice that just loaded, or the choice itself when this poll left before the tab asked
        let wait = if self.open.web && (self.web.estimated.is_none() || !self.web.loaded) && info.error.is_none() { 0 } else { 2 };
        self.web.next = (self.open.web || info.running || child).then(|| Instant::now() + Duration::from_secs(wait));
        self.web.info = Some(info);
    }

    /// New ranked PP maps (user 2026-11-13): at start and every 24 h when on, never during a play or another crawl.
    fn pp_auto_tick(&mut self) {
        if self.web.pp_due.is_some_and(|t| Instant::now() < t) || !kv_get(&self.db, ms_rec::worker::AUTO_RANKED).and_then(|v| v.as_bool()).unwrap_or(false) {
            return;
        }
        let busy = self.game.load(std::sync::atomic::Ordering::Relaxed) || self.web.alive() || webmaps::running();
        let Some(floor) = kv_get(&self.db, ms_rec::worker::PP_FLOOR).and_then(|v| v.as_f64()).filter(|_| !busy && !self.s.keys["pp"].is_empty()) else { return };
        let (Ok(exe), Ok(log)) = (std::env::current_exe(), log()) else { return };
        // keymodes without a play of the player's are left out: nothing says which of their maps could enter the top 100 (user 2026-11-13)
        let played = ms_data::store::played_keys(&self.db, i64::MAX, 0.).unwrap_or_default();
        let list = self.s.keys["pp"].iter().filter(|k| played.contains(k)).map(i64::to_string).collect::<Vec<_>>().join(",");
        if list.is_empty() {
            return;
        }
        let mut c = Command::new(exe);
        c.args(["--webmaps", "crawl", "--keys", &list, "--pp-floor", &format!("{floor:.2}")]);
        c.stdin(Stdio::null()).stderr(log.try_clone().map_or(Stdio::null(), Stdio::from)).stdout(log);
        self.web.child = c.spawn().ok();
        self.web.pp = true;
        self.web.pp_due = Some(Instant::now() + Duration::from_secs(86400));
        self.web_poll();
    }

    pub fn web_tick(&mut self, ctx: &egui::Context) {
        self.pp_auto_tick();
        if self.web.ready.is_none() && !self.web.polling {
            self.web_poll(); // a paused download shows on the button after a restart
        }
        if let Some(at) = self.web.next {
            let left = at.saturating_duration_since(Instant::now());
            if left.is_zero() {
                self.web.next = None;
                self.web_poll();
            } else {
                ctx.request_repaint_after(left);
            }
        }
    }

    fn web_run(&mut self, command: &str, wants: &[Want]) {
        let Ok(exe) = std::env::current_exe() else { return };
        let Ok(log) = log() else { return };
        let mut c = Command::new(exe);
        c.args(["--webmaps", command, "--wants", &serde_json::to_string(wants).unwrap_or_default()]);
        c.stdin(Stdio::null()).stderr(log.try_clone().map_or(Stdio::null(), Stdio::from)).stdout(log);
        self.web.child = c.spawn().ok();
        self.web.pp = false;
        self.web_poll();
    }

    pub fn web_pause(&mut self) {
        if self.web.alive() && !self.web.pp { // paused, not killed: the next launch continues
            let _ = std::fs::write(webmaps::web_dir().join("stop"), "");
            if let Ok(db) = webmaps::state_db() {
                let _ = webmaps::kv_set(&db, "reopen", &Value::Bool(true));
            }
        }
    }

    fn web_ui(&mut self, ui: &mut Ui) {
        if !self.web.saved {
            (self.web.saved, self.web.loaded) = (true, false);
            self.web_poll();
        }
        if !self.web.loaded {
            theme::dim(ui, "Loading…"); // read off this thread: `web_show` brings it
            return;
        }
        ui.label("Adds maps you don't have from osu.direct to the NPS and Skills lists.").on_hover_text("Downloads the chart text only (no audio) and analyses it at low priority; Next offers the download in osu!. Closing ManiaScope pauses it; it continues the next time you open the app.");
        let mut auto = kv_get(&self.db, ms_rec::worker::AUTO_DOWNLOAD).and_then(|v| v.as_bool()).unwrap_or(true);
        if theme::checkbox(ui, &mut auto, "Next downloads maps I don't have").on_hover_text("On: a picked website map or website convert is downloaded and imported into osu! by itself.\nOff: Next opens the map in osu! so you can look at it first; a website convert opens its original, then Settings → Convert this map adds the convert.").changed() {
            let _ = ms_data::store::kv_set(&self.db, ms_rec::worker::AUTO_DOWNLOAD, &Value::Bool(auto));
        }
        let mut ranked = kv_get(&self.db, ms_rec::worker::AUTO_RANKED).and_then(|v| v.as_bool()).unwrap_or(false);
        if theme::checkbox(ui, &mut ranked, "Fetch new ranked PP maps").on_hover_text("By itself, at start and every 24 hours, never while you play: ranked maps newer than the population data, on your PP tab's keymodes you have played, from the star rating where a perfect play enters your top 100. No Start needed.").changed() {
            let _ = ms_data::store::kv_set(&self.db, ms_rec::worker::AUTO_RANKED, &Value::Bool(ranked));
            self.web.pp_due = None;
        }
        let (running, phase) = self.web.info.as_ref().map_or((false, String::new()), |i| (i.running, i.progress["phase"].as_str().unwrap_or("").to_string()));
        let per_key = self.web.info.as_ref().and_then(|i| i.estimate.as_ref()).map(|e| e.per_key.clone()).unwrap_or_default();
        let c = ms_rec::engine::commas;
        let i64n = |v: i64| c(v.max(0) as usize);
        let mut changed = false;
        // a table can't wrap: narrower than it, it scrolls sideways
        egui::ScrollArea::horizontal().show(ui, |ui| ui.add_enabled_ui(!running, |ui| {
            egui::Grid::new("wants").spacing([10., 4.]).show(ui, |ui| {
                ui.label("");
                ui.label("Ranked").on_hover_text("Ranked, approved, qualified and loved: a few minutes to list");
                ui.label("Unranked").on_hover_text("Pending, WIP and graveyard: most of the ~100k mania sets, 15–30 min to list the first time");
                ui.label("osu! ★ at 1.0×");
                ui.label("Min plays").on_hover_text("Least playcount on osu!: keeps only maps people play");
                ui.label("New");
                ui.end_row();
                for k in SUPPORTED_KEYS {
                    let suggest = self.web.suggest.get(&k).copied();
                    let w = self.web_want(k);
                    changed |= theme::checkbox(ui, &mut w.on, format!("{k}K")).changed();
                    if w.on {
                        for (name, _) in webmaps::GROUPS {
                            let group = webmaps::statuses(&[name]);
                            let mut on = group.iter().all(|s| w.statuses.contains(s));
                            if theme::checkbox(ui, &mut on, "").on_hover_text(name).changed() {
                                w.statuses.retain(|s| !group.contains(s));
                                if on {
                                    w.statuses.extend(&group);
                                }
                                changed = true;
                            }
                        }
                        ui.horizontal(|ui| {
                            let tip = suggest.map_or_else(|| "− / +, drag or type".to_string(), |(a, b)| format!("− / +, drag or type. Your plays suggest {a}–{b}."));
                            let (mut lo, mut hi) = w.stars;
                            changed |= stepper(ui, &mut lo, 0.0..=hi, &tip);
                            ui.label("–");
                            changed |= stepper(ui, &mut hi, lo..=15., &tip);
                            w.stars = (lo, hi);
                        });
                        changed |= ui.add(egui::DragValue::new(&mut w.min_plays).range(0..=10_000_000).speed(20.)).on_hover_text("Drag or type").changed();
                        ui.label(per_key.get(&k).map_or_else(|| "0".to_string(), |n| i64n(*n)));
                    }
                    ui.end_row();
                }
            });
        }));
        if changed {
            self.web_poll();
        }
        let mut go_ok = false;
        let on = self.web.wants.iter().any(|w| w.on && !w.statuses.is_empty());
        match &self.web.info {
            None => { ui.label("Counting…"); }
            Some(info) => {
                let p = &info.progress;
                let num = |k: &str| p[k].as_i64().unwrap_or(0);
                if let Some(e) = &info.error {
                    ui.label(format!("Could not read the website chart state: {e}"));
                } else if running && phase == "check" {
                    ui.label(format!("Listing osu.direct: {} map sets…", i64n(num("sets"))));
                } else if running {
                    ui.label(format!("Downloaded {} / {} · analysed {} / {} · {} left", i64n(num("fetched")), i64n(num("fetch_total")), i64n(num("analysed")), i64n(num("analyse_total")), fmt::duration(info.left)));
                    theme::bar(ui, info.done as f32);
                } else if let Some(est) = &info.estimate {
                    if phase == "failed" {
                        ui.label(format!("Stopped: {}. Resume retries.", p["error"].as_str().unwrap_or("?")));
                    }
                    if !est.unlisted.is_empty() {
                        ui.label(format!("{} not listed yet: Start lists it first.", keys_label(&est.unlisted)));
                    }
                    if !on {
                        ui.label("Tick a keymode and a status.");
                    } else if est.fetch > 0 || est.analyse > 0 {
                        ui.label(format!("{} maps · {} · about {}", i64n(est.fetch), fmt::size(est.new_bytes as f64), fmt::duration(est.seconds)))
                            .on_hover_text(format!("{} to analyse. On disk now {}, afterwards about {}.", i64n(est.analyse), fmt::size(est.disk as f64), fmt::size(est.disk as f64 + est.new_bytes as f64)));
                    } else if est.unlisted.is_empty() {
                        ui.label(format!("Up to date · {} maps in the lists", i64n(est.ready)));
                    }
                    if let Some(t) = est.checked.filter(|t| *t != 0.) {
                        theme::dim(ui, format!("Listing checked {} ago", fmt::duration(now() - t)));
                    }
                    // Start lists unchecked keymodes itself: a first run needs no separate Check (fresh install).
                    go_ok = on && (est.fetch > 0 || est.analyse > 0 || !est.unlisted.is_empty() || phase == "paused");
                }
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min).with_main_wrap(true), |ui| {
            let label = if running { "Pause" } else if matches!(phase.as_str(), "check" | "download" | "analyse" | "paused" | "failed") { "Resume" } else { "Start" };
            if ui.add_enabled(running || go_ok, theme::Btn::new(label)).clicked() {
                if running {
                    let _ = std::fs::write(webmaps::web_dir().join("stop"), ""); // pauses after the current chart
                } else {
                    self.web_run("crawl", &self.web.wants.clone());
                }
            }
            if ui.add_enabled(!running && on, theme::Btn::new("Check for new charts")).on_hover_text("Only reads the listing, to see the counts first").clicked() {
                self.web_run("check", &self.web.wants.clone());
            }
        });
        // deleting them: a second click confirms (user 2026-11-21: with an "are you sure")
        let disk = self.web.info.as_ref().and_then(|i| i.estimate.as_ref()).map_or(0, |e| e.disk);
        if self.web.delete_asked {
            ui.add(egui::Label::new(RichText::new(format!("Delete every downloaded website map and its analysis ({})? Start downloads them again.", fmt::size(disk as f64))).color(theme::CHIP_OFF)).wrap());
            ui.horizontal_wrapped(|ui| {
                if theme::button(ui, "Yes, delete").clicked() {
                    self.web.delete_asked = false;
                    self.web.deleted = Some(webmaps::delete_all().map_or_else(|e| e, |n| format!("{} website maps deleted", ms_rec::engine::commas(n))));
                    self.q.put(Task::WebUpdated);
                    self.web_poll();
                }
                if theme::button(ui, "Cancel").clicked() {
                    self.web.delete_asked = false;
                }
            });
        } else if ui.add_enabled(!running && disk > 0, theme::Btn::new(format!("Delete website maps ({})…", fmt::size(disk as f64)))).on_hover_text("Frees the disk space of the downloaded website charts (text and analyses, no audio). Maps already in osu! stay.").clicked() {
            self.web.delete_asked = true;
        }
        if let Some(text) = &self.web.deleted {
            theme::dim(ui, text);
        }
    }

    /// The keymode's want, made with the ★ range its plays suggest the first time.
    fn web_want(&mut self, k: i64) -> &mut Want {
        let i = match self.web.wants.iter().position(|w| w.keys == k) {
            Some(i) => i,
            None => {
                let stars = self.web.suggest.get(&k).copied().unwrap_or((5., 10.));
                self.web.wants.push(Want { on: false, ..Want::new(k, stars) });
                self.web.wants.len() - 1
            }
        };
        &mut self.web.wants[i]
    }

    /// The Settings window: General, and the website charts (one window, user 2026-11-21).
    pub fn menu_ui(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.add(theme::Btn::new("General").selected(!self.open.web)).clicked() {
                (self.open.web, self.web.saved, self.web.loaded) = (false, false, false);
            }
            if ui.add(theme::Btn::new(self.web_label().replace('…', "")).selected(self.open.web)).on_hover_text("Maps you don't have, from the osu! website, for the NPS and Skills playlists").clicked() {
                self.open.web = true;
            }
        });
        ui.add_space(4.);
        ui.spacing_mut().item_spacing.y = 5.;
        if self.open.web {
            theme::card(ui, |ui| { ui.set_width(ui.available_width()); self.web_ui(ui) });
            return;
        }
        let mut close = false;
        let selected = self.analysis().map(|a| (a.keys, a.path.clone()));
        theme::card(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Convert this map to");
        ui.horizontal_wrapped(|ui| {
            for k in SUPPORTED_KEYS.filter(|k| selected.as_ref().map(|s| s.0) != Some(*k)) {
                if ui.add_enabled(selected.is_some(), theme::Btn::new(format!("{k}K"))).on_hover_text("Adds it to osu! as its own difficulty, without the playlist").clicked() {
                    if let Some((_, path)) = &selected {
                        self.q.put(Task::ConvertSelected { path: path.clone(), keys: k });
                    }
                    close = true;
                }
            }
        });
        });
        theme::card(ui, |ui| {
        ui.set_width(ui.available_width());
        if theme::checkbox(ui, &mut self.s.auto_next, "Auto next after finishing the pick").on_hover_text("Finishing the recommended map picks the next one and copies its search, as if you pressed Next").changed() {
            self.request_save();
        }
        let mut above = self.s.keep_above;
        if theme::checkbox(ui, &mut above, "Always on top").on_hover_text("Keep ManiaScope and the playlist above other windows. Not above exclusive fullscreen; for that add ManiaScope to tosu's in-game overlay.").changed() {
            self.s.keep_above = above;
            self.request_save();
            ui.ctx().send_viewport_cmd_to(ViewportId::ROOT, ViewportCommand::WindowLevel(level(above)));
            ui.ctx().send_viewport_cmd_to(playlist_id(), ViewportCommand::WindowLevel(level(above)));
        }
        if theme::checkbox(ui, &mut self.s.black_bg, "Black background (colour key for streaming)").on_hover_text("The main window and the playlist show nothing but black between their cards: key black out in OBS and only the cards stay.").changed() {
            self.request_save();
        }
        if cfg!(windows) {
            let mut on = self.s.overlay;
            if theme::checkbox(ui, &mut on, "ManiaScope in your tosu overlay").on_hover_text("If you run tosu yourself: ManiaScope adds its page there. In osu! press Ctrl+Shift+Space, right-click, add ManiaScope, then press Ctrl+Shift+Space again.").changed() {
                self.toggle_overlay(on);
            }
        }
        let tip = "Saves your own osu!mania replays and the charts they were played on in one zip, for improving the difficulty calculator. Replays include your player name. Nothing is uploaded: send the zip yourself if you want to.";
        // side by side (user 2026-11-22)
        ui.horizontal_wrapped(|ui| {
            if theme::button(ui, "osu! profile…").on_hover_text("Your username: your public top 100 plays fill in bests that lazer doesn't have locally").clicked() {
                self.open.profile = if self.open.profile.is_some() { None } else { Some(false) };
                self.open.profile_text = kv_get(&self.db, "website_username").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
            }
            if !self.open.share && theme::button(ui, "Share replays for calibration…").on_hover_text(tip).clicked() {
                self.open.share = true;
            }
        });
        // a second click confirms (user 2026-11-21), as deleting the website maps does
        if self.open.share {
            // on its own wrapped line: beside the buttons it widened the window (user 2026-11-22)
            ui.add(egui::Label::new("Save all your replays and their charts in one zip? It holds your player name; nothing is uploaded.").wrap());
            ui.horizontal_wrapped(|ui| {
                if theme::button(ui, "Yes, save").clicked() {
                    self.open.share = false;
                    self.share_replays();
                    close = true;
                }
                if theme::button(ui, "Cancel").clicked() {
                    self.open.share = false;
                }
            });
        }
        if self.open.profile == Some(false) {
            self.profile_ui(ui);
        }
        });
        theme::card(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::section(ui, "Main window shows");
            self.display_ui(ui);
        });
        if self.debug && theme::button(ui, "Play & prediction history…").clicked() {
            self.open.history = true;
            self.q.put(Task::History(false));
            close = true;
        }
        if close {
            self.open.menu = false;
        }
    }

    fn profile_ui(&mut self, ui: &mut Ui) {
        ui.label("Your osu! username or profile number. ManiaScope reads your public top 100 mania plays so PP suggestions know your bests. Empty = off.");
        let edit = ui.text_edit_singleline(&mut self.open.profile_text);
        let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let first = self.open.profile == Some(true);
        ui.horizontal_wrapped(|ui| {
            if theme::button(ui, "Save").clicked() || enter {
                self.q.put(Task::WebsiteUser(self.open.profile_text.clone()));
                self.open.profile = None;
            }
            if theme::button(ui, if first { "Skip" } else { "Cancel" }).clicked() {
                self.open.profile = None;
            }
        });
    }

    fn display_ui(&mut self, ui: &mut Ui) {
        let d = &mut self.s.display;
        let mut changed = false;
        // In the window's row order, one switch per row (user 2026-11-02).
        changed |= theme::checkbox(ui, &mut d.title, "Title and artist").changed();
        changed |= theme::checkbox(ui, &mut d.version, "Difficulty and mapper").changed();
        changed |= theme::checkbox(ui, &mut d.info, "Keys, dan, rate, length, LN %").changed();
        changed |= theme::checkbox(ui, &mut d.o2jam, "O2Jam level (7K)").on_hover_text("Right after the dan: the map's O2Jam level at the rate you play, read off this calculator's ratings (no extra calculation).").changed();
        changed |= theme::checkbox(ui, &mut d.prediction, "Expected accuracy, form, PP gain").on_hover_text("Shown once you have 50 plays in the keymode. PP gain on the PP tab, or for a PP map the NPS list mixed in.").changed();
        changed |= theme::checkbox(ui, &mut d.rating, "Difficulty rating").changed();
        ui.horizontal(|ui| {
            ui.label("Skills shown");
            ui.spacing_mut().slider_width = (ui.available_width() - 60.).clamp(40., ui.spacing().slider_width); // narrows with the window
            // A slider, not a drag field: a number box gave no hint it could change (user 2026-11-02). 0 hides the skills.
            changed |= ui.add(egui::Slider::new(&mut d.skills, 0..=MAX_SKILLS)).on_hover_text(format!("Default {DEFAULT_SKILLS}")).changed();
        });
        changed |= theme::checkbox(ui, &mut d.nps, "NPS").changed();
        changed |= theme::checkbox(ui, &mut d.timeline, "Difficulty timeline").changed();
        changed |= theme::checkbox(ui, &mut d.status, "Status line").on_hover_text("The small line at the bottom: import and analysis progress").changed();
        if changed {
            self.request_save();
        }
    }

    pub fn side_windows(&mut self, ctx: &egui::Context) {
        let above = self.s.keep_above;
        if self.open.profile == Some(true) && window(ctx, "profile", "Welcome to ManiaScope", [380., 150.], above, |ui| self.profile_ui(ui)) {
            self.open.profile = None;
        }
        if !self.open.menu && self.open.web {
            (self.open.web, self.web.saved, self.web.loaded) = (false, false, false); // closed by its button
        }
        if self.open.menu {
            // as big as its wider tab needs, Website charts, from the start (user 2026-11-22: it grew on the click)
            self.open.menu &= !window(ctx, "menu", "ManiaScope · Settings", [584., 560.], above, |ui| self.menu_ui(ui));
            if !self.open.menu {
                self.open.profile = self.open.profile.filter(|first| *first);
                (self.open.web, self.web.saved, self.web.loaded) = (false, false, false);
            }
        }
        if self.open.history {
            self.open.history = !window(ctx, "history", "Play & prediction history", [760., 520.], above, |ui| {
                if theme::checkbox(ui, &mut self.open.history_only, "Only unusually high or low results (±1.5σ)").changed() {
                    self.q.put(Task::History(self.open.history_only));
                }
                ui.label(RichText::new("Latest 100 plays; unusual results sorted by prediction error. Accuracy is lazer accuracy.").color(DIM));
                ScrollArea::vertical().show(ui, |ui| ui.label(&self.open.history_text));
            });
        }
    }
}

pub fn level(above: bool) -> egui::WindowLevel {
    if above { egui::WindowLevel::AlwaysOnTop } else { egui::WindowLevel::Normal }
}

pub fn playlist_id() -> ViewportId {
    ViewportId::from_hash_of("playlist")
}

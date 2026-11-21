//! The worker owns the recommendation state and every write to its event log; the UI only queues tasks and renders
//! what `emit` hands back.
use crate::builder::{self, rank, ranked_pick, taste, web_factor, Builder, Published, RankInputs, Request, Update, DEFAULT_TARGET, STRUCTURE_VERSION};
use crate::catalog::{one_by_md5, one_by_sha, Inst, Pick};
use crate::engine::{forecast_chart, hashes, refit, Card, Engine, Now, Refit, CALC_ID, PREDICTOR_ID};
use crate::evidence::y_of;
use crate::history::{client_name, default_client, history_text, play_history, recover_results, Shown};
use crate::navigation::{perform, same_map, ACTIONS};
use crate::pp::{self, PpSettings, PESSIMISM};
use crate::stats;
use ms_calc::features::features_batch;
use ms_data::imports::{self, add_live_score, import_library, Imported};
use ms_data::public::Public;
use ms_data::store::{acc_lazer, acc320, connect, kv_get, kv_set, log_event, now, parse_stats, predictable_context, stable_miss_ratio, timestamp, tracking_allowed, Db, Result, Stats, SUPPORTED_KEYS};
use ms_data::{replays, webmaps};
use ms_model::choices::{deal, key_shares, ChoiceList, Prepared, Shares, Tags, CONVERT, PP, REAL, WEB, WEB_CONVERT};
use crate::converts::{self, Setting};
use ms_model::predict::{Consts, FeatStore, HashMap};
use ms_model::session::SESSION_GAP;
use ms_model::skills;
use rusqlite::OptionalExtension;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

pub const MODES: [&str; 3] = ["pp", "nps", "skills"];
const CONVERT_WAIT: Duration = Duration::from_secs(2); // skipping through a list imports nothing; a Next after a play waits for nothing
const NOT_RUNNING: &str = "osu! isn't running";
/// kv: Next downloads maps you don't have (default) or opens them in osu! to look at first.
pub const AUTO_DOWNLOAD: &str = "auto_download";
pub const AUTO_RANKED: &str = "auto_ranked"; // fetch new ranked PP maps (website charts window, default off)
pub const PP_FLOOR: &str = "pp_floor"; // ★ from which a perfect play enters the top 100 (`pp::top100_stars`), for that fetch
const NPS_CLOCK: f64 = 300.; // s: a break this long re-predicts NPS/Skills once for the cooled warm-up

#[derive(Clone, Debug, PartialEq)]
pub struct PpMix {
    pub on: bool,
    pub every: usize,
    pub min_gain: f64,
    pub chance: f64,
    pub ht_chance: f64,
}

impl Default for PpMix {
    fn default() -> Self {
        PpMix { on: false, every: 4, min_gain: 3., chance: 0.4, ht_chance: 0.7 }
    }
}

#[derive(Clone, Default)]
pub struct Config {
    pub mode: String,
    pub keys: std::collections::HashMap<String, Vec<i64>>,
    pub action: String,
    pub revision: u64,
    pub practice_skills: Option<Vec<String>>,
    pub nps_focus: Option<f64>,
    pub skills_focus: Option<f64>, // the open custom playlist's slider; below 0: off
    pub skills_mix: Option<PpMix>,
    pub acc_targets: std::collections::HashMap<String, f64>,
    pub rates: std::collections::HashMap<String, (f64, f64)>, // NPS and Skills HT/DT ranges
    pub web_bias: std::collections::HashMap<String, f64>,
    pub pp_settings: Option<PpSettings>,
    pub pp_mix: Option<PpMix>,
    pub skill_weights: Option<std::collections::HashMap<String, f64>>,
    pub skill_tuning: Option<skills::Tuning>,
    pub web_weighted: std::collections::HashMap<String, bool>,
    pub key_weights: std::collections::HashMap<String, BTreeMap<i64, f64>>, // tab → keymode → share multiplier
    pub by_count: std::collections::HashMap<String, bool>, // tab → keymode shares follow map counts
    pub converts: std::collections::HashMap<String, BTreeMap<i64, Setting>>,
}

const SELECTED: &str = "selected"; // a convert's mode when it comes from the selected map, not a list

pub enum Task {
    ConvertDone { pick: Box<Pick>, mode: String, error: Option<String> },
    ConvertSelected { path: PathBuf, keys: i64 }, // the shown map: needs no worker state, so it runs during the startup carry too
    Configure(Box<Config>),
    Selected { path: Option<PathBuf>, rate: Option<f64>, mods: Option<Vec<Value>> },
    FeatureReady { sha: String, features: Option<Vec<(f64, ms_calc::val::Dict)>>, refit: bool },
    Start(Value),
    Abort(Value),
    Void,
    Score { score: Value, path: PathBuf, fresh: bool, play: Option<Value> },
    Next { skip: bool, mode: Option<String>, revision: Option<u64>, candidate: Option<Box<Pick>>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64 },
    Open { pick: Box<Pick>, mode: Option<String>, revision: Option<u64>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64 },
    Publish,
    Refresh,
    Carried(bool), // the background carry moved more stored ratings to this calculator; true: all of them
    WebUpdated,
    Stats(bool),
    Snooze { pick: Box<Pick>, mode: String }, // "Skip for 12 h": out of every list until then; the current pick moves on
    Profile, // the Stats pages without opening Stats: the main window compares maps with them
    NpsUpdate(Update),
    SkipWarmup,
    Lock(Option<i64>),
    Import,
    Client(String),
    WebsiteUser(String),
    Imported { result: Option<Imported>, recovered: usize, changed: bool, error: Option<String> },
    Tracking,
    History(bool),
    Refit,
    RefitDone(Option<Box<Refit>>),
    ReplayDone { pending: bool, error: Option<String>, fresh: Vec<String> },
}

/// A navigation the UI already performed (the clicked moment's clipboard stays on its thread).
#[derive(Clone)]
pub struct Done {
    pub copy: Option<String>,
    pub msg: String,
    pub sent: bool,
    pub unavailable: bool,
}

pub struct Verdict {
    pub title: String,
    pub text: String,
    pub acc: f64,
    pub expected: Option<f64>,
    pub z: Option<f64>,
    pub rate: Option<f64>,
    pub when: Option<f64>,
    pub news: Vec<String>,
    pub targets: Vec<String>,
}

pub enum Out {
    Status { text: String, progress: Option<(usize, usize)>, blocking: bool, learning: Option<bool> },
    Pool { mode: String, revision: u64, selection_id: u64, phase: String, shown: Vec<Pick>, note: String, summary: String, progress: Option<(usize, usize)>, key_counts: BTreeMap<i64, usize>, skill_counts: BTreeMap<String, usize>, nps_pool: BTreeMap<i64, Vec<f64>> },
    Progress { mode: String, progress: Option<(usize, usize)> }, // analysis between publications: the list stays as it is
    /// The picked map on its way into the game, under its row (user 2026-11-14: "put some downloading bar"): what runs, share
    /// done when known; empty text = done.
    Delivery { identity: String, text: String, progress: Option<f32>, failed: bool },
    Target { mode: String, revision: u64, selection_id: u64, c: Option<Box<Pick>>, msg: String, copy: Option<String> },
    Copy { text: String, msg: String },
    Card { c: Option<Box<Card>>, note: Option<String>, path: Option<PathBuf>, rate: Option<f64> },
    Result(Box<Verdict>),
    History(String),
    Stats(Box<stats::Current>),
}

/// FIFO, except the selected map's card: it no longer waits behind post-play rebuilds (G835LX 2026-10-05: score, publish
/// and abort queued 10–30 s ahead of it). Post-play rebuilds go after every user action.
#[derive(Default)]
pub struct Queue {
    lanes: Mutex<[VecDeque<Task>; 3]>,
    wake: Condvar,
    pub config_revision: Mutex<u64>,
    pub selected_at: Mutex<Option<Instant>>,
}

impl Queue {
    pub fn put(&self, task: Task) {
        let lane = match task {
            Task::Selected { .. } | Task::FeatureReady { .. } => 0,
            Task::Refresh => 2,
            _ => 1,
        };
        match &task {
            Task::Configure(c) => {
                let mut r = self.config_revision.lock().unwrap();
                *r = (*r).max(c.revision);
            }
            Task::Selected { .. } => *self.selected_at.lock().unwrap() = Some(Instant::now()),
            _ => {}
        }
        let mut lanes = self.lanes.lock().unwrap();
        if matches!(task, Task::Selected { .. }) {
            // only the latest selection matters: scrolling song select queued one full card per chart passed (2026-11-17 audit)
            lanes[lane].retain(|t| !matches!(t, Task::Selected { .. }));
        }
        lanes[lane].push_back(task);
        drop(lanes);
        self.wake.notify_one();
    }

    pub fn get(&self, timeout: Duration) -> Option<Task> {
        let mut lanes = self.lanes.lock().unwrap();
        if lanes.iter().all(VecDeque::is_empty) {
            lanes = self.wake.wait_timeout(lanes, timeout).unwrap().0;
        }
        lanes.iter_mut().find_map(VecDeque::pop_front)
    }

    fn urgent(&self) -> Option<Task> {
        self.lanes.lock().unwrap()[0].pop_front()
    }

    /// The queued tasks `pick` takes, in order; the rest keep their places.
    fn take(&self, pick: impl Fn(&Task) -> bool) -> Vec<Task> {
        let mut lanes = self.lanes.lock().unwrap();
        let (taken, rest): (VecDeque<Task>, VecDeque<Task>) = std::mem::take(&mut lanes[1]).into_iter().partition(|t| pick(t));
        lanes[1] = rest;
        taken.into()
    }
}

#[derive(Clone, Default)]
struct ListState {
    candidates: Arc<ChoiceList>,
    catalog: Arc<HashMap<String, Arc<Inst>>>,
    total: usize,
    analyzed: usize,
    failed: usize,
    error: String,
    restored: bool,
    build_failed: bool,
}

impl ListState {
    /// Charts that cannot be analysed (converts, broken files) are done too, or the bar waits forever.
    fn progress(&self) -> Option<(usize, usize)> {
        (!self.build_failed && self.total != 0 && self.analyzed + self.failed < self.total).then_some((self.analyzed + self.failed, self.total))
    }
}

type Check = Arc<dyn Fn() -> bool + Send + Sync>;

/// What the UI gives the worker.
#[derive(Clone)]
pub struct Hooks {
    pub emit: Arc<dyn Fn(Out) + Send + Sync>,
    pub busy: Check, // a map is being played
    pub game: Check, // the game is running
    pub list_open: Check, // the playlist window is open: lists are built and published only then (user 2026-11-08)
}

struct BuildSlot {
    request: Mutex<Option<Request>>,
    wake: Condvar,
}

pub struct Worker {
    q: Arc<Queue>,
    h: Hooks,
    halt: Arc<AtomicBool>,
    db: Db,
    db_path: PathBuf,
    c: Arc<Consts>,
    rec: Option<Engine>,
    public: Option<Arc<Public>>,
    store: Arc<RwLock<FeatStore>>,
    slot: Arc<BuildSlot>,
    generation: u64,
    local_request: u64,
    mode: String,
    action: String,
    revision: u64,
    selection_id: u64,
    keys: std::collections::HashMap<&'static str, Vec<i64>>,
    target: Option<Box<Pick>>,
    targets: std::collections::HashMap<String, Box<Pick>>,
    playlists: std::collections::HashMap<&'static str, Vec<Pick>>,
    nps_focus: f64,
    skills_focus: f64,
    skills_mix: PpMix,
    web_bias: std::collections::HashMap<&'static str, f64>,
    web_weighted: std::collections::HashMap<&'static str, bool>,
    key_weights: std::collections::HashMap<&'static str, BTreeMap<i64, f64>>,
    by_count: std::collections::HashMap<&'static str, bool>,
    converts: std::collections::HashMap<&'static str, BTreeMap<i64, Setting>>,
    mix_logged: HashMap<&'static str, String>, // the last list mix written to the log, per tab
    convert_turn: Arc<AtomicU64>, // the newest convert to deliver: a skipped one never reaches the game
    ahead: Option<i64>,           // the website set being fetched for the upcoming pick
    skill_weights: std::collections::HashMap<String, f64>,
    skill_tuning: skills::Tuning,
    acc_targets: std::collections::HashMap<&'static str, f64>,
    rates: std::collections::HashMap<&'static str, (f64, f64)>,
    pp_mix: PpMix,
    settings: PpSettings,
    practice_skills: Vec<String>,
    states: std::collections::HashMap<&'static str, ListState>,
    restored: HashSet<String>,
    shown: Shown,
    session_cache: Option<((Option<i64>, i64), f64, Arc<Now>)>,
    events_cache: Option<(Option<i64>, f64, Vec<Arc<ms_data::store::Event>>)>,
    taste_cache: (Option<i64>, HashMap<(i64, String), f64>),
    nps_pool: HashMap<&'static str, BTreeMap<i64, Vec<f64>>>, // per tab: eligible maps' NPS per keymode, sorted (choices::nps_eligible)
    sel: (Option<PathBuf>, Option<f64>, Option<Vec<Value>>),
    loading: Option<String>,
    learning: Arc<Mutex<Option<String>>>,
    first_import: bool,
    refresh_queued: bool,
    rest_pending: bool, // `carry_rest` waits for the first list: built beside it, that list read whatever rows had landed
    refit_due: bool,
    refit_running: bool,
    refit_again: bool,
    last_refit: f64,
    last_idle_check: Instant,
    nps_requested: Instant,
    clock_due: bool, // a play since the last break's re-prediction
    replay_due: Option<Instant>,
    replay_retries: u32,
    jobs: Arc<Jobs>,
    feature_errors: Arc<Mutex<std::collections::HashMap<(String, u64), (Instant, String)>>>,
    feature_pending: Arc<Mutex<(std::collections::VecDeque<(String, PathBuf, f64, bool)>, bool)>>,
    stats_visible: bool,
    stats_cache: Option<(u64, Arc<stats::Profile>)>,
    current: Option<String>,
    yielding: bool,
}

#[derive(Default)]
struct Jobs {
    importing: AtomicBool,
    archiving: AtomicBool,
    filling: AtomicBool,
}

/// "0.75–1.50×", or "1.00× only" when HT/DT is off.
pub fn rate_text((lo, hi): (f64, f64)) -> String {
    if (hi - lo).abs() < 1e-9 { format!("{lo:.2}× only") } else { format!("{lo:.2}–{hi:.2}×") }
}

fn key_mode(mode: &str) -> &'static str {
    MODES.into_iter().find(|m| *m == mode).unwrap_or("pp")
}

fn normalize_keys(v: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = v.iter().copied().filter(|k| SUPPORTED_KEYS.contains(k)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn verdict_words(z: f64) -> &'static str {
    if z > 1.5 { "way above expected" } else if z > 0.5 { "better than expected" } else if z < -1.5 { "way below expected" } else if z < -0.5 { "below expected" } else { "about as expected" }
}

fn reason(c: &Pick) -> String {
    let chance = 100. * c.p_up.unwrap_or(0.);
    match c.best.filter(|&b| b != 0.) {
        Some(b) => format!("refarm · best {b:.0}pp · {chance:.0}% to beat"),
        None => format!("new map · {chance:.0}% to count"),
    }
}

fn parse_mods(v: Option<&Value>) -> Option<Vec<Value>> {
    v.and_then(Value::as_array).cloned()
}

/// The chart a selection or a play is of once its mods are applied (`modcharts`); a plain chart is itself.
struct Played {
    sha: String,
    md5: String,
    path: PathBuf,
    /// The rate to rate it at: the mods' constant rate, 1× where the rate follows the player (Adaptive Speed).
    rate: f64,
    m: ms_calc::mods::Mods,
}

fn as_played(db: &Db, path: &Path, data: &[u8], rate: Option<f64>, mods: &[Value]) -> Played {
    let m = ms_calc::mods::plan(mods);
    let (sha, md5) = hashes(data);
    let rate = rate.or(m.rate).unwrap_or(1.);
    let tag = m.tag();
    match (!tag.is_empty()).then(|| crate::modcharts::ensure(db, &sha, data, &tag).ok()).flatten() {
        Some((sha, md5, path)) => Played { sha, md5, path, rate, m },
        None => Played { sha, md5, path: path.to_path_buf(), rate, m },
    }
}

/// Random before its seed is drawn (plan 2026-11-20 D): the chart under each of the three fixed shuffles; empty otherwise.
fn shuffles(db: &Db, data: &[u8], played: &Played) -> Vec<Played> {
    if played.m.random != Some(None) {
        return Vec::new();
    }
    let source = hashes(data).0;
    ms_calc::mods::RANDOM_SEEDS.iter().filter_map(|&seed| {
        let m = ms_calc::mods::Mods { random: Some(Some(seed)), ..played.m.clone() };
        let (sha, md5, path) = crate::modcharts::ensure(db, &source, data, &m.tag()).ok()?;
        Some(Played { sha, md5, path, rate: played.rate, m })
    }).collect()
}

/// The unseeded-Random card: the three shuffles' forecasts averaged (accuracy, pp), the skill most of them lead with.
fn mean_card(mut plain: Card, cards: &[Card]) -> Card {
    let n = cards.len() as f64;
    let mean = |f: &dyn Fn(&Card) -> f64| cards.iter().map(f).sum::<f64>() / n;
    (plain.acc_mid, plain.acc_lo, plain.acc_hi) = (mean(&|c| c.acc_mid), mean(&|c| c.acc_lo), mean(&|c| c.acc_hi));
    plain.pp_expected = cards.iter().map(|c| c.pp_expected).collect::<Option<Vec<f64>>>().map(|v| v.iter().sum::<f64>() / n);
    plain.pp = cards.iter().map(|c| c.pp).collect::<Option<Vec<_>>>().map(|v| {
        let m = |f: fn(&(f64, f64, Option<f64>, f64, f64)) -> f64| v.iter().map(f).sum::<f64>() / n;
        (m(|p| p.0), m(|p| p.1), v[0].2, m(|p| p.3), m(|p| p.4))
    });
    plain.skill = cards.iter().map(|c| &c.skill).max_by_key(|s| cards.iter().filter(|c| &c.skill == *s).count()).cloned().flatten();
    plain.unread = cards[0].unread.clone(); // the same SV and reads under every shuffle
    plain.without.retain(|w| w != "RD");
    plain.random = true;
    plain
}

/// Spawned on its own thread: runs until `stop`. `public` is opened by the caller (the UI thread never waits on it).
pub fn spawn(public: impl FnOnce(&Hooks) -> std::result::Result<Arc<Public>, String> + Send + 'static, q: Arc<Queue>, h: Hooks, halt: Arc<AtomicBool>, db_path: PathBuf) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new().name("recommend".into()).spawn(move || {
        let db = match connect(&db_path) {
            Ok(db) => db,
            Err(e) => {
                (h.emit)(Out::Status { text: format!("Could not open your data: {e}"), progress: None, blocking: true, learning: None });
                return;
            }
        };
        let mut w = Worker::new(q, h, halt, db, db_path);
        w.run(public);
    }).expect("worker thread")
}

impl Worker {
    fn new(q: Arc<Queue>, h: Hooks, halt: Arc<AtomicBool>, db: Db, db_path: PathBuf) -> Worker {
        let c = Arc::new(Consts::standard());
        let store = FeatStore::new(skills::feat_trees(), builder::profile(), c.clone(), STRUCTURE_VERSION).expect("skill trees");
        let client = kv_get(&db, "client").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| default_client().to_string());
        let miss_ratio = kv_get(&db, "stable_miss_ratio").and_then(|v| v.as_f64()).unwrap_or(0.88);
        let first_import = kv_get(&db, "realm_import").is_none();
        let empty = || MODES.into_iter().map(|m| (m, Vec::new())).collect();
        Worker {
            q, h, halt, db, db_path, c, rec: None, public: None, store: Arc::new(RwLock::new(store)), slot: Arc::new(BuildSlot { request: Mutex::new(None), wake: Condvar::new() }), generation: 0, local_request: 0,
            mode: "pp".into(), action: "auto".into(), revision: 0, selection_id: 0,
            keys: [("pp", SUPPORTED_KEYS.collect()), ("nps", vec![7]), ("skills", vec![7])].into_iter().collect(),
            target: None, targets: Default::default(), playlists: empty(), nps_focus: 0.5, skills_focus: 0., skills_mix: PpMix { on: false, ..PpMix::default() }, web_bias: [("nps", 0.5), ("skills", 0.5)].into_iter().collect(), skill_weights: Default::default(), skill_tuning: Default::default(), web_weighted: [("nps", true), ("skills", true)].into_iter().collect(), key_weights: Default::default(), by_count: Default::default(),
            converts: [("nps", BTreeMap::new()), ("skills", BTreeMap::new())].into_iter().collect(), mix_logged: HashMap::default(), ahead: None, convert_turn: Arc::new(AtomicU64::new(0)),
            acc_targets: [("nps", DEFAULT_TARGET), ("skills", DEFAULT_TARGET)].into_iter().collect(), rates: [("nps", builder::DEFAULT_RATES), ("skills", builder::DEFAULT_RATES)].into_iter().collect(), pp_mix: PpMix::default(), settings: PpSettings::default(), practice_skills: Vec::new(),
            states: [("nps", ListState::default()), ("skills", ListState::default())].into_iter().collect(), restored: HashSet::new(),
            shown: Shown { client, miss_ratio }, session_cache: None, events_cache: None, taste_cache: (None, HashMap::default()), nps_pool: HashMap::default(), sel: (None, None, None), loading: None, learning: Arc::default(), first_import, refresh_queued: false, rest_pending: false,
            refit_due: false, refit_running: false, refit_again: false, last_refit: now(), last_idle_check: Instant::now(), nps_requested: Instant::now(), clock_due: true, replay_due: Some(Instant::now() + Duration::from_secs(10)),
            replay_retries: 0, jobs: Arc::default(), feature_errors: Arc::default(), feature_pending: Arc::default(), stats_visible: false, stats_cache: None, current: None, yielding: false,
        }
    }

    fn status(&self, text: impl Into<String>) {
        (self.h.emit)(Out::Status { text: text.into(), progress: None, blocking: false, learning: None });
    }

    /// What startup is waiting on: the status line, its progress bar, and the selected map's card.
    fn stage(&mut self, text: Option<String>, blocking: bool) {
        self.loading.clone_from(&text);
        (self.h.emit)(Out::Status { text: text.unwrap_or_default(), progress: None, blocking, learning: None });
        if self.rec.is_none() && self.sel.0.is_some() {
            self.card();
        }
    }

    fn run(&mut self, public: impl FnOnce(&Hooks) -> std::result::Result<Arc<Public>, String>) {
        self.stage(Some("Opening your saved plays and recommendations…".into()), false);
        // The UI queues its real tab and filters before start: honoured before any invisible local-playlist work.
        for task in self.q.take(|t| matches!(t, Task::Configure(_))) {
            self.run_task(task);
        }
        match self.startup(public) {
            Ok(()) => self.stage(None, false),
            Err(e) => {
                self.stage(Some(format!("Recommendations unavailable: {}", e.chars().take(240).collect::<String>())), true);
                // the playlist window shows why; the main window drops the prediction line and keeps the map's own ratings
                self.loading = None;
                self.card();
            }
        }
        self.begin_import();
        while !self.halt.load(Ordering::Relaxed) {
            if self.refit_due && !(self.h.busy)() {
                self.t_refit();
            }
            self.catch_up_replays();
            let Some(task) = self.q.get(Duration::from_secs(5)) else {
                self.idle();
                continue;
            };
            self.run_task(task);
        }
    }

    fn idle(&mut self) {
        let busy = (self.h.busy)();
        if self.rec.is_some() && !busy {
            let last: Option<f64> = self.db.query_row("SELECT MAX(t) FROM events WHERE kind IN ('finish','abort')", [], |r| r.get(0)).unwrap_or(None);
            if last.is_some_and(|l| self.last_refit < l && now() - l > SESSION_GAP) {
                self.t_refit();
            }
            if self.clock_due && self.nps_requested.elapsed().as_secs_f64() > NPS_CLOCK {
                self.clock_due = false;
                self.request_nps(); // warm-up cooled during a break: one new list, then none until the next play
            }
        }
        if !busy && self.last_idle_check.elapsed().as_secs() > 3600 {
            self.last_idle_check = Instant::now();
            if now() - kv_get(&self.db, "snapshot_check").and_then(|v| v["t"].as_f64()).unwrap_or(0.) > 86400. {
                if let Ok(Some(newest)) = ms_data::public::check_index(&self.db) {
                    self.status(format!("newer public snapshot {} — refresh --download (personal analysis; see licence)", &newest[..10]));
                }
            }
        }
    }

    fn run_task(&mut self, task: Task) {
        let started = Instant::now();
        if let Err(e) = self.dispatch(task) {
            eprintln!("recommendation task failed: {e}");
            self.status("recommendation task failed (see maniascope.log)");
        }
        let took = started.elapsed().as_secs_f64();
        if took > 1.5 {
            eprintln!("slow recommender task {took:.1} s"); // every later tab switch or Next waits behind it
        }
    }

    fn startup(&mut self, public: impl FnOnce(&Hooks) -> std::result::Result<Arc<Public>, String>) -> std::result::Result<(), String> {
        let public = public(&self.h)?;
        // the per-note abilities need only the plays and the net: solved beside the carry and the fit, done before any list
        let pernote = crate::pernote::PerNote::load().map(Arc::new);
        let filling = pernote.clone().map(|p| {
            let path = self.db_path.clone();
            std::thread::spawn(move || connect(&path).and_then(|db| p.fill(&db, crate::pernote::today() - crate::pernote::WINDOW)))
        });
        let carrying = self.carry_played()?;
        self.prune_calculators();
        self.stage(Some("Fitting your prediction model to your scores…".into()), false);
        ms_data::imports::import_public_user(&self.db, &public).map_err(|e| e.to_string())?;
        let r = refit(&self.db, &public, &self.c).map_err(|e| e.to_string())?;
        self.public = Some(public.clone());
        self.rec = Some(Engine::new(public, self.c.clone(), r, self.settings.clone()));
        self.rec.as_mut().unwrap().pernote = pernote;
        if carrying {
            self.carry_near()?;
        }
        if let Some(Ok(Err(e))) = filling.map(std::thread::JoinHandle::join) {
            eprintln!("per-note abilities: {e}");
        }
        self.save_pp_floor();
        self.generation += 1;
        self.spawn_builder();
        self.restore_local(&self.mode.clone()).map_err(|e| e.to_string())?;
        self.request_nps();
        self.publish(None).map_err(|e| e.to_string())?;
        self.start_fill();
        // the first list is made from the near rows alone, so two starts of the same data give the same list (audit 2026-11-20:
        // carried beside the build, the list's order changed run to run); a closed window builds nothing, so nothing waits
        if carrying && (self.h.list_open)() {
            self.rest_pending = true;
        } else if carrying {
            self.carry_rest();
        }
        Ok(())
    }

    /// A calculator change that only moved its calibration re-rates what is stored instead of analysing it again; the wait
    /// before the first list after an update is this (user 2026-11-19: "the most important essentially (for speed gains)").
    /// In two phases: before the first list the played rows (the fit reads them; here) and the charts near the player's
    /// level (`carry_near`); the rest of the library and the website store in the background (`carry_rest`).
    /// true: this launch still has that to do.
    fn carry_played(&mut self) -> std::result::Result<bool, String> {
        if kv_get(&self.db, &format!("carried_{CALC_ID}")).is_some() {
            // another version ran in between and published the website list for its calculator: publish ours again
            if ms_data::webmaps::published_calc().is_some_and(|c| c != CALC_ID) && !ms_data::webmaps::running() {
                if let Err(e) = crate::engine::carry_web(&self.db) {
                    eprintln!("website list not republished: {e}");
                }
            }
            return Ok(false);
        }
        self.carry_now(crate::engine::played_names(&self.db).map_err(|e| e.to_string())?)?;
        Ok(true)
    }

    /// The stored rows a first list can offer: each tab's keymodes, within NEAR stars of the rating the player's level puts at
    /// the tab's accuracy target (the previous calculator's ratings choose them; a calibration change moves a rating by far
    /// less than the window).
    fn carry_near(&mut self) -> std::result::Result<(), String> {
        const NEAR: f64 = 1.5;
        let model = self.rec.as_ref().unwrap().model();
        let mut near: std::collections::HashMap<i64, (f64, f64)> = Default::default();
        for mode in ["nps", "skills"] {
            let target = self.acc_targets.get(mode).copied().unwrap_or(DEFAULT_TARGET);
            for &k in self.keys[mode].iter().filter(|k| model.level.contains_key(k)) {
                let at = builder::level_rating(model, k, target, &self.c);
                let w = near.entry(k).or_insert((at - NEAR, at + NEAR));
                *w = (w.0.min(at - NEAR), w.1.max(at + NEAR));
            }
        }
        let names = crate::engine::app_names(&self.db).and_then(|all| ms_data::store::near_names(&self.db, &all, &near)).map_err(|e| e.to_string())?;
        self.carry_now(names)
    }

    /// The rows `names` re-rated now, before anything else: on its own connection while this thread counts them (so it
    /// doesn't look stuck) and serves the map menu's converts, which need no fitted model (user 2026-11-15: a convert waited
    /// behind 102,840 charts).
    fn carry_now(&mut self, names: HashSet<String>) -> std::result::Result<(), String> {
        self.stage(Some("Updating stored ratings (once)…".into()), false);
        let path = self.db_path.clone();
        let before = ms_data::store::CARRIED.each_ref().map(|a| a.load(Ordering::Relaxed));
        std::thread::scope(|s| {
            let carry = s.spawn(|| connect(&path).map_err(|e| e.to_string()).and_then(|db| crate::engine::carry_rows(&db, Some(&names))));
            while !carry.is_finished() {
                ms_data::store::GAME_UP.store((self.h.game)(), Ordering::Relaxed);
                for task in self.q.take(|t| matches!(t, Task::ConvertSelected { .. })) {
                    self.run_task(task);
                }
                let [d, t] = [0, 1].map(|i| ms_data::store::CARRIED[i].load(Ordering::Relaxed) - before[i]);
                if t > 0 {
                    let text = format!("Updating stored ratings (once) · {} / {} charts", crate::engine::commas(d.min(t)), crate::engine::commas(t));
                    // counted, so both windows draw it as their progress bar (user 2026-11-13), gone with the next status
                    (self.h.emit)(Out::Status { text, progress: Some((d.min(t), t)), blocking: false, learning: None });
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            carry.join().map_err(|_| "stored ratings not updated: the update stopped".to_string())?
        })
    }

    /// The rest of the library and the website store, re-rated in the background at low priority (the carry's threads are):
    /// the lists offer what is re-rated so far, and each part that lands widens them (`Task::Carried`).
    fn carry_rest(&self) {
        ms_data::store::CARRYING.store(true, Ordering::Relaxed);
        let (path, q, h, halt) = (self.db_path.clone(), self.q.clone(), self.h.clone(), self.halt.clone());
        std::thread::Builder::new().name("carry".into()).spawn(move || {
            let before = ms_data::store::CARRIED.each_ref().map(|a| a.load(Ordering::Relaxed));
            let all = std::thread::scope(|s| {
                let carry = s.spawn(|| connect(&path).map_err(|e| e.to_string()).and_then(|db| Self::carry_all(&db)));
                let (mut told, mut widened) = (0, Instant::now());
                while !carry.is_finished() && !halt.load(Ordering::Relaxed) {
                    ms_data::store::GAME_UP.store((h.game)(), Ordering::Relaxed);
                    let [d, t] = [0, 1].map(|i| ms_data::store::CARRIED[i].load(Ordering::Relaxed) - before[i]);
                    if t > 0 && d != told {
                        told = d;
                        let text = format!("Updating the rest of your library in the background · {} / {} charts", crate::engine::commas(d.min(t)), crate::engine::commas(t));
                        (h.emit)(Out::Status { text, progress: Some((d.min(t), t)), blocking: false, learning: None });
                        if widened.elapsed().as_secs() >= 15 {
                            widened = Instant::now();
                            q.put(Task::Carried(false));
                        }
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
                carry.join().unwrap_or_else(|_| Err("the update stopped".into()))
            });
            ms_data::store::CARRYING.store(false, Ordering::Relaxed);
            match all {
                Ok(done) => q.put(Task::Carried(done)),
                Err(e) => eprintln!("stored ratings not all updated: {e}"), // the next launch goes on
            }
            (h.emit)(Out::Status { text: String::new(), progress: None, blocking: false, learning: None });
        }).expect("carry thread");
    }

    /// Ok(true) once both stores carried. The own rows' error stops the carry; the website's waits for the next launch.
    fn carry_all(db: &Db) -> std::result::Result<bool, String> {
        crate::engine::carry_rows(db, Some(&crate::engine::app_names(db).map_err(|e| e.to_string())?))?;
        // A crawl from before the update can still hold it.
        if ms_data::webmaps::running() {
            return Ok(false);
        }
        if let Err(e) = crate::engine::carry_web(db) {
            eprintln!("website features not carried: {e}");
            return Ok(false);
        }
        Ok(true)
    }

    /// Once, after the carry: a player never needs the older calculators' rows again (user 2026-11-01).
    fn prune_calculators(&mut self) {
        let (carried, done) = (format!("carried_{CALC_ID}"), format!("pruned_{CALC_ID}"));
        if kv_get(&self.db, &carried).is_none() || kv_get(&self.db, &done).is_some() {
            return;
        }
        self.stage(Some("Compacting stored ratings (once)…".into()), false);
        match ms_data::store::prune_feats(&self.db, CALC_ID) {
            Ok(None) => {} // an older database re-analyses instead of carrying: its old rows go next launch
            Ok(Some(_)) => {
                let _ = kv_set(&self.db, &done, &json!(true));
                if let Err(e) = crate::converts::prune_kept(&self.db) {
                    eprintln!("kept converts not pruned: {e}");
                }
            }
            Err(e) => eprintln!("old ratings kept for now: {e}"),
        }
    }

    fn spawn_builder(&self) {
        let (slot, q, h, store, path, halt) = (self.slot.clone(), self.q.clone(), self.h.clone(), self.store.clone(), self.db_path.clone(), self.halt.clone());
        let prm = self.rec.as_ref().unwrap().prm.clone();
        std::thread::Builder::new().name("nps-build".into()).spawn(move || {
            let mut builder = Builder::new(store, prm);
            let (s, stop) = (slot.clone(), halt.clone());
            builder.wanted = Arc::new(move || stop.load(Ordering::Relaxed) || s.request.lock().unwrap().is_some());
            builder.busy = { let (b, o) = (h.busy.clone(), h.list_open.clone()); Arc::new(move || b() || !o()) }; // a closed window pauses a running build
            builder.game = h.game.clone();
            let out = q.clone();
            builder.emit = Arc::new(move |u| out.put(Task::NpsUpdate(u)));
            let Ok(db) = connect(&path) else { return };
            while !halt.load(Ordering::Relaxed) {
                let request = {
                    let mut pending = slot.request.lock().unwrap();
                    while pending.is_none() && !halt.load(Ordering::Relaxed) {
                        pending = slot.wake.wait_timeout(pending, Duration::from_secs(2)).unwrap().0;
                    }
                    pending.take()
                };
                let Some(r) = request else { continue };
                if let Err(e) = builder.build(&db, &r) {
                    builder.reset();
                    (builder.emit)(Update::Failed { error: e.to_string(), mode: r.mode.clone(), revision: r.revision, generation: r.generation });
                }
                r.public.release();
                ms_data::store::release_memory();
            }
        }).expect("builder thread");
    }

    fn dispatch(&mut self, task: Task) -> Result<()> {
        match task {
            Task::Configure(c) => self.t_configure(*c)?,
            Task::ConvertDone { pick, mode, error } => self.t_convert_done(pick, mode, error)?,
            Task::Selected { path, rate, mods } => self.t_selected(path, rate, mods)?,
            Task::FeatureReady { sha, features, refit } => {
                if let Some(rec) = self.rec.as_mut() {
                    if let Some(fs) = &features {
                        rec.store(&self.db, &sha, fs)?;
                    }
                    self.card();
                    if refit {
                        self.refresh_later(); // the play's one list request waited for its chart (t_score)
                    }
                }
            }
            Task::Start(play) => self.t_start(&play)?,
            Task::Abort(play) => self.t_abort(&play)?,
            Task::Void => {
                log_event(&self.db, "void", self.current.as_deref(), None, &json!({}))?;
            }
            Task::Score { score, path, fresh, play } => self.t_score(&score, &path, fresh, play.as_ref())?,
            Task::Next { skip, mode, revision, candidate, result, selected_at, selection_id } => self.t_next(skip, mode, revision, candidate, result, selected_at, selection_id)?,
            Task::Open { pick, mode, revision, result, selected_at, selection_id } => self.t_open(pick, mode, revision, result, selected_at, selection_id)?,
            Task::Publish => self.publish(None)?,
            Task::Carried(all) => {
                if all {
                    kv_set(&self.db, &format!("carried_{CALC_ID}"), &json!(true))?;
                    self.prune_calculators();
                    self.start_fill(); // what it left to the carry and still lacks a row
                }
                if let Some(rec) = self.rec.as_mut() {
                    rec.forget_variants();
                }
                self.request_nps(); // the builder reads the rows that landed since its last request
            }
            Task::Refresh => {
                self.refresh_queued = false;
                self.request_nps();
                if self.mode == "pp" {
                    self.publish(None)?; // NPS/Skills: the Builder's first publication of this request follows
                }
            }
            Task::WebUpdated => {
                // new website charts join the next play's list; an empty list takes them now
                if self.playlists.get(key_mode(&self.mode)).is_none_or(Vec::is_empty) {
                    self.request_nps();
                }
            }
            Task::ConvertSelected { path, keys } => self.t_convert_selected(path, keys),
            Task::Stats(v) => self.t_stats(v)?,
            Task::Profile => self.emit_stats()?,
            Task::Snooze { pick, mode } => self.t_snooze(&pick, mode)?,
            Task::NpsUpdate(u) => self.t_nps_update(u)?,
            Task::SkipWarmup => {
                log_event(&self.db, "skip_warmup", None, None, &json!({}))?;
                self.playlists.values_mut().for_each(|l| l.retain(|c| c.purpose != "warmup"));
                self.request_nps();
                self.publish(None)?;
            }
            Task::Lock(keys) => {
                let m = key_mode(&self.mode);
                self.keys.insert(m, keys.map_or_else(|| SUPPORTED_KEYS.collect(), |k| normalize_keys(&[k])));
                self.request_nps();
                self.publish(None)?;
            }
            Task::Import => self.begin_import(),
            Task::Client(name) => self.t_client(&name)?,
            Task::WebsiteUser(name) => {
                let name = name.trim();
                kv_set(&self.db, "website_username", &if name.is_empty() { Value::Null } else { json!(name) })?;
                kv_set(&self.db, "website_user", &json!({}))?;
                self.begin_import();
            }
            Task::Imported { result, recovered, changed, error } => self.t_imported(result, recovered, changed, error)?,
            Task::Tracking | Task::Refit => self.t_refit(),
            Task::History(outliers) => (self.h.emit)(Out::History(history_text(&play_history(&self.db, 100, outliers, None)?, &self.shown))),
            Task::RefitDone(r) => self.t_refit_done(r)?,
            Task::ReplayDone { pending, error, fresh } => self.t_replay_done(pending, error, fresh)?,
        }
        Ok(())
    }

    fn target_of(&self, mode: &str) -> f64 {
        self.shown.model(self.acc_targets.get(mode).copied().unwrap_or(DEFAULT_TARGET))
    }

    fn session(&mut self) -> Result<Arc<Now>> {
        let id: Option<i64> = self.db.query_row("SELECT MAX(id) FROM events", [], |r| r.get(0))?;
        let stamp = (id, (now() / 2.) as i64);
        let target = self.target_of(&self.mode.clone());
        if let Some((s, t, n)) = &self.session_cache {
            if *s == stamp && *t == target {
                return Ok(n.clone());
            }
        }
        // The events only change with a new row or a new minute of window: re-reading ~8 MB of event JSON (14 days of starts'
        // feature records) every 2 s of tasks was most of a state (2026-11-17 audit); the session itself is rebuilt for now().
        let current = now();
        let since = crate::engine::Engine::events_since(current);
        if self.events_cache.as_ref().is_none_or(|c| c.0 != id || c.1 != since) {
            self.events_cache = Some((id, since, ms_data::events::load_events(&self.db, since)?));
        }
        let events = self.events_cache.as_ref().unwrap().2.clone();
        let n = Arc::new(self.rec.as_ref().expect("a fitted model").state_with(events, target, current)?);
        self.session_cache = Some((stamp, target, n.clone()));
        Ok(n)
    }

    fn publish(&mut self, mode: Option<String>) -> Result<()> {
        let Some(rec) = self.rec.as_ref() else { return Ok(()) };
        if self.stats_visible || !(self.h.list_open)() {
            return Ok(());
        }
        // The player's level is still unknown: an empty list says why, not beginner maps.
        if rec.r.fitted.rows.rows.len() < 30 && (self.learning.lock().unwrap().is_some() || self.first_import) {
            return Ok(());
        }
        let mode = key_mode(mode.as_deref().unwrap_or(&self.mode));
        let st = self.states.get(mode).cloned();
        let (phase, shown, note) = self.list(mode)?;
        let progress = st.as_ref().and_then(ListState::progress);
        (self.h.emit)(Out::Pool { mode: mode.to_string(), revision: self.revision, selection_id: self.selection_id, phase, shown, note, summary: self.summary(), progress, key_counts: self.key_counts(mode), skill_counts: if mode == "skills" { self.skill_counts(mode) } else { BTreeMap::new() }, nps_pool: self.nps_pool.get(mode).cloned().unwrap_or_default() });
        Ok(())
    }

    fn summary(&self) -> String {
        let Some(rec) = self.rec.as_ref() else { return String::new() };
        let levels: Vec<String> = SUPPORTED_KEYS.filter_map(|k| rec.model().level.get(&k).map(|l| format!("{k}K {:+.2}", l.0))).collect();
        format!("ledger {:.1}pp over {} maps · {} lazer scores without a local chart · {} evidence rows · levels {} · retry {:.3}", rec.r.ledger.total(), rec.r.ledger.best.len(), rec.r.missing,
                rec.r.fitted.rows.rows.len(), levels.join(", "), rec.model().retry)
    }

    /// The open tab's keymode shares (`choices::key_shares`) from its catalogue: your maps and converts, website charts out.
    fn key_shares(&self, mode: &str) -> BTreeMap<i64, f64> {
        key_shares(&self.keys[mode], &self.key_weights.get(mode).cloned().unwrap_or_default(), &self.key_counts(mode), self.by_count.get(mode).copied().unwrap_or(false))
    }

    /// Your maps and converts per keymode in the tab's catalogue (the settings page computes the shares from these live).
    fn key_counts(&self, mode: &str) -> BTreeMap<i64, usize> {
        let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
        for i in self.states.get(mode).map(|s| s.catalog.values()).into_iter().flatten().filter(|i| i.installed || i.convert_of.is_some()) {
            *counts.entry(i.keys).or_default() += 1;
        }
        counts
    }

    /// Your maps and converts per leading included skill ("" for none) in the Skills catalogue: the skill sliders' counts.
    fn skill_counts(&self, mode: &str) -> BTreeMap<String, usize> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for c in self.states.get(mode).map(|s| s.candidates.items.iter()).into_iter().flatten().filter(|c| (c.installed || c.convert) && self.keys[mode].contains(&c.keys)) {
            *counts.entry(c.matched.first().cloned().unwrap_or_default()).or_default() += 1;
        }
        counts
    }

    /// The session's last offers as `choices::deal` counts them, oldest first: this tab's by keymode, skill slot and stream; a
    /// PP map (in this list or the PP tab) restarts the PP count; another tab's offer only counts toward the next PP map.
    fn ledger(n: &Now, mode: &str, by_skill: bool) -> Vec<Tags> {
        let all: Vec<Tags> = n.session.all.iter().filter(|e| e.kind == "offer").map(|e| {
            let i = &e.info;
            let stream = if i["convert"].as_bool() == Some(true) { if i["source_web"].as_bool() == Some(true) { WEB_CONVERT } else { CONVERT } } else if i["web"].as_bool() == Some(true) { WEB } else { REAL };
            match i["mode"].as_str() {
                Some("pp") => Tags { stream: PP, ..Tags::default() },
                Some(m) if m == mode => Tags { keys: i["keys"].as_i64().unwrap_or(0), skill: i["matched_skills"].get(0).and_then(Value::as_str).filter(|_| by_skill).unwrap_or("").to_string(), stream },
                _ => Tags::default(),
            }
        }).collect();
        // far enough back for a small share to come due: over 40 offers a keymode at 0.2 % was never owed a turn (300 presses on
        // the user's settings, 2026-11-19); what a slot is owed stays within one turn, so an old run can't flood the list
        all[all.len().saturating_sub(400)..].to_vec()
    }

    /// The charts of the session's last 20 offers, oldest first: a list takes them again only once nothing else is left.
    fn offered(n: &Now) -> Vec<String> {
        let all: Vec<String> = n.session.all.iter().filter(|e| e.kind == "offer").filter_map(|e| e.beatmap.clone()).collect();
        all[all.len().saturating_sub(20)..].to_vec()
    }

    /// The sets of what was just played or offered, and of the converts among them: one song, whichever side came first.
    fn recent_sets(&self, n: &Now) -> Result<(HashSet<i64>, HashSet<i64>)> {
        let played = n.session.attempts.iter().rev().take(20).filter_map(|a| a.beatmap.clone());
        let (mut sets, mut convert_sets) = (HashSet::new(), HashSet::new());
        let mut library = self.db.prepare_cached("SELECT set_id FROM installed WHERE md5=? OR sha256=?")?;
        let mut converts = self.db.prepare_cached("SELECT set_id FROM converts WHERE md5=? OR sha=?")?;
        for id in Self::offered(n).into_iter().chain(played) {
            let hash = id.split_once(':').map_or(id.as_str(), |x| x.1);
            let rows: Vec<Option<i64>> = library.query_map([hash, hash], |r| r.get(0))?.collect::<Result<_>>()?;
            sets.extend(rows.into_iter().flatten().filter(|&s| s > 0));
            let rows: Vec<Option<i64>> = converts.query_map([hash, hash], |r| r.get(0))?.collect::<Result<_>>()?;
            convert_sets.extend(rows.into_iter().flatten().filter(|&s| s > 0));
        }
        for t in self.targets.values() {
            if let Some(set) = t.set_id.filter(|&s| s > 0) {
                sets.insert(set);
                if t.convert_of.is_some() {
                    convert_sets.insert(set);
                }
            }
        }
        sets.extend(convert_sets.iter().copied());
        Ok((sets, convert_sets))
    }

    fn local_shas(&self) -> Arc<HashSet<String>> {
        self.rec.as_ref().expect("a fitted model").r.local_shas.clone()
    }

    fn rate_range(&self, mode: &str) -> (f64, f64) {
        self.rates.get(key_mode(mode)).copied().unwrap_or(builder::DEFAULT_RATES)
    }

    fn prepared(&self, mode: &'static str) -> (ListState, Prepared) {
        let st = self.states[mode].clone();
        let shas = self.local_shas();
        let p = st.candidates.prepare(&self.keys[mode], self.rate_range(mode), &|s| shas.contains(s));
        (st, p)
    }

    /// The tab's list. One allocator owns what comes next (user 2026-11-18): the rows that still stand stay where they are; a
    /// place that opened and the end of the list are dealt by `choices::deal`, which counts the tab's offers and the rows above.
    /// A row leaves when it is offered, when its chart is gone, when it no longer fits the target or the tab's maps, and when a
    /// setting that decides the list changes (`t_configure` empties it).
    fn list(&mut self, mode: &'static str) -> Result<(String, Vec<Pick>, String)> {
        enum Row {
            Ranked((usize, f64, f64, f64)), // an entry of the tab's prepared pool
            Made(Pick),                     // a PP map
        }
        let n = self.session()?;
        if self.keys[mode].is_empty() {
            self.playlists.insert(mode, Vec::new());
            return Ok((n.session.phase().to_string(), Vec::new(), "Select at least one keymode".to_string()));
        }
        let offered = Self::offered(&n);
        let local = (mode != "pp").then(|| self.prepared(mode));
        let selected = self.practice_skills.clone();
        let miss_ratio = (self.shown.client == "stable").then_some(self.shown.miss_ratio);
        let (mut phase, mut note) = (n.session.phase().to_string(), String::new());
        let (mut pool, mut tags): (Vec<Row>, Vec<Tags>) = Default::default();
        let (mut shares, mut ledger, mut relax, mut floor) = (Shares::default(), Vec::new(), false, BTreeMap::new());
        let mut pp_rows: Vec<Pick> = Vec::new();
        if let Some((st, prepared)) = &local {
            // Explicit skill preferences must not be erased by unrelated NPS taste.
            let none = HashMap::default();
            if mode == "nps" {
                let id: Option<i64> = self.db.query_row("SELECT MAX(id) FROM events", [], |r| r.get(0))?;
                if id != self.taste_cache.0 {
                    self.taste_cache = (id, taste(&self.db, now())?);
                }
            }
            self.yield_urgent(); // a selected map's card goes before the library-wide weighting
            let focus = if mode == "nps" { self.nps_focus } else { self.skills_focus };
            // a custom playlist's slider at off has no shuffle and no skill turns: the best skill match first
            let skill_shares = (mode == "skills" && focus >= 0.).then(|| skills::sliders(&self.practice_skills, &self.skill_weights, self.skill_tuning.other)).flatten().map(|sl| {
                let counts = self.skill_counts(mode);
                let entries: Vec<(String, f64, usize)> = sl.into_iter().map(|(k, w)| { let n = counts.get(&k).copied().unwrap_or(0); (k, w, n) }).collect();
                ms_model::choices::mix_shares(&entries, self.skill_tuning.by_count).into_iter().collect::<std::collections::HashMap<_, _>>()
            });
            let by_skill = skill_shares.is_some();
            let active: Vec<&Setting> = self.keys[mode].iter().filter_map(|k| self.converts[mode].get(k)).filter(|s| s.on && !s.sources.is_empty()).collect();
            let convert = if active.is_empty() { 0. } else { web_factor(active.iter().map(|s| s.often).sum::<f64>() / active.len() as f64) }; // several keymodes share one list
            let (recent_sets, recent_convert_sets) = if active.is_empty() { Default::default() } else { self.recent_sets(&n)? };
            let web = web_factor(self.web_bias[mode]);
            // the slider counts the maps its floor keeps from this, while dragging, without asking the worker
            let nps_pool = prepared.nps_pool(&prepared.family_best(&self.keys[mode], web, convert));
            floor = nps_pool.iter().map(|(k, ns)| (*k, ms_model::choices::nps_floor(ns, if focus >= 0. { focus } else { 0.5 }))).collect();
            self.nps_pool.insert(mode, nps_pool);
            shares = Shares { keys: self.key_shares(mode), skills: skill_shares, sliders: [web, convert], weighted: [self.web_weighted[mode], active.iter().any(|s| s.weighted)], pp_every: 0 };
            ledger = Self::ledger(&n, mode, by_skill);
            let seed = Some(n.session.shuffle_seed().to_string()); // one shuffle per run of offers: a quit (user 2026-11-15) or a pick must not re-draw it
            let (rows, relaxed) = rank(prepared, &st.catalog, &n.session, RankInputs {
                keys: self.keys[mode].clone(), mode, web, convert, recent_sets: &recent_sets, recent_convert_sets: &recent_convert_sets, taste: if mode == "nps" { &self.taste_cache.1 } else { &none },
                offered: &offered.iter().cloned().collect(), seed, focus, skills: by_skill,
            });
            relax = relaxed;
            for row in rows.into_iter().filter(|r| st.catalog.contains_key(&prepared.list.items[r.0].sha)) {
                tags.push(prepared.tags(row.0, by_skill));
                pool.push(Row::Ranked(row));
            }
            if self.mix_of(mode).on {
                shares.pp_every = self.mix_of(mode).every;
                pp_rows = self.pp_rows(mode, &n)?;
            }
        } else {
            let keys = self.keys["pp"].clone();
            let (at, rows, text) = self.rec.as_mut().unwrap().pool(&self.db, &keys, &n)?;
            (phase, note) = (at.to_string(), text);
            pp_rows = rows;
        }

        // the rows already shown: each stays in its place if it still stands, with its newest numbers
        let shas = self.local_shas();
        let (warm, target) = (n.view.warm_flag, n.target);
        let by_skill = shares.skills.is_some();
        let mut places: Vec<Option<(Pick, Tags)>> = Vec::new();
        for mut c in self.playlists[mode].clone() {
            let key = c.identity().unwrap_or_default();
            let fresh = pp_rows.iter().position(|p| p.identity().as_deref() == Some(key.as_str()));
            let rec = self.rec.as_ref().unwrap();
            places.push((|| {
                if c.purpose == "nps_mix" {
                    return Some((pp_rows.remove(fresh?), Tags { stream: PP, ..Tags::default() })); // still a likely gain
                }
                if !self.keys[mode].contains(&c.keys) || (mode == "skills" && c.practice_skills != selected) {
                    return None;
                }
                let sha = c.sha.clone().unwrap_or_default();
                if c.installed && !shas.contains(&sha) {
                    return None;
                }
                let alternative = match &local {
                    Some((st, p)) => p.find(&key).and_then(|i| builder::prepared_pick(p, i, &st.catalog, mode, &selected, &self.store.read().unwrap())),
                    None => fresh.map(|at| pp_rows.remove(at)),
                };
                if let Some(a) = alternative.as_ref().filter(|a| a.rate == c.rate && (c.md5.is_some() || !sha.is_empty())) {
                    let (why, weight, value) = (c.why.clone(), c.weight, c.value);
                    c = a.clone(); // its newest exact features
                    if local.is_some() {
                        (c.why, c.weight, c.value) = (why, weight, value);
                    }
                }
                let f = c.f.clone()?;
                let chart = c.md5.clone().unwrap_or_else(|| sha.clone());
                let mut expected = rec.predict(&f, &chart, c.bid, c.rate, &n)?;
                if let Some(a) = alternative.as_ref().filter(|a| a.rate != c.rate) {
                    let central = rec.model().central_target(rec.model().target_for(&f, target), warm);
                    if (a.pred.acc_mid - central).abs() < (expected.shown.acc_mid - central).abs() {
                        if let Some(e) = a.f.as_ref().and_then(|af| rec.predict(af, &chart, c.bid, a.rate, &n)) {
                            c = a.clone();
                            expected = e;
                        }
                    }
                }
                if mode == "skills" && alternative.is_none() {
                    return None; // no longer matches the selection: the kernel left it out
                }
                if local.is_some() {
                    // off the target now, or far under the tab's density floor (G835LX 2026-10-22: dEKA, 6 NPS, queued under a
                    // cold start's list, was offered after seven such maps)
                    let thin = c.nps.is_some_and(|x| x > 0. && x < ms_model::choices::NPS_CUT * floor.get(&c.keys).copied().unwrap_or(0.));
                    if thin || !rec.model().eligible(&expected.shown, rec.model().target_for(c.f.as_ref()?, target), warm) {
                        return None;
                    }
                }
                c.pred = expected.shown.clone();
                c.pp_pred = Some(expected.pp.clone());
                if local.is_none() && c.purpose != "warmup" {
                    if let Some(scale) = c.scale {
                        let (lo, hi, best, gain, up) = pp::estimates_at(&expected.pp, scale, c.bid, &rec.r.ledger);
                        (c.pp_lo, c.pp_hi, c.best, c.gain, c.p_up) = (Some(lo), Some(hi), best, Some(gain), Some(up));
                    }
                }
                let tags = if local.is_some() { Tags { keys: c.keys, skill: c.matched_skills.first().filter(|_| by_skill).cloned().unwrap_or_default(), stream: c.stream() } } else { Tags::default() };
                Some((c, tags))
            })());
        }

        // the pool: what the list doesn't hold yet (a chart family once, as the ranking keeps it)
        let listed: HashSet<String> = places.iter().flatten().filter_map(|p| p.0.identity()).collect();
        if let Some((_, prepared)) = &local {
            let families: HashSet<(i64, &str)> = places.iter().flatten().filter_map(|p| Some((p.0.keys, p.0.family.as_deref()?))).collect();
            let free: Vec<bool> = pool.iter().map(|r| {
                let Row::Ranked(r) = r else { return true };
                let c = &prepared.list.items[r.0];
                !listed.contains(&c.identity()) && !families.contains(&(c.keys, c.family.as_str()))
            }).collect();
            let mut keep = free.iter();
            pool.retain(|_| *keep.next().unwrap());
            let mut keep = free.iter();
            tags.retain(|_| *keep.next().unwrap());
        }
        // PP maps: not the ones just offered, unless nothing else is left (a small library cycles; never the last one again)
        let is_free = |p: &Pick, skip: &[String]| p.identity().is_none_or(|k| !listed.contains(&k) && !skip.contains(&k));
        let mut free: Vec<Pick> = pp_rows.iter().filter(|p| is_free(p, &offered)).cloned().collect();
        if free.is_empty() && pool.is_empty() && listed.is_empty() {
            free = pp_rows.iter().filter(|p| is_free(p, &offered[offered.len().saturating_sub(1)..])).cloned().collect();
            if free.is_empty() {
                free = pp_rows.iter().take(1).cloned().collect();
            }
        }
        for p in free {
            tags.push(if local.is_some() { Tags { stream: PP, ..Tags::default() } } else { Tags::default() });
            pool.push(Row::Made(p));
        }

        let mut queue: Vec<Option<Tags>> = places.iter().map(|p| p.as_ref().map(|x| x.1.clone())).collect();
        queue.resize(20, None);
        let mut dealt = deal(&tags, &shares, &ledger, &queue).into_iter();
        let mut pool: Vec<Option<Row>> = pool.into_iter().map(Some).collect();
        let mut shown: Vec<Pick> = Vec::new();
        for place in places.into_iter().chain(std::iter::repeat_with(|| None)).take(queue.len()) {
            let row = match place {
                Some((c, _)) => Some(c),
                None => dealt.next().and_then(|i| match (pool[i].take()?, &local) {
                    (Row::Ranked(r), Some((st, prepared))) => ranked_pick(prepared, r, relax, &st.catalog, mode, &selected, &self.store.read().unwrap(), miss_ratio),
                    (Row::Made(p), _) => Some(p),
                    _ => None,
                }),
            };
            shown.extend(row);
        }
        // Next plays the first row from this snapshot without asking the worker: it gets its target rate here, with the newest results.
        if let Some(head) = shown.first_mut() {
            self.aim(head, mode)?;
        }
        self.playlists.insert(mode, shown.clone());
        if mode == self.mode {
            self.fetch_ahead(shown.first());
        }
        let Some((st, prepared)) = local else { return Ok((phase, shown, note)) };
        self.log_mix(mode, &prepared, &shown, &shares, &tags);
        let target = self.acc_targets.get(mode).copied().unwrap_or(DEFAULT_TARGET);
        note = format!("target ~{:.0}%", 100. * target);
        let incomplete = st.analyzed + st.failed < st.total;
        if st.build_failed {
            note += " · local map analysis stopped";
        } else if st.total == 0 {
            note += " · no maps of these keymodes found yet";
        } else if !incomplete {
            note += &format!(" · {} maps", crate::engine::commas(st.total)); // else the window's bottom bar counts them
        }
        let missing = shown.iter().filter(|c| c.stream() == WEB).count();
        if missing != 0 {
            note += &format!(" · {missing} to download");
        }
        if mode == "skills" {
            note = format!("{} · {note}", skills::label(&self.practice_skills));
        }
        if shown.is_empty() && !st.build_failed {
            note += &if incomplete { " · looking for maps that fit your target…".to_string() } else { format!(" · no map fits your accuracy target at {} (try another target, rate range or keymode)", rate_text(self.rate_range(mode))) };
        }
        if st.failed != 0 {
            note += &format!(" · {} unreadable", st.failed);
        }
        if !st.error.is_empty() {
            note += &format!(" · {}", st.error);
        }
        if st.restored && !st.build_failed {
            note += " · showing the last list while it updates";
        }
        Ok((phase, shown, note))
    }

    /// One log line per changed list (user 2026-11-13: lists stuck on one skill, "add debugging"): each pattern group's
    /// eligible maps, its picks among the 20 shown with their mean value and weight, and its NPS taste factor.
    fn log_mix(&mut self, mode: &'static str, prepared: &ms_model::choices::Prepared, shown: &[Pick], shares: &Shares, pool: &[Tags]) {
        let sliders = shares.sliders;
        let mut eligible: BTreeMap<(i64, String), (usize, usize)> = BTreeMap::new(); // maps, of them converts
        for i in prepared.family_best(&self.keys[mode], sliders[0], sliders[1]) {
            let c = &prepared.list.items[i];
            let e = eligible.entry((c.keys, c.group.clone())).or_default();
            (e.0, e.1) = (e.0 + 1, e.1 + usize::from(c.convert));
        }
        let mut picked: BTreeMap<(i64, String), (usize, f64, f64, usize)> = BTreeMap::new();
        for p in shown {
            let e = picked.entry((p.keys, p.group.clone().unwrap_or_default())).or_default();
            (e.0, e.1, e.2, e.3) = (e.0 + 1, e.1 + p.value, e.2 + p.weight, e.3 + usize::from(p.convert_of.is_some()));
        }
        let groups: Vec<String> = eligible.iter().map(|(k, &(n, nc))| {
            let (c, v, w, cc) = picked.get(k).copied().unwrap_or_default();
            let taste = if mode == "nps" { self.taste_cache.1.get(k).map_or(String::new(), |t| format!(" taste {:.2}", (0.65 * t).exp())) } else { String::new() };
            let shown = if c > 0 { format!(" shown {c} ({cc} converts) value {:.2} weight {:.3}", v / c as f64, w / c as f64) } else { String::new() };
            format!("{}K {} {n} ({nc} converts){shown}{taste}", k.0, k.1)
        }).collect();
        // the shown order, which Next takes from the top: keymode, `c` a convert, `w` a website chart
        let order: Vec<String> = shown.iter().map(|p| format!("{}{}", p.keys, if p.convert_of.is_some() { if p.source_web { "cw" } else { "c" } } else if p.stream() == WEB { "w" } else { "" })).collect();
        // what the list is dealt by (`choices::deal`): keymode and skill shares, each keymode's stream shares over the rows left
        let keys: Vec<String> = shares.keys.iter().map(|(k, v)| format!("{k}K {v:.3}")).collect();
        let mut skills: Vec<String> = shares.skills.iter().flatten().map(|(k, v)| format!("{} {v:.3}", if k.is_empty() { "other" } else { k })).collect();
        skills.sort();
        let streams: Vec<String> = self.keys[mode].iter().filter_map(|k| {
            let own: Vec<u8> = pool.iter().filter(|t| t.keys == *k && t.stream != PP).map(|t| t.stream).collect();
            let odds = ms_model::choices::stream_odds(sliders, shares.weighted, &own);
            let total: f64 = (0..4).filter(|s| own.contains(&(*s as u8))).map(|s| odds[s]).sum();
            (total > 0.).then(|| format!("{k}K {}", (0..4).map(|s| format!("{:.2}", if own.contains(&(s as u8)) { odds[s] / total } else { 0. })).collect::<Vec<_>>().join("/")))
        }).collect();
        let line = format!("list {mode} focus {:.2} sliders web/convert {:.2}/{:.2} order {}: {} · shares keys {} · skills {} · own/web/convert/web-convert {} · PP 1 in {}", if mode == "nps" { self.nps_focus } else { self.skills_focus },
                           sliders[0], sliders[1], order.join(" "), groups.join(" | "), keys.join(", "), skills.join(", "), streams.join(", "), shares.pp_every);
        if self.mix_logged.get(mode) != Some(&line) {
            eprintln!("{line}");
            self.mix_logged.insert(mode, line);
        }
    }

    fn mix_of(&self, mode: &str) -> &PpMix {
        if mode == "nps" { &self.pp_mix } else { &self.skills_mix }
    }

    /// The PP maps past the mix bars that the NPS list or a custom playlist may take, the likeliest gains first (user 2026-10-30,
    /// 2026-11-06). `choices::deal` gives them one place in `every`.
    fn pp_rows(&mut self, mode: &'static str, n: &Now) -> Result<Vec<Pick>> {
        let keys = self.keys[mode].clone();
        let cands = self.rec.as_mut().unwrap().scored(&self.db, &keys, n)?;
        let m = self.mix_of(mode).clone();
        // A row carries its list's skill selection: Next and a click drop rows of another selection (user 2026-11-19: a PP row
        // on top of a custom playlist could not be selected, and a finished play stayed on its map).
        let selected = if mode == "skills" { self.practice_skills.clone() } else { Vec::new() };
        // Time-based freshness (no pool): an offered PP map stays back ~half an hour, a played one ~6 plays.
        let mut picks: Vec<Pick> = cands.into_iter().filter(|c| {
            let id = c.identity().unwrap_or_default();
            c.ranked && c.gain.unwrap_or(0.) >= m.min_gain && c.p_up.unwrap_or(0.) >= (if c.rate < 1. { m.ht_chance } else { m.chance })
                && n.session.freshness(&id, c.length, Some("pp"), None, ms_model::session::EXPOSURE_S) >= 0.5 && !n.session.finished_now(&id)
        }).map(|mut c| {
            c.why = format!("{} · PP map in this list", reason(&c));
            c.purpose = "nps_mix".into();
            c.practice_skills.clone_from(&selected);
            c
        }).collect();
        let quality = |c: &Pick| c.p_up.unwrap_or(0.) * (0.3 + (c.gain.unwrap_or(0.) / 0.05).ln_1p().powf(0.65));
        picks.sort_by(|a, b| quality(b).total_cmp(&quality(a)));
        Ok(picks)
    }

    fn refresh_later(&mut self) {
        if !self.refresh_queued {
            self.refresh_queued = true;
            self.q.put(Task::Refresh);
        }
    }

    /// Run a waiting map/rate selection inside a long rebuild, so its card never waits for it.
    fn yield_urgent(&mut self) {
        if self.yielding {
            return;
        }
        self.yielding = true;
        while let Some(task) = self.q.urgent() {
            self.run_task(task);
        }
        self.yielding = false;
    }

    fn request_of(&mut self, mode: &str) -> Result<Request> {
        let n = self.session()?;
        let rec = self.rec.as_ref().unwrap();
        let view = n.view.clone();
        Ok(Request { model: rec.r.fitted.display.as_ref().map_or_else(|| rec.r.fitted.pp.model.clone(), |d| d.model.clone()), tables_for: Arc::new(move |k| view.tables(k)), session: n.session.clone(),
                     public: rec.public.clone(), calc: CALC_ID.to_string(), predictor: PREDICTOR_ID.to_string(), client: self.shown.client.clone(), target: n.target, rates: self.rate_range(mode), keys: self.keys["nps"].clone(),
                     skills_keys: self.keys["skills"].clone(), selected_skills: self.practice_skills.clone(), tuning: self.skill_tuning.clone(), converts: self.converts.get(key_mode(mode)).map_or_else(Vec::new, |s| converts::pairs(s, &self.keys[key_mode(mode)])), mode: mode.to_string(),
                     nps_weight: if key_mode(mode) == "skills" { self.skills_focus.max(0.) } else { 0. }, revision: self.local_request, generation: self.generation })
    }

    fn restore_local(&mut self, mode: &str) -> Result<()> {
        if self.rec.is_none() || mode == "pp" || !self.restored.insert(mode.to_string()) {
            return Ok(());
        }
        let m = key_mode(mode);
        let Some(saved) = kv_get(&self.db, &format!("playlist_cache:{mode}")).filter(|s| s["version"] == 1) else { return Ok(()) };
        let refs: Vec<(String, f64)> = saved["choices"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|x| Some((x[0].as_str()?.to_string(), x[1].as_f64()?))).collect());
        let installed: Vec<Arc<Inst>> = self.rec.as_ref().unwrap().r.installed.values().filter(|i| self.keys[m].contains(&i.keys)).cloned().collect();
        let request = self.request_of(mode)?;
        let mut builder = Builder::new(self.store.clone(), self.rec.as_ref().unwrap().prm.clone());
        let (candidates, catalog) = builder.restore(&self.db, &request, &refs, &installed)?;
        let total = installed.len();
        self.states.insert(m, ListState { candidates, catalog, total, analyzed: total.min(saved["analyzed"].as_u64().unwrap_or(0) as usize), restored: true, ..ListState::default() });
        Ok(())
    }

    fn request_nps(&mut self) {
        if self.rec.is_none() || !(self.h.list_open)() {
            return;
        }
        self.local_request += 1;
        self.nps_requested = Instant::now();
        let mode = if self.stats_visible { "stats".to_string() } else { self.mode.clone() };
        if let Ok(r) = self.request_of(&mode) {
            *self.slot.request.lock().unwrap() = Some(r);
            self.slot.wake.notify_all();
        }
    }

    fn t_stats(&mut self, visible: bool) -> Result<()> {
        self.stats_visible = visible;
        self.request_nps();
        if !visible {
            return self.publish(None);
        }
        self.emit_stats()
    }

    fn emit_stats(&mut self) -> Result<()> {
        if self.rec.is_none() {
            return Ok(());
        }
        if self.stats_cache.as_ref().is_none_or(|c| c.0 != self.generation) {
            let profile = stats::build(self.rec.as_ref().unwrap(), &self.db)?;
            self.stats_cache = Some((self.generation, Arc::new(profile)));
        }
        let n = self.session()?;
        let profile = self.stats_cache.as_ref().unwrap().1.clone();
        (self.h.emit)(Out::Stats(Box::new(stats::current(&profile, &n, self.rec.as_ref().unwrap()))));
        Ok(())
    }

    fn t_nps_update(&mut self, update: Update) -> Result<()> {
        if self.rest_pending && matches!(update, Update::Ready(_) | Update::Failed { .. }) {
            self.rest_pending = false;
            self.carry_rest();
        }
        match update {
            Update::Failed { error, mode, revision, generation } => {
                if generation != self.generation || revision != self.local_request {
                    return Ok(());
                }
                // Keep the usable shortlist; a later ready publication replaces this terminal status.
                let m = key_mode(&mode);
                if let Some(st) = self.states.get_mut(m) {
                    st.error = error;
                    st.build_failed = true;
                }
                if mode == self.mode {
                    self.publish(Some(mode))?;
                }
            }
            Update::Progress { mode, revision, generation, total, analyzed, failed } => {
                let m = key_mode(&mode);
                if generation != self.generation || revision != self.local_request {
                    return Ok(());
                }
                if let Some(st) = self.states.get_mut(m) {
                    (st.total, st.analyzed, st.failed) = (total, analyzed, failed);
                    (self.h.emit)(Out::Progress { mode, progress: st.progress() });
                }
            }
            Update::Ready(p) => {
                if p.generation != self.generation || p.revision != self.local_request || p.keys_by_mode.0 != self.keys["nps"] || p.keys_by_mode.1 != self.keys["skills"] || p.selected_skills != self.practice_skills {
                    return Ok(());
                }
                self.accept(p)?;
            }
        }
        Ok(())
    }

    fn accept(&mut self, p: Published) -> Result<()> {
        let mode = key_mode(&p.mode);
        // The restart cache holds no website charts; the first real build replaces that placeholder list instead
        // of keeping it, or website charts never show up.
        if self.states.get(mode).is_some_and(|s| s.restored) {
            self.playlists.insert(mode, Vec::new());
        }
        // Exact new features of a chart that left the candidates: the retained queue predicts and reads from them at once.
        let mut refreshed = HashMap::default();
        {
            let store = self.store.read().unwrap();
            let practice = mode == "skills";
            for c in self.playlists.get_mut(mode).into_iter().flatten() {
                let Some(sha) = c.sha.clone() else { continue };
                if let Some(f) = store.feats(&sha, c.rate) {
                    c.f = Some(Arc::new(f.clone()));
                }
                let Some(rec) = self.rec.as_ref() else { continue };
                let Some(mut m) = store.metadata(&sha, c.rate, &self.practice_skills, &self.skill_tuning, practice, rec.model()) else { continue };
                m.family = Some(m.family).filter(|f| !f.is_empty()).or(c.family.clone()).unwrap_or_default();
                m.nps = Some(m.nps).filter(|n| *n != 0.).or(c.nps).unwrap_or(0.);
                if c.refresh(&m, practice) {
                    if let Some(key) = c.identity() {
                        refreshed.insert((key, c.rate.to_bits()), m);
                    }
                }
            }
        }
        // The features may have moved on after the Builder queued this publication: its same-rate choice must not undo that.
        let mut p = p;
        if !refreshed.is_empty() {
            p.candidates = Arc::new(p.candidates.patched(&refreshed));
        }
        let refs: Vec<Value> = p.candidates.restart_refs().into_iter().map(|(s, r)| json!([s, r])).collect();
        kv_set(&self.db, &format!("playlist_cache:{mode}"), &json!({"version": 1, "calc": CALC_ID, "choices": refs, "analyzed": p.analyzed}))?;
        self.states.insert(mode, ListState { candidates: p.candidates, catalog: p.catalog, total: p.total, analyzed: p.analyzed, failed: p.failed, error: p.error, restored: false, build_failed: false });
        if mode == self.mode {
            self.publish(Some(mode.to_string()))?;
        }
        Ok(())
    }

    fn t_configure(&mut self, c: Config) -> Result<()> {
        // Fast clicks need one search for the final selection, not a full catalogue pass for every intermediate state.
        if !MODES.contains(&c.mode.as_str()) || c.revision < self.revision.max(*self.q.config_revision.lock().unwrap()) {
            return Ok(());
        }
        let old = (self.keys["nps"].clone(), self.keys["skills"].clone(), self.practice_skills.clone(), self.skill_tuning.clone());
        let old_mode = self.mode.clone();
        self.mode.clone_from(&c.mode);
        self.action = if ACTIONS.contains(&c.action.as_str()) { c.action.clone() } else { "auto".to_string() };
        self.revision = c.revision;
        for m in MODES {
            self.keys.insert(m, c.keys.get(m).map_or_else(|| if m == "pp" { SUPPORTED_KEYS.collect() } else { vec![7] }, |k| normalize_keys(k)));
        }
        self.practice_skills = skills::normalize(c.practice_skills.as_deref().unwrap_or(&[]));
        if let Some(focus) = c.nps_focus.map(builder::normalize_focus).filter(|f| *f != self.nps_focus) {
            self.nps_focus = focus;
            self.playlists.insert("nps", Vec::new());
        }
        for (m, value) in &c.web_bias {
            let m = key_mode(m);
            if self.web_bias.get(m).is_some_and(|v| v != value) {
                self.web_bias.insert(m, *value);
                self.playlists.insert(m, Vec::new());
            }
        }
        let mut rebuild = false;
        if let Some(focus) = c.skills_focus.filter(|f| f.is_finite() && *f != self.skills_focus) {
            self.skills_focus = focus.min(1.);
            self.playlists.insert("skills", Vec::new());
            rebuild = true; // the slider's NPS share is part of each map's value
        }
        if let Some(m) = c.skills_mix.filter(|m| *m != self.skills_mix) {
            self.skills_mix = m;
            self.playlists.insert("skills", Vec::new());
        }
        for (m, value) in &c.key_weights {
            let m = key_mode(m);
            if self.key_weights.get(m) != Some(value) {
                self.key_weights.insert(m, value.clone());
                self.playlists.insert(m, Vec::new());
            }
        }
        for (m, on) in &c.by_count {
            let m = key_mode(m);
            if self.by_count.get(m) != Some(on) {
                self.by_count.insert(m, *on);
                self.playlists.insert(m, Vec::new());
            }
        }
        for (m, on) in &c.web_weighted {
            let m = key_mode(m);
            if self.web_weighted.get(m).is_some_and(|v| v != on) {
                self.web_weighted.insert(m, *on);
                self.playlists.insert(m, Vec::new());
            }
        }
        for (m, value) in c.converts {
            let m = key_mode(&m);
            if self.converts.get(m).is_some_and(|v| *v != value) {
                rebuild = true;
                self.converts.insert(m, value);
                self.playlists.insert(m, Vec::new());
            }
        }
        if let Some(w) = c.skill_weights.filter(|w| *w != self.skill_weights) {
            self.skill_weights = w;
            self.playlists.insert("skills", Vec::new());
        }
        if let Some(t) = c.skill_tuning.filter(|t| *t != self.skill_tuning) {
            self.skill_tuning = t;
            self.playlists.insert("skills", Vec::new());
        }
        let mut retarget = false;
        for (m, value) in &c.acc_targets {
            let (m, value) = (key_mode(m), builder::normalize_target(*value));
            if self.acc_targets.get(m).is_some_and(|v| *v != value) {
                self.acc_targets.insert(m, value);
                retarget = true;
                self.playlists.insert(m, Vec::new());
            }
        }
        for (m, value) in &c.rates {
            let (m, value) = (key_mode(m), builder::normalize_rates(*value));
            if self.rates.get(m).is_some_and(|v| *v != value) {
                self.rates.insert(m, value);
                retarget = true;
                self.playlists.insert(m, Vec::new());
            }
        }
        if let Some(s) = c.pp_settings.filter(|s| *s != self.settings) {
            self.settings = s.clone();
            self.playlists.insert("pp", Vec::new());
            if let Some(rec) = self.rec.as_mut() {
                rec.settings = s;
            }
        }
        if let Some(m) = c.pp_mix.filter(|m| *m != self.pp_mix) {
            self.pp_mix = m;
            self.playlists.insert("nps", Vec::new());
        }
        self.restore_local(&c.mode)?;
        if rebuild || retarget || old != (self.keys["nps"].clone(), self.keys["skills"].clone(), self.practice_skills.clone(), self.skill_tuning.clone()) || old_mode != self.mode {
            self.request_nps();
        }
        self.publish(None)
    }

    fn t_start(&mut self, play: &Value) -> Result<()> {
        let t = play["t"].as_f64().unwrap_or_else(now);
        if !tracking_allowed(&self.db, Some(t), None)? {
            return Ok(());
        }
        let sha = play["sha"].as_str().unwrap_or("").to_string();
        let path = play["path"].as_str().filter(|p| !p.is_empty()).map(PathBuf::from);
        let rate = play["rate"].as_f64();
        let mods = parse_mods(play.get("mods"));
        let data = path.as_ref().and_then(|p| std::fs::read(p).ok());
        let md5 = data.as_ref().map(|d| hashes(d).1);
        let predictable = self.rec.is_some() && mods.as_ref().is_some_and(|m| rate.is_some_and(|r| predictable_context(m, r)));
        let mut exp = None;
        if let (true, Some(p), Some(data), Some(m)) = (predictable, path.as_deref(), data.as_deref(), mods.as_deref()) {
            let (n, played) = (self.session()?, as_played(&self.db, p, data, rate, m)); // the forecast of the chart as played with its mods
            exp = self.rec.as_mut().unwrap().expectation(&self.db, &played.sha, &played.md5, played.rate, &n)?;
        }
        let inst = one_by_sha(&self.db, &sha)?;
        let meta = if inst.is_none() { path.as_deref().and_then(ms_data::lazer_index::metadata) } else { None };
        let title = inst.as_ref().map(Inst::name).or_else(|| meta.as_ref().filter(|m| m.contains_key("Title")).map(|m| {
            let g = |k: &str| m.get(k).map_or("", String::as_str);
            format!("{} - {} [{}]", g("Artist"), g("Title"), g("Version"))
        })).unwrap_or_else(|| sha.clone());
        let bid = inst.as_ref().and_then(|i| i.beatmap_id).or_else(|| meta.as_ref().and_then(|m| m.get("BeatmapID")?.parse().ok()));
        let identity_md5 = md5.clone().or_else(|| inst.as_ref().and_then(|i| i.md5.clone()));
        let mut c = self.target.clone().filter(|c| same_map(c, Some(&sha), md5.as_deref(), bid));
        if c.is_none() {
            // Restore the offer linkage after an application restart. Only the latest preceding offer can match; an
            // unrelated manual pick cannot.
            let offer: Option<(i64, String)> = self.db.query_row("SELECT id,info FROM events WHERE kind='offer' AND t<=? AND t>=? ORDER BY t DESC,id DESC LIMIT 1", [t, t - 1800.], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
            if let Some((id, info)) = offer {
                let o: Value = serde_json::from_str(&info).unwrap_or(Value::Null);
                let p = Pick { sha: o["sha"].as_str().map(str::to_string), md5: o["md5"].as_str().map(str::to_string), bid: o["bid"].as_i64(), ..Pick::default() };
                if same_map(&p, Some(&sha), identity_md5.as_deref(), bid) {
                    let mut p = p;
                    p.offer_id = Some(id);
                    p.purpose = o["purpose"].as_str().unwrap_or("").to_string();
                    p.mode = o["mode"].as_str().unwrap_or("").to_string();
                    p.rate = o["rate"].as_f64().unwrap_or(0.);
                    p.pred.acc_mid = o["accuracy"].as_f64().unwrap_or(f64::NAN);
                    p.pred.opening_acc = o["opening_accuracy"].as_f64().unwrap_or(f64::NAN);
                    p.selection = o["selection"].as_str().unwrap_or("").to_string();
                    p.practice_skills = o["practice_skills"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect());
                    c = Some(Box::new(p));
                }
            }
        }
        let matched = c.is_some();
        let keys = exp.as_ref().map(|e| e.2.keys).or(inst.as_ref().map(|i| i.keys));
        let length = exp.as_ref().map(|e| e.1.f("length") / 1000. / rate.unwrap_or(1.)).unwrap_or_else(|| inst.as_ref().and_then(|i| i.length).zip(rate).map_or(0., |(l, r)| l / 1000. / r));
        let key = if let Some(m) = identity_md5.as_ref().filter(|m| !m.is_empty()) { format!("md5:{m}") } else { format!("sha:{sha}") };
        for list in self.playlists.values_mut() {
            list.retain(|p| p.identity().as_deref() != Some(&key));
        }
        let (mut pattern, mut family) = (None, None);
        if let Some(f) = exp.as_ref().map(|e| &e.1) {
            if f.dict("nps").is_some_and(|n| n.get("version").is_some_and(|v| v.truthy())) {
                pattern = Some(ms_model::predict::profile_of(f, &builder::profile()));
                family = f.dict("nps").and_then(|n| n.get("family")).map(ms_data::codec::to_json);
            }
        }
        let rec = self.rec.as_ref();
        let chart = identity_md5.clone().unwrap_or_else(|| sha.clone());
        let forecast = exp.as_ref().and_then(|e| forecast_chart(e.0.forecast(&chart, rate.unwrap_or(1.)), Some(&sha), identity_md5.as_deref()));
        let mut info = json!({
            "sha": sha, "rate": rate, "purpose": c.as_ref().map_or("manual", |c| c.purpose.as_str()), "mode": c.as_ref().map_or("manual", |c| c.mode.as_str()), "offer_id": c.as_ref().and_then(|c| c.offer_id),
            "offered_rate": c.as_ref().map(|c| c.rate), "offered_accuracy": c.as_ref().map(|c| c.pred.acc_mid).filter(|a| !a.is_nan()), "offered_opening_accuracy": c.as_ref().map(|c| c.pred.opening_acc).filter(|a| !a.is_nan()),
            "practice_skills": c.as_ref().map(|c| c.practice_skills.clone()).filter(|_| matched), "choice": c.as_ref().map(|c| c.selection.clone()),
            "profile": pattern.or_else(|| c.as_ref().map(|c| c.profile.clone()).filter(|p| !p.is_empty())), "family": family.or_else(|| c.as_ref().and_then(|c| c.family.clone()).map(Value::from)),
            "mods": mods.as_ref().map(|m| m.iter().map(|x| x["acronym"].clone()).collect::<Vec<_>>()).unwrap_or_default(), "mods_list": mods, "title": title, "predicted_at": now(),
            "predictor": forecast.as_ref().map_or(json!(PREDICTOR_ID), |f| f["predictor"].clone()), "calculator": forecast.as_ref().map_or(json!(CALC_ID), |f| f["calculator"].clone()),
            "snapshot": rec.map(|r| r.public.meta.snapshot.clone()), "features": exp.as_ref().map(|e| ms_data::codec::to_json(&ms_calc::val::Val::Dict((*e.1).clone()))),
            "difficulty": exp.as_ref().map(|e| e.1.f("overall")), "expected": exp.as_ref().map(|e| e.0.shown.acc_mid), "expected_lo": exp.as_ref().map(|e| e.0.shown.acc_lo),
            "expected_hi": exp.as_ref().map(|e| e.0.shown.acc_hi), "accuracy_forecast": forecast, "accuracy_forecast_recorded_at": now(), "shown_mu": exp.as_ref().map(|e| e.0.pp.mu),
            "md5": identity_md5, "keys": keys, "length": length,
        });
        if let Some((e, f, _, b)) = &exp {
            let rec = rec.unwrap();
            let map = info.as_object_mut().unwrap();
            map.extend(e.fields());
            map.insert("skill".into(), json!(crate::engine::top_skill(f)));
            let meta = b.and_then(|b| rec.public.map_info(b));
            map.insert("ranked".into(), json!(meta.is_some_and(|m| ms_data::store::RANKED_STATUS.contains(&m.approved))));
        }
        log_event(&self.db, "start", Some(&key), Some(t), &info)?;
        self.current = Some(key);
        Ok(())
    }

    fn t_abort(&mut self, play: &Value) -> Result<()> {
        let t = play["t"].as_f64();
        let end = play["end"].as_f64();
        if !tracking_allowed(&self.db, Some(end.unwrap_or_else(now)), t)? {
            return Ok(());
        }
        let start: Option<(i64, Option<String>)> = self.db.query_row("SELECT id,beatmap FROM events WHERE kind='start' AND t=? ORDER BY id DESC LIMIT 1", [t], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
        let beatmap = start.as_ref().and_then(|s| s.1.clone()).or_else(|| self.current.clone());
        log_event(&self.db, "abort", beatmap.as_deref(), end, &json!({"start_id": start.map(|s| s.0), "progress": ms_calc::py::round_digits(play["progress"].as_f64().unwrap_or(0.), 3),
                                                                     "played_seconds": play["played_seconds"].as_f64().unwrap_or(0.), "partial_stats": play.get("partial_stats")}))?;
        self.refresh_later();
        Ok(())
    }

    fn t_score(&mut self, score: &Value, path: &Path, fresh: bool, play: Option<&Value>) -> Result<()> {
        let md5 = std::fs::read(path).ok().map(|d| hashes(&d).1); // None: still stored; the next import fills the ids
        let inst = match &md5 {
            Some(m) => one_by_md5(&self.db, m)?,
            None => None,
        };
        let bid = inst.as_ref().and_then(|i| i.beatmap_id);
        let sha = score["sha256"].as_str().unwrap_or("").to_string();
        let rate = score["rate"].as_f64();
        let before = match self.rec.is_some() {
            true => {
                self.session_cache = None;
                let n = self.session()?;
                let rec = self.rec.as_ref().unwrap();
                Some((rec.r.ledger.total(), bid.map_or(0., |b| rec.r.ledger.best_of(b)), n.session.phase()))
            }
            false => None,
        };
        let mut row = score.clone();
        row["md5"] = json!(md5);
        row["beatmap_id"] = json!(bid);
        row["status"] = json!(inst.as_ref().and_then(|i| i.status));
        let (key, new) = add_live_score(&self.db, &row, fresh, play)?;
        let Some(key) = key else {
            self.status("play excluded: tracking was paused");
            return Ok(());
        };
        let t = play.and_then(|p| p["t"].as_f64());
        let mut start: Option<(i64, Option<String>, Value)> = match t {
            Some(t) => self.db.query_row("SELECT id,beatmap,info FROM events WHERE kind='start' AND t=? ORDER BY id DESC LIMIT 1", [t], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, String>(2)?))).optional()?
                .map(|(a, b, c)| (a, b, serde_json::from_str(&c).unwrap_or(Value::Null))),
            None => None,
        };
        if start.as_ref().is_some_and(|s| s.2["sha"].as_str() != Some(&sha) || s.2["rate"].as_f64() != rate) {
            start = None;
        }
        let mut info = start.as_ref().map_or(Value::Null, |s| s.2.clone());
        // a delayed startup must not masquerade as a pre-play prediction
        if info["predicted_at"].as_f64().unwrap_or(0.) > score["played"].as_str().and_then(timestamp).unwrap_or_else(now) {
            info = Value::Null;
        }
        let title = inst.as_ref().map(Inst::name).or_else(|| info["title"].as_str().map(str::to_string)).unwrap_or_else(|| sha.clone());
        if fresh {
            self.replay_due = Some(Instant::now() + Duration::from_secs(3));
            self.replay_retries = 0;
            let (acc, stats): (f64, String) = self.db.query_row("SELECT acc,stats FROM scores WHERE key=?", [&key], |r| Ok((r.get(0)?, r.get(1)?)))?;
            let display = parse_stats(&stats).map_or(0., |s| acc_lazer(&s));
            // form compares this play with base_mu in lazer units (evidence::STABLE_LN_Y)
            let held = if score["client"] == "stable" { info["ln"].as_f64().unwrap_or(0.) } else { 0. };
            if self.db.query_row("SELECT 1 FROM events WHERE kind='finish' AND json_extract(info,'$.key')=?", [&key], |_| Ok(())).optional()?.is_some() {
                return Ok(());
            }
            log_event(&self.db, "finish", start.as_ref().and_then(|s| s.1.as_deref()), None, &json!({"key": key, "start_id": start.as_ref().map(|s| s.0), "prediction_valid": info["mu"].is_number(), "y": y_of(acc) - crate::evidence::STABLE_LN_Y320 * held,
                                                                                                  "y_lazer": y_of(display) - crate::evidence::STABLE_LN_Y * held, "pauses": play.and_then(|p| p.get("pauses")).filter(|p| !p.is_null()).cloned().unwrap_or(json!([]))}))?;
            // Show the saved play even if subsequent feature calculation/refitting fails.
            let mut v = self.verdict(&key, bid, None, &info, &title)?;
            v.targets = self.targets.iter().filter(|(_, c)| same_map(c, Some(&sha), md5.as_deref(), bid)).map(|x| x.0.clone()).collect();
            (self.h.emit)(Out::Result(Box::new(v)));
        }
        if self.rec.is_some() && (new || fresh) {
            let mods = parse_mods(score.get("mods_list"));
            // A play asks for one new list: after its chart's features when they are still being computed.
            let played = mods.as_ref().filter(|m| rate.is_some_and(|r| predictable_context(m, r))).and_then(|m| Some(as_played(&self.db, path, &std::fs::read(path).ok()?, rate, m)));
            let waiting = match played {
                Some(p) => self.request_feature(&p.sha, &p.path, Some(p.rate), true, false)?,
                None => false,
            };
            self.rec.as_mut().unwrap().observe(&self.db, &key)?;
            if let (true, Some((_, b0, _)), Some(s), Some(b)) = (fresh, before.as_ref(), start.as_ref(), bid) {
                log_event(&self.db, "pp_result", s.1.as_deref(), None, &json!({"start_id": s.0, "before": b0, "after": self.rec.as_ref().unwrap().r.ledger.best_of(b)}))?;
            }
            if !fresh && new {
                self.t_refit(); // a newly viewed old score is historical evidence, unlike a live result handled by form
            }
            self.solve_abilities();
            self.card();
            if fresh {
                self.retarget(Some((&sha, md5.as_deref(), bid)))?;
            }
            if !waiting {
                self.refresh_later();
            }
            self.clock_due = true;
            if self.stats_visible {
                self.t_stats(true)?;
            }
            if fresh {
                let v = self.verdict(&key, bid, before, &info, &title)?;
                (self.h.emit)(Out::Result(Box::new(v)));
            }
        }
        Ok(())
    }

    /// A finished play against what was shown, and what it did.
    fn verdict(&mut self, key: &str, bid: Option<i64>, before: Option<(f64, f64, &'static str)>, info: &Value, title: &str) -> Result<Verdict> {
        let text: String = self.db.query_row("SELECT stats FROM scores WHERE key=?", [key], |r| r.get(0))?;
        let st: Stats = parse_stats(&text).unwrap_or([0; 6]);
        let (acc, _) = acc320(&st);
        // Shown as the connected game shows it; the verdict compares lazer units (stable: shown acc of the forecast).
        let shown = self.shown.play(&st, acc_lazer(&st));
        let mut out = vec![format!("{:.2}%", 100. * shown)];
        let expected = info["expected"].as_f64().map(|e| self.shown.acc(e));
        let mut z = None;
        if let (Some(mu), Some(sd)) = (info["mu"].as_f64(), info["sd"].as_f64().filter(|&s| s != 0.)) {
            let shown_mu = info["shown_mu"].as_f64().unwrap_or(mu + PESSIMISM * info["sdm"].as_f64().unwrap_or(0.));
            let mut value = (shown_mu - y_of(acc)) / sd;
            if let (Some(m), Some(s)) = (info["display_mu"].as_f64(), info["display_sd"].as_f64().filter(|&s| s != 0.)) {
                value = (m - y_of(acc_lazer(&st))) / s;
            }
            z = Some(value);
            out.push(format!("{}{}", expected.map_or_else(String::new, |e| format!("expected ~{:.2}% · ", 100. * e)), verdict_words(value)));
        }
        let mut news = Vec::new();
        if let Some((t0, b0, ph0)) = before {
            let rec = self.rec.as_ref().unwrap();
            let total = rec.r.ledger.total();
            let b1 = bid.map_or(0., |b| rec.r.ledger.best_of(b));
            if b1 > b0 + 0.05 {
                news.push(if b0 != 0. { format!("new best {b1:.0}pp (was {b0:.0}) · account {:+.1}pp", total - t0) } else { format!("new map {b1:.0}pp · account {:+.1}pp", total - t0) });
            }
            self.session_cache = None;
            let ph1 = self.session()?.session.phase();
            if ph1 != ph0 {
                news.push(match ph1 {
                    "build" => "warmed up — building",
                    "push" => "in form — pushing",
                    "recover" => "easing off",
                    "fading" => "fading",
                    _ => "warming up",
                }.to_string());
            }
            out.extend(news.iter().cloned());
        }
        Ok(Verdict { title: title.to_string(), text: out.join(" · "), acc: shown, expected, z, rate: info["rate"].as_f64(), when: Some(now()), news, targets: Vec::new() })
    }

    fn card(&mut self) {
        let (path, rate, mods) = self.sel.clone();
        let Some(path) = path.filter(|_| self.rec.is_some()) else {
            let note = self.sel.0.as_ref().and(self.loading.clone());
            (self.h.emit)(Out::Card { c: None, note, path: self.sel.0.clone(), rate });
            return;
        };
        let (mut c, mut note) = (None, None);
        match mods {
            Some(m) => match std::fs::read(&path) {
                Err(_) => note = Some("Selected chart file is unavailable".to_string()),
                Ok(data) => {
                    // Every mod list gets a forecast: of the chart as its modelled mods make it, without the others (named on the card).
                    let played = as_played(&self.db, &path, &data, rate, &m);
                    let r = played.rate;
                    if let Ok(n) = self.session() {
                        c = self.rec.as_mut().unwrap().card(&self.db, &played.path, r, &n).ok().flatten();
                    }
                    // unseeded Random: the plain card until the three shuffles are rated, then their mean
                    let shuffled = shuffles(&self.db, &data, &played);
                    if let (false, Some(plain), Ok(n)) = (shuffled.is_empty(), c.as_ref(), self.session()) {
                        let rec = self.rec.as_mut().unwrap();
                        let cards: Vec<Card> = shuffled.iter().filter_map(|p| rec.card(&self.db, &p.path, p.rate, &n).ok().flatten()).collect();
                        if cards.len() == shuffled.len() {
                            let plain = Card { mods: played.m.labels(), without: played.m.without.clone(), ..plain.clone() };
                            c = Some(mean_card(plain, &cards));
                        }
                    }
                    match c.as_mut() {
                        Some(c) if c.random => {}
                        Some(c) => (c.mods, c.without) = (played.m.labels(), played.m.without),
                        None => {
                            let failed = self.feature_errors.lock().unwrap().get(&(played.sha, ms_calc::py::round_digits(r, 3).to_bits())).map(|x| x.1.clone());
                            note = Some(failed.map_or_else(|| format!("Calculating prediction at {r:.2}×…"), |f| format!("Prediction unavailable at {r:.2}×: {f}")));
                        }
                    }
                }
            },
            None => note = Some("Waiting for osu! to report the playback rate and mods".to_string()),
        }
        if c.is_some() {
            if let Some(asked) = self.q.selected_at.lock().unwrap().take() {
                if asked.elapsed().as_secs_f64() > 1. {
                    eprintln!("selected card latency {:.1} s", asked.elapsed().as_secs_f64());
                }
            }
        }
        (self.h.emit)(Out::Card { c: c.map(Box::new), note, path: Some(path), rate });
    }

    /// After a play the current pick is aimed again with the newest results (`aim`); only the current pick, the rest of
    /// the list keeps its offer. A pick that was just played keeps its result.
    fn retarget(&mut self, played: Option<(&str, Option<&str>, Option<i64>)>) -> Result<()> {
        let Some(c) = self.target.clone() else { return Ok(()) };
        let Some(mode) = self.targets.iter().find(|(_, t)| t.identity() == c.identity()).map(|(m, _)| m.clone()) else { return Ok(()) };
        if !MODES.contains(&mode.as_str()) || mode == "pp" || self.rec.is_none() || played.is_some_and(|(sha, md5, bid)| same_map(&c, Some(sha), md5, bid)) {
            return Ok(());
        }
        let mut aimed = (*c).clone();
        self.aim(&mut aimed, &mode)?;
        if aimed.rate == c.rate && (aimed.pred.acc_mid - c.pred.acc_mid).abs() < 1e-4 {
            return Ok(());
        }
        let msg = if aimed.rate == c.rate { "Expected accuracy updated with your latest results".to_string() } else { format!("Rate {:.2}× → {:.2}× for your target with your latest results", c.rate, aimed.rate) };
        let aimed = Box::new(aimed);
        self.target = Some(aimed.clone());
        self.targets.insert(mode.clone(), aimed.clone());
        (self.h.emit)(Out::Target { mode, revision: self.revision, selection_id: self.selection_id, c: Some(aimed), msg, copy: None });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn t_next(&mut self, skip: bool, mode: Option<String>, revision: Option<u64>, candidate: Option<Box<Pick>>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64) -> Result<()> {
        let mode = mode.unwrap_or_else(|| self.mode.clone());
        let m = key_mode(&mode);
        if !MODES.contains(&mode.as_str()) || (result.is_none() && (revision.is_some_and(|r| r != self.revision) || self.keys[m].is_empty())) || self.rec.is_none() {
            return Ok(());
        }
        if skip {
            if let Some(t) = self.targets.get(&mode) {
                log_event(&self.db, "skip", t.identity().as_deref(), None, &json!({"mode": mode}))?;
            }
        }
        match candidate.or_else(|| self.playlists[m].first().cloned().map(Box::new)) {
            Some(pick) => {
                log_event(&self.db, "next", pick.identity().as_deref(), selected_at, &json!({"purpose": pick.purpose, "var": pick.var, "mode": mode}))?;
                self.open(*pick, &mode, "next", result, selected_at, selection_id, skip)?;
            }
            None => (self.h.emit)(Out::Target { mode, revision: self.revision, selection_id: self.selection_id, c: None, msg: "Preparing the next playlist…".to_string(), copy: None }),
        }
        // Let pending selections log before refreshing the remainder.
        self.q.put(Task::Publish);
        Ok(())
    }

    fn t_snooze(&mut self, pick: &Pick, mode: String) -> Result<()> {
        let key = pick.identity();
        log_event(&self.db, "snooze", key.as_deref(), None, &json!({"mode": mode, "md5": pick.md5, "sha": pick.sha}))?;
        for list in self.playlists.values_mut() {
            list.retain(|p| p.identity() != key);
        }
        if self.targets.get(key_mode(&mode)).is_some_and(|t| t.identity() == key) {
            return self.t_next(false, Some(mode), None, None, None, None, self.selection_id);
        }
        self.publish(None)
    }

    fn t_open(&mut self, pick: Box<Pick>, mode: Option<String>, revision: Option<u64>, result: Option<Done>, selected_at: Option<f64>, selection_id: u64) -> Result<()> {
        let mode = mode.or_else(|| Some(pick.mode.clone()).filter(|m| !m.is_empty())).unwrap_or_else(|| self.mode.clone());
        if !MODES.contains(&mode.as_str()) {
            return Ok(());
        }
        let m = key_mode(&mode);
        if result.is_none() && (revision.is_some_and(|r| r != self.revision) || !self.keys[m].contains(&pick.keys) || (mode == "skills" && pick.practice_skills != self.practice_skills)) {
            return Ok(());
        }
        self.open(*pick, &mode, "list", result, selected_at, selection_id, false)?;
        self.publish(None)
    }

    /// The picked chart moved to the rate whose prediction is the accuracy target (user 2026-11-06: "at least make it so
    /// that the selected map in the playlist gets the correct rate"; the list itself keeps its analysed rates). Each rate is
    /// predicted now, with the newest results, as the card does (user 2026-11-08: "SET 1.00× ~96.0 %" while the card showed
    /// 96.3 %, and 0.82× ~95.9 % when 0.81× gave 96.0 %): a secant on ln(1 − acc) over ln rate from the prediction's own
    /// slope, then the 0.01× neighbours of the best, ≤ 7 analyses of one chart, ~5 ms each.
    fn aim(&mut self, c: &mut Pick, mode: &str) -> Result<()> {
        let tol = ms_model::predict::ACC_TOL;
        let (Some(sha), Some(inst)) = (c.sha.clone(), c.sha.as_ref().and_then(|s| self.states.get(mode)?.catalog.get(s).cloned())) else { return Ok(()) };
        if self.shown.client == "stable" || c.purpose == "nps_mix" || c.purpose == "warmup" || c.f.is_none() {
            return Ok(()); // stable has HT/NM/DT only; PP and warm-up picks aren't aimed at the target
        }
        let n = self.session()?;
        let model = self.rec.as_ref().unwrap().model();
        let target = model.central_target(c.f.as_ref().map_or(n.target, |f| model.target_for(f, n.target)), n.view.warm_flag);
        let (lo, hi) = self.rate_range(mode);
        let root = crate::catalog::local_origin(&self.shown.client).1;
        let chart = c.md5.clone().unwrap_or_else(|| sha.clone());
        let mut data = None;
        // A chart not yet in the game's store (a convert built here, a website download) is read here: the per-note blend
        // reads it too, as the card does once it's imported (user 2026-11-14: convert SET ~96.0 % while the card showed
        // 93.5 %; 2026-11-15: website SET 1.05× ~96.2 % while the card showed 96.4 %, 1.06× 96.0 %).
        let local = (!inst.installed).then(|| match inst.convert_of {
            Some(_) => crate::converts::bytes(&inst, &root),
            None => inst.path.as_ref().and_then(|p| std::fs::read(p).ok()),
        }).flatten();
        // The prediction at `rate` with its features, analysing the chart there first if needed.
        let mut at = |w: &mut Self, rate: f64| -> Result<Option<(Arc<ms_model::predict::Feats>, crate::engine::Expected)>> {
            if !w.store.read().unwrap().has_rate(&sha, rate) {
                let bytes = data.get_or_insert_with(|| local.clone().or_else(|| inst.path.clone().or_else(|| ms_data::paths::local_file(&sha, &root)).and_then(|p| std::fs::read(p).ok())));
                let Some(bytes) = bytes.as_ref() else { return Ok(None) };
                let Ok(fs) = ms_calc::features::chart_features(bytes, &[rate], &w.rec.as_ref().unwrap().prm) else { return Ok(None) };
                let mut store = w.store.write().unwrap();
                for (r, f) in &fs {
                    w.db.execute("INSERT OR REPLACE INTO feats VALUES (?,?)", rusqlite::params![format!("{sha}@{r:.3}|{CALC_ID}"), ms_data::codec::encode(&ms_calc::val::Val::Dict(f.clone()))])?;
                    store.put(&sha, *r, f);
                }
            }
            let Some(f) = w.store.read().unwrap().feats(&sha, rate).cloned() else { return Ok(None) };
            Ok(w.rec.as_ref().unwrap().expected(&w.db, &f, &sha, &chart, c.bid, rate, &n, local.as_deref()).map(|e| (Arc::new(f), e)))
        };
        let gap = |acc: f64| (1. - acc).max(1e-6).ln() - (1. - target).ln();
        let off = |e: &crate::engine::Expected| (e.shown.acc_mid - target).abs();
        let mut tried: Vec<(f64, Arc<ms_model::predict::Feats>, crate::engine::Expected)> = Vec::new();
        if let Some((f, e)) = at(self, c.rate)? {
            tried.push((c.rate, f, e));
        }
        let mut slope = tried.first().map_or(c.pred.slope, |t| t.2.shown.slope).clamp(0.5, 8.);
        for _ in 0..4 {
            let Some((r, _, e)) = tried.last() else { break };
            if off(e) <= tol / 2. {
                break;
            }
            let (r, g) = (*r, gap(e.shown.acc_mid));
            if let [.., (r0, _, e0), _] = &tried[..] {
                let s = (g - gap(e0.shown.acc_mid)) / (r / r0).ln();
                if s.is_finite() && s > 0.2 {
                    slope = s.min(8.);
                }
            }
            let next = ms_calc::py::round_digits((r * (-g / slope).exp()).clamp(lo, hi), 2);
            if tried.iter().any(|t| t.0 == next) {
                break;
            }
            let Some((f, e)) = at(self, next)? else { break };
            tried.push((next, f, e));
        }
        // Step 0.01 at a time until both neighbours of the best are no nearer the target.
        while let Some(best) = tried.iter().min_by(|a, b| off(&a.2).total_cmp(&off(&b.2))).map(|t| t.0) {
            let mut grew = false;
            for next in [best - 0.01, best + 0.01].map(|r| ms_calc::py::round_digits(r, 2)) {
                if (lo - 1e-9..=hi + 1e-9).contains(&next) && !tried.iter().any(|t| t.0 == next) {
                    if let Some((f, e)) = at(self, next)? {
                        tried.push((next, f, e));
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }
        if let Some((rate, f, e)) = tried.into_iter().min_by(|a, b| off(&a.2).total_cmp(&off(&b.2))) {
            let old = c.rate;
            (c.rate, c.var, c.f, c.pred, c.pp_pred) = (rate, crate::catalog::rate_label(rate), Some(f), e.shown, Some(e.pp));
            c.nps = c.nps.map(|v| v * rate / old);
            c.length *= old / rate;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn open(&mut self, mut c: Pick, mode: &str, selection: &str, result: Option<Done>, selected_at: Option<f64>, selection_id: u64, skipped: bool) -> Result<()> {
        c.mode = mode.to_string();
        if result.is_none() && mode != "pp" {
            self.aim(&mut c, mode)?;
        }
        c.selection = selection.to_string();
        let performed = result.is_some();
        let auto = kv_get(&self.db, AUTO_DOWNLOAD).and_then(|v| v.as_bool()).unwrap_or(true);
        let done = result.unwrap_or_else(|| {
            let action = if (self.h.busy)() { "copy" } else { self.action.as_str() };
            let p = perform(&mut c, action, &self.shown.client, auto);
            Done { copy: p.copy, msg: p.msg, sent: p.sent, unavailable: p.unavailable }
        });
        self.selection_id = self.selection_id.max(selection_id);
        let key = c.identity();
        let logged = if c.purpose == "nps_mix" { "pp" } else { mode }; // a PP map in the NPS list is a PP offer
        for list in self.playlists.values_mut() {
            list.retain(|p| p.identity() != key);
        }
        if done.unavailable {
            self.status(done.msg);
            return Ok(());
        }
        let shown = (mode == "skills").then(|| (c.practice_skills.clone(), c.matched_skills.clone()));
        let forecast = self.rec.as_ref().filter(|r| r.r.fitted.display.is_some()).and_then(|_| {
            let e = crate::engine::Expected { pp: c.pred.clone(), shown: c.pred.clone(), direct: true };
            forecast_chart(e.forecast(c.chart(), c.rate), c.sha.as_deref(), c.md5.as_deref())
        });
        let info = json!({
            "mode": logged, "keys": c.keys, "group": c.group, "bid": c.bid, "rate": c.rate, "selection": selection, "length": c.length, "accuracy": c.pred.acc_mid, "title": c.title,
            "opening_accuracy": c.pred.opening_acc, "activation_minutes": c.pred.activation, "purpose": c.purpose, "md5": c.md5, "sha": c.sha, "web": c.stream() == WEB, "source_web": c.stream() == WEB_CONVERT, "convert": c.convert_of.is_some(), "p_up": c.p_up, "gain": c.gain, "best_pp": c.best,
            "pp_mid": c.pp_mid, "practice_skills": shown.as_ref().map(|s| &s.0), "matched_skills": shown.as_ref().map(|s| &s.1), "accuracy_forecast": forecast, "accuracy_forecast_recorded_at": now(),
            "predictor": forecast.as_ref().map_or(json!(PREDICTOR_ID), |f| f["predictor"].clone()), "calculator": forecast.as_ref().map_or(json!(CALC_ID), |f| f["calculator"].clone()),
        });
        c.offer_id = Some(log_event(&self.db, "offer", key.as_deref(), selected_at, &info)?);
        let boxed = Box::new(c);
        self.target = Some(boxed.clone());
        self.targets.insert(mode.to_string(), boxed.clone());
        self.convert_turn.fetch_add(1, Ordering::SeqCst); // a delivery still running for the previous pick stops
        if crate::navigation::delivers(&boxed, auto) {
            self.deliver(boxed.clone(), mode.to_string(), skipped);
        }
        if done.sent {
            log_event(&self.db, "open", key.as_deref(), selected_at, &json!({"var": boxed.var, "mode": mode}))?;
        }
        let msg = format!("{} · set {} manually", done.msg, boxed.var);
        (self.h.emit)(Out::Target { mode: mode.to_string(), revision: self.revision, selection_id: self.selection_id, c: Some(boxed), msg, copy: if performed { None } else { done.copy } });
        Ok(())
    }

    /// A convert's turn: written and imported once no map is being played; after a skip, only if it stays the pick
    /// for a moment (user 2026-11-03: waited seconds after a play for the import to start).
    /// The upcoming pick's website set is downloaded while the current one is played: Next then only imports it.
    fn fetch_ahead(&mut self, next: Option<&Pick>) {
        let auto = kv_get(&self.db, AUTO_DOWNLOAD).and_then(|v| v.as_bool()).unwrap_or(true);
        let stable = self.shown.client == "stable";
        let set = next.filter(|p| crate::navigation::delivers(p, auto)).and_then(|p| converts::set_to_download(p, &self.db, stable));
        if set == self.ahead || cfg!(test) { // tests stay off the network and the real cache folder
            return;
        }
        self.ahead = set;
        let keep = self.target.as_ref().and_then(|t| t.set_id);
        let halt = self.halt.clone();
        let _ = std::thread::Builder::new().name("ahead".into()).spawn(move || {
            if let Err(e) = converts::fetch_ahead(set, keep, &|| !halt.load(Ordering::Relaxed)) {
                eprintln!("fetch ahead {set:?}: {e}");
            }
        });
    }

    fn deliver(&mut self, pick: Box<Pick>, mode: String, skipped: bool) {
        let turn = self.convert_turn.fetch_add(1, Ordering::SeqCst) + 1;
        let (q, h, halt, turns, path, client) = (self.q.clone(), self.h.clone(), self.halt.clone(), self.convert_turn.clone(), self.db_path.clone(), self.shown.client.clone());
        let _ = std::thread::Builder::new().name("convert".into()).spawn(move || {
            let picked = Instant::now();
            if skipped {
                std::thread::sleep(CONVERT_WAIT);
            }
            while (h.busy)() && !halt.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
            }
            if turns.load(Ordering::SeqCst) != turn || halt.load(Ordering::Relaxed) {
                return;
            }
            let root = crate::catalog::local_origin(&client).1;
            let identity = pick.identity().unwrap_or_default();
            // Next while it runs moves the turn on: the download stops and its line goes (user 2026-11-14).
            let last = std::cell::RefCell::new((Instant::now() - Duration::from_secs(1), String::new()));
            let report = |text: &str, progress: Option<f32>| {
                let mut last = last.borrow_mut();
                let step = text.split('…').next().unwrap_or("");
                if last.0.elapsed() >= Duration::from_millis(200) || last.1 != step { // a few lines a second, every new step at once
                    (h.emit)(Out::Delivery { identity: identity.clone(), text: text.to_string(), progress, failed: false });
                    *last = (Instant::now(), step.to_string());
                }
                turns.load(Ordering::SeqCst) == turn && !halt.load(Ordering::Relaxed)
            };
            let error = if crate::navigation::game_running(&client) {
                connect(&path).map_err(|e| e.to_string()).and_then(|db| converts::deliver(&pick, &db, &client, &root, picked, &report)).err()
            } else {
                Some(NOT_RUNNING.to_string())
            };
            if error.as_deref() == Some(converts::CANCELLED) {
                return;
            }
            (h.emit)(Out::Delivery { identity, text: error.clone().map_or_else(String::new, |e| format!("Couldn't get it into osu!: {e} — Next skips it")), progress: None, failed: error.is_some() });
            q.put(Task::ConvertDone { pick, mode, error });
        });
    }

    /// The map selected in osu! as a convert, on request and outside any list: delivered like a list's.
    fn t_convert_selected(&mut self, path: PathBuf, target: i64) {
        let converted = std::fs::read(&path).map_err(|e| e.to_string()).and_then(|data| Ok((hashes(&data).0, ms_convert::convert(&data, target as usize).map_err(|e| e.to_string())?)));
        let (source, out) = match converted {
            Ok(done) => done,
            Err(e) => return self.status(format!("Couldn't convert that map: {e}")),
        };
        let m = ms_data::lazer_index::metadata(&path).unwrap_or_default();
        let (sha, md5) = hashes(&out.osu);
        let text = |k: &str| m.get(k).cloned();
        let pick = Pick { sha: Some(sha), md5: Some(md5), title: text("Title").unwrap_or_default(), artist: text("Artist"), song: text("Title"), version: Some(out.version), creator: text("Creator"),
                          keys: target, set_id: text("BeatmapSetID").and_then(|s| s.parse().ok()), convert_of: Some(source), ..Pick::default() };
        self.status("Converting…");
        self.deliver(Box::new(pick), SELECTED.to_string(), false);
    }

    fn t_convert_done(&mut self, pick: Box<Pick>, mode: String, error: Option<String>) -> Result<()> {
        let current = self.targets.get(&mode).is_some_and(|t| t.identity() == pick.identity());
        match error {
            None if mode == SELECTED => {
                let first = if self.shown.client == "stable" { "press F5 in song select, then Ctrl+V" } else { "press Ctrl+V in song select" };
                (self.h.emit)(Out::Copy { text: crate::navigation::song_search(&pick, &self.shown.client), msg: format!("Ready — {first}") });
            }
            None if current => {
                let mut c = pick;
                c.source_web |= !c.installed; // a downloaded website map keeps its WEB badge (user 2026-11-15)
                c.installed = true;
                let copy = Some(crate::navigation::song_search(&c, &self.shown.client));
                let first = if self.shown.client == "stable" { "press F5 in song select, then Ctrl+V" } else { "press Ctrl+V in song select" };
                let msg = format!("Ready — {first} · set {} manually", c.var);
                self.targets.insert(mode.clone(), c.clone());
                self.target = Some(c.clone());
                (self.h.emit)(Out::Target { mode, revision: self.revision, selection_id: self.selection_id, c: Some(c), msg, copy });
            }
            None => {}
            Some(e) if e == NOT_RUNNING => self.status(format!("{e} — start it, then press Next")),
            Some(e) => {
                eprintln!("delivery failed: {} {e}", pick.title);
                // The pick stays: moving on by itself pulled the map from under the player (user 2026-11-11). Next skips it.
                self.status(format!("Couldn't {} that map: {e} — press Next to skip it", if pick.convert_of.is_some() { "convert" } else { "download" }));
            }
        }
        Ok(())
    }

    fn t_selected(&mut self, path: Option<PathBuf>, rate: Option<f64>, mods: Option<Vec<Value>>) -> Result<()> {
        let previous = self.sel.0.clone();
        self.sel = (path.clone(), rate, mods.clone());
        let mut hashed = None;
        if let Some(p) = &path {
            if let Ok(data) = std::fs::read(p) {
                let (sha, md5) = hashes(&data);
                if let Some(m) = &mods {
                    let played = as_played(&self.db, p, &data, rate, m); // any mods: the forecast is of what the calculator can build
                    self.request_feature(&played.sha, &played.path, Some(played.rate), false, false)?;
                    for s in shuffles(&self.db, &data, &played) {
                        self.request_feature(&s.sha, &s.path, Some(s.rate), false, true)?;
                    }
                }
                if self.rec.is_some() && previous.as_ref() != Some(p) && !self.local_shas().contains(&sha) {
                    self.begin_import();
                }
                hashed = Some((sha, md5));
            }
        }
        self.card();
        let (Some(target), Some(p), Some((sha, md5))) = (self.target.clone(), path, hashed) else { return Ok(()) };
        let bid = ms_data::lazer_index::metadata(&p).and_then(|m| m.get("BeatmapID")?.parse().ok());
        if same_map(&target, Some(&sha), Some(&md5), bid) {
            let updated = target.md5.as_deref() != Some(md5.as_str());
            log_event(&self.db, "selected", Some(&format!("md5:{md5}")), None, &json!({"offered_revision": target.identity(), "updated": updated}))?;
            let msg = format!("✓ {}chart selected — intended {}; {}", if updated { "updated " } else { "" }, target.var, rate.map_or_else(|| "actual rate unknown".to_string(), |r| format!("observed {r:.2}×")));
            (self.h.emit)(Out::Target { mode: target.mode.clone(), revision: self.revision, selection_id: self.selection_id, c: Some(target), msg, copy: None });
        }
        Ok(())
    }

    /// True: a FeatureReady follows. `keep`: beside the pending requests (a selection's Random shuffles), else in their place.
    fn request_feature(&mut self, sha: &str, path: &Path, rate: Option<f64>, refit: bool, keep: bool) -> Result<bool> {
        let (Some(rate), Some(rec)) = (rate, self.rec.as_mut()) else { return Ok(false) };
        if sha.is_empty() || rec.features(&self.db, Some(sha), None, None, rate)?.0.is_some() {
            return Ok(false);
        }
        let id = (sha.to_string(), ms_calc::py::round_digits(rate, 3).to_bits());
        {
            let mut errors = self.feature_errors.lock().unwrap();
            if errors.get(&id).is_some_and(|f| f.0 > Instant::now()) {
                return Ok(false);
            }
            errors.remove(&id);
        }
        let mut pending = self.feature_pending.lock().unwrap();
        let refit = refit || pending.0.iter().any(|p| p.3); // a replaced play's list request rides on the newest
        if !keep {
            pending.0.clear();
        }
        pending.0.push_back((sha.to_string(), path.to_path_buf(), rate, refit));
        if pending.1 {
            return Ok(true); // the newest request replaces a pending old selection
        }
        pending.1 = true;
        let (queue, errors, slot, halt, h, prm) = (self.q.clone(), self.feature_errors.clone(), self.feature_pending.clone(), self.halt.clone(), self.h.clone(), self.rec.as_ref().unwrap().prm.clone());
        std::thread::Builder::new().name("selected-feature".into()).spawn(move || loop {
            let task = {
                let mut p = slot.lock().unwrap();
                let task = p.0.pop_front();
                if task.is_none() || halt.load(Ordering::Relaxed) {
                    p.1 = false;
                    return;
                }
                task.unwrap()
            };
            // Not deferred during play: waiting left the card on "Calculating…" for the whole song after an in-game
            // rate change (2026-10-10).
            let (sha, path, rate, refit) = task;
            let id = (sha.clone(), ms_calc::py::round_digits(rate, 3).to_bits());
            match Engine::analyse(&path, ms_calc::py::round_digits(rate, 3), &prm) {
                Ok(fs) => {
                    errors.lock().unwrap().remove(&id);
                    queue.put(Task::FeatureReady { sha, features: Some(fs), refit });
                }
                Err(e) => {
                    let mut list = errors.lock().unwrap();
                    if list.len() > 64 {
                        list.clear();
                    }
                    list.insert(id, (Instant::now() + Duration::from_secs(30), e.chars().take(240).collect()));
                    drop(list);
                    (h.emit)(Out::Status { text: format!("Selected chart prediction unavailable: {}", e.chars().take(160).collect::<String>()), progress: None, blocking: false, learning: None });
                    queue.put(Task::FeatureReady { sha, features: None, refit });
                }
            }
        }).expect("feature thread");
        Ok(true)
    }

    fn t_client(&mut self, name: &str) -> Result<()> {
        if name != "lazer" && name != "stable" {
            return Ok(());
        }
        kv_set(&self.db, "client", &json!(name))?;
        if name == self.shown.client {
            return Ok(());
        }
        self.shown.client = name.to_string();
        self.shown.miss_ratio = stable_miss_ratio(&self.db)?;
        self.first_import = true;
        self.begin_import();
        Ok(())
    }

    /// Restored recommendations are usable before the game's files are read. One import is coalesced; the live event and
    /// model owner remains this worker.
    fn begin_import(&mut self) {
        if self.jobs.importing.swap(true, Ordering::SeqCst) {
            return;
        }
        let game = client_name(&self.shown.client);
        self.status(if self.first_import { format!("Reading your {game} scores and installed maps (first time: about a minute)…") } else { format!("Using saved history · checking {game} for new plays in the background…") });
        let (q, h, halt, jobs, path, client) = (self.q.clone(), self.h.clone(), self.halt.clone(), self.jobs.clone(), self.db_path.clone(), self.shown.client.clone());
        std::thread::Builder::new().name("score-import".into()).spawn(move || {
            while (h.busy)() && !halt.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(1));
            }
            if halt.load(Ordering::Relaxed) {
                return;
            }
            let run = || -> std::result::Result<(Imported, usize, bool), String> {
                let db = connect(&path).map_err(|e| e.to_string())?;
                let result = import_library(&db, &client)?;
                let recovered = recover_results(&db).map_err(|e| e.to_string())?;
                let mut changed = result.changed;
                // A website chart now in the client drops its stored .osu right away, not at the next launch (user 2026-11-06).
                // A running crawl does the same at its start and owns state.db meanwhile.
                if changed && !ms_data::webmaps::running() {
                    if let Err(e) = ms_data::webmaps::forget_in_client() {
                        eprintln!("website charts not forgotten: {e}");
                    }
                }
                // Website top 100 for a named profile: bests from stable/other PCs (user 2026-10-01).
                let name = kv_get(&db, "website_username").and_then(|v| v.as_str().map(str::to_string));
                let last = kv_get(&db, "website_user").and_then(|v| v["t"].as_f64()).unwrap_or(0.);
                if let Some(name) = name.filter(|_| now() - last > 6. * 3600.) {
                    match imports::import_website_best(&db, &name) {
                        Ok(_) => changed = true,
                        Err(e) => (h.emit)(Out::Status { text: format!("osu! profile scores unavailable: {e}").chars().take(160).collect(), progress: None, blocking: false, learning: None }),
                    }
                }
                Ok((result, recovered, changed))
            };
            let task = match run() {
                Ok((result, recovered, changed)) => Task::Imported { result: Some(result), recovered, changed, error: None },
                Err(e) => Task::Imported { result: None, recovered: 0, changed: false, error: Some(e.lines().next().unwrap_or("").chars().take(160).collect()) },
            };
            jobs.importing.store(false, Ordering::SeqCst);
            q.put(task);
        }).expect("import thread");
    }

    fn t_imported(&mut self, result: Option<Imported>, recovered: usize, changed: bool, error: Option<String>) -> Result<()> {
        let first = std::mem::take(&mut self.first_import);
        let game = client_name(&self.shown.client);
        if let Some(e) = error {
            if first {
                self.stage(Some(format!("Could not read your {game} scores: {e}")), true);
            } else {
                self.status(format!("Saved recommendations retained; {game} import pending: {e}"));
            }
            return Ok(());
        }
        let Some(r) = result else { return Ok(()) };
        if let Some(m) = r.miss_ratio {
            self.shown.miss_ratio = m;
        }
        let head = if r.new != 0 { format!("Read {} new plays and {} installed maps from {game}", crate::engine::commas(r.new as usize), crate::engine::commas(r.installed)) } else { format!("Up to date with {game} ({} maps)", crate::engine::commas(r.installed)) };
        self.status(format!("{head}{}", if recovered != 0 { format!(" · {recovered} predictions recovered") } else { String::new() }));
        if r.new != 0 || recovered != 0 {
            let rows = play_history(&self.db, 1, false, None)?;
            if let Some(p) = rows.first() {
                (self.h.emit)(Out::Result(Box::new(Verdict { title: p.title.clone(), text: history_text(&rows, &self.shown), acc: self.shown.play(&p.stats, p.actual), expected: p.expected.map(|e| self.shown.acc(e)), z: p.z, rate: p.rate,
                                                             when: p.played.as_deref().and_then(timestamp), news: Vec::new(), targets: Vec::new() })));
            }
        }
        if r.new != 0 && self.rec.is_some() {
            self.start_fill(); // before the refit: its publish must already know the level is pending
        }
        if r.new != 0 || recovered != 0 || changed {
            self.t_refit();
        }
        Ok(())
    }

    fn first_learning(&self) -> bool {
        self.rec.as_ref().is_none_or(|r| r.r.fitted.rows.rows.len() < 30)
    }

    /// Features for the player's own charts that the public table lacks (custom rates, unranked maps). A first run imports
    /// thousands of plays after this started once with none (fresh install 2026-10-02: 5,632 chart/rates never analysed,
    /// so PP suggested 4K Beginner maps to a 7K player). One at a time; an import that brings new plays starts another.
    fn start_fill(&mut self) {
        if self.jobs.filling.swap(true, Ordering::SeqCst) {
            return;
        }
        let first = self.first_learning();
        if first {
            *self.learning.lock().unwrap() = Some("Checking which of your played charts still need analysing…".to_string());
        }
        let rec = self.rec.as_ref().unwrap();
        let (q, h, halt, jobs, path, public, prm, learning, c) = (self.q.clone(), self.h.clone(), self.halt.clone(), self.jobs.clone(), self.db_path.clone(), rec.public.clone(), rec.prm.clone(), self.learning.clone(), self.c.clone());
        let installed = rec.r.installed.clone();
        std::thread::Builder::new().name("features".into()).spawn(move || {
            let mut filled = false;
            let mut run = || -> Result<()> {
                let db = connect(&path)?;
                let mut source = crate::evidence::FeatureSource { public: public.clone(), calc: CALC_ID.to_string(), cache: HashMap::default() };
                let scored: Vec<(String, Option<String>, Option<i64>, Option<f64>, Option<String>)> = db.prepare("SELECT sha256, md5, beatmap_id, rate, mods FROM scores WHERE sha256 IS NOT NULL")?
                    .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?.collect::<Result<_>>()?;
                // a modded play is a play of its own chart: built here from the source file (modcharts), then analysed like any other
                let mut rows: Vec<(Option<String>, Option<String>, Option<i64>, Option<f64>)> = Vec::with_capacity(scored.len());
                for (sha, md5, bid, rate, mods) in scored {
                    let Some(m) = crate::evidence::evidence_mods(mods.as_deref()) else { continue };
                    let (rate, tag) = (rate.or(m.rate), m.tag());
                    if tag.is_empty() {
                        rows.push((Some(sha), md5, bid, rate));
                    } else if let Some(data) = pp::chart_path(Some(&sha)).and_then(|p| std::fs::read(p).ok()) {
                        rows.push((Some(crate::modcharts::ensure(&db, &sha, &data, &tag)?.0), None, None, rate));
                    }
                }
                source.prefetch(&db, rows.iter().filter_map(|r| Some(format!("{}@{:.3}|{CALC_ID}", r.0.as_ref()?, ms_calc::py::round_digits(r.3.filter(|&x| x != 0.).unwrap_or(1.), 3)))))?;
                let mut need: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
                let mut want = |source: &mut crate::evidence::FeatureSource, sha: &str, md5: Option<&str>, bid: Option<i64>, rate: f64| -> Result<()> {
                    // An analysis of another structure version is redone too (2026-11-06: rows without the LN types).
                    let stale = source.of(&db, Some(sha), md5, bid, Some(rate))?.0.is_none_or(|f| f.dict("nps").and_then(|n| n.get("version")).map(ms_calc::val::Val::f) != Some(STRUCTURE_VERSION as f64));
                    if stale && crate::modcharts::chart_file(&db, sha).is_some() {
                        let rate = ms_calc::py::round_digits(rate, 3);
                        let list = need.entry(sha.to_string()).or_default();
                        if !list.contains(&rate) {
                            list.push(rate);
                        }
                    }
                    Ok(())
                };
                for r in &rows {
                    want(&mut source, r.0.as_deref().unwrap(), r.1.as_deref(), r.2, r.3.filter(|&x| x != 0.).unwrap_or(1.))?;
                }
                // while the background carry runs, a chart without a row is its to bring (`carry_rest` fills again after it)
                for inst in pp::newer_installed(&public, &installed).into_iter().filter(|_| !ms_data::store::CARRYING.load(Ordering::Relaxed)) {
                    for (rate, _) in pp::VARIANTS {
                        want(&mut source, &inst.sha256, inst.md5.as_deref(), None, rate)?;
                    }
                }
                let _ = &c;
                let jobs_list: Vec<(String, Vec<f64>)> = need.into_iter().collect();
                if jobs_list.is_empty() {
                    return Ok(());
                }
                filled = true;
                let (t0, total) = (Instant::now(), jobs_list.len());
                let mut refits = 0;
                let emit = |done: usize| {
                    let speed = done as f64 / t0.elapsed().as_secs_f64().max(1e-3);
                    let left = if speed > 0. { (total - done) as f64 / speed } else { 0. };
                    let eta = if left >= 90. { format!(" · ~{:.0} min left", left / 60.) } else if left > 0. { format!(" · ~{left:.0} s left") } else { String::new() };
                    let (d, t) = (crate::engine::commas(done), crate::engine::commas(total));
                    let text = if first { format!("Analysing the charts you have played, to learn your level · {d} / {t} charts{eta}") } else { format!("Analysing {t} new or updated maps in the background · {d} done{eta}") };
                    *learning.lock().unwrap() = first.then(|| text.clone());
                    (h.emit)(Out::Status { text, progress: Some((done, total)), blocking: false, learning: Some(first) });
                };
                emit(0);
                // First learning is all the player waits for: every core but two, paused while playing.
                let workers = if first { webmaps::workers(None) } else { 1 };
                let chunk = 12.max(4 * workers);
                for (i, part) in jobs_list.chunks(chunk).enumerate() {
                    while (h.busy)() && !halt.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_secs(2));
                    }
                    if halt.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let data: Vec<(Vec<u8>, Vec<f64>)> = part.iter().map(|(sha, rates)| (crate::modcharts::chart_file(&db, sha).and_then(|p| std::fs::read(p).ok()).unwrap_or_default(), rates.clone())).collect();
                    for ((sha, _), res) in part.iter().zip(features_batch(&data, workers, &prm)) {
                        let Ok(fs) = res else { continue };
                        let tx = db.unchecked_transaction()?;
                        for (rate, f) in fs {
                            tx.execute("INSERT OR REPLACE INTO feats VALUES (?, ?)", rusqlite::params![source.key(sha, rate), ms_data::codec::encode(&ms_calc::val::Val::Dict(f))])?;
                        }
                        tx.commit()?;
                    }
                    let done = total.min((i + 1) * chunk);
                    emit(done);
                    // Only while the level is unknown: each refit restarts the NPS/Skills build.
                    if first && done - refits >= 200.max(total / 5) && done < total {
                        refits = done;
                        q.put(Task::Refit); // suggestions sharpen while the rest is analysed
                    }
                }
                Ok(())
            };
            let outcome = run();
            *learning.lock().unwrap() = None;
            jobs.filling.store(false, Ordering::SeqCst);
            match outcome {
                Ok(()) if filled => {
                    q.put(Task::Refit);
                    (h.emit)(Out::Status { text: String::new(), progress: None, blocking: false, learning: Some(false) });
                }
                Ok(()) => {}
                Err(e) => {
                    eprintln!("feature fill: {e}");
                    (h.emit)(Out::Status { text: "Some score analyses could not be completed; available recommendations are kept".to_string(), progress: None, blocking: false, learning: Some(false) });
                }
            }
        }).expect("feature fill thread");
    }

    /// Archive independently: neither a busy queue nor slow game-file I/O blocks Next.
    fn catch_up_replays(&mut self) {
        if self.replay_due.is_none_or(|d| Instant::now() < d) || (self.h.busy)() || self.jobs.importing.load(Ordering::SeqCst) || self.jobs.archiving.swap(true, Ordering::SeqCst) {
            return;
        }
        self.replay_due = None;
        let (q, jobs, path, client) = (self.q.clone(), self.jobs.clone(), self.db_path.clone(), self.shown.client.clone());
        std::thread::Builder::new().name("replay-archive".into()).spawn(move || {
            let run = || -> std::result::Result<(bool, Vec<String>), String> {
                let e = |x: rusqlite::Error| x.to_string();
                let db = connect(&path).map_err(e)?;
                if client == "stable" {
                    recover_results(&db).map_err(e)?;
                    replays::archive_stable(&db, None).map_err(e)?; // stable's scores came live; its replay file follows
                } else {
                    // A small post-result catch-up: recent scores only, no full library scan.
                    let since = chrono::Utc::now().timestamp() - 3 * 86400;
                    let raw = imports::snapshot(None, false, Some((since, 0)))?;
                    let user = kv_get(&db, "user_id").and_then(|v| v.as_i64());
                    for row in imports::realm_rows(&raw, user) {
                        ms_data::store::upsert_score(&db, row).map_err(e)?;
                    }
                    recover_results(&db).map_err(e)?;
                    replays::archive(&db, &raw, "lazer").map_err(e)?;
                }
                // retry only for the play just finished: one older score without a replay retried 3 more full catch-ups (each a
                // ~200 MB realm copy) after every later play (2026-11-17)
                let pending = db.query_row("SELECT 1 FROM replay_evidence WHERE status='missing' AND julianday(played)>=julianday('now','-30 minutes') LIMIT 1", [], |_| Ok(())).optional().map_err(e)?.is_some();
                let fresh: Vec<String> = db.prepare("SELECT key FROM scores WHERE fresh=1 AND julianday(played)>=julianday('now','-3 days')").map_err(e)?.query_map([], |r| r.get(0)).map_err(e)?.flatten().collect();
                Ok((pending, fresh))
            };
            let task = match run() {
                Ok((pending, fresh)) => Task::ReplayDone { pending, error: None, fresh },
                Err(e) => Task::ReplayDone { pending: true, error: Some(e.chars().take(120).collect()), fresh: Vec::new() },
            };
            jobs.archiving.store(false, Ordering::SeqCst);
            q.put(task);
        }).expect("replay thread");
    }

    /// The per-note forecast's ability history (pernote::PerNote::fill) before anything reads it: on a thread it raced the
    /// lists, which read whatever plays were solved so far (audit 2026-11-20: two pinned starts gave different forecasts).
    fn solve_abilities(&self) {
        let Some(p) = self.rec.as_ref().and_then(|r| r.pernote.clone()) else { return };
        if let Err(e) = p.fill(&self.db, crate::pernote::today() - crate::pernote::WINDOW) {
            eprintln!("per-note abilities: {e}");
        }
    }

    fn t_replay_done(&mut self, pending: bool, error: Option<String>, fresh: Vec<String>) -> Result<()> {
        if self.rec.is_some() && !fresh.is_empty() {
            for key in &fresh {
                self.rec.as_mut().unwrap().observe(&self.db, key)?;
            }
            // no new list: the replay joins the next play's request (one list per play)
        }
        if pending {
            self.replay_retries += 1;
            if self.replay_retries <= 3 {
                self.replay_due = Some(Instant::now() + Duration::from_secs(20));
            }
        } else {
            self.replay_retries = 0;
        }
        if let Some(e) = error {
            self.status(format!("Replay archive pending: {e}"));
        }
        Ok(())
    }

    /// A refit runs on its own connection and thread; the worker keeps answering cards from the previous model (a 15–40 s
    /// refit held the selected card that long, G835LX run.log 2026-10).
    fn t_refit(&mut self) {
        if (self.h.busy)() {
            self.refit_due = true;
            return;
        }
        self.refit_due = false;
        let Some(rec) = self.rec.as_ref() else { return };
        self.last_refit = now();
        if self.refit_running {
            self.refit_again = true; // newer evidence arrived: fit once more after this one
            return;
        }
        self.refit_running = true;
        *rec.observing.lock().unwrap() = Some(Vec::new());
        let (q, public, c, path) = (self.q.clone(), rec.public.clone(), self.c.clone(), self.db_path.clone());
        std::thread::Builder::new().name("refit".into()).spawn(move || {
            let result = connect(&path).and_then(|db| refit(&db, &public, &c));
            if let Err(e) = &result {
                eprintln!("refit: {e}");
            }
            q.put(Task::RefitDone(result.ok().map(Box::new)));
        }).expect("refit thread");
    }

    fn save_pp_floor(&self) {
        if let Some(rec) = self.rec.as_ref() {
            let _ = kv_set(&self.db, PP_FLOOR, &json!(pp::top100_stars(&rec.r.ledger)));
        }
    }

    fn t_refit_done(&mut self, r: Option<Box<Refit>>) -> Result<()> {
        self.refit_running = false;
        if let (Some(r), Some(rec)) = (r, self.rec.as_mut()) {
            let observed = rec.observing.lock().unwrap().take().unwrap_or_default();
            rec.apply(&self.db, *r, observed)?;
            self.save_pp_floor();
            self.generation += 1;
            self.session_cache = None;
            self.request_nps();
            if self.mode == "pp" {
                self.publish(None)?;
            }
            self.card();
            if self.stats_visible {
                self.t_stats(true)?;
            }
        } else if let Some(rec) = self.rec.as_ref() {
            *rec.observing.lock().unwrap() = None;
        }
        if std::mem::take(&mut self.refit_again) {
            self.t_refit();
        }
        Ok(())
    }
}

/// Moves the pick with this identity to the front; false when it isn't listed.
#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;

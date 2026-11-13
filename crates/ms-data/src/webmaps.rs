//! Website charts for the NPS and Skills lists: every status, analysed like installed maps. osu.direct metadata →
//! chart text (.osu only, no audio) for charts whose density can reach the NPS range at 0.70–1.50× → features on a
//! rate grid → WEB_DIR/web.msp, the fields the lists read, one blob per rate. Treated as not installed: Next sends
//! osu://b/<id> so the client offers the download. A calculator change needs another crawl (only analysis reruns;
//! files and metadata are cached). `stop` in WEB_DIR pauses a crawl, the next one resumes from state.db.
use crate::blobs::{self, Blobs};
use crate::codec;
use crate::paths;
use crate::pickle::{self, P};
use crate::public::value;
use crate::store::now;
use ms_calc::features::{features_batch};
use ms_calc::params::Params;
use ms_calc::val::{Dict, Val};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

const SEARCH: &str = "https://osu.direct/api/v2/search?mode=3&amount=100&sort=last_updated:desc";
// osu.direct first; ppy only for what the mirror lacks, slowly: both laptops share the home IP that the client uses,
// so a throttled IP at ppy would hurt play (user concern 2026-09-30).
pub const FILES: [(&str, f64); 2] = [("https://osu.direct/api/osu/", 0.6), ("https://osu.ppy.sh/osu/", 3.)];
const UA: &str = "maniascope/1.0 (personal osu!mania practice recommender)";
// Listing order: ranked, approved, qualified, loved, then pending, wip, graveyard. Ranked first: the graveyard is most of
// the ~100k mania sets (user 2026-11-06: "the search currently goes through like 100k maps … it searches non-ranked too").
const STATUSES: [i64; 7] = [1, 2, 3, 4, 0, -1, -2];
/// The window's status choices: (name the app and `--status` use, osu! statuses).
pub const GROUPS: [(&str, &[i64]); 2] = [("ranked", &[1, 2, 3, 4]), ("unranked", &[0, -1, -2])];

/// osu! statuses of the named groups.
pub fn statuses(groups: &[&str]) -> Vec<i64> {
    GROUPS.iter().filter(|g| groups.contains(&g.0)).flat_map(|g| g.1.iter().copied()).collect()
}

/// One keymode's choice in the website charts window (user 2026-11-13: 4K unranked at 5–7★ is 160,000 maps, 7K ranked is
/// not): statuses, ★ range at 1.0×, least plays. `on`: this keymode downloads; an off keymode keeps its settings and the
/// maps it already has in the playlists.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Want {
    pub keys: i64,
    pub on: bool,
    pub statuses: Vec<i64>,
    pub stars: (f64, f64),
    pub min_plays: i64,
}

impl Want {
    pub fn new(keys: i64, stars: (f64, f64)) -> Want {
        Want { keys, on: true, statuses: statuses(&["ranked"]), stars, min_plays: 0 }
    }

    /// Rows listed before the status column count as unranked; charts without a listed rating stay in (the filter only
    /// drops what it can see).
    fn takes(&self, status: Option<i64>, notes: i64, length: f64, stars: Option<f64>, plays: i64) -> bool {
        self.statuses.contains(&status.unwrap_or(-2)) && reachable(notes, length, self.keys) && stars.is_none_or(|s| self.stars.0 <= s && s <= self.stars.1)
            && plays >= self.min_plays
    }
}

/// The keymode's want takes this row (`on_only`: only keymodes that download now).
#[allow(clippy::too_many_arguments)]
fn wanted(wants: &[Want], on_only: bool, keys: i64, status: Option<i64>, notes: i64, length: f64, stars: Option<f64>, plays: i64) -> bool {
    wants.iter().any(|w| w.keys == keys && (w.on || !on_only) && w.takes(status, notes, length, stars, plays))
}

/// The window's saved choice. Crawls before per-keymode choices saved one keymode set, status set and ★ range: each keymode
/// with listed rows gets those (off unless it was chosen).
pub fn wants(db: &Connection) -> Vec<Want> {
    if let Some(w) = kv_get(db, "wants").and_then(|v| serde_json::from_value(v).ok()) {
        return w;
    }
    let on: HashSet<i64> = kv_get(db, "want_keys").and_then(|v| v.as_array().map(|a| a.iter().filter_map(Value::as_i64).collect())).unwrap_or_default();
    let statuses = kv_get(db, "want_status").and_then(|v| v.as_array().map(|a| a.iter().filter_map(Value::as_i64).collect())).unwrap_or_else(|| STATUSES.to_vec());
    let stars = kv_get(db, "stars").and_then(|v| Some((v[0].as_f64()?, v[1].as_f64()?))).unwrap_or((0., 99.));
    let listed: HashSet<i64> = db.prepare("SELECT DISTINCT keys FROM diffs").and_then(|mut q| q.query_map([], |r| r.get(0))?.collect()).unwrap_or_default();
    let mut keys: Vec<i64> = listed.union(&on).copied().collect();
    keys.sort_unstable();
    keys.into_iter().map(|k| Want { keys: k, on: on.contains(&k), statuses: statuses.clone(), stars, min_plays: 0 }).collect()
}
// HT/NM/DT only (user 2026-10-02: ~3x faster crawl, ~3x smaller file). NPS/Skills compute the exact rate a candidate
// needs on demand, as for installed maps.
pub const RATES: [f64; 3] = [0.75, 1., 1.5];
pub const PAUSE: f64 = 0.6;   // osu.direct allows 120 requests/minute
const PAGES: i64 = 4;     // pages in flight (~3 s each server-side → ~60/min)
// Before the first run on this PC (measured on G533QR 2026-10-01); runs learn their own.
const SEC_FETCH: f64 = 2.4;
const SEC_ANALYSE: f64 = 0.7;
const CHART_BYTES: i64 = 150_000;
const MAGIC: &[u8; 8] = b"MSWEB001";

pub fn web_dir() -> PathBuf {
    paths::rec_dir().join("web")
}

pub fn workers(n: Option<usize>) -> usize {
    n.unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |x| x.get()).saturating_sub(2).max(1))
}

#[derive(Debug)]
pub enum CrawlError {
    Stopped,
    Failed(String),
}

impl<E: std::fmt::Display> From<E> for CrawlError {
    fn from(e: E) -> CrawlError {
        CrawlError::Failed(e.to_string())
    }
}

pub fn get(url: &str) -> Result<Vec<u8>, String> {
    let mut last = String::new();
    for attempt in 0..4u64 {
        match ureq::get(url).set("User-Agent", UA).timeout(Duration::from_secs(30)).call() {
            Ok(r) => {
                let mut out = Vec::new();
                match std::io::Read::read_to_end(&mut r.into_reader(), &mut out) {
                    Ok(_) => return Ok(out),
                    Err(e) => last = e.to_string(),
                }
            }
            Err(ureq::Error::Status(404, _)) => return Ok(Vec::new()),
            Err(ureq::Error::Status(429, _)) => {
                last = "429".into();
                std::thread::sleep(Duration::from_secs(60)); // rate limited, e.g. another client on this IP
            }
            Err(e) => last = e.to_string(),
        }
        std::thread::sleep(Duration::from_secs(5 * (attempt + 1)));
    }
    Err(last)
}

pub fn state_db() -> rusqlite::Result<Connection> {
    let dir = web_dir();
    let _ = std::fs::create_dir_all(dir.join("osu"));
    let db = Connection::open(dir.join("state.db"))?;
    db.busy_timeout(Duration::from_secs(5))?;
    db.execute_batch("CREATE TABLE IF NOT EXISTS diffs(bid INTEGER PRIMARY KEY, set_id INTEGER, status INTEGER, keys INTEGER,
          md5 TEXT, notes INTEGER, length REAL, playcount INTEGER, updated TEXT, artist TEXT, title TEXT,
          version TEXT, creator TEXT, sha TEXT, file_md5 TEXT, fetched TEXT, error TEXT, stars REAL);
        CREATE TABLE IF NOT EXISTS feats(sha TEXT, calc TEXT, data BLOB, PRIMARY KEY(sha, calc));
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);")?;
    let has_stars: bool = db.prepare("PRAGMA table_info(diffs)")?.query_map([], |r| r.get::<_, String>(1))?.flatten().any(|n| n == "stars");
    if !has_stars {
        db.execute("ALTER TABLE diffs ADD COLUMN stars REAL", [])?; // nomod star rating from the listing
    }
    Ok(db)
}

pub fn kv_get(db: &Connection, k: &str) -> Option<Value> {
    db.query_row("SELECT v FROM kv WHERE k=?", [k], |r| r.get::<_, String>(0)).ok().and_then(|v| serde_json::from_str(&v).ok())
}

pub fn kv_set(db: &Connection, k: &str, v: &Value) -> rusqlite::Result<()> {
    db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", params![k, v.to_string()]).map(|_| ())
}

fn check_stop() -> Result<(), CrawlError> {
    if web_dir().join("stop").exists() { Err(CrawlError::Stopped) } else { Ok(()) }
}

/// The OS releases it when the crawl exits however it ends; None while another crawl holds it.
pub fn lock() -> Option<Connection> {
    let lock = Connection::open(web_dir().join("crawl.lock")).ok()?;
    lock.busy_timeout(Duration::ZERO).ok()?;
    lock.execute_batch("BEGIN EXCLUSIVE").ok().map(|()| lock)
}

pub fn running() -> bool {
    lock().is_none()
}

fn progress(db: &Connection, fields: Value) -> rusqlite::Result<()> {
    let mut p = kv_get(db, "progress").unwrap_or_else(|| json!({}));
    p["t"] = json!(now());
    if let (Some(o), Some(f)) = (p.as_object_mut(), fields.as_object()) {
        o.extend(f.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    kv_set(db, "progress", &p)
}

/// Time left for these counts at this PC's measured pace (defaults before its first run).
pub fn seconds(db: &Connection, fetch: i64, analyse: i64) -> f64 {
    let number = |k: &str| kv_get(db, k).and_then(|v| v.as_f64()).filter(|&x| x != 0.);
    fetch as f64 * number("sec_fetch").unwrap_or(SEC_FETCH) + analyse as f64 * number("sec_analyse").unwrap_or(SEC_ANALYSE / workers(None) as f64)
}

pub fn disk_bytes() -> u64 {
    let size = |d: PathBuf| std::fs::read_dir(d).into_iter().flatten().flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum::<u64>();
    size(web_dir()) + size(web_dir().join("osu"))
}

/// New ranked maps for the PP list (user 2026-11-13): ranked or approved, the PP tab's keymodes, from the ★ where a perfect play
/// enters the player's top 100. Kept beside the window's own choice: one crawl's publish never drops the other's maps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PpAuto {
    pub keys: Vec<i64>,
    pub floor: f64,
}

fn pp_auto(db: &Connection) -> Option<PpAuto> {
    kv_get(db, "pp_auto").and_then(|v| serde_json::from_value(v).ok())
}

fn pp_wanted(auto: Option<&PpAuto>, status: Option<i64>, keys: i64, stars: Option<f64>) -> bool {
    auto.is_some_and(|a| crate::store::RANKED_STATUS.contains(&status.unwrap_or(-2)) && a.keys.contains(&keys) && stars.is_some_and(|s| s >= a.floor))
}

/// The metadata density can reach the NPS range at 1.5× (the NPS ranking's density floor ≈ 20·k/7).
fn reachable(notes: i64, length: f64, keys: i64) -> bool {
    notes >= 100 && length >= 30. && notes as f64 / length * 1.5 >= 0.6 * 20. * keys as f64 / 7.
}

/// Keymodes whose listing of `status` is complete. `keys` (no status) is from crawls that listed every status at once.
fn listed_keys(db: &Connection, status: i64) -> rusqlite::Result<HashSet<i64>> {
    let saved = |k: &str| kv_get(db, k).and_then(|v| v.as_array().cloned()).map(|a| a.iter().filter_map(Value::as_i64).collect::<HashSet<i64>>());
    match (saved(&format!("keys{status}")), saved("keys")) {
        // Crawls before either listed every status together: their rows' keymodes.
        (None, None) => db.prepare("SELECT DISTINCT keys FROM diffs")?.query_map([], |r| r.get(0))?.collect(),
        (own, all) => Ok(own.unwrap_or_default().union(&all.unwrap_or_default()).copied().collect()),
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Estimate {
    pub fetch: i64,
    pub analyse: i64,
    pub ready: i64,
    pub new_bytes: i64,
    pub seconds: f64,
    pub disk: u64,
    pub checked: Option<f64>,
    pub unlisted: Vec<i64>,
    pub per_key: BTreeMap<i64, i64>, // charts to download per keymode
}

/// What lazer already holds, by revision and by map. The website index only adds what isn't in the client (user 2026-11-02:
/// "weren't website maps supposed to only download the maps that aren't in the client"); 35 % of a real crawl was copies.
#[derive(Default)]
struct InClient {
    md5: HashSet<String>,
    bid: HashSet<i64>,
    sources: HashSet<String>, // website charts converts are remade from: `converts::bytes` reads their file
}

impl InClient {
    fn read() -> Self {
        let read = || -> rusqlite::Result<Self> {
            let db = Connection::open_with_flags(paths::rec_dir().join("rec.db"), rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            db.busy_timeout(Duration::from_secs(5))?;
            let mut out = Self::default();
            for row in db.prepare("SELECT md5,beatmap_id FROM installed")?.query_map([], |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<i64>>(1)?)))? {
                let (md5, bid) = row?;
                out.md5.extend(md5);
                out.bid.extend(bid.filter(|b| *b > 0));
            }
            out.sources = db.prepare("SELECT DISTINCT source_sha FROM converts WHERE web!=0")?.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
            Ok(out)
        };
        read().unwrap_or_default() // no rec.db yet: nothing held
    }

    fn holds(&self, bid: i64, md5: &str) -> bool {
        self.md5.contains(md5) || self.bid.contains(&bid)
    }
}

/// Drops what earlier crawls kept of charts now in the client, so they stop being carried and published. Comes back if the map
/// leaves lazer: the next crawl finds it missing.
pub fn forget_in_client() -> Result<usize, String> {
    if !web_dir().join("state.db").is_file() {
        return Ok(0);
    }
    let db = state_db().map_err(|e| e.to_string())?;
    forget(&db, &InClient::read(), &web_dir().join("osu")).map_err(|e| e.to_string())
}

fn forget(db: &Connection, client: &InClient, osu: &Path) -> rusqlite::Result<usize> {
    let held: Vec<(i64, String)> = db.prepare("SELECT bid,md5,sha FROM diffs WHERE sha IS NOT NULL")?
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default(), r.get::<_, String>(2)?)))?
        .filter_map(Result::ok).filter(|r| client.holds(r.0, &r.1)).map(|r| (r.0, r.2)).collect();
    let tx = db.unchecked_transaction()?;
    for (bid, sha) in &held {
        tx.execute("UPDATE diffs SET sha=NULL, fetched=NULL WHERE bid=?", [bid])?;
        tx.execute("DELETE FROM feats WHERE sha=?", [sha])?;
    }
    tx.commit()?;
    for (_, sha) in held.iter().filter(|h| !client.sources.contains(&h.1)) {
        let _ = std::fs::remove_file(osu.join(format!("{sha}.osu")));
    }
    Ok(held.len())
}

/// What a crawl of the keymodes that download still has to do.
pub fn estimate(db: &Connection, wants: &[Want], calc: &str) -> rusqlite::Result<Estimate> {
    let (mut fetch, mut analyse, mut ready, mut per_key) = (0, 0, 0, BTreeMap::new());
    let mut q = db.prepare("SELECT d.keys,d.notes,d.length,d.stars,d.sha,d.fetched,d.md5,f.sha IS NOT NULL,d.bid,d.status,d.playcount FROM diffs d LEFT JOIN feats f ON f.sha=d.sha AND f.calc=? WHERE d.error IS NULL")?;
    let rows = q.query_map([calc], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, Option<f64>>(3)?, r.get::<_, Option<String>>(4)?,
                                           r.get::<_, Option<String>>(5)?, r.get::<_, Option<String>>(6)?, r.get::<_, bool>(7)?, r.get::<_, i64>(8)?, r.get::<_, Option<i64>>(9)?,
                                           r.get::<_, Option<i64>>(10)?)))?;
    let (client, auto) = (InClient::read(), pp_auto(db));
    for row in rows {
        let (k, notes, length, s, sha, fetched, md5, done, bid, status, plays) = row?;
        if !(wanted(wants, true, k, status, notes, length, s, plays.unwrap_or(0)) || pp_wanted(auto.as_ref(), status, k, s)) || client.holds(bid, md5.as_deref().unwrap_or("")) {
            continue;
        }
        if sha.is_none() || fetched != md5 {
            fetch += 1;
            *per_key.entry(k).or_insert(0) += 1;
        } else if !done {
            analyse += 1;
        } else {
            ready += 1;
        }
    }
    let mut unlisted = Vec::new();
    for w in wants.iter().filter(|w| w.on) {
        for &status in &w.statuses {
            if !listed_keys(db, status)?.contains(&w.keys) {
                unlisted.push(w.keys);
            }
        }
    }
    unlisted.sort_unstable();
    unlisted.dedup();
    Ok(Estimate { fetch, analyse: fetch + analyse, ready, new_bytes: fetch * CHART_BYTES, seconds: seconds(db, fetch, fetch + analyse), disk: disk_bytes(),
                  checked: kv_get(db, "checked").and_then(|v| v.as_f64()), unlisted, per_key })
}

/// Newest-first per status; an incremental crawl stops a day past the last watermark.
/// Where a listing for these keymodes stopped: (status index, offset, sets, newest). A pause or a closed app lost a full pass
/// before this, so a keymode that needs one never got listed (user 2026-11-01: "it always checks the whole listing").
fn listing_at(db: &Connection, keys: &HashSet<i64>, order: &[i64]) -> Option<(usize, i64, i64, String)> {
    let mut wanted: Vec<i64> = keys.iter().copied().collect();
    wanted.sort_unstable();
    let l = kv_get(db, "listing").filter(|l| l["keys"] == json!(wanted) && l["statuses"] == json!(order))?;
    Some((l["status"].as_u64()? as usize, l["offset"].as_i64()?, l["sets"].as_i64()?, l["newest"].as_str()?.to_string()))
}

fn save_listing(db: &Connection, keys: &HashSet<i64>, order: &[i64], at: (usize, i64, i64, &str)) -> rusqlite::Result<()> {
    let mut wanted: Vec<i64> = keys.iter().copied().collect();
    wanted.sort_unstable();
    kv_set(db, "listing", &json!({"keys": wanted, "statuses": order, "status": at.0, "offset": at.1, "sets": at.2, "newest": at.3}))
}

fn crawl_meta(db: &Connection, keys: &HashSet<i64>, statuses: &[i64], full: bool) -> Result<(), CrawlError> {
    let order: Vec<i64> = STATUSES.into_iter().filter(|s| statuses.contains(s)).collect();
    // Rows crawled before the stars column need one full pass to learn their rating.
    let backfill = kv_get(db, "stars_backfilled").is_none_or(|v| !v.as_bool().unwrap_or(false));
    let stopped = listing_at(db, keys, &order);
    let whole = |status: i64| -> rusqlite::Result<bool> { Ok(full || backfill || stopped.is_some() || !keys.is_subset(&listed_keys(db, status)?)) };
    let mut sets = stopped.as_ref().map_or(0, |s| s.2 as usize);
    let any_whole = order.iter().map(|&s| whole(s)).collect::<rusqlite::Result<Vec<bool>>>()?.contains(&true);
    progress(db, json!({"phase": "check", "sets": sets, "full": any_whole}))?;
    for (index, &status) in order.iter().enumerate() {
        if stopped.as_ref().is_some_and(|s| s.0 > index) {
            continue;
        }
        let full = whole(status)?;
        let mark: String = db.query_row("SELECT v FROM kv WHERE k=?", [format!("mark{status}")], |r| r.get(0)).optional()?.unwrap_or_default();
        let stop = if full || mark.is_empty() { None } else {
            chrono::DateTime::parse_from_rfc3339(&mark.replace('Z', "+00:00")).ok().map(|t| (t - chrono::Duration::days(1)).format("%Y-%m-%dT%H:%M:%SZ").to_string())
        };
        let (mut newest, mut offset) = match stopped.as_ref().filter(|s| s.0 == index) {
            Some(s) => (mark.max(s.3.clone()), s.1),
            None => (mark, 0),
        };
        loop {
            let urls: Vec<String> = (0..PAGES).map(|i| format!("{SEARCH}&status={status}&offset={}", offset + 100 * i)).collect();
            let batch: Vec<Result<Vec<u8>, String>> = std::thread::scope(|s| {
                let handles: Vec<_> = urls.iter().map(|u| s.spawn(move || get(u))).collect();
                handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("page thread panicked".into()))).collect()
            });
            std::thread::sleep(Duration::from_secs_f64(PAUSE));
            let mut pages: Vec<Vec<Value>> = Vec::new();
            for page in batch {
                let raw = page.map_err(CrawlError::Failed)?;
                if let Ok(Value::Array(p)) = serde_json::from_slice::<Value>(&raw) {
                    if !p.is_empty() {
                        pages.push(p);
                    }
                }
            }
            sets += pages.iter().map(Vec::len).sum::<usize>();
            progress(db, json!({"phase": "check", "sets": sets}))?;
            check_stop()?;
            let done = (pages.len() as i64) < PAGES || pages.iter().any(|p| p.len() < 100);
            if pages.is_empty() {
                break;
            }
            let tx = db.unchecked_transaction()?;
            for s in pages.iter().flatten() {
                for b in s["beatmaps"].as_array().map_or(&[][..], Vec::as_slice) {
                    let cs = b["cs"].as_f64().unwrap_or(0.) as i64;
                    if b["mode_int"].as_i64() != Some(3) || b["convert"].as_bool().unwrap_or(false) || !keys.contains(&cs) {
                        continue;
                    }
                    let notes = b["count_circles"].as_i64().unwrap_or(0) + b["count_sliders"].as_i64().unwrap_or(0);
                    tx.execute("INSERT INTO diffs(bid,set_id,status,keys,md5,notes,length,playcount,updated,artist,title,version,creator,stars) \
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(bid) DO UPDATE SET status=excluded.status, md5=excluded.md5, \
                                notes=excluded.notes, length=excluded.length, playcount=excluded.playcount, updated=excluded.updated, stars=excluded.stars, \
                                error=CASE WHEN diffs.md5 IS excluded.md5 THEN diffs.error END",
                               params![b["id"].as_i64(), s["id"].as_i64(), b["ranked"].as_i64().or_else(|| s["ranked"].as_i64()), cs, b["checksum"].as_str(), notes,
                                       b["hit_length"].as_f64().unwrap_or(0.), b["playcount"].as_i64().unwrap_or(0),
                                       b["last_updated"].as_str().or_else(|| s["last_updated"].as_str()), s["artist"].as_str(), s["title"].as_str(), b["version"].as_str(),
                                       s["creator"].as_str(), b["difficulty_rating"].as_f64()])?;
                }
            }
            tx.commit()?;
            newest = newest.max(pages[0][0]["last_updated"].as_str().unwrap_or("").to_string());
            let oldest = pages.last().and_then(|p| p.last()).and_then(|x| x["last_updated"].as_str()).unwrap_or("");
            if done || stop.as_deref().is_some_and(|s| oldest < s) {
                break;
            }
            offset += 100 * PAGES;
            if full {
                save_listing(db, keys, &order, (index, offset, sets as i64, &newest))?;
            }
        }
        db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", params![format!("mark{status}"), newest])?;
        if full {
            save_listing(db, keys, &order, (index + 1, 0, sets as i64, ""))?;
        }
        let mut done: Vec<i64> = keys.union(&listed_keys(db, status)?).copied().collect();
        done.sort_unstable();
        kv_set(db, &format!("keys{status}"), &json!(done))?;
    }
    db.execute("DELETE FROM kv WHERE k='listing'", [])?;
    if backfill {
        kv_set(db, "stars_backfilled", &json!(true))?;
    }
    kv_set(db, "checked", &json!(now()))?;
    Ok(())
}

/// Chart text once per revision, most-played first; empty body = unavailable.
fn fetch(db: &Connection, wants: &[Want], limit: usize) -> Result<usize, CrawlError> {
    let (client, auto) = (InClient::read(), pp_auto(db));
    let mut q = db.prepare("SELECT bid,md5,keys,notes,length,stars,status,playcount FROM diffs WHERE error IS NULL AND (sha IS NULL OR fetched IS NOT md5) ORDER BY playcount DESC")?;
    let rows: Vec<(i64, String)> = q.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default(), r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, f64>(4)?, r.get::<_, Option<f64>>(5)?, r.get::<_, Option<i64>>(6)?, r.get::<_, Option<i64>>(7)?)))?
        .filter_map(Result::ok).filter(|r| (wanted(wants, true, r.2, r.6, r.3, r.4, r.5, r.7.unwrap_or(0)) || pp_wanted(auto.as_ref(), r.6, r.2, r.5)) && !client.holds(r.0, &r.1))
        .take(limit).map(|r| (r.0, r.1)).collect();
    let (t0, done) = (std::time::Instant::now(), kv_get(db, "progress").and_then(|p| p["fetched"].as_u64()).unwrap_or(0) as usize);
    let total = rows.len();
    let (send, receive) = std::sync::mpsc::sync_channel::<(i64, Vec<u8>)>(1); // one ahead of the store loop, ~60/min
    std::thread::scope(|s| -> Result<(), CrawlError> {
        let rows = &rows;
        s.spawn(move || {
            for (bid, md5) in rows {
                // Dump copies for ranked and loved charts: no download for what is already here.
                let mut data = crate::paths::frozen_chart(*bid, Some(md5)).and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
                for (url, pause) in FILES {
                    if !data.is_empty() {
                        break;
                    }
                    data = get(&format!("{url}{bid}")).unwrap_or_default();
                    std::thread::sleep(Duration::from_secs_f64(pause));
                }
                if send.send((*bid, data)).is_err() {
                    return; // a pause drops the queued downloads, not just stops reading them
                }
            }
        });
        for (i, (bid, data)) in receive.iter().enumerate() {
            store(db, bid, &data)?;
            progress(db, json!({"phase": "download", "fetched": done + i + 1}))?;
            if i >= 20 && i % 20 == 0 {
                kv_set(db, "sec_fetch", &json!(t0.elapsed().as_secs_f64() / (i + 1) as f64))?;
            }
            check_stop()?;
        }
        Ok(())
    })?;
    Ok(total)
}

fn store(db: &Connection, bid: i64, data: &[u8]) -> Result<(), CrawlError> {
    if data.iter().all(u8::is_ascii_whitespace) {
        db.execute("UPDATE diffs SET error='unavailable', fetched=md5 WHERE bid=?", [bid])?;
        return Ok(());
    }
    let sha: String = Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(web_dir().join("osu").join(format!("{sha}.osu")), data)?;
    db.execute("UPDATE diffs SET sha=?, file_md5=?, fetched=md5 WHERE bid=?", params![sha, format!("{:x}", md5::compute(data)), bid])?;
    Ok(())
}

/// A state.db feats cell: {rate: features} written by this program or the Python crawler; Err = its analysis failed.
fn read_cell(data: &[u8]) -> Result<Vec<(f64, Dict)>, String> {
    let rates = |items: Vec<(f64, Val)>| -> Vec<(f64, Dict)> {
        items.into_iter().filter_map(|(r, v)| match v { Val::Dict(d) => Some((r, d)), _ => None }).collect()
    };
    if data.first() == Some(&0x80) {
        let p = pickle::load(data)?;
        if p.get("error").is_some() {
            return Err("analysis failed".into());
        }
        return Ok(rates(p.items().iter().filter_map(|(r, f)| Some((r.f()?, value(f)))).collect()));
    }
    let Val::Dict(d) = codec::decode(data)? else { return Err("bad cell".into()) };
    if d.has("error") {
        return Err("analysis failed".into());
    }
    Ok(rates(d.0.into_iter().filter_map(|(k, v)| match k {
        ms_calc::val::Key::S(s) => Some((s.parse().ok()?, v)),
        ms_calc::val::Key::F(r) => Some((r, v)),
        ms_calc::val::Key::B(_) => None,
    }).collect()))
}

fn write_cell(fs: &[(f64, Dict)]) -> Vec<u8> {
    let mut d = Dict::new();
    fs.iter().for_each(|(r, f)| d.setf(*r, Val::Dict(f.clone())));
    codec::encode(&Val::Dict(d))
}

fn analyse(db: &Connection, calc: &str, prm: &Params, threads: Option<usize>) -> Result<(), CrawlError> {
    let todo: Vec<String> = db.prepare("SELECT DISTINCT sha FROM diffs WHERE sha IS NOT NULL AND sha NOT IN (SELECT sha FROM feats WHERE calc=?)")?
        .query_map([calc], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
    let (t0, done) = (std::time::Instant::now(), kv_get(db, "progress").and_then(|p| p["analysed"].as_u64()).unwrap_or(0) as usize);
    for (n, chunk) in todo.chunks(64).enumerate() { // a pause stops between batches
        let jobs: Vec<(Vec<u8>, Vec<f64>)> = chunk.iter().map(|sha| (std::fs::read(web_dir().join("osu").join(format!("{sha}.osu"))).unwrap_or_default(), RATES.to_vec())).collect();
        let results = features_batch(&jobs, workers(threads), prm);
        let tx = db.unchecked_transaction()?;
        for (sha, fs) in chunk.iter().zip(results) {
            let cell = match fs {
                Ok(fs) => write_cell(&fs),
                Err(e) => {
                    let mut d = Dict::new();
                    d.set("error", e);
                    codec::encode(&Val::Dict(d))
                }
            };
            tx.execute("INSERT OR REPLACE INTO feats VALUES(?,?,?)", params![sha, calc, cell])?;
        }
        tx.commit()?;
        let i = n * 64 + chunk.len();
        progress(db, json!({"phase": "analyse", "analysed": done + i}))?;
        kv_set(db, "sec_analyse", &json!(t0.elapsed().as_secs_f64() / i as f64))?;
        check_stop()?;
    }
    Ok(())
}

// What NPS/Skills predictions, eligibility and practice matching read from a website chart. Full dicts cost ~90 KB per
// chart in memory.
const NPS_FIELDS: [&str; 13] = ["active", "burst", "chord", "drill", "family", "nps", "overlap", "play_span", "repeat", "rhythm", "version", "vibro_focused", "wide"];

fn compact(f: &Dict) -> Dict {
    let mut out = Dict::new();
    for k in ["keys", "overall", "od", "ln", "notes", "length", "stam", "description", "sk", "stars", "hits"] {
        out.set(k, f.get(k).cloned().unwrap_or(Val::None));
    }
    let mut endurance = Dict::new();
    endurance.set("load", f.dict("endurance").map_or(0., |e| e.fget("load", 0.)));
    out.set("endurance", endurance);
    let mut execution = Dict::new();
    execution.set("residual_vector", f.dict("execution").and_then(|e| e.get("residual_vector")).cloned().unwrap_or(Val::None));
    out.set("execution", execution);
    out.set("calibration_features", f.get("calibration_features").cloned().unwrap_or(Val::List(Vec::new())));
    let mut nps = Dict::new();
    if let Some(st) = f.dict("nps") {
        for k in NPS_FIELDS {
            if let Some(v) = st.get(k) {
                nps.set(k, v.clone());
            }
        }
    }
    out.set("nps", nps);
    out.set("tech_profile", f.get("tech_profile").cloned().unwrap_or_else(|| Val::Dict(Dict::new())));
    out
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WebMap {
    pub sha: String,
    pub bid: i64,
    pub set_id: i64,
    pub status: Option<i64>,
    pub keys: i64,
    pub file_md5: String,
    pub playcount: i64,
    pub artist: String,
    pub title: String,
    pub version: String,
    pub creator: String,
    pub overall: Vec<(f64, f64)>,
}

/// Stored website features moved to the calculator `to`, web.msp with them: a calculator change would else crawl everything again.
pub fn carry(from: &str, to: &str, after: &Params, chart: &(dyn Fn(&str) -> Option<ms_calc::chart::Chart> + Sync)) -> Result<(), String> {
    let text = |e: rusqlite::Error| e.to_string();
    if !web_dir().join("state.db").is_file() {
        return Ok(());
    }
    let db = state_db().map_err(text)?;
    let todo: Vec<String> = db.prepare("SELECT sha FROM feats WHERE calc=? AND sha NOT IN (SELECT sha FROM feats WHERE calc=?)").map_err(text)?
        .query_map([from, to], |r| r.get(0)).map_err(text)?.collect::<rusqlite::Result<_>>().map_err(text)?;
    let threads = (std::thread::available_parallelism().map_or(2, |x| x.get()) / 2).max(1);
    crate::store::CARRIED[1].fetch_add(todo.len(), std::sync::atomic::Ordering::Relaxed);
    for chunk in todo.chunks(500) {
        crate::store::CARRIED[0].fetch_add(chunk.len(), std::sync::atomic::Ordering::Relaxed);
        let mut stored: Vec<(&String, Vec<u8>)> = Vec::with_capacity(chunk.len());
        for sha in chunk {
            stored.push((sha, db.query_row("SELECT data FROM feats WHERE sha=? AND calc=?", [sha.as_str(), from], |r| r.get(0)).map_err(text)?));
        }
        let cells = ms_calc::background::background_map(&stored, threads, |(sha, data)| {
            let file = std::cell::OnceCell::new();
            let chart = || file.get_or_init(|| chart(sha).map(std::rc::Rc::new)).clone();
            let fs = read_cell(data).unwrap_or_default();
            let moved: Vec<Option<Dict>> = fs.iter().map(|(r, f)| ms_calc::features::carried(f, after, *r, &chart)).collect();
            Ok(if moved.iter().any(Option::is_some) {
                write_cell(&fs.iter().zip(moved).map(|((r, f), m)| (*r, m.unwrap_or_else(|| f.clone()))).collect::<Vec<_>>())
            } else {
                data.clone()
            })
        });
        let tx = db.unchecked_transaction().map_err(text)?;
        for ((sha, _), cell) in stored.iter().zip(cells) {
            tx.execute("INSERT OR REPLACE INTO feats VALUES(?,?,?)", params![sha, to, cell?]).map_err(text)?;
        }
        tx.commit().map_err(text)?;
    }
    // another version's web.msp (user 2026-11-12: back on 1.9.0 after 1.10.1, no website charts): rebuilt for this calculator
    // even when nothing needed carrying
    let stale = published_calc().is_some_and(|c| c != to);
    if !todo.is_empty() || stale {
        publish(&db, to, &wants(&db)).map_err(|e| format!("{e:?}"))?;
    }
    Ok(())
}

/// The calculator the website list (web.msp) was published for.
pub fn published_calc() -> Option<String> {
    Web::open(&web_dir().join("web.msp")).ok().map(|w| w.calc)
}

#[derive(Serialize, Deserialize)]
struct WebMeta {
    calc: String,
    built: f64,
    maps: Vec<WebMap>,
}

/// Every keymode's want, on or off: turning a keymode's downloads off keeps its maps in the playlists.
fn publish(db: &Connection, calc: &str, wants: &[Want]) -> Result<(), CrawlError> {
    let mut q = db.prepare("SELECT d.sha,d.bid,d.set_id,d.status,d.keys,d.file_md5,d.playcount,d.artist,d.title,d.version,d.creator,d.stars,f.data,d.notes,d.length \
                            FROM diffs d JOIN feats f ON f.sha=d.sha AND f.calc=? WHERE d.error IS NULL")?;
    let mut rows = q.query([calc])?;
    let text = |r: &rusqlite::Row<'_>, i: usize| r.get::<_, Option<String>>(i).map(Option::unwrap_or_default);
    let (mut maps, mut records, mut seen, auto) = (Vec::new(), Vec::new(), HashSet::new(), pp_auto(db));
    while let Some(r) = rows.next()? {
        let sha = text(r, 0)?;
        let plays = r.get::<_, Option<i64>>(6)?.unwrap_or(0);
        let wanted = wanted(wants, false, r.get(4)?, r.get(3)?, r.get::<_, Option<i64>>(13)?.unwrap_or(0), r.get::<_, Option<f64>>(14)?.unwrap_or(0.), r.get(11)?, plays)
            || pp_wanted(auto.as_ref(), r.get(3)?, r.get(4)?, r.get(11)?);
        if !wanted || !seen.insert(sha.clone()) {
            continue;
        }
        let Ok(fs) = read_cell(&r.get::<_, Vec<u8>>(12)?) else { continue };
        let at = maps.len() as i64;
        maps.push(WebMap { sha, bid: r.get(1)?, set_id: r.get::<_, Option<i64>>(2)?.unwrap_or(0), status: r.get(3)?, keys: r.get(4)?, file_md5: text(r, 5)?,
                           playcount: r.get::<_, Option<i64>>(6)?.unwrap_or(0), artist: text(r, 7)?, title: text(r, 8)?, version: text(r, 9)?, creator: text(r, 10)?,
                           overall: fs.iter().map(|(rate, f)| (*rate, f.f("overall"))).collect() });
        let mut sorted: Vec<_> = fs.iter().map(|(rate, f)| (at, *rate, codec::encode(&Val::Dict(compact(f))))).collect();
        sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
        records.extend(sorted);
    }
    write_file(&web_dir().join("web.msp"), &WebMeta { calc: calc.to_string(), built: now(), maps }, &records)?;
    progress(db, json!({"ready": seen.len()}))?;
    Ok(())
}

fn write_file(path: &Path, meta: &WebMeta, records: &[(i64, f64, Vec<u8>)]) -> Result<(), String> {
    blobs::write(path, MAGIC, &serde_json::to_vec(meta).map_err(|e| e.to_string())?, records)
}

/// web.pkl of a Python install → web.msp, so an existing crawl keeps working.
pub fn convert_pickle(from: &Path, to: &Path) -> Result<(), String> {
    let root = pickle::load(&std::fs::read(from).map_err(|e| e.to_string())?)?;
    if root.get("format").and_then(P::i) != Some(3) {
        return Err("web.pkl has an old format".into());
    }
    let feats: HashMap<&str, &P> = root.get("feats").map_or(&[][..], P::items).iter().filter_map(|(k, v)| Some((k.s()?, v))).collect();
    let (mut maps, mut records) = (Vec::new(), Vec::new());
    for (sha, m) in root.get("maps").map_or(&[][..], P::items) {
        let Some(sha) = sha.s() else { continue };
        let num = |k: &str| m.get(k).and_then(P::i);
        let text = |k: &str| m.get(k).and_then(P::s).unwrap_or("").to_string();
        let at = maps.len() as i64;
        let mut rates: Vec<(f64, Vec<u8>)> = Vec::new();
        for (rate, blob) in feats.get(sha).map_or(&[][..], |f| f.items()) {
            if let (Some(rate), P::Bytes(b)) = (rate.f(), blob) {
                rates.push((rate, codec::encode(&value(&pickle::load(b)?))));
            }
        }
        rates.sort_by(|a, b| a.0.total_cmp(&b.0));
        records.extend(rates.into_iter().map(|(rate, blob)| (at, rate, blob)));
        maps.push(WebMap { sha: sha.to_string(), bid: num("bid").unwrap_or(0), set_id: num("set_id").unwrap_or(0), status: num("status"), keys: num("keys").unwrap_or(0),
                           file_md5: text("file_md5"), playcount: num("playcount").unwrap_or(0), artist: text("artist"), title: text("title"), version: text("version"),
                           creator: text("creator"), overall: m.get("overall").map_or(&[][..], P::items).iter().filter_map(|(r, o)| Some((r.f()?, o.f()?))).collect() });
    }
    let meta = WebMeta { calc: root.get("calc").and_then(P::s).unwrap_or("").to_string(), built: root.get("built").and_then(P::f).unwrap_or(0.), maps };
    write_file(to, &meta, &records)
}

pub struct Web {
    pub maps: Vec<WebMap>,
    pub calc: String,
    blobs: Blobs,
}

impl Web {
    pub fn open(path: &Path) -> Result<Web, String> {
        let (blobs, header) = Blobs::open(path, MAGIC)?;
        let meta: WebMeta = serde_json::from_slice(blobs.header(header)).map_err(|e| e.to_string())?;
        Ok(Web { maps: meta.maps, calc: meta.calc, blobs })
    }

    pub fn feature(&self, map: usize, rate: f64) -> Option<Val> {
        self.blobs.feature(map as i64, rate)
    }

    /// The rates to analyse a website chart from: those whose rating can land near the player's target. Two nearest
    /// the window's centre anchor the exact-rate refinement: all in-window rates were ~190k predictions after every
    /// score and the GUI lagged for minutes (G835LX 2026-10-02). None = outside every window.
    pub fn window_rates(&self, map: usize, reach: Option<(f64, f64)>) -> Option<Vec<f64>> {
        window_rates(&self.maps[map].overall, reach)
    }
}

fn window_rates(ov: &[(f64, f64)], reach: Option<(f64, f64)>) -> Option<Vec<f64>> {
    let (lo, hi) = reach.unwrap_or((0., f64::INFINITY));
    let mut rates: Vec<(f64, f64)> = ov.iter().copied().filter(|x| lo <= x.1 && x.1 <= hi).collect();
    if rates.is_empty() {
        // The window falls between two rated rates: both anchor an exact refinement.
        let below = ov.iter().copied().filter(|x| x.1 < lo).max_by(|a, b| a.1.total_cmp(&b.1));
        let above = ov.iter().copied().filter(|x| x.1 > hi).min_by(|a, b| a.1.total_cmp(&b.1));
        rates = vec![below?, above?];
    }
    if reach.is_some() && rates.len() > 2 {
        let mid = (lo * hi).ln() / 2.;
        rates.sort_by(|a, b| (a.1.ln() - mid).abs().total_cmp(&(b.1.ln() - mid).abs()));
        rates.truncate(2);
    }
    Some(rates.into_iter().map(|x| x.0).collect())
}

/// Everything a crawl does: listing, chart text, analysis, publication. `check` stops after the listing, so the app
/// can show counts, size and time before Start.
/// `wants`: the window's choice (saved; the app reopens with it). `pp`: the new-ranked-PP-maps run (`PpAuto`, `wants` unused)
/// instead: it fetches only its own maps, and every publish keeps the window's choice.
pub fn crawl(wants: &[Want], check: bool, full: bool, threads: Option<usize>, calc: &str, prm: &Params, pp: Option<PpAuto>) -> Result<(), CrawlError> {
    let db = state_db()?;
    let Some(_held) = lock() else { return Err(CrawlError::Failed("another crawl is running".into())) };
    let _ = std::fs::remove_file(web_dir().join("stop"));
    let (keys, list, fetching): (HashSet<i64>, Vec<i64>, Vec<Want>) = match &pp {
        Some(a) => {
            kv_set(&db, "pp_auto", &json!(a))?;
            (a.keys.iter().copied().collect(), crate::store::RANKED_STATUS.to_vec(), Vec::new())
        }
        None => {
            kv_set(&db, "wants", &json!(wants))?;
            kv_set(&db, "command", &json!(if check { "check" } else { "crawl" }))?;
            db.execute("DELETE FROM kv WHERE k='reopen'", [])?;
            let on: Vec<&Want> = wants.iter().filter(|w| w.on).collect();
            let mut list: Vec<i64> = on.iter().flat_map(|w| w.statuses.iter().copied()).collect();
            list.sort_unstable();
            list.dedup();
            (on.iter().map(|w| w.keys).collect(), list, wants.to_vec())
        }
    };
    let all = self::wants(&db); // saved above: the window's whole choice, also for a PP run
    let (keys, fetching, all) = (&keys, &fetching[..], &all[..]);
    let run = || -> Result<(), CrawlError> {
        crawl_meta(&db, keys, &list, full)?;
        forget(&db, &InClient::read(), &web_dir().join("osu"))?;
        if check {
            progress(&db, json!({"phase": "checked"}))?;
            return Ok(());
        }
        let plan = estimate(&db, fetching, calc)?;
        progress(&db, json!({"phase": "download", "fetched": 0, "fetch_total": plan.fetch, "analysed": 0, "analyse_total": plan.analyse, "error": null}))?;
        // Publish every chunk; small first chunks so a new player sees website maps within minutes, not after 5,000
        // downloads (fresh-install run 2026-10-02).
        for size in [300, 1500].into_iter().chain(std::iter::repeat(5000)) {
            let more = fetch(&db, fetching, size)?;
            analyse(&db, calc, prm, threads)?;
            publish(&db, calc, all)?;
            if more == 0 {
                break;
            }
        }
        progress(&db, json!({"phase": "done"}))?;
        Ok(())
    };
    match run() {
        Err(CrawlError::Stopped) => progress(&db, json!({"phase": "paused"})).map_err(Into::into),
        Err(CrawlError::Failed(e)) => {
            progress(&db, json!({"phase": "failed", "error": e}))?;
            Err(CrawlError::Failed(e))
        }
        Ok(()) => Ok(()),
    }
}

/// {key: value} of the crawl's saved state, for the app's window.
pub fn state(db: &Connection) -> BTreeMap<String, Value> {
    ["progress", "checked", "reopen", "command"].iter().filter_map(|k| Some((k.to_string(), kv_get(db, k)?))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pp_fetch_takes_ranked_maps_of_its_keymodes_from_the_floor() {
        let a = PpAuto { keys: vec![7], floor: 5. };
        assert!(pp_wanted(Some(&a), Some(1), 7, Some(5.2)) && pp_wanted(Some(&a), Some(2), 7, Some(5.)));
        assert!(!pp_wanted(Some(&a), Some(4), 7, Some(6.)), "loved pays no pp");
        assert!(!pp_wanted(Some(&a), Some(1), 4, Some(6.)) && !pp_wanted(Some(&a), Some(1), 7, Some(4.9)) && !pp_wanted(Some(&a), Some(1), 7, None));
        assert!(!pp_wanted(None, Some(1), 7, Some(9.)), "off: nothing");
    }

    #[test]
    fn a_chart_in_the_client_is_forgotten_and_a_convert_source_keeps_its_file() {
        let dir = std::env::temp_dir().join(format!("ms-web-forget-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE diffs(bid INTEGER PRIMARY KEY, md5 TEXT, sha TEXT, fetched TEXT); CREATE TABLE feats(sha TEXT, calc TEXT, data BLOB);
            INSERT INTO diffs VALUES (1,'a','s1','a'),(2,'b','s2','b'),(3,'c','s3','c'),(4,'d','s4','d');
            INSERT INTO feats VALUES ('s1','x',x''),('s2','x',x''),('s3','x',x''),('s4','x',x'');").unwrap();
        for sha in ["s1", "s2", "s3", "s4"] {
            std::fs::write(dir.join(format!("{sha}.osu")), "").unwrap();
        }
        let client = InClient { md5: ["a".to_string(), "d".to_string()].into(), bid: [2].into(), sources: ["s4".to_string()].into() };
        assert_eq!(forget(&db, &client, &dir).unwrap(), 3, "same revision, older revision of the map, convert source");
        let left: Vec<String> = db.prepare("SELECT sha FROM feats").unwrap().query_map([], |r| r.get(0)).unwrap().flatten().collect();
        assert_eq!(left, ["s3"]);
        let fetched: i64 = db.query_row("SELECT count(*) FROM diffs WHERE sha IS NOT NULL", [], |r| r.get(0)).unwrap();
        assert_eq!(fetched, 1);
        assert!(!dir.join("s1.osu").exists() && !dir.join("s2.osu").exists());
        assert!(dir.join("s3.osu").exists() && dir.join("s4.osu").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn state_carries_what_the_app_needs_to_resume_a_paused_batch() {
        let db = Connection::open_in_memory().unwrap();
        db.execute("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT)", []).unwrap();
        kv_set(&db, "reopen", &json!(true)).unwrap();
        kv_set(&db, "command", &json!("crawl")).unwrap();
        let s = state(&db);
        assert_eq!(s.get("reopen"), Some(&json!(true)));
        assert_eq!(s.get("command"), Some(&json!("crawl")));
    }

    #[test]
    fn rates_out_of_reach_are_never_loaded_and_two_anchor_a_window() {
        let ov = [(1., 6.), (1.2, 9.)];
        assert_eq!(window_rates(&ov, Some((5., 7.))), Some(vec![1.]));
        assert_eq!(window_rates(&ov, Some((10., 12.))), None);
        assert_eq!(window_rates(&ov, Some((7., 8.))), Some(vec![1., 1.2]), "a window between two rated rates keeps both");
        let many = [(0.8, 4.), (1., 6.), (1.2, 9.), (1.4, 11.)];
        let mut near = window_rates(&many, Some((3.5, 12.))).unwrap();
        near.sort_by(f64::total_cmp);
        assert_eq!(near, [1., 1.2], "only the two nearest the centre are predicted");
        assert_eq!(window_rates(&many, None).unwrap().len(), 4);
    }

    #[test]
    fn a_stopped_listing_continues_only_for_the_same_keymodes() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT);").unwrap();
        let (ten, seven): (HashSet<i64>, HashSet<i64>) = ([10].into(), [7, 10].into());
        let (ranked, all) = (statuses(&["ranked"]), statuses(&["ranked", "unranked"]));
        assert_eq!(listing_at(&db, &ten, &ranked), None);
        save_listing(&db, &ten, &ranked, (3, 4800, 52800, "2026-11-01T08:37:21Z")).unwrap();
        assert_eq!(listing_at(&db, &ten, &ranked), Some((3, 4800, 52800, "2026-11-01T08:37:21Z".to_string())));
        assert_eq!(listing_at(&db, &seven, &ranked), None, "other keymodes read the listing for themselves");
        assert_eq!(listing_at(&db, &ten, &all), None, "and other statuses");
    }

    #[test]
    fn the_estimate_counts_what_a_crawl_still_has_to_do() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT); CREATE TABLE feats(sha TEXT, calc TEXT);
                          CREATE TABLE diffs(bid INTEGER PRIMARY KEY, status INTEGER, keys INTEGER, md5 TEXT, notes INTEGER, length REAL, playcount INTEGER, stars REAL, sha TEXT, fetched TEXT, error TEXT);").unwrap();
        for bid in 1..5 {
            db.execute("INSERT INTO diffs(bid,status,keys,md5,notes,length,playcount,stars) VALUES(?,1,7,?,2000,120,?,?)", rusqlite::params![bid, format!("m{bid}"), bid, if bid < 4 { 8. } else { 3. }]).unwrap();
        }
        db.execute("INSERT INTO diffs(bid,status,keys,md5,notes,length,playcount,stars) VALUES(9,-2,7,'m9',2000,120,9,8)", []).unwrap();
        let want = |k: i64, s: &[&str]| Want { statuses: statuses(s), ..Want::new(k, (6., 12.)) };
        let estimate = |w: &[Want]| estimate(&db, w, "c").unwrap();
        assert_eq!(estimate(&[want(7, &["ranked"])]).unlisted, Vec::<i64>::new(), "older crawls: their rows' keymodes");
        assert_eq!(estimate(&[want(7, &["unranked"])]).fetch, 1, "a graveyard chart only when unranked is ticked");
        assert_eq!(estimate(&[want(7, &["ranked", "unranked"])]).fetch, 4);
        assert_eq!(estimate(&[Want { min_plays: 3, ..want(7, &["ranked", "unranked"]) }]).fetch, 2, "plays 3 and 9");
        assert_eq!(estimate(&[Want { stars: (2., 4.), ..want(7, &["ranked"]) }]).fetch, 1, "each keymode its own ★ range");
        assert_eq!(estimate(&[Want { on: false, ..want(7, &["ranked"]) }]).fetch, 0, "an off keymode downloads nothing");
        kv_set(&db, "keys", &json!([7])).unwrap();
        let e = estimate(&[want(4, &["ranked"]), want(7, &["ranked"])]);
        assert_eq!((e.fetch, e.analyse, e.ready, e.unlisted.clone()), (3, 3, 0, vec![4]));
        assert_eq!(e.per_key, BTreeMap::from([(7, 3)]));
        assert_eq!(e.new_bytes, 3 * CHART_BYTES);
        assert!(e.seconds > 0.);
        assert_eq!(estimate(&[want(4, &["ranked"])]).fetch, 0);
        db.execute("UPDATE diffs SET sha='s1', fetched='m1' WHERE bid=1", []).unwrap();
        assert_eq!(estimate(&[want(7, &["ranked"])]).fetch, 2, "a resume fetches only the rest");
        db.execute("INSERT INTO feats VALUES('s1','c')", []).unwrap();
        assert_eq!(estimate(&[want(7, &["ranked"])]).ready, 1);
    }

    #[test]
    fn old_crawls_become_one_want_per_listed_keymode() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE kv(k TEXT PRIMARY KEY, v TEXT); CREATE TABLE diffs(bid INTEGER PRIMARY KEY, keys INTEGER);
                          INSERT INTO diffs VALUES(1,4),(2,10);").unwrap();
        kv_set(&db, "want_keys", &json!([7])).unwrap();
        kv_set(&db, "stars", &json!([5., 7.])).unwrap();
        let w = wants(&db);
        assert_eq!(w.iter().map(|w| (w.keys, w.on)).collect::<Vec<_>>(), [(4, false), (7, true), (10, false)]);
        assert!(w.iter().all(|w| w.stars == (5., 7.) && w.statuses == STATUSES));
        kv_set(&db, "wants", &json!([Want::new(4, (1., 2.))])).unwrap();
        assert_eq!(wants(&db), [Want::new(4, (1., 2.))]);
    }
}

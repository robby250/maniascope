//! Keymode converts: which charts become which, and handing a convert to the client when its turn comes. Difficulty is
//! known ahead: a chart is converted in memory and analysed like any other, and nothing is written until its turn
//! (user 2026-11-01: "since converts are deterministic we should be able to know/calculate the difficulty ... beforehand").
use crate::catalog::{Inst, Pick};
use crate::engine::hashes;
use ms_calc::features::chart_features;
use ms_calc::params::Params;
use ms_calc::val::Dict;
use ms_convert::osz::{self, Entry};
use ms_data::paths;
use ms_data::store::{Db, Result, SUPPORTED_KEYS};
use ms_data::webmaps::{web_dir, Web};
use ms_model::predict::HashMap;
use rusqlite::params;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// catboy.best first: measured 2026-11-11 first byte 0.2 s and 10.8 MB/s vs osu.direct 1.2–2 s and 8.2 MB/s, and osu.direct
// answered 503 on a set catboy served. osu.direct only when catboy fails.
const MIRRORS: [&str; 2] = ["https://catboy.best/d/", "https://osu.direct/api/d/"];
const UA: &str = "maniascope/1.0 (personal osu!mania practice recommender)";
const CONFIRM: Duration = Duration::from_secs(10);

/// One target keymode's converts in one tab.
#[derive(Clone, PartialEq, Debug)]
pub struct Setting {
    pub on: bool,
    pub sources: Vec<i64>,
    pub often: f64,
    pub weighted: bool,
}

/// Within 3 keys: 10K ← 7, 8, 9; further ones convert worse.
pub fn default_sources(target: i64) -> Vec<i64> {
    SUPPORTED_KEYS.filter(|&k| k != target && (k - target).abs() <= 3).collect()
}

impl Setting {
    pub fn new(target: i64) -> Setting {
        Setting { on: false, sources: default_sources(target), often: 0.5, weighted: true }
    }
}

/// The enabled (target, source keymodes) among a tab's keymodes.
pub fn pairs(settings: &BTreeMap<i64, Setting>, keys: &[i64]) -> Vec<(i64, Vec<i64>)> {
    keys.iter().filter_map(|k| settings.get(k).filter(|s| s.on && !s.sources.is_empty()).map(|s| (*k, s.sources.clone()))).collect()
}

/// One chart to convert.
#[derive(Clone)]
pub struct Job {
    pub source: String,
    pub target: i64,
    pub keys: i64,
    pub path: PathBuf,
    pub artist: String,
    pub title: String,
    pub version: String,
    pub creator: Option<String>,
    pub set_id: Option<i64>,
    pub bid: Option<i64>,
    pub od: Option<f64>,
    pub length: Option<f64>,
    pub playcount: Option<i64>,
    pub web: bool,
}

fn known(db: &Db, target: i64) -> Result<HashSet<String>> {
    let mut q = db.prepare_cached("SELECT source_sha FROM converts WHERE target=? AND converter=?")?;
    let rows = q.query_map(params![target, ms_convert::CONVERTER_ID], |r| r.get(0))?;
    rows.collect()
}

/// What the offered rates (0.7–1.5×) can do to a 1× rating, with margin.
const RATE_REACH: (f64, f64) = (0.55, 2.);

/// A chart rated `overall` at 1× can fit `window` at some rate. Skipped charts stay unconverted, so they come in when reach moves.
pub(crate) fn in_window(overall: f64, window: (f64, f64)) -> bool {
    overall * RATE_REACH.1 >= window.0 && overall * RATE_REACH.0 <= window.1
}

/// Up to `size` charts not converted yet, your own first, only near the target: a convert is denser than its source, so the
/// window opens downwards. `rated`: a source's stored 1× rating; an unrated one converts anyway.
pub fn pending(db: &Db, library: &[Inst], web: Option<&Web>, installed_md5: &HashSet<String>, pairs: &[(i64, Vec<i64>)], reach: &HashMap<i64, (f64, f64)>, rated: &mut dyn FnMut(&str) -> Option<f64>, root: &Path, size: usize) -> Result<Vec<Job>> {
    let mut out = Vec::new();
    for (target, sources) in pairs {
        let done = known(db, *target)?;
        let window = reach.get(target).map(|&(lo, hi)| (lo / 1.5, hi));
        for i in library.iter().filter(|i| sources.contains(&i.keys) && !done.contains(&i.sha256)) {
            if window.zip(rated(&i.sha256)).is_some_and(|(w, overall)| !in_window(overall, w)) {
                continue;
            }
            let Some(path) = paths::local_file(&i.sha256, root) else { continue };
            out.push(Job { source: i.sha256.clone(), target: *target, keys: i.keys, path, artist: i.artist.clone(), title: i.title.clone(), version: i.version.clone(), creator: i.creator.clone(),
                           set_id: i.set_id, bid: i.beatmap_id, od: i.od, length: i.length, playcount: i.playcount, web: false });
            if out.len() == size {
                return Ok(out);
            }
        }
        let Some(web) = web else { continue };
        for (n, m) in web.maps.iter().enumerate() {
            if !sources.contains(&m.keys) || done.contains(&m.sha) || installed_md5.contains(&m.file_md5) || web.window_rates(n, window).is_none() {
                continue;
            }
            let path = web_dir().join("osu").join(format!("{}.osu", m.sha));
            if path.is_file() {
                out.push(Job { source: m.sha.clone(), target: *target, keys: m.keys, path, artist: m.artist.clone(), title: m.title.clone(), version: m.version.clone(), creator: Some(m.creator.clone()),
                               set_id: Some(m.set_id), bid: Some(m.bid), od: None, length: None, playcount: Some(m.playcount), web: true });
                if out.len() == size {
                    return Ok(out);
                }
            }
        }
    }
    Ok(out)
}

/// What a job gives: the output's identity and its features at 1×.
pub struct Indexed {
    pub sha: String,
    pub md5: String,
    pub version: String,
    pub features: Vec<(f64, Dict)>,
}

pub fn run(job: &Job, prm: &Params) -> std::result::Result<Indexed, String> {
    let data = std::fs::read(&job.path).map_err(|e| e.to_string())?;
    let out = ms_convert::convert(&data, job.target as usize).map_err(|e| e.to_string())?;
    let (sha, md5) = hashes(&out.osu);
    let features = chart_features(&out.osu, &[1.], prm)?;
    Ok(Indexed { sha, md5, version: out.version, features })
}

/// A failed chart gets a row too, without an output, so it isn't tried on every build.
pub fn store(db: &Db, job: &Job, done: Option<&Indexed>) -> Result<()> {
    db.prepare_cached("INSERT OR REPLACE INTO converts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")?.execute(params![
        job.source, job.target, ms_convert::CONVERTER_ID, job.keys, done.map(|d| &d.sha), done.map(|d| &d.md5), job.artist, job.title, done.map(|d| &d.version), job.creator,
        job.set_id, job.bid, job.od, job.length, job.playcount, job.web])?;
    Ok(())
}

pub fn inst(job: &Job, done: &Indexed) -> Inst {
    Inst { sha256: done.sha.clone(), set_id: job.set_id, md5: Some(done.md5.clone()), keys: job.target, od: job.od, length: job.length, title: job.title.clone(), artist: job.artist.clone(),
           version: done.version.clone(), creator: job.creator.clone(), playcount: job.playcount, convert_of: Some(job.source.clone()), source_keys: job.keys, source_web: job.web, ..Inst::default() }
}

/// Every indexed convert: the row as a list entry, its target keymode and source keymode.
pub fn rows(db: &Db) -> Result<Vec<Inst>> {
    let mut q = db.prepare_cached("SELECT source_sha,target,source_keys,sha,md5,artist,title,version,creator,set_id,od,length,playcount,web FROM converts WHERE converter=? AND sha IS NOT NULL")?;
    let rows = q.query_map([ms_convert::CONVERTER_ID], |r| {
        let text = |i: usize| r.get::<_, Option<String>>(i).map(Option::unwrap_or_default);
        Ok(Inst { convert_of: Some(r.get(0)?), keys: r.get(1)?, source_keys: r.get(2)?, sha256: r.get(3)?, md5: r.get(4)?, artist: text(5)?, title: text(6)?, version: text(7)?, creator: r.get(8)?,
                  set_id: r.get(9)?, od: r.get(10)?, length: r.get(11)?, playcount: r.get(12)?, source_web: r.get::<_, i64>(13)? != 0, ..Inst::default() })
    })?;
    rows.collect()
}

fn source_path(source: &str, web: bool, root: &Path) -> Option<PathBuf> {
    if web { Some(web_dir().join("osu").join(format!("{source}.osu"))) } else { paths::local_file(source, root) }
}

/// The convert's chart, made again (a rate the index has no analysis of, a new calculator's re-rating) or read back from
/// `kept`: rebuilding converts was 61 % of a calculator update's re-rating (~24 ms each, ~49k on the main PC, 2026-11-15).
pub fn bytes(i: &Inst, root: &Path) -> Option<Vec<u8>> {
    bytes_kept(i, root, &kept_dir())
}

fn bytes_kept(i: &Inst, root: &Path, dir: &Path) -> Option<Vec<u8>> {
    let path = dir.join(format!("{}.osu.z", i.sha256));
    let back = std::fs::read(&path).ok().and_then(|z| {
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::DeflateDecoder::new(&z[..]), &mut out).ok()?;
        Some(out)
    });
    // a convert's sha is the hash of its bytes: a kept file is right or it is made again
    if let Some(data) = back.filter(|d| hashes(d).0 == i.sha256) {
        return Some(data);
    }
    let data = std::fs::read(source_path(i.convert_of.as_ref()?, i.source_web, root)?).ok()?;
    let osu = ms_convert::convert(&data, i.keys as usize).ok()?.osu;
    if hashes(&osu).0 == i.sha256 {
        let mut z = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
        let part = path.with_extension("part");
        let written = std::io::Write::write_all(&mut z, &osu).ok().and_then(|_| z.finish().ok())
            .and_then(|z| std::fs::create_dir_all(path.parent()?).and_then(|_| std::fs::write(&part, z)).and_then(|_| std::fs::rename(&part, &path)).ok());
        if written.is_none() {
            let _ = std::fs::remove_file(&part);
        }
    }
    Some(osu)
}

/// Where converts' bytes are kept, deflated, by their sha.
fn kept_dir() -> PathBuf {
    paths::cache().join("converts")
}

/// Kept converts no row names any more go (after a calculator's re-rating, with its old rows).
pub fn prune_kept(db: &Db) -> rusqlite::Result<usize> {
    let names: std::collections::HashSet<String> = db.prepare("SELECT sha FROM converts WHERE sha IS NOT NULL")?.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
    let Ok(dir) = std::fs::read_dir(kept_dir()) else { return Ok(0) };
    let gone: Vec<PathBuf> = dir.flatten().map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).and_then(|n| n.strip_suffix(".osu.z")).is_none_or(|sha| !names.contains(sha))).collect();
    gone.iter().for_each(|p| { let _ = std::fs::remove_file(p); });
    Ok(gone.len())
}

fn is_chart(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".osu")
}

/// Called while a delivery runs: (what it does, share done when known); false = the player moved on, stop.
pub type Report<'a> = &'a dyn Fn(&str, Option<f32>) -> bool;
pub const CANCELLED: &str = "cancelled";

/// Where the upcoming website pick's set waits (user 2026-11-19: "predownloads the next map from the playlist when it's a
/// web pick … when it becomes current selection it just gets imported … if it's skipped or not used … deleted later").
fn ahead_dir() -> PathBuf {
    paths::cache().join("ahead")
}

/// The set this pick's delivery would download, if any.
pub fn set_to_download(p: &Pick, db: &Db, stable: bool) -> Option<i64> {
    let set = p.set_id.filter(|s| *s > 0)?;
    (!p.installed && (p.convert_of.is_none() || needs_original(p, db, stable))).then_some(set)
}

/// Fetches the upcoming pick's set ahead; sets fetched for other picks go, but for `keep` (the current pick's, which its
/// delivery may be about to take). `live` false stops it.
pub fn fetch_ahead(set: Option<i64>, keep: Option<i64>, live: &dyn Fn() -> bool) -> std::result::Result<(), String> {
    let dir = ahead_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    prune_ahead(&dir, &[set, keep]);
    let Some(set) = set else { return Ok(()) };
    let to = dir.join(format!("{set}.osz"));
    if to.is_file() || to.with_extension("osz.part").is_file() {
        return Ok(()); // here, or on its way
    }
    first_ok(&MIRRORS, |m| fetch(&format!("{m}{set}"), &to, &|_, _| live()))
}

fn prune_ahead(dir: &Path, keep: &[Option<i64>]) {
    let kept: Vec<String> = keep.iter().flatten().map(|s| format!("{s}.osz")).collect();
    for f in std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()) {
        let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if !kept.iter().any(|k| name == *k || name == format!("{k}.part")) {
            let _ = std::fs::remove_file(&f);
        }
    }
}

/// The set fetched ahead, moved to `to`; one still on its way is waited for. False: nothing ahead, download it.
fn take_ahead(dir: &Path, set: i64, to: &Path, report: Report) -> bool {
    let ahead = dir.join(format!("{set}.osz"));
    while !ahead.is_file() && ahead.with_extension("osz.part").is_file() {
        if !report("Downloading (started ahead)…", None) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    ahead.is_file() && (std::fs::rename(&ahead, to).is_ok() || std::fs::copy(&ahead, to).and_then(|_| std::fs::remove_file(&ahead)).is_ok())
}

fn download(set: i64, to: &Path, report: Report) -> std::result::Result<(), String> {
    if take_ahead(&ahead_dir(), set, to, report) {
        let _ = ms_data::webmaps::delivered(set);
        return Ok(());
    }
    let got = first_ok(&MIRRORS, |m| fetch(&format!("{m}{set}"), to, report));
    if got.is_ok() {
        let _ = ms_data::webmaps::delivered(set);
    }
    got.map_err(|e| {
        match ms_data::webmaps::strike(set, refused(&e)) {
            Ok(true) => format!("{e} · gone from the mirrors, removed from your lists"),
            _ => e,
        }
    })
}

/// A mirror answered and refused this set (403, 404, 410, 451): the map, not the server. 429, 5xx, timeouts and dropped
/// connections are the server's.
fn refused(errors: &str) -> bool {
    errors.split(" · ").any(|e| e.rsplit("status code ").next().filter(|_| e.contains("status code ")).and_then(|c| c.trim().parse::<u16>().ok())
        .is_some_and(|c| matches!(c, 403 | 404 | 410 | 451)))
}

/// The first mirror that delivers; every mirror's error when none does.
fn first_ok(mirrors: &[&str], mut get: impl FnMut(&str) -> std::result::Result<(), String>) -> std::result::Result<(), String> {
    let mut errors = Vec::new();
    for m in mirrors {
        match get(m) {
            Ok(()) => return Ok(()),
            Err(e) if e == CANCELLED => return Err(e),
            Err(e) => errors.push(e),
        }
    }
    Err(errors.join(" · "))
}

fn fetch(url: &str, to: &Path, report: Report) -> std::result::Result<(), String> {
    use std::io::{Read, Write};
    let host = url.split('/').nth(2).unwrap_or(url);
    if !report(&format!("Downloading from {host}…"), None) {
        return Err(CANCELLED.into());
    }
    let response = ureq::get(url).set("User-Agent", UA).timeout(Duration::from_secs(120)).call().map_err(|e| e.to_string())?;
    let total = response.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).filter(|&t| t > 0);
    let part = to.with_extension("osz.part");
    let (mut reader, mut file) = (response.into_reader(), std::fs::File::create(&part).map_err(|e| e.to_string())?);
    let (mut buf, mut got) = (vec![0u8; 1 << 16], 0u64);
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        got += n as u64;
        let text = format!("Downloading from {host}… {:.1} MB{}", got as f64 / 1e6, total.map_or(String::new(), |t| format!(" of {:.1}", t as f64 / 1e6)));
        if !report(&text, total.map(|t| (got as f64 / t as f64).min(1.) as f32)) {
            drop(file);
            let _ = std::fs::remove_file(&part);
            return Err(CANCELLED.into());
        }
    }
    std::fs::rename(&part, to).map_err(|e| e.to_string())
}

const ENTRY_MAX: usize = 256 << 20;

/// A website set's files from its .osz, its charts only when asked.
pub fn archive_files(path: &Path, charts: bool) -> std::result::Result<Vec<(String, Vec<u8>)>, String> {
    let mut zip = zip::ZipArchive::new(std::fs::File::open(path).map_err(|e| e.to_string())?).map_err(|e| format!("not a beatmap archive: {e}"))?;
    let mut out = Vec::new();
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(|e| e.to_string())?;
        if f.is_file() && (charts || !is_chart(f.name())) {
            // the size is the archive's claim: capacity and read stay under a set's largest sane file
            let mut data = Vec::with_capacity((f.size() as usize).min(ENTRY_MAX));
            std::io::Read::read_to_end(&mut std::io::Read::take(&mut f, ENTRY_MAX as u64 + 1), &mut data).map_err(|e| e.to_string())?;
            if data.len() > ENTRY_MAX {
                return Err(format!("{} is over {} MB", f.name(), ENTRY_MAX >> 20));
            }
            out.push((f.name().to_string(), data));
        }
    }
    Ok(out)
}

/// A set's files in lazer, by content hash.
fn local_set(source: &str) -> std::result::Result<Vec<(String, PathBuf)>, String> {
    let root = paths::lazer_data();
    let files = ms_data::imports::with_copy(&root.join("client.realm"), |copy| ms_data::realm::set_files(copy, source));
    Ok(files?.into_iter().filter(|(name, _)| !is_chart(name)).filter_map(|(name, hash)| Some((name, paths::local_file(&hash, &root)?))).collect())
}

fn import_dir(client: &str) -> std::result::Result<PathBuf, String> {
    // Inside lazer's own folder: its Flatpak sandbox can't read anywhere else.
    let dir = if client == "stable" { ms_data::imports::stable_songs(&paths::stable_data()) } else { paths::lazer_data().join("maniascope") };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// The set folder a convert joins on stable, once the set is there.
fn stable_folder(p: &Pick, source: &str, db: &Db) -> std::result::Result<Option<PathBuf>, String> {
    if p.source_web {
        return Ok(Some(ms_data::imports::stable_songs(&paths::stable_data()).join(p.set_id.unwrap_or(0).to_string())).filter(|f| f.is_dir()));
    }
    let chart: String = db.query_row("SELECT path FROM stable_files WHERE sha256=?", [source], |r| r.get(0)).map_err(|_| format!("{} is no longer in Songs", p.title))?;
    Ok(Path::new(&chart).parent().map(Path::to_path_buf))
}

/// A website source stays marked so once its set is in lazer; importing it again only shows the same name twice (user 2026-11-01).
fn needs_original(p: &Pick, db: &Db, stable: bool) -> bool {
    p.source_web && (stable || db.query_row("SELECT 1 FROM installed WHERE sha256=?", [p.convert_of.as_deref().unwrap_or("")], |_| Ok(())).is_err())
}

/// stable: an extra diff in the set's folder, sharing its audio and background; F5 in song select shows it (real stable,
/// 2026-11-01, no ids needed). A website set arrives as one .osz with the diff inside. lazer has no such folder and an
/// import can't add to a set: the same online id replaces the set (online ids reset, offsets and last played lost), so
/// a convert is its own set there; its audio and images are stored once by hash (one extra file, 36 KB, same test).
/// For a website map on lazer the original goes first: the whole set is imported beside the convert, which reuses its
/// audio and images (user 2026-11-02: "fine if it also imports the set").
pub fn deliver(p: &Pick, db: &Db, client: &str, root: &Path, picked: Instant, report: Report) -> std::result::Result<(), String> {
    let stage = |what: &str| eprintln!("convert {}: {what} at {} ms", p.title, picked.elapsed().as_millis()); // where the wait after Next goes
    stage("start");
    if let (Some(source), true) = (p.invert_of.as_deref(), client == "stable") {
        return deliver_invert(p, source, db, root);
    }
    let Some(source) = p.convert_of.as_deref() else { return deliver_website(p, client, report) };
    let set = p.set_id.unwrap_or(0);
    let data = std::fs::read(source_path(source, p.source_web, root).ok_or("the source chart is gone")?).map_err(|e| e.to_string())?;
    let out = ms_convert::convert(&data, p.keys as usize).map_err(|e| e.to_string())?;
    if p.sha.as_deref() != Some(&hashes(&out.osu).0) {
        return Err("the converter changed: the list will catch up".to_string());
    }
    stage("converted");
    let stable = client == "stable";
    if stable {
        if let Some(folder) = stable_folder(p, source, db)? {
            return std::fs::write(paths::long_path(&folder.join(&out.file_name)), &out.osu).map_err(|e| e.to_string());
        }
    }
    let dir = import_dir(client)?;
    let name = format!("{} {}", p.sha.as_deref().unwrap_or("")[..12].to_owned(), out.file_name.trim_end_matches(".osu"));
    let name = if stable { set.to_string() } else { name.chars().take(120).collect::<String>().trim_end().to_string() };
    let path = dir.join(format!("{name}.osz"));
    let mut owned: Vec<(String, Vec<u8>)> = Vec::new();
    let mut local = Vec::new();
    if needs_original(p, db, stable) {
        let holder = if stable { paths::cache() } else { dir }; // stable would take the original from Songs by itself
        std::fs::create_dir_all(&holder).map_err(|e| e.to_string())?;
        let original = holder.join(format!("{set}.osz"));
        download(set, &original, report)?;
        stage("website set downloaded");
        owned = archive_files(&original, stable)?;
        if stable {
            let _ = std::fs::remove_file(&original);
        } else {
            crate::navigation::hand_over(&original, client)?;
        }
    } else {
        local = local_set(source)?;
    }
    let mut entries = vec![(out.file_name.clone(), Entry::Bytes(&out.osu))];
    entries.extend(owned.iter().map(|(n, d)| (n.clone(), Entry::Bytes(d))));
    entries.extend(local.iter().map(|(n, f)| (n.clone(), Entry::File(f))));
    let part = path.with_extension("part"); // stable would take a half-written .osz
    osz::write(&part, &entries).map_err(|e| e.to_string())?;
    std::fs::rename(&part, &path).map_err(|e| e.to_string())?;
    stage("written");
    crate::navigation::hand_over(&path, client)?;
    stage("handed over");
    if client == "stable" {
        return Ok(()); // stable writes its database when it closes: nothing to confirm against
    }
    imported(p, "osu! didn't import the convert (is it running?)", report)?;
    stage("in osu!'s store");
    let _ = std::fs::remove_file(&path);
    Ok(())
}

/// stable's Invert pick: the inverted chart beside its source in Songs (F5 shows it), as a convert of a local map is.
fn deliver_invert(p: &Pick, source: &str, db: &Db, root: &Path) -> std::result::Result<(), String> {
    let data = std::fs::read(source_path(source, false, root).ok_or("the source chart is gone")?).map_err(|e| e.to_string())?;
    let out = ms_convert::invert(&data).map_err(|e| e.to_string())?;
    if p.sha.as_deref() != Some(&hashes(&out.osu).0) {
        return Err("the Invert chart changed: the list will catch up".to_string());
    }
    let folder = stable_folder(p, source, db)?.ok_or_else(|| format!("{} is no longer in Songs", p.title))?;
    let path = paths::long_path(&folder.join(&out.file_name));
    if std::fs::read(&path).is_ok_and(|d| d == out.osu) {
        return Ok(()); // written for an earlier offer
    }
    std::fs::write(path, &out.osu).map_err(|e| e.to_string())
}

/// A website chart that isn't a convert: its set from the mirror into the client, as a convert's original arrives
/// (user 2026-11-06: natives opened osu://b for the in-game download while converts arrived by themselves).
fn deliver_website(p: &Pick, client: &str, report: Report) -> std::result::Result<(), String> {
    let set = p.set_id.filter(|s| *s > 0).ok_or("no beatmap set id")?;
    let path = import_dir(client)?.join(format!("{set}.osz"));
    download(set, &path, report)?;
    crate::navigation::hand_over(&path, client)?;
    if client == "stable" {
        return Ok(()); // stable imports the .osz from Songs on entering song select
    }
    // The website chart's sha is its .osu's: the set landed when lazer stores that file (an updated map never does).
    imported(p, "osu! didn't import the map (is it running, or was it updated on the website?)", report)?;
    let _ = std::fs::remove_file(&path);
    Ok(())
}

/// lazer stores a chart under its hash: the import landed when the pick's file is there.
fn imported(p: &Pick, timeout: &str, report: Report) -> std::result::Result<(), String> {
    let (wanted, start) = (paths::local_file(p.sha.as_deref().unwrap_or(""), &paths::lazer_data()).ok_or("no sha")?, Instant::now());
    while !wanted.is_file() {
        if start.elapsed() > CONFIRM {
            return Err(timeout.to_string());
        }
        if !report("Importing into osu!…", None) {
            return Err(CANCELLED.into());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use ms_calc::features::standard_params;

    #[test]
    fn a_set_fetched_ahead_is_taken_and_the_others_are_pruned() {
        let dir = std::env::temp_dir().join(format!("ms-ahead-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["11.osz", "22.osz", "33.osz", "33.osz.part", "44.osz.part"] {
            std::fs::write(dir.join(name), name).unwrap();
        }
        prune_ahead(&dir, &[Some(11), None, Some(33)]);
        let mut left: Vec<String> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, ["11.osz", "33.osz", "33.osz.part"]);
        let to = dir.join("out.osz");
        assert!(take_ahead(&dir, 11, &to, &|_, _| true));
        assert_eq!((std::fs::read_to_string(&to).unwrap().as_str(), dir.join("11.osz").exists()), ("11.osz", false));
        assert!(!take_ahead(&dir, 55, &to, &|_, _| true), "nothing ahead: the caller downloads");
        std::fs::remove_file(dir.join("33.osz")).unwrap();
        assert!(!take_ahead(&dir, 33, &to, &|_, _| false), "one on its way is waited for until the player moves on");
        let _ = std::fs::remove_dir_all(&dir);
    }
    use std::sync::Arc;

    fn chart() -> String {
        let mut osu = "osu file format v14\n\n[General]\nAudioFilename: a.mp3\nMode: 3\n\n[Metadata]\nTitle:T\nArtist:A\nCreator:C\nVersion:V\n\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n\n[TimingPoints]\n0,500,4,1,0,60,1,0\n\n[HitObjects]\n".to_string();
        for step in 0..600 {
            osu.push_str(&format!("{},192,{},1,0,0:0:0:0:\n", (step * 3 % 7) * 512 / 7 + 36, step * 125));
        }
        osu
    }

    #[test]
    fn only_a_mirror_refusing_the_set_counts_against_the_map() {
        assert!(refused("https://catboy.best/d/1: status code 403 · https://osu.direct/api/d/1: status code 503"));
        assert!(!refused("https://catboy.best/d/1: status code 503 · https://osu.direct/api/d/1: status code 429"));
        assert!(!refused("https://catboy.best/d/1: Connection Failed: tls connection init failed · x: timed out"));
    }

    #[test]
    fn a_failed_mirror_falls_through_to_the_next() {
        let mut asked = Vec::new();
        assert_eq!(first_ok(&["a", "b"], |m| { asked.push(m.to_string()); if m == "a" { Err("503".into()) } else { Ok(()) } }), Ok(()));
        assert_eq!(asked, ["a", "b"]);
        assert_eq!(first_ok(&["a", "b"], |m| Err(format!("{m} down"))), Err("a down · b down".to_string()));
    }

    #[test]
    fn a_source_out_of_reach_waits_until_reach_moves() {
        let dir = std::env::temp_dir().join(format!("ms-pending-{}", std::process::id()));
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        let chart = |c: char| Inst { sha256: c.to_string().repeat(64), keys: 7, ..Inst::default() };
        let library = [chart('a'), chart('b'), chart('c')];
        let mut rated = |sha: &str| match &sha[..1] { "a" => Some(1.5), "b" => Some(9.), _ => None };
        let mut jobs = |level: f64| {
            let reach = [(10, (0.55 * level, 1.1 * level))].into_iter().collect();
            let got = pending(&db, &library, None, &HashSet::new(), &[(10, vec![7])], &reach, &mut rated, &dir, 10).unwrap();
            got.iter().map(|j| j.source[..1].to_string()).collect::<Vec<_>>().join("")
        };
        assert_eq!(jobs(4.), "ac", "9 stars can't fit a level 4 player, at any rate");
        assert_eq!(jobs(6.), "abc", "the same chart once they improve");
        assert_eq!(jobs(40.), "bc", "1.5 stars can't fit a level 40 player");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_convert_is_made_once_and_read_back_by_its_sha() {
        let dir = std::env::temp_dir().join(format!("ms-kept-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let osu = ms_convert::convert(chart().as_bytes(), 10).unwrap().osu;
        let sha = hashes(&osu).0;
        let i = Inst { sha256: sha.clone(), keys: 10, convert_of: Some("x".into()), source_web: true, ..Inst::default() };
        let kept = dir.join("kept");
        std::fs::create_dir_all(&kept).unwrap();
        let mut z = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut z, &osu).unwrap();
        std::fs::write(kept.join(format!("{sha}.osu.z")), z.finish().unwrap()).unwrap();
        assert_eq!(bytes_kept(&i, &dir, &kept).unwrap(), osu, "read back without its source");
        std::fs::write(kept.join(format!("{sha}.osu.z")), b"garbage").unwrap();
        assert!(bytes_kept(&i, &dir, &kept).is_none(), "a wrong kept file is never used (no source here to make it again)");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_index_knows_what_the_written_chart_will_be() {
        let dir = std::env::temp_dir().join(format!("ms-converts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("source.osu");
        std::fs::write(&path, chart()).unwrap();
        let job = Job { source: "source".into(), target: 10, keys: 7, path, artist: "A".into(), title: "T".into(), version: "V".into(), creator: None, set_id: None, bid: None, od: None, length: None, playcount: None, web: false };
        let prm = standard_params();
        let done = run(&job, &prm).unwrap();
        // what `deliver` writes
        let written = ms_convert::convert(chart().as_bytes(), 10).unwrap().osu;
        assert_eq!((hashes(&written).0, hashes(&written).1), (done.sha.clone(), done.md5.clone()));
        let direct = chart_features(&written, &[1.], &prm).unwrap();
        assert_eq!(ms_data::codec::encode(&ms_calc::val::Val::Dict(direct[0].1.clone())), ms_data::codec::encode(&ms_calc::val::Val::Dict(done.features[0].1.clone())));
        assert_eq!(direct[0].1.f("keys") as i64, 10);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_stable_convert_joins_its_set_folder() {
        let dir = std::env::temp_dir().join(format!("ms-stable-convert-{}", std::process::id()));
        let (store, folder) = (dir.join("store"), dir.join("Songs/1 A - T"));
        std::fs::create_dir_all(&folder).unwrap();
        let source = hashes(chart().as_bytes()).0;
        let linked = paths::local_file(&source, &store).unwrap();
        std::fs::create_dir_all(linked.parent().unwrap()).unwrap();
        std::fs::write(&linked, chart()).unwrap();
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        db.execute("CREATE TABLE IF NOT EXISTS stable_files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER, sha256 TEXT)", []).unwrap();
        db.execute("INSERT INTO stable_files VALUES (?,0,0,?)", rusqlite::params![folder.join("a.osu").to_str().unwrap(), source]).unwrap();
        let out = ms_convert::convert(chart().as_bytes(), 10).unwrap();
        let p = Pick { convert_of: Some(source), keys: 10, sha: Some(hashes(&out.osu).0), set_id: Some(1), title: "T".into(), ..Pick::default() };
        deliver(&p, &db, "stable", &store, Instant::now(), &|_, _| true).unwrap();
        assert_eq!(std::fs::read(folder.join(&out.file_name)).unwrap(), out.osu);
        assert_eq!(std::fs::read_dir(&folder).unwrap().count(), 1, "no archive beside it");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_stable_invert_is_written_beside_its_source() {
        let dir = std::env::temp_dir().join(format!("ms-stable-invert-{}", std::process::id()));
        let (store, folder) = (dir.join("store"), dir.join("Songs/1 A - T"));
        std::fs::create_dir_all(&folder).unwrap();
        let source = hashes(chart().as_bytes()).0;
        let linked = paths::local_file(&source, &store).unwrap();
        std::fs::create_dir_all(linked.parent().unwrap()).unwrap();
        std::fs::write(&linked, chart()).unwrap();
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        db.execute("CREATE TABLE IF NOT EXISTS stable_files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER, sha256 TEXT)", []).unwrap();
        db.execute("INSERT INTO stable_files VALUES (?,0,0,?)", rusqlite::params![folder.join("a.osu").to_str().unwrap(), source]).unwrap();
        let out = ms_convert::invert(chart().as_bytes()).unwrap();
        let p = Pick { invert_of: Some(source), sha: Some(hashes(&out.osu).0), set_id: Some(1), title: "T".into(), installed: true, ..Pick::default() };
        deliver(&p, &db, "stable", &store, Instant::now(), &|_, _| true).unwrap();
        assert_eq!(std::fs::read(folder.join(&out.file_name)).unwrap(), out.osu);
        deliver(&p, &db, "stable", &store, Instant::now(), &|_, _| true).unwrap(); // offered again: the same file
        assert_eq!(std::fs::read_dir(&folder).unwrap().count(), 1);
        // the written chart rates as lazer's Invert mod does
        let prm = standard_params();
        let rate = |d: &[u8]| chart_features(d, &[1.], &prm).unwrap()[0].1.f("overall");
        let (written, modded) = (rate(&out.osu), rate(&ms_calc::mods::tagged(chart().as_bytes(), "IN")));
        assert!((written / modded - 1.).abs() < 0.01, "{written} {modded}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_convert_play_counts_for_the_model_never_for_pp() {
        let dir = std::env::temp_dir().join(format!("ms-convert-play-{}", std::process::id()));
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        let prm = standard_params();
        let written = ms_convert::convert(chart().as_bytes(), 10).unwrap().osu;
        let (sha, md5) = hashes(&written);
        let feats = chart_features(&written, &[1.], &prm).unwrap();
        db.execute("INSERT INTO feats VALUES (?,?)", params![format!("{sha}@1.000|calc"), ms_data::codec::encode(&ms_calc::val::Val::Dict(feats[0].1.clone()))]).unwrap();
        // lazer's row for a set with no online ids, as strong as a score can look to the ledger
        db.execute("INSERT INTO scores(key,src,client,sha256,md5,beatmap_id,status,online_id,played,mods,rate,ranked,stats,acc,n,pp)
                    VALUES ('s','realm','lazer-live',?,?,-1,1,5,'2026-11-01T10:00:00','[]',1,1,'{\"perfect\":500}',0.97,500,100)", params![sha, md5]).unwrap();
        ms_data::public::write(&dir.join("public.msp"), &ms_data::public::Meta::default(), &[]).unwrap();
        let public = Arc::new(ms_data::public::Public::open(&dir.join("public.msp")).unwrap());
        let mut source = crate::evidence::FeatureSource { public: public.clone(), calc: "calc".into(), cache: HashMap::default() };
        let c = ms_model::predict::Consts::standard();
        let rows = crate::evidence::evidence_rows(&db, &mut source, &c).unwrap();
        assert_eq!(rows.rows.iter().map(|r| (r.key.as_str(), r.keys)).collect::<Vec<_>>(), [("s", 10)]);
        let (ledger, _) = crate::pp::build_ledger(&db, &public, &HashMap::default()).unwrap();
        assert!(ledger.best.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_installed_website_source_is_not_imported_again() {
        let dir = std::env::temp_dir().join(format!("ms-original-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        let pick = Pick { source_web: true, convert_of: Some("source".into()), ..Pick::default() };
        assert!(needs_original(&pick, &db, false), "a new website map brings its original");
        db.execute("INSERT INTO installed(sha256, keys) VALUES ('source', 7)", []).unwrap();
        assert!(!needs_original(&pick, &db, false));
        assert!(needs_original(&pick, &db, true), "stable's copy comes from the same download");
        assert!(!needs_original(&Pick { source_web: false, ..pick }, &db, false));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn default_sources_are_within_three_keys() {
        assert_eq!(default_sources(10), [7, 8, 9]);
        assert_eq!(default_sources(4), [5, 6, 7]);
        assert_eq!(default_sources(7), [4, 5, 6, 8, 9, 10]);
    }
}

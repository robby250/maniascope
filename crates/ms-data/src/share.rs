//! Opt-in replay export for calibration (user 2026-11-13): the player's own mania replays and the charts they were
//! played on, in one zip laid out as a lazer realm dump — `plays.jsonl` (realm_dump's line shape, user_id −1) and
//! `files/<a>/<ab>/<sha256>` — so `calib rows --realm plays.jsonl --files files --user -1` reads lazer and stable
//! alike (`--files` = the unzipped `files` folder). Replay headers carry the player name; nothing else about the player is written.
use crate::paths::{lazer_data, local_file, local_origin, long_path, stable_data};
use crate::stable::osr_header;
use crate::store::{kv_get, parse_stats, Db};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

fn hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// stable's mod bits as lazer's mod list; mods the reconstruction can't judge (key mods, ScoreV2, …) keep a name it refuses.
pub fn stable_mods(bits: i64) -> Value {
    const NAMES: [(i64, &str); 12] = [(1, "NF"), (8, "HD"), (16, "HR"), (32, "SD"), (64, "DT"), (256, "HT"), (512, "NC"), (1024, "FL"), (16384, "PF"),
                                      (1 << 20, "FI"), (1 << 21, "RD"), (1 << 30, "MR")];
    let mut out: Vec<Value> = NAMES.iter().filter(|(b, n)| bits & b != 0 && !(*n == "DT" && bits & 512 != 0)).map(|(_, n)| json!({"acronym": n})).collect();
    let known: i64 = NAMES.iter().map(|x| x.0).sum();
    if bits & !known != 0 {
        out.push(json!({"acronym": format!("bits{}", bits & !known)}));
    }
    Value::Array(out)
}

struct Zip {
    w: zip::ZipWriter<std::fs::File>,
    seen: HashSet<String>,
}

impl Zip {
    fn put(&mut self, data: &[u8]) -> Result<String, String> {
        let sha = hex(data);
        if self.seen.insert(sha.clone()) {
            let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
            self.w.start_file(format!("files/{}/{}/{sha}", &sha[..1], &sha[..2]), opts).map_err(|e| e.to_string())?;
            self.w.write_all(data).map_err(|e| e.to_string())?;
        }
        Ok(sha)
    }
}

/// Writes the zip to `to`; returns how many plays went in.
pub fn export(db: &Db, to: &Path) -> Result<usize, String> {
    let file = std::fs::File::create(to).map_err(|e| format!("{}: {e}", to.display()))?;
    let mut z = Zip { w: zip::ZipWriter::new(file), seen: HashSet::new() };
    let mut lines: Vec<Value> = Vec::new();
    // lazer: every own play whose replay its file store still holds
    let user = kv_get(db, "user_id").and_then(|v| v.as_i64());
    let files = lazer_data().join("files");
    let realm = lazer_data().join("client.realm");
    for item in if realm.is_file() { crate::realm::dump(&realm, false, None).unwrap_or_default() } else { Vec::new() } {
        let (Some(replay), Some(chart)) = (item["replay_sha256"].as_str(), item["hash"].as_str()) else { continue };
        if user.is_none() || item["user_id"].as_i64() != user {
            continue;
        }
        let read = |sha: &str| local_file(sha, files.parent().unwrap_or(Path::new("."))).and_then(|p| std::fs::read(p).ok());
        let (Some(r), Some(c)) = (read(replay), read(chart)) else { continue };
        let (r, c) = (z.put(&r)?, z.put(&c)?);
        lines.push(json!({"id": item["id"], "user_id": -1, "date": item["date"], "hash": c, "replay_sha256": r, "mods": item["mods"], "stats": item["stats"], "pauses": item["pauses"], "client": "lazer"}));
    }
    // stable: the local replays (Data/r/<map md5>-<ticks>.osr) of the player's own mania scores
    let store = local_origin("stable").1;
    // the player's own stable scores as the import kept them: a replay is theirs when map, score and judgements match
    type Own = std::collections::HashMap<(String, i64, [i64; 6]), String>;
    let own: Own = db.prepare("SELECT md5,total,stats,sha256 FROM scores WHERE src='stable' AND md5 IS NOT NULL AND sha256 IS NOT NULL")
        .and_then(|mut s| s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<String>>(2)?, r.get::<_, String>(3)?)))?.collect::<Result<Vec<_>, _>>())
        .unwrap_or_default().into_iter().filter_map(|(m, t, st, sha)| Some(((m, t, st.as_deref().and_then(parse_stats)?), sha))).collect();
    for entry in std::fs::read_dir(long_path(&stable_data().join("Data").join("r"))).into_iter().flatten().flatten() {
        let Ok(bytes) = std::fs::read(entry.path()) else { continue };
        let Ok(h) = osr_header(&bytes) else { continue };
        let Some(sha) = own.get(&(h.md5.clone(), i64::from(h.score), h.stats.map(i64::from))).filter(|_| h.mode == 3) else { continue };
        let Some(chart) = local_file(sha, &store).and_then(|p| std::fs::read(p).ok()) else { continue };
        let (r, c) = (z.put(&bytes)?, z.put(&chart)?);
        let date = chrono::DateTime::from_timestamp(h.played.floor() as i64, 0).map(|t| t.format("%Y-%m-%dT%H:%M:%S").to_string());
        let s = h.stats;
        lines.push(json!({"id": r, "user_id": -1, "date": date, "hash": c, "replay_sha256": r, "mods": stable_mods(i64::from(h.mods)).to_string(),
            "stats": json!({"perfect": s[0], "great": s[1], "good": s[2], "ok": s[3], "meh": s[4], "miss": s[5]}).to_string(), "pauses": [], "client": "stable"}));
    }
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    z.w.start_file("plays.jsonl", opts).map_err(|e| e.to_string())?;
    for l in &lines {
        z.w.write_all(format!("{l}\n").as_bytes()).map_err(|e| e.to_string())?;
    }
    z.w.finish().map_err(|e| e.to_string())?;
    Ok(lines.len())
}

#[cfg(test)]
mod tests {
    #[test]
    fn stable_mod_bits_become_lazer_acronyms_and_unknown_bits_stay_refusable() {
        let names = |b| super::stable_mods(b).as_array().unwrap().iter().map(|m| m["acronym"].as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert_eq!(names(64 | 512), ["NC"], "Nightcore implies Double Time");
        assert_eq!(names(256 | 8), ["HD", "HT"]);
        assert_eq!(names(1 << 15), ["bits32768"], "a key mod is not judged");
    }
}

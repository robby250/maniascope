//! Charts as played with mods (`ms_calc::mods`): the source file plus its tag line, built in memory when needed (a cheap
//! one-line edit) and never written: 54k Invert copies filled 4.3 GB (user 2026-11-21). `mod_charts` keeps the source, tag
//! and md5 of each modded sha, so a score (source chart + mod list) finds its chart and its plays find their rows.
use crate::engine::hashes;
use ms_data::store::{Db, Result};
use rusqlite::OptionalExtension;

/// osu!stable has no Invert mod: its Invert charts are real files (`ms_convert::invert`), written into the source's set
/// folder when offered (user 2026-11-21: "it just imports the map in the game just like it does with key converts").
pub const STABLE_INVERT: &str = "INW";

/// `source`'s bytes as `tag` makes them; None: Invert has nothing to hold on this chart.
pub fn build(data: &[u8], tag: &str) -> Option<Vec<u8>> {
    if tag == STABLE_INVERT { ms_convert::invert(data).ok().map(|c| c.osu) } else { Some(ms_calc::mods::tagged(data, tag)) }
}

/// `modded` (`source` under `tag`) registered → its (sha, md5).
pub fn register(db: &Db, source: &str, tag: &str, modded: &[u8]) -> Result<(String, String)> {
    let (sha, md5) = hashes(modded);
    db.execute("INSERT INTO mod_charts VALUES (?, ?, ?, ?) ON CONFLICT(sha) DO UPDATE SET md5=excluded.md5", [&sha, source, tag, &md5])?;
    Ok((sha, md5))
}

/// `source` (its sha, its bytes) played with mod `tag` (`ms_calc::mods`) → (sha, md5, bytes) of the modded chart, registered once.
pub fn ensure(db: &Db, source: &str, data: &[u8], tag: &str) -> Result<(String, String, Vec<u8>)> {
    let modded = ms_calc::mods::tagged(data, tag);
    let (sha, md5) = register(db, source, tag, &modded)?;
    Ok((sha, md5, modded))
}

/// The sha of `source` played with `tag`, once `ensure` has built it.
pub fn find(db: &Db, source: &str, tag: &str) -> Result<Option<String>> {
    db.query_row("SELECT sha FROM mod_charts WHERE source=? AND tag=?", [source, tag], |r| r.get(0)).optional()
}

/// The source chart's sha of a modded chart; None for a chart as written.
pub fn source_of(db: &Db, sha: &str) -> Option<String> {
    db.query_row("SELECT source FROM mod_charts WHERE sha=?", [sha], |r| r.get(0)).optional().ok()?
}

/// A chart's bytes by sha: an installed chart's file, else a modded one built from its source.
pub fn bytes(db: &Db, sha: &str) -> Option<Vec<u8>> {
    if let Some(p) = crate::pp::chart_path(Some(sha)) {
        return std::fs::read(p).ok();
    }
    let (source, tag): (String, String) = db.query_row("SELECT source, tag FROM mod_charts WHERE sha=?", [sha], |r| Ok((r.get(0)?, r.get(1)?))).optional().ok()??;
    build(&std::fs::read(crate::pp::chart_path(Some(&source))?).ok()?, &tag)
}

/// Invert charts' stored ratings without their "invert" flag get it back, once: the carry to calc cc36 rebuilt them without
/// it (`features::carried`), so 52k Invert charts read as their source's skills and Vibro picks came out inverted (user
/// 2026-11-21). The ratings themselves were made from the Invert chart and stand.
pub fn restore_invert_flags(db: &Db) -> Result<usize> {
    const DONE: &str = "invert_flags_restored";
    if ms_data::store::kv_get(db, DONE).is_some() {
        return Ok(0);
    }
    let rows: Vec<(String, Vec<u8>)> = db.prepare("SELECT f.key, f.data FROM mod_charts m JOIN feats f ON f.key >= m.sha || '@' AND f.key < m.sha || 'A' WHERE m.tag IN ('IN', ?)")?
        .query_map([STABLE_INVERT], |r| Ok((r.get(0)?, r.get::<_, Vec<u8>>(1).unwrap_or_default())))?.collect::<Result<_>>()?;
    let tx = db.unchecked_transaction()?;
    let mut fixed = 0;
    for (key, data) in rows {
        if let Ok(ms_calc::val::Val::Dict(mut f)) = ms_data::codec::decode(&data) {
            if f.get("invert").is_none() {
                f.set("invert", 1.);
                tx.execute("UPDATE feats SET data=? WHERE key=?", rusqlite::params![ms_data::codec::encode(&ms_calc::val::Val::Dict(f)), key])?;
                fixed += 1;
            }
        }
    }
    ms_data::store::kv_set(&tx, DONE, &serde_json::json!(fixed))?;
    tx.commit()?;
    Ok(fixed)
}

/// The modded-chart files earlier versions wrote (`<rec>/mods`), removed once.
pub fn remove_old_files() {
    let dir = ms_data::paths::rec_dir().join("mods");
    if dir.is_dir() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_modded_chart_is_found_again_by_its_source_and_tag() {
        let dir = crate::testkit::sandbox().join("modcharts");
        std::fs::create_dir_all(&dir).unwrap();
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        let osu = b"osu file format v14\n[General]\nMode: 3\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[HitObjects]\n64,192,0,128,0,500:0:0:0:0:\n192,192,250,1,0,0:0:0:0:\n";
        let (source, _) = hashes(osu);
        assert_eq!(find(&db, &source, "HO").unwrap(), None);
        let (sha, md5, data) = ensure(&db, &source, osu, "HO").unwrap();
        assert!(sha != source && md5.len() == 32 && hashes(&data).0 == sha);
        assert_eq!(find(&db, &source, "HO").unwrap().as_deref(), Some(sha.as_str()));
        let stored: String = db.query_row("SELECT md5 FROM mod_charts WHERE sha=?", [&sha], |r| r.get(0)).unwrap();
        assert_eq!(stored, md5, "a restore reads the md5 instead of hashing the file");
        assert_eq!(find(&db, &source, "IN").unwrap(), None);
        let chart = ms_calc::chart::parse(&data).unwrap();
        assert_eq!(chart.end, chart.t, "Hold Off: the hold is a tap");
        assert_eq!(ensure(&db, &source, osu, "HO").unwrap().0, sha, "the same chart both times");
    }

    #[test]
    fn carried_invert_ratings_get_their_flag_back_once() {
        let dir = crate::testkit::sandbox().join("modflags");
        std::fs::create_dir_all(&dir).unwrap();
        let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
        let notes: String = (0..64).map(|i| format!("{},192,{},1,0,0:0:0:0:\n", 64 + 128 * (i % 4), 125 * i)).collect();
        let osu = format!("osu file format v14\n[General]\nMode: 3\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[HitObjects]\n{notes}").into_bytes();
        let osu = osu.as_slice();
        let (source, _) = hashes(osu);
        let (sha, _, data) = ensure(&db, &source, osu, "IN").unwrap();
        let (_, carried) = ms_calc::features::chart_features(&data, &[1.], &ms_calc::features::standard_params()).unwrap().remove(0);
        assert_eq!(carried.fget("invert", 0.), 1., "rated under Invert");
        let mut carried = carried;
        carried.0.retain(|(k, _)| !matches!(k, ms_calc::val::Key::S(s) if s == "rules")); // rated under other rules: re-rated
        let carried = ms_calc::features::carried(&carried, &ms_calc::features::standard_params(), 1., &|| ms_calc::chart::parse(&data).ok().map(std::rc::Rc::new)).unwrap();
        assert_eq!(carried.fget("invert", 0.), 1., "a re-rating keeps the flag");
        let mut lost = carried.clone();
        lost.0.retain(|(k, _)| !matches!(k, ms_calc::val::Key::S(s) if s == "invert"));
        db.execute("INSERT INTO feats VALUES (?, ?)", rusqlite::params![format!("{sha}@1.000|x"), ms_data::codec::encode(&ms_calc::val::Val::Dict(lost))]).unwrap();
        assert_eq!(restore_invert_flags(&db).unwrap(), 1);
        let back: Vec<u8> = db.query_row("SELECT data FROM feats WHERE key=?", [format!("{sha}@1.000|x")], |r| r.get(0)).unwrap();
        assert!(matches!(ms_data::codec::decode(&back), Ok(ms_calc::val::Val::Dict(f)) if f.fget("invert", 0.) == 1.));
        assert_eq!(restore_invert_flags(&db).unwrap(), 0, "once");
    }
}

//! Charts as played with mods (`ms_calc::mods`): the source file plus its tag line, kept as `<rec>/mods/<sha>.osu`
//! under the modded bytes' own sha. `mod_charts` remembers the source and tag of each, so a cleared cache is rebuilt
//! and a score (source chart + mod list) finds its chart again.
use crate::engine::hashes;
use ms_data::store::{Db, Result};
use rusqlite::OptionalExtension;
use std::path::PathBuf;

pub(crate) fn file(sha: &str) -> PathBuf {
    ms_data::paths::rec_dir().join("mods").join(format!("{sha}.osu"))
}

/// `source` (its sha, its bytes) played with `tag` → (sha, md5, file) of the modded chart, written and registered once.
pub fn ensure(db: &Db, source: &str, data: &[u8], tag: &str) -> Result<(String, String, PathBuf)> {
    let modded = ms_calc::mods::tagged(data, tag);
    let (sha, md5) = hashes(&modded);
    let path = file(&sha);
    if !path.is_file() {
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        let _ = std::fs::write(&path, &modded); // unwritable: the registry still rebuilds the bytes
    }
    db.execute("INSERT OR IGNORE INTO mod_charts VALUES (?, ?, ?)", [&sha, source, tag])?;
    Ok((sha, md5, path))
}

/// The sha of `source` played with `tag`, once `ensure` has built it.
pub fn find(db: &Db, source: &str, tag: &str) -> Result<Option<String>> {
    db.query_row("SELECT sha FROM mod_charts WHERE source=? AND tag=?", [source, tag], |r| r.get(0)).optional()
}

/// The source chart's sha of a modded chart; None for a chart as written.
pub fn source_of(db: &Db, sha: &str) -> Option<String> {
    db.query_row("SELECT source FROM mod_charts WHERE sha=?", [sha], |r| r.get(0)).optional().ok()?
}

/// A chart file by sha: an installed chart, else a modded one (rebuilt from its source when its file is gone).
pub fn chart_file(db: &Db, sha: &str) -> Option<PathBuf> {
    if let Some(p) = crate::pp::chart_path(Some(sha)) {
        return Some(p);
    }
    let path = file(sha);
    if path.is_file() {
        return Some(path);
    }
    let (source, tag): (String, String) = db.query_row("SELECT source, tag FROM mod_charts WHERE sha=?", [sha], |r| Ok((r.get(0)?, r.get(1)?))).optional().ok()??;
    let data = std::fs::read(crate::pp::chart_path(Some(&source))?).ok()?;
    ensure(db, &source, &data, &tag).ok().filter(|m| m.0 == sha).map(|m| m.2)
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
        let (sha, md5, path) = ensure(&db, &source, osu, "HO").unwrap();
        assert!(sha != source && md5.len() == 32 && path.is_file());
        assert_eq!(find(&db, &source, "HO").unwrap().as_deref(), Some(sha.as_str()));
        assert_eq!(find(&db, &source, "IN").unwrap(), None);
        let chart = ms_calc::chart::parse(&std::fs::read(chart_file(&db, &sha).unwrap()).unwrap()).unwrap();
        assert_eq!(chart.end, chart.t, "Hold Off: the hold is a tap");
        assert_eq!(ensure(&db, &source, osu, "HO").unwrap().0, sha, "the same chart both times");
    }
}

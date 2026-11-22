//! Keymode conversion: Torosu's key converter (MIT, see LICENSE-Torosu), commit d4d97c9, reduced to what ManiaScope uses.
//! Every convert is a separate new set on both clients, with no online ids: lazer marks an installed set for deletion
//! when an import has its OnlineID (BeatmapImporter.PreImport), so Torosu's merge into the original set must never
//! reach it, and one output per chart keeps its sha and md5 the same on either client.
mod analysis;
mod key_converter;
mod mania;
mod osu_fixups;
pub mod osz;
mod timing;

use anyhow::{bail, Result};
use key_converter::{Density, KeyConverterParams};

/// Changes whenever the output bytes of any chart can change: stored with each index row.
pub const CONVERTER_ID: &str = "torosu-d4d97c9-1";

pub const KEYS: std::ops::RangeInclusive<usize> = 1..=mania::STABLE_MAX_KEYS;

pub struct Converted {
    pub osu: Vec<u8>,
    pub file_name: String,
    pub version: String,
    pub keys: usize,
}

/// Going up adds notes: krrcream's 10K preset does too (rows of 2 to 8), Torosu's default None leaves maps thinner than
/// real ones of that keymode, and Full keeps handstreams as handstreams. One fixed setting: "too many options, better to
/// have good defaults" (user 2026-11-01). Going down never adds notes.
fn density(source: usize, target: usize) -> Density {
    if target > source { Density::Full } else { Density::None }
}

/// The `[Events]` lines rosu-map doesn't encode, with `[Variables]` expanded: storyboard and video.
fn raw_events(text: &str) -> Vec<String> {
    let (mut section, mut variables, mut events) = ("", Vec::<(&str, &str)>::new(), Vec::new());
    for line in text.lines() {
        let line = line.trim_start_matches('\u{feff}').trim_end();
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = trimmed;
            continue;
        }
        match section {
            "[Variables]" => {
                if let Some((name, value)) = trimmed.split_once('=').filter(|x| x.0.starts_with('$')) {
                    variables.push((name, value));
                }
            }
            "[Events]" if !trimmed.is_empty() && !trimmed.starts_with("//") => {
                let encoded = !line.starts_with([' ', '_']) && matches!(line.split(',').next().map(str::trim), Some("0" | "Background" | "2" | "Break"));
                if !encoded {
                    events.push(line.to_string());
                }
            }
            _ => {}
        }
    }
    variables.sort_by_key(|v| std::cmp::Reverse(v.0.len())); // so $ab is not replaced by the value of $a
    for line in &mut events {
        for (name, value) in &variables {
            if line.contains(name) {
                *line = line.replace(name, value);
            }
        }
    }
    events
}

/// Appended at the end of the `[Events]` section of an encoded `.osu`.
fn inject_events(osu: String, events: &[String]) -> String {
    const HEADER: &str = "[Events]\n";
    let Some(start) = osu.find(HEADER).filter(|_| !events.is_empty()) else { return osu };
    let body = start + HEADER.len();
    let at = osu[body..].find("\n[").map_or(osu.len(), |p| body + p);
    let mut out = String::with_capacity(osu.len() + events.iter().map(|e| e.len() + 1).sum::<usize>());
    out.push_str(&osu[..at]);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    for line in events {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&osu[at..]);
    out
}

/// rosu-map reads `key:value` by splitting at every colon and keeps the second piece (0.2.1 `KeyValue::parse`): a value is
/// cut at its own first colon, so lazer got "mona (CV" for `mona (CV: Natsukawa Shiina)` and the copied search, built
/// from the full name, never found the convert (user 2026-11-21). → the `[Metadata]` values it cuts, whole.
fn colon_values(text: &str) -> impl Iterator<Item = (&str, &str)> {
    const TEXTS: [&str; 8] = ["Title", "TitleUnicode", "Artist", "ArtistUnicode", "Creator", "Version", "Source", "Tags"];
    let lines = text.lines().map(|l| l.trim_start_matches('\u{feff}').trim()).skip_while(|l| *l != "[Metadata]").skip(1).take_while(|l| !l.starts_with('['));
    lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim(), v.trim())).filter(|(k, v)| v.contains(':') && TEXTS.contains(k))
}

/// A convert or Invert chart of `source` made before 2026-11-21 has cut metadata: its bytes, and so its sha, changed.
pub fn metadata_was_cut(source: &[u8]) -> bool {
    colon_values(&String::from_utf8_lossy(source)).next().is_some()
}

fn restore_metadata(map: &mut rosu_map::Beatmap, text: &str) {
    for (key, value) in colon_values(text) {
        let field = match key {
            "Title" => &mut map.title,
            "TitleUnicode" => &mut map.title_unicode,
            "Artist" => &mut map.artist,
            "ArtistUnicode" => &mut map.artist_unicode,
            "Creator" => &mut map.creator,
            "Version" => &mut map.version,
            "Source" => &mut map.source,
            _ => &mut map.tags,
        };
        *field = value.to_string();
    }
}

fn sanitize_filename(name: &str) -> String {
    name.chars().filter(|c| !matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') && !c.is_control()).collect()
}

/// A chart at another key count. Deterministic: the same bytes give the same bytes.
pub fn convert(source: &[u8], target: usize) -> Result<Converted> {
    // Website charts are anyone's files: an odd one fails alone, not the thread indexing thousands.
    std::panic::catch_unwind(|| convert_chart(source, target)).unwrap_or_else(|_| bail!("the converter failed on this chart"))
}

fn convert_chart(source: &[u8], target: usize) -> Result<Converted> {
    let mut map: rosu_map::Beatmap = rosu_map::from_bytes(source)?;
    restore_metadata(&mut map, &String::from_utf8_lossy(source));
    let keys = mania::key_count(&map);
    if keys == target {
        bail!("the map is already {target}K");
    }
    key_converter::apply(&mut map, &KeyConverterParams { target_keys: target as u8, density: density(keys, target), keep_jacks: true, playstyle: None })?;
    map.version = format!("{} ({keys}K to {target}K)", map.version);
    let events = raw_events(&String::from_utf8_lossy(source));
    let encoded = map.encode_to_string()?;
    let osu = osu_fixups::sanitize(inject_events(encoded, &events));
    let file_name = sanitize_filename(&format!("{} - {} ({}) [{}].osu", map.artist, map.title, map.creator, map.version));
    Ok(Converted { osu: osu.into_bytes(), file_name, version: map.version, keys: target })
}

/// The chart with lazer's Invert applied, as a chart of its own for osu!stable (no Invert mod there): in each column every
/// note start becomes a hold to the next start, shortened by a quarter beat (never below half the gap); the column's last
/// note goes. The same rule as `ms_calc::mods` (ManiaModInvert), which lazer plays rate by.
pub fn invert(source: &[u8]) -> Result<Converted> {
    std::panic::catch_unwind(|| invert_chart(source)).unwrap_or_else(|_| bail!("Invert failed on this chart"))
}

fn invert_chart(source: &[u8]) -> Result<Converted> {
    use rosu_map::section::hit_objects::{HitObject, HitObjectHold, HitObjectKind};
    let mut map: rosu_map::Beatmap = rosu_map::from_bytes(source)?;
    restore_metadata(&mut map, &String::from_utf8_lossy(source));
    let keys = mania::key_count(&map);
    let beats: Vec<(f64, f64)> = map.control_points.timing_points.iter().filter(|p| p.beat_len > 0.).map(|p| (p.time, p.beat_len)).collect();
    let beat_at = |t: f64| beats.iter().rev().find(|b| b.0 <= t).or(beats.first()).map_or(1000., |b| b.1); // lazer's default: 1000 ms
    let mut starts: Vec<(usize, f64)> = map.hit_objects.iter().filter_map(|o| mania::note(o).map(|(x, _)| (mania::column(x, keys), o.start_time))).collect();
    starts.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut out: Vec<HitObject> = Vec::new();
    for c in 0..keys {
        let column: Vec<f64> = starts.iter().filter(|s| s.0 == c).map(|s| s.1).collect();
        for pair in column.windows(2) {
            let gap = pair[1] - pair[0];
            let duration = (gap / 2.).max(gap - beat_at(pair[1]) / 4.);
            out.push(HitObject { start_time: pair[0], kind: HitObjectKind::Hold(HitObjectHold { pos_x: mania::column_x(c, keys), duration }), samples: Vec::new() });
        }
    }
    if out.is_empty() {
        bail!("nothing to hold under Invert");
    }
    out.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
    map.hit_objects = out;
    map.version = format!("{} (Invert)", map.version);
    let events = raw_events(&String::from_utf8_lossy(source));
    let osu = osu_fixups::sanitize(inject_events(map.encode_to_string()?, &events));
    let file_name = sanitize_filename(&format!("{} - {} ({}) [{}].osu", map.artist, map.title, map.creator, map.version));
    Ok(Converted { osu: osu.into_bytes(), file_name, version: map.version, keys })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_notes_are_one_note() {
        let mut osu = source();
        for _ in 0..14 {
            osu.push_str("36,192,30000,1,0,0:0:0:0:\n");
        }
        let out = String::from_utf8(convert(osu.as_bytes(), 10).unwrap().osu).unwrap();
        assert_eq!(out.lines().filter(|l| l.split(',').nth(2) == Some("30000")).count(), 1);
    }

    fn source() -> String {
        let mut osu = "osu file format v14\n\n[General]\nAudioFilename: a.mp3\nMode: 3\n\n[Metadata]\nTitle:T\nArtist:A\nCreator:C\nVersion:Insane\nBeatmapID:123\nBeatmapSetID:456\n\n\
                       [Difficulty]\nCircleSize:7\nOverallDifficulty:8\n\n[Variables]\n$x=99\n\n[Events]\n0,0,\"bg.jpg\",0,0\nSprite,Foreground,Centre,\"s.png\",$x,240\n _F,0,0,1000,1,0\n\n\
                       [TimingPoints]\n0,500,4,1,0,60,1,0\n\n[HitObjects]\n".to_string();
        for step in 0..200 {
            osu.push_str(&format!("{},192,{},1,0,0:0:0:0:\n", (step * 3 % 7) * 512 / 7 + 36, step * 125));
        }
        osu
    }

    #[test]
    fn a_value_keeps_what_follows_its_colon() {
        let osu = source().replace("Title:T", "Title:Re:A\nTags:x:y z").replace("Artist:A", "Artist:mona (CV: X)").replace("Version:Insane", "Version:a:b");
        let value = |text: &str, key: &str| text.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix(':').map(|v| v.trim().to_string())).unwrap();
        for out in [convert(osu.as_bytes(), 4).unwrap(), invert(osu.as_bytes()).unwrap()] {
            let text = String::from_utf8(out.osu).unwrap();
            assert_eq!((value(&text, "Title"), value(&text, "Artist"), value(&text, "Tags")), ("Re:A".into(), "mona (CV: X)".into(), "x:y z".into()));
            assert_eq!(value(&text, "Version"), out.version);
            assert!(metadata_was_cut(osu.as_bytes()) && !metadata_was_cut(source().as_bytes()));
            assert!(out.version.starts_with("a:b (") && out.file_name.starts_with("mona (CV X) - ReA (C) [ab ("), "{} {}", out.version, out.file_name);
        }
    }

    #[test]
    fn same_bytes_give_same_bytes() {
        assert_eq!(convert(source().as_bytes(), 10).unwrap().osu, convert(source().as_bytes(), 10).unwrap().osu);
    }

    /// lazer marks an installed set for deletion when an import has its OnlineID.
    #[test]
    fn a_convert_has_no_online_ids() {
        let out = convert(source().as_bytes(), 10).unwrap();
        let text = String::from_utf8(out.osu).unwrap();
        assert!(!text.lines().any(|l| l.starts_with("BeatmapID") || l.starts_with("BeatmapSetID")), "{text}");
    }

    #[test]
    fn keeps_the_storyboard_and_names_the_convert() {
        let out = convert(source().as_bytes(), 4).unwrap();
        assert_eq!(out.version, "Insane (7K to 4K)");
        assert_eq!(out.file_name, "A - T (C) [Insane (7K to 4K)].osu");
        let text = String::from_utf8(out.osu).unwrap();
        assert!(text.contains("Sprite,Foreground,Centre,\"s.png\",99,240\r\n _F,0,0,1000,1,0\r\n"), "{text}");
        assert!(text.lines().any(|l| l.replace(" ", "") == "CircleSize:4") && text.lines().any(|l| l == "Version: Insane (7K to 4K)" || l == "Version:Insane (7K to 4K)"), "{text}");
    }

    #[test]
    fn invert_holds_each_note_to_the_next_in_its_column() {
        let out = invert(source().as_bytes()).unwrap();
        assert_eq!(out.version, "Insane (Invert)");
        let text = String::from_utf8(out.osu).unwrap();
        let holds: Vec<&str> = text.split("[HitObjects]").nth(1).unwrap().lines().filter(|l| !l.trim().is_empty()).collect();
        // 200 notes over 7 columns: each column's last note goes
        assert_eq!(holds.len(), 200 - 7);
        assert!(holds.iter().all(|l| l.split(',').nth(3) == Some("128")), "{holds:?}");
        assert!(!text.lines().any(|l| l.starts_with("BeatmapID")), "{text}");
    }

    #[test]
    fn refuses_the_same_key_count() {
        assert!(convert(source().as_bytes(), 7).is_err());
    }

    #[test]
    fn going_up_adds_notes_going_down_never() {
        let rows = |osu: &[u8]| String::from_utf8_lossy(osu).split("[HitObjects]").nth(1).unwrap().lines().filter(|l| !l.is_empty()).count();
        let original = rows(source().as_bytes());
        assert!(rows(&convert(source().as_bytes(), 10).unwrap().osu) > original);
        assert!(rows(&convert(source().as_bytes(), 4).unwrap().osu) <= original);
    }
}

//! `.osu` parsing: notes (start, end, column) and the lazer mania scroll multiplier.
use crate::py;
use std::collections::BTreeMap;

#[derive(Debug)]
pub enum ChartError {
    /// Not a usable native osu!mania chart (`skill_calc.ChartError`).
    Chart(String),
    /// Python would have raised something else (ValueError/OverflowError) while parsing.
    Value(String),
}

#[derive(Clone, Debug, Default)]
pub struct Chart {
    pub keys: usize,
    /// Structure of arrays, sorted by (t, col, end); ms (integral for parsed charts, fractional for
    /// generated charts).
    pub t: Vec<f64>,
    pub end: Vec<f64>,
    pub col: Vec<u8>,
    /// (ms, multiplier); empty = constant scroll.
    pub sv: Vec<(f64, f64)>,
    pub od: f64,
    /// The OD stands for a window mods set (Hard Rock, Easy, Difficulty Adjust, a time ramp): read as it is, past 0–10 too.
    pub od_free: bool,
    /// ln(GREAT window as played / the chart's own): a mod moved the windows (0 on a plain chart). The forecast reads it
    /// (`mods::window_law`); the rating reads the played window itself (`great_window`).
    pub window_shift: f64,
    /// No Release: a hold's tail is not timed.
    pub no_release: bool,
    /// Judgements no press can get (a hold counts head and tail): notes on a note already at that column and time. One
    /// press takes one note, so even auto misses them (user 2026-11-16: Tentacle VIP [Inertia VIP], 195 stacked, auto ~80 %).
    pub stacked: f64,
    pub title: String,
    pub artist: String,
    pub version: String,
    pub creator: String,
}

impl Chart {
    pub fn len(&self) -> usize {
        self.t.len()
    }
    /// The great window in seconds. A file's OD counts within 0–10; a window mods set is taken as given.
    pub fn great_window(&self) -> f64 {
        if self.od_free { ((64.0 - 3.0 * self.od) / 1000.0).max(0.005) } else { crate::gestures::great_window(self.od) }
    }

    pub fn last_end(&self) -> f64 {
        self.end.iter().copied().fold(None, |m: Option<f64>, x| Some(match m { Some(m) if m >= x => m, _ => x })).unwrap_or(0.0)
    }

    /// The chart as played at `rate` (MASH_SPAN of the great window, in real time; see `mashed`). None: as written.
    pub fn played(&self, rate: f64, k: &crate::params::Knobs) -> Option<(Chart, Vec<Gesture>)> {
        (k.MASH_SPAN > 0.0).then(|| self.mashed(k.MASH_SPAN * self.great_window() * 1000.0 * rate, k.MASH_COVER)).flatten()
    }

    /// Consecutive rows on columns not yet hit, all within `span` ms of the first, are one gesture,
    /// struck at once at the middle of the span (a 7-key chord rolled over 27 ms is mashed as a chord: user 2026-11-15,
    /// A Fool Moon Night's fake streams and brackets). A column already in the gesture starts the next. Only gestures over `cover` of the keys merge: a 2–3 note flam inside
    /// a stream merged into fake fast jacks and LN shields. None: nothing merged. With the chart: each merged gesture.
    pub fn mashed(&self, span: f64, cover: f64) -> Option<(Chart, Vec<Gesture>)> {
        let n = self.len();
        let mut rows: Vec<(usize, usize, u64)> = Vec::new(); // (first note, end, columns)
        for i in 0..n {
            match rows.last_mut() {
                Some(r) if self.t[r.0] == self.t[i] => (r.1, r.2) = (i + 1, r.2 | 1 << self.col[i]),
                _ => rows.push((i, i + 1, 1 << self.col[i])),
            }
        }
        let (mut t, mut merged, mut r) = (self.t.clone(), Vec::new(), 0);
        while r < rows.len() {
            let (start, mut mask, mut q) = (self.t[rows[r].0], rows[r].2, r + 1);
            while q < rows.len() && self.t[rows[q].0] - start <= span && mask & rows[q].2 == 0 {
                mask |= rows[q].2;
                q += 1;
            }
            if q > r + 1 && f64::from(mask.count_ones()) >= cover * self.keys as f64 {
                let last = self.t[rows[q - 1].0];
                let mid = 0.5 * (start + last);
                t[rows[r].0..rows[q - 1].1].iter_mut().for_each(|x| *x = mid);
                merged.push(Gesture { t: mid, rows: (q - r - 1) as f64, inside: 1.0 - (last - start) / span });
            }
            r = q;
        }
        (!merged.is_empty()).then_some(())?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| t[a].total_cmp(&t[b]).then(self.col[a].cmp(&self.col[b])));
        let end = |x: usize| if self.end[x] > self.t[x] { self.end[x].max(t[x]) } else { t[x] };
        Some((Chart { t: order.iter().map(|&x| t[x]).collect(), end: order.iter().map(|&x| end(x)).collect(), col: order.iter().map(|&x| self.col[x]).collect(), ..self.clone() }, merged))
    }
}

/// Written rows struck as one (`Chart::mashed`): when (chart ms), how many rows went into another, and how far inside the
/// span the roll ends (0 = it just fits, 1 = a chord).
pub struct Gesture {
    pub t: f64,
    pub rows: f64,
    pub inside: f64,
}

/// Python text mode: utf-8 with errors="ignore", universal newlines. Slices of `text`: no copy per line. A lone \r also
/// ends a line; the empty pieces this split leaves are blank lines, which the parser skips.
fn lines(text: &str) -> Vec<&str> {
    // One byte scan: \r\n, \n and a lone \r each end a line (same pieces as split('\n') then split('\r')).
    let (b, mut out, mut s, mut i) = (text.as_bytes(), Vec::new(), 0, 0);
    while i < b.len() {
        if b[i] == b'\n' || b[i] == b'\r' {
            out.push(&text[s..i]);
            i += if b[i] == b'\r' && b.get(i + 1) == Some(&b'\n') { 2 } else { 1 };
            s = i;
        } else {
            i += 1;
        }
    }
    out.push(&text[s..]);
    if text.ends_with('\n') {
        out.pop(); // "a\n" is one line, as before
    }
    out
}

/// Python `str.strip()` (whitespace as `str.isspace`).
fn strip(s: &str) -> &str {
    // nearly every line starts and ends on a printable ASCII byte: nothing to trim, no char decoding
    if let (Some(&a), Some(&z)) = (s.as_bytes().first(), s.as_bytes().last()) {
        if a > b' ' && a < 0x80 && z > b' ' && z < 0x80 {
            return s;
        }
    }
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

pub fn parse(data: &[u8]) -> Result<Chart, ChartError> {
    // Valid UTF-8 without U+FFFD (nearly every chart) is read in place; the copy only drops bad bytes.
    let text: std::borrow::Cow<str> = match std::str::from_utf8(data) {
        Ok(t) if !t.contains('\u{FFFD}') => t.into(),
        _ => String::from_utf8_lossy(data).replace('\u{FFFD}', "").into(),
    };
    crate::tick::lap("p.utf8");
    let lines = lines(&text);
    crate::tick::lap("p.lines");
    let first = lines.first().map(|l| l.trim_start_matches('\u{FEFF}')).unwrap_or("");
    if !first.starts_with("osu file format") {
        return Err(ChartError::Chart("not an .osu file".into()));
    }
    let mut meta: Vec<(String, String)> = Vec::new();
    let mut keys: Option<i64> = None;
    let mut od = 5.0f64;
    let mut mode = 0i64;
    let mut mods: Option<crate::mods::Mods> = None;
    // the section as a number: 1 HitObjects, 2 TimingPoints, 3 General, 4 Difficulty, 5 Metadata, 0 any other (it was a
    // string, compared up to four times per line)
    let mut section = 0u8;
    let mut raw: Vec<(i64, i64, i64)> = Vec::new();
    let mut tps: Vec<(f64, f64, bool)> = Vec::new();
    for line in &lines[1..] {
        let line = strip(line);
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            section = match line { "[HitObjects]" => 1, "[TimingPoints]" => 2, "[General]" => 3, "[Difficulty]" => 4, "[Metadata]" => 5, _ => 0 };
            continue;
        }
        if section == 1 {
            // fields 0–5 by one byte scan, no Vec per note; the hit sample after them is never read
            let (mut p, mut len, mut s) = ([""; 6], 0, 0);
            for (i, &c) in line.as_bytes().iter().chain([&b',']).enumerate() {
                if c == b',' {
                    p[len] = &line[s..i];
                    len += 1;
                    s = i + 1;
                    if len == 6 {
                        break;
                    }
                }
            }
            if len < 4 {
                continue;
            }
            let x = match py::int_float(p[0]) {
                Ok(v) => v,
                Err(true) => return Err(ChartError::Value("cannot convert float infinity to integer".into())),
                Err(false) => continue,
            };
            let t = match py::int_float(p[2]) {
                Ok(v) => v,
                Err(true) => return Err(ChartError::Value("cannot convert float infinity to integer".into())),
                Err(false) => continue,
            };
            let typ = match py::int(p[3]) {
                Some(v) => v,
                None => continue,
            };
            let end = if typ & 128 != 0 {
                if len < 6 {
                    continue;
                }
                match py::int_float(&p[5][..p[5].bytes().position(|c| c == b':').unwrap_or(p[5].len())]) {
                    Ok(v) => v,
                    Err(true) => return Err(ChartError::Value("cannot convert float infinity to integer".into())),
                    Err(false) => continue,
                }
            } else {
                t
            };
            raw.push((t, end.max(t), x));
        } else if section == 2 {
            let p: Vec<&str> = line.split(',').collect();
            if p.len() < 2 {
                continue;
            }
            let (Some(a), Some(b)) = (py::float(p[0]), py::float(p[1])) else { continue };
            let red = if p.len() > 6 { !strip(p[6]).starts_with('0') } else { true };
            tps.push((a, b, red));
        } else if section == 0 {
        } else if let Some((k, v)) = line.split_once(':') {
            if section == 3 && k == "Mode" {
                let v = strip(v);
                mode = if v.is_empty() { 0 } else { py::int(v).ok_or_else(|| ChartError::Value(format!("invalid literal for int(): {v:?}")))? };
            } else if section == 4 && k == "CircleSize" {
                let f = py::float(v).ok_or_else(|| ChartError::Value(format!("could not convert string to float: {v:?}")))?;
                if !f.is_finite() {
                    return Err(ChartError::Value("cannot convert float to integer".into()));
                }
                keys = Some(py::round(f) as i64);
            } else if section == 4 && k == "OverallDifficulty" {
                od = py::float(v).ok_or_else(|| ChartError::Value(format!("could not convert string to float: {v:?}")))?;
            } else if section == 4 && k == "ManiaScopeMods" {
                mods = Some(crate::mods::Mods::from_tag(strip(v))); // a modded chart's own file (mods::tagged)
            } else if section == 5 {
                let v = strip(v).to_string();
                match meta.iter_mut().find(|(mk, _)| mk == k) {
                    Some(e) => e.1 = v,
                    None => meta.push((k.to_string(), v)),
                }
            }
        }
    }
    crate::tick::lap("p.loop");
    if mode != 3 {
        return Err(ChartError::Chart("not a native osu!mania chart (converts are unsupported)".into()));
    }
    let keys = match keys {
        Some(k) if (1..=18).contains(&k) => k,
        k => return Err(ChartError::Chart(format!("unsupported key count: {}", k.map_or("None".into(), |k| k.to_string())))),
    };
    // Normalise: column, stacked notes out, order-independent sort, no same-column overlap.
    let mut notes: Vec<(i64, i64, i64)> = raw
        .into_iter()
        .map(|(t, e, x)| (t, e, (keys - 1).min(0i64.max(py::floordiv(x.saturating_mul(keys), 512)))))
        .collect();
    notes.sort_unstable_by_key(|&(t, e, c)| (t, c, e)); // equal keys are equal notes: order among them is moot
    let mut ch = Chart { keys: keys as usize, ..Default::default() };
    let mut last: Vec<Option<usize>> = vec![None; keys as usize];
    for (t, e, c) in notes {
        if let Some(i) = last[c as usize] {
            if ch.t[i] == t as f64 {
                ch.stacked += if e > t { 2. } else { 1. };
                continue; // a stacked note: kept out of the patterns, counted as a sure miss
            }
            if ch.end[i] > t as f64 {
                ch.end[i] = t as f64; // impossible overlap: cut the earlier hold
            }
        }
        last[c as usize] = Some(ch.t.len());
        ch.t.push(t as f64);
        ch.end.push(e as f64);
        ch.col.push(c as u8);
    }
    crate::tick::lap("p.normalise");
    ch.od = if od.is_finite() { od } else { 5.0 };
    let get = |k: &str| meta.iter().find(|(mk, _)| mk == k).map(|(_, v)| v.clone());
    let nonempty = |k: &str| get(k).filter(|v| !v.is_empty());
    ch.title = nonempty("TitleUnicode").or_else(|| get("Title")).unwrap_or_default();
    ch.artist = nonempty("ArtistUnicode").or_else(|| get("Artist")).unwrap_or_default();
    ch.version = get("Version").unwrap_or_default();
    ch.creator = get("Creator").unwrap_or_default();
    ch.sv = scroll(&tps, ch.last_end());
    if let Some(m) = mods {
        let beats: Vec<(f64, f64)> = tps.iter().filter(|p| p.2 && p.1 > 0.0).map(|p| (p.0, p.1)).collect();
        m.apply(&mut ch, &beats);
    }
    Ok(ch)
}

/// [(ms, scroll multiplier)] as osu!lazer's mania playfield scrolls (ppy/osu 577d29f,
/// DrawableScrollingRuleset + LegacyBeatmapDecoder): multiplier = SV (100 / -beatLength of a
/// green line, 1 at a red line, clamped 0.01-10) × most common beat length / current beat length;
/// last point at a timestamp wins; points after the last object are dropped. [] = constant.
fn scroll(tps: &[(f64, f64, bool)], last: f64) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    let mut bl: Option<f64> = None;
    let mut sv = 1.0f64;
    for &(t, v, red) in tps {
        if red && v > 0.0 {
            bl = Some(60000f64.min(6f64.max(v)));
            sv = 1.0;
        } else if !red {
            sv = if v < 0.0 { 10f64.min(0.01f64.max(100.0 / -v)) } else { 1.0 };
        }
        if let Some(b) = bl {
            if t <= last {
                pts.push((t, b, sv));
            }
        }
    }
    let reds: Vec<(f64, f64)> = tps.iter().filter(|&&(_, v, red)| red && v > 0.0).map(|&(t, v, _)| (t, v)).collect();
    if pts.is_empty() || reds.is_empty() {
        return Vec::new();
    }
    // most common beat length by duration (first red starts at 0); dict order = first appearance
    let mut dur: Vec<(f64, f64)> = Vec::new();
    for (i, &(t, v)) in reds.iter().enumerate() {
        if t > last {
            continue;
        }
        let end = if i + 1 < reds.len() { reds[i + 1].0 } else { last };
        let k = py::round_digits(60000f64.min(6f64.max(v)), 3);
        let add = 0f64.max(end.min(last) - if i == 0 { 0.0 } else { t });
        match dur.iter_mut().find(|(dk, _)| *dk == k) {
            Some(e) => e.1 += add,
            None => dur.push((k, add)),
        }
    }
    let base = if dur.is_empty() {
        pts[0].1
    } else {
        let mut best = dur[0];
        for &d in &dur[1..] {
            if d.1 > best.1 {
                best = d;
            }
        }
        best.0
    };
    let mut out: BTreeMap<u64, (f64, f64)> = BTreeMap::new(); // keyed by ordered t bits
    for &(t, b, v) in &pts {
        out.insert(order_key(t), (t, v * base / b));
    }
    let sv: Vec<(f64, f64)> = out.into_values().collect();
    if sv.iter().all(|&(_, m)| (m - 1.0).abs() < 1e-3) {
        Vec::new()
    } else {
        sv
    }
}

/// Total order of finite doubles as u64 (for sorted maps keyed by time).
fn order_key(x: f64) -> u64 {
    let x = if x == 0.0 { 0.0 } else { x }; // -0.0 == 0.0 as a dict key
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn line_endings() {
        assert_eq!(super::lines("a\r\nb\rc\n\nd\r\r\ne\n"), ["a", "b", "c", "", "d", "", "e"]);
        assert_eq!(super::lines("a\r"), ["a", ""]);
        assert_eq!(super::lines(""), [""]);
    }

    #[test]
    fn stacked_notes_are_sure_misses() {
        // column 0: a tap, the same tap twice more and a hold on it; column 1: one hold
        let osu = "osu file format v14\n[General]\nMode: 3\n[Difficulty]\nCircleSize:4\nOverallDifficulty:8\n[HitObjects]\n\
                   64,192,1000,1,0,0:0:0:0:\n64,192,1000,1,0,0:0:0:0:\n64,192,1000,1,0,0:0:0:0:\n64,192,1000,128,0,1500:0:0:0:0:\n\
                   192,192,1000,128,0,1500:0:0:0:0:\n";
        let c = super::parse(osu.as_bytes()).unwrap();
        assert_eq!((c.t.len(), c.stacked), (2, 4.));
    }
}

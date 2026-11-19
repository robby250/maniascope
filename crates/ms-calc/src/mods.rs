//! What each osu! mod does to the chart a player meets: the one table of acronyms (`plan`) and the one place a chart is
//! changed (`Mods::apply`, called by `chart::parse`). A modded chart is its file plus one `[Difficulty]` line
//! (`ManiaScopeMods:<tag>`, written by `tagged`): its own bytes and hashes, so every reader of charts (features, the
//! per-note forecast, a calculator update's re-rating) gets the chart as played without knowing about mods.
//! Rules from ppy/osu (osu.Game.Rulesets.Mania/Mods, ModTimeRamp, ManiaHitWindows), read 2026-11-19.
use crate::chart::Chart;
use serde_json::Value;

/// Hard Rock divides every hit window by this, Easy multiplies (ManiaModHardRock.HIT_WINDOW_DIFFICULTY_MULTIPLIER).
const WINDOW_STEP: f64 = 1.4;
/// A time ramp reaches its final rate this far through the chart (ModTimeRamp.FINAL_RATE_PROGRESS).
const RAMP_END: f64 = 0.75;

#[derive(Clone, Debug, PartialEq)]
pub struct Mods {
    /// DT/HT's constant rate; None when the rate follows the player (Adaptive Speed).
    pub rate: Option<f64>,
    /// Wind Up / Wind Down: (initial, final) rate.
    pub ramp: Option<(f64, f64)>,
    pub invert: bool,
    pub hold_off: bool,
    pub constant_speed: bool,
    pub no_release: bool,
    /// Random's seed. Some(None): Random is on and the game has not drawn its seed yet.
    pub random: Option<Option<i32>>,
    /// Difficulty Adjust's accuracy slider.
    pub od: Option<f64>,
    /// Hit windows × this: 1/1.4 Hard Rock, 1.4 Easy.
    pub window: f64,
    /// Mods the forecast is shown without: nothing here models them.
    pub without: Vec<String>,
    /// Whether a play with these mods is the player's own result on the chart `apply` builds.
    pub tracked: bool,
    /// Whether osu! ranks a score with them (osu! wiki: custom speeds and any other setting are unranked; adjust_pitch
    /// alone keeps a score ranked).
    pub ranked: bool,
}

impl Default for Mods {
    fn default() -> Mods {
        Mods { rate: Some(1.), ramp: None, invert: false, hold_off: false, constant_speed: false, no_release: false, random: None, od: None, window: 1.,
               without: Vec::new(), tracked: true, ranked: true }
    }
}

const RANKED: [&str; 13] = ["NF", "EZ", "HD", "FI", "FL", "MR", "SD", "PF", "HT", "DC", "DT", "NC", "CL"];
/// Mods that leave the chart and its judgement as they are. Classic only narrows the Perfect window (16 ms flat
/// against 22.4–13.9 by OD: at most ~.15 pp of accuracy); Mirror is the same chart for two hands.
const PLAIN: [&str; 13] = ["NF", "SD", "PF", "AC", "HD", "FI", "FL", "CO", "MU", "CL", "MR", "SV2", "TD"];

/// A lazer mod list (`[{"acronym": "DT", "settings": {"speed_change": 1.2}}, …]`, acronyms alone too) as what it does.
pub fn plan(mods: &[Value]) -> Mods {
    let mut m = Mods::default();
    for entry in mods {
        let acronym = entry.get("acronym").or(Some(entry)).and_then(Value::as_str).unwrap_or("").to_uppercase();
        let setting = |k: &str| entry.get("settings").and_then(|s| s.get(k));
        let number = |k: &str| setting(k).and_then(Value::as_f64);
        let plain_settings = entry.get("settings").and_then(Value::as_object).is_none_or(|s| s.keys().all(|k| k == "adjust_pitch"));
        m.ranked &= RANKED.contains(&acronym.as_str()) && plain_settings;
        match acronym.as_str() {
            "DT" | "NC" | "HT" | "DC" => {
                let default = if matches!(acronym.as_str(), "DT" | "NC") { 1.5 } else { 0.75 };
                let rate = number("speed_change").unwrap_or(default);
                m.ranked &= entry.get("settings").and_then(Value::as_object).is_none_or(|s| rate == default && s.keys().all(|k| k == "speed_change" || k == "adjust_pitch"));
                m.rate = Some(rate);
            }
            "WU" => m.ramp = Some((number("initial_rate").unwrap_or(1.), number("final_rate").unwrap_or(1.5))),
            "WD" => m.ramp = Some((number("initial_rate").unwrap_or(1.), number("final_rate").unwrap_or(0.75))),
            "IN" => m.invert = true,
            "HO" => m.hold_off = true,
            "CS" => m.constant_speed = true,
            "NR" => m.no_release = true,
            "RD" => m.random = Some(setting("seed").and_then(Value::as_i64).map(|s| s as i32)),
            "DA" => m.od = number("overall_difficulty"),
            "HR" => m.window = 1. / WINDOW_STEP,
            "EZ" => m.window = WINDOW_STEP,
            a if PLAIN.contains(&a) => {}
            // Auto and Cinema play by themselves; Adaptive Speed's rate follows the hits; the rest is unknown here
            a => {
                m.tracked = false;
                if a == "AS" {
                    m.rate = None;
                }
                if !a.is_empty() {
                    m.without.push(a.to_string());
                }
            }
        }
    }
    if m.random == Some(None) {
        m.without.push("RD".to_string()); // before the play: the shuffle is not known
    }
    m
}

impl Mods {
    /// The chart-changing part, in one fixed order: "" for a chart played as written.
    pub fn tag(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.invert {
            parts.push("IN".into());
        }
        if self.hold_off {
            parts.push("HO".into());
        }
        if self.constant_speed {
            parts.push("CS".into());
        }
        if self.no_release {
            parts.push("NR".into());
        }
        if let Some(Some(seed)) = self.random {
            parts.push(format!("RD{seed}"));
        }
        if let Some((a, b)) = self.ramp {
            parts.push(format!("TR{a:.2}-{b:.2}"));
        }
        if let Some(od) = self.od {
            parts.push(format!("OD{od:.1}"));
        }
        if self.window < 1. {
            parts.push("HR".into());
        } else if self.window > 1. {
            parts.push("EZ".into());
        }
        parts.join("+")
    }

    /// `tag`'s inverse; an unknown part is skipped.
    pub fn from_tag(tag: &str) -> Mods {
        let mut m = Mods::default();
        for part in tag.split('+') {
            match part {
                "IN" => m.invert = true,
                "HO" => m.hold_off = true,
                "CS" => m.constant_speed = true,
                "NR" => m.no_release = true,
                "HR" => m.window = 1. / WINDOW_STEP,
                "EZ" => m.window = WINDOW_STEP,
                p if p.starts_with("RD") => m.random = p[2..].parse().ok().map(Some),
                p if p.starts_with("TR") => m.ramp = p[2..].split_once('-').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))),
                p if p.starts_with("OD") => m.od = p[2..].parse().ok(),
                _ => {}
            }
        }
        m
    }

    /// The mods as the card lists them ("IN", "DA OD 9.5", "WU 1.00→1.50×", "RD").
    pub fn labels(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (on, name) in [(self.invert, "IN"), (self.hold_off, "HO"), (self.constant_speed, "CS"), (self.no_release, "NR"), (self.random.is_some_and(|s| s.is_some()), "RD")] {
            if on {
                out.push(name.into());
            }
        }
        if let Some((a, b)) = self.ramp {
            out.push(format!("{} {a:.2}→{b:.2}×", if b >= a { "WU" } else { "WD" }));
        }
        if let Some(od) = self.od {
            out.push(format!("DA OD {od:.1}"));
        }
        if self.window != 1. {
            out.push(if self.window < 1. { "HR" } else { "EZ" }.into());
        }
        out
    }

    /// The chart as played. `beats`: (ms, beat length) of the uninherited timing points, in file order.
    pub fn apply(&self, ch: &mut Chart, beats: &[(f64, f64)]) {
        if self.hold_off {
            ch.end.clone_from(&ch.t);
        }
        if self.invert {
            invert(ch, beats);
        }
        if let Some(Some(seed)) = self.random {
            let to = shuffled_columns(seed, ch.keys);
            ch.col.iter_mut().for_each(|c| *c = to[*c as usize] as u8);
        }
        let mut window = self.window;
        if let Some((from, to)) = self.ramp {
            window /= ramp(ch, from, to); // a time ramp leaves the windows in chart time: in real time they shrink as it speeds up
        }
        if self.constant_speed {
            ch.sv.clear();
        }
        if self.od.is_some() || window != 1. {
            // every reader takes the great window as 64 − 3·OD ms: the mods' window travels as the OD that gives it
            let (od, own) = (self.od.unwrap_or(ch.od.clamp(0., 10.)), ch.great_window());
            ch.od = (64. - (64. - 3. * od) * window) / 3.;
            ch.od_free = true;
            ch.window_shift = (ch.great_window() / own).ln();
        }
        ch.no_release = self.no_release;
        sort(ch);
    }
}

/// How ln(1 − accuracy) moves with ln(GREAT window ratio) x when only the windows change: A·x + B·x². Measured by judging
/// 789 own lazer replays again under windows × 0.71 … 1.4 and OD −6 … +3 (`calib window_law`, 2026-11-19: rms 0.167; a
/// tighter window costs more than a wider one gives; left per keymode as a slope: 4K −0.37, 7K −0.14, 10K +0.42, one
/// player). The forecast had no window term for mods: its fitted window weight is 0 (the rating carries what the public
/// scores show of OD), so Hard Rock, Easy and Difficulty Adjust moved it only through the rating's mash and release terms.
pub const WINDOW_LAW: [f64; 2] = [-1.2937, 0.8765];

/// The change of ln(1 − accuracy) a window shift (`Chart::window_shift`) brings; flat past the parabola's vertex.
pub fn window_law(shift: f64) -> f64 {
    let [a, b] = WINDOW_LAW;
    let x = shift.min(-a / (2. * b));
    a * x + b * x * x
}

/// `data` with the mods' tag in its `[Difficulty]` section: the modded chart's own file.
pub fn tagged(data: &[u8], tag: &str) -> Vec<u8> {
    const SECTION: &[u8] = b"[Difficulty]";
    let line = format!("ManiaScopeMods:{tag}");
    let at = data.windows(SECTION.len()).position(|w| w == SECTION).and_then(|p| data[p..].iter().position(|&c| c == b'\n').map(|n| p + n + 1));
    let mut out = Vec::with_capacity(data.len() + line.len() + 16);
    match at {
        Some(at) => {
            out.extend_from_slice(&data[..at]);
            out.extend_from_slice(line.as_bytes());
            out.extend_from_slice(if data[..at].ends_with(b"\r\n") { b"\r\n" } else { b"\n" });
            out.extend_from_slice(&data[at..]);
        }
        None => {
            out.extend_from_slice(data);
            out.extend_from_slice(format!("\n[Difficulty]\n{line}\n").as_bytes());
        }
    }
    out
}

fn sort(ch: &mut Chart) {
    let mut order: Vec<usize> = (0..ch.t.len()).collect();
    order.sort_by(|&a, &b| ch.t[a].total_cmp(&ch.t[b]).then(ch.col[a].cmp(&ch.col[b])).then(ch.end[a].total_cmp(&ch.end[b])));
    (ch.t, ch.end, ch.col) = (order.iter().map(|&i| ch.t[i]).collect(), order.iter().map(|&i| ch.end[i]).collect(), order.iter().map(|&i| ch.col[i]).collect());
}

/// ManiaModInvert: in each column every note start becomes a hold that lasts to the next start, shortened by a quarter
/// beat (never below half the gap); the column's last note has nothing to hold to and goes.
fn invert(ch: &mut Chart, beats: &[(f64, f64)]) {
    let beat_at = |t: f64| beats.iter().rev().find(|b| b.0 <= t).or(beats.first()).map_or(1000., |b| b.1); // lazer's default timing point: 1000 ms
    let (mut t, mut end, mut col) = (Vec::new(), Vec::new(), Vec::new());
    for c in 0..ch.keys as u8 {
        let starts: Vec<f64> = (0..ch.t.len()).filter(|&i| ch.col[i] == c).map(|i| ch.t[i]).collect(); // in time order: the chart is sorted
        for pair in starts.windows(2) {
            let gap = pair[1] - pair[0];
            t.push(pair[0]);
            end.push(pair[0] + (gap / 2.).max(gap - beat_at(pair[1]) / 4.));
            col.push(c);
        }
    }
    (ch.t, ch.end, ch.col) = (t, end, col);
}

/// ModTimeRamp: the rate runs from `from` to `to` over the first three quarters of the chart, in steps of 0.01. Every
/// time becomes the real time it is played at; → the mean rate over the notes.
fn ramp(ch: &mut Chart, from: f64, to: f64) -> f64 {
    let (Some(&first), last) = (ch.t.first(), ch.last_end()) else { return 1. };
    let span = (RAMP_END * (last - first)).max(1.);
    let rate = |t: f64| ((from + (to - from) * ((t - first) / span).clamp(0., 1.)) * 100.).round() / 100.;
    // real time of a chart time: 1 ms steps of chart time at the rate in force (the rate moves by 0.01 at a time)
    let mut real = Vec::with_capacity((last - first) as usize + 2);
    let mut at = 0f64;
    let mut ms = first;
    while ms <= last + 1. {
        real.push(at);
        at += 1. / rate(ms).max(0.01);
        ms += 1.;
    }
    let map = |t: f64| {
        let x = (t - first).max(0.);
        let i = (x as usize).min(real.len() - 2);
        if t < first { (t - first) / rate(first).max(0.01) } else { real[i] + (x - i as f64) * (real[i + 1] - real[i]) }
    };
    let mean = ch.t.iter().map(|&t| rate(t)).sum::<f64>() / ch.t.len() as f64;
    ch.t.iter_mut().chain(ch.end.iter_mut()).for_each(|t| *t = map(*t));
    ch.sv.iter_mut().for_each(|p| p.0 = map(p.0));
    mean
}

/// ManiaModRandom: `Enumerable.Range(0, keys).OrderBy(_ => rng.Next())` with .NET's seeded generator; a note in column c
/// moves to column `[c]`. The replay tool rebuilt 2,345 own Random replays to their saved judgement counts with it.
pub fn shuffled_columns(seed: i32, keys: usize) -> Vec<usize> {
    let mut rng = DotNetRandom::new(seed);
    let mut keyed: Vec<(i32, usize)> = (0..keys).map(|c| (rng.next(), c)).collect();
    keyed.sort_by_key(|k| k.0); // stable, as OrderBy
    keyed.into_iter().map(|k| k.1).collect()
}

/// `new System.Random(seed)`: Knuth's subtractive generator, which .NET keeps for seeded instances.
struct DotNetRandom {
    seeds: [i32; 56],
    next: usize,
    nextp: usize,
}

impl DotNetRandom {
    fn new(seed: i32) -> DotNetRandom {
        let mut seeds = [0i32; 56];
        let mut mj = 161_803_398 - if seed == i32::MIN { i32::MAX } else { seed.abs() };
        seeds[55] = mj;
        let (mut mk, mut ii) = (1i32, 0usize);
        for _ in 1..55 {
            ii += 21;
            if ii >= 55 {
                ii -= 55;
            }
            seeds[ii] = mk;
            mk = mj - mk;
            if mk < 0 {
                mk += i32::MAX;
            }
            mj = seeds[ii];
        }
        for _ in 1..5 {
            for i in 1..56 {
                let mut n = i + 30;
                if n >= 55 {
                    n -= 55;
                }
                seeds[i] = seeds[i].wrapping_sub(seeds[1 + n]);
                if seeds[i] < 0 {
                    seeds[i] += i32::MAX;
                }
            }
        }
        DotNetRandom { seeds, next: 0, nextp: 21 }
    }

    fn next(&mut self) -> i32 {
        let step = |i: usize| if i + 1 >= 56 { 1 } else { i + 1 };
        (self.next, self.nextp) = (step(self.next), step(self.nextp));
        let mut v = self.seeds[self.next] - self.seeds[self.nextp];
        if v == i32::MAX {
            v -= 1;
        }
        if v < 0 {
            v += i32::MAX;
        }
        self.seeds[self.next] = v;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chart(keys: usize, notes: &[(f64, f64, u8)]) -> Chart {
        Chart { keys, t: notes.iter().map(|n| n.0).collect(), end: notes.iter().map(|n| n.1).collect(), col: notes.iter().map(|n| n.2).collect(), od: 8., ..Default::default() }
    }

    #[test]
    fn the_seeded_generator_is_dotnets() {
        let mut r = DotNetRandom::new(42);
        assert_eq!([r.next(), r.next(), r.next()], [1_434_747_710, 302_596_119, 269_548_474]);
        let mut r = DotNetRandom::new(0);
        assert_eq!([r.next(), r.next()], [1_559_595_546, 1_755_192_844]);
        let to = shuffled_columns(42, 7);
        let mut seen = to.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..7).collect::<Vec<_>>(), "a permutation");
        assert_ne!(to, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn every_acronym_has_one_outcome_and_none_refuses_a_forecast() {
        let of = |v: Value| plan(v.as_array().unwrap());
        assert_eq!(of(json!([])), Mods::default());
        assert_eq!(of(json!([{"acronym": "DT", "settings": {"speed_change": 1.2}}])).rate, Some(1.2));
        assert!(of(json!([{"acronym": "DT", "settings": {"adjust_pitch": true}}])).ranked && !of(json!([{"acronym": "DT", "settings": {"speed_change": 1.2}}])).ranked);
        assert_eq!((of(json!([{"acronym": "HT"}, {"acronym": "NF"}])).rate, of(json!([{"acronym": "HT"}, {"acronym": "NF"}])).ranked), (Some(0.75), true));
        let auto = of(json!([{"acronym": "AT"}, {"acronym": "IN"}]));
        assert!(!auto.tracked && auto.invert && auto.without == ["AT"] && auto.rate == Some(1.), "Auto: the forecast without it, never the player's play");
        assert!(!of(json!([{"acronym": "CN"}])).tracked && !of(json!([{"acronym": "ZZ"}])).tracked);
        assert_eq!(of(json!([{"acronym": "AS"}])).rate, None);
        for plain in PLAIN {
            let m = of(json!([{"acronym": plain}]));
            assert!(m.tracked && m.tag().is_empty() && m.without.is_empty(), "{plain}");
        }
        // every mania mod lazer lists (ms-osumem's CATEGORIES): part of the chart, plain, or named as left out
        for a in ["EZ", "NF", "HT", "DC", "NR", "HR", "SD", "PF", "DT", "NC", "FI", "HD", "CO", "FL", "AC", "RD", "DS", "MR", "DA", "CL", "IN", "CS", "HO", "1K", "2K", "3K", "4K", "5K", "6K", "7K",
                  "8K", "9K", "10K", "AT", "CN", "WU", "WD", "MU", "AS", "TD", "SV2"] {
            let m = of(json!([{"acronym": a, "settings": {"seed": 7, "overall_difficulty": 9.0}}]));
            let (changes, plain, left_out) = (!m.tag().is_empty() || m.rate.is_some_and(|r| r != 1.), PLAIN.contains(&a), m.without == [a]);
            assert_eq!(u8::from(changes) + u8::from(plain) + u8::from(left_out), 1, "{a}: {m:?}");
            assert_eq!(m.tracked, !left_out, "{a}");
        }
        let wu = of(json!([{"acronym": "WU"}]));
        assert_eq!((wu.ramp, wu.rate, wu.ranked), (Some((1., 1.5)), Some(1.), false));
        assert_eq!(of(json!([{"acronym": "WD", "settings": {"final_rate": 0.6}}])).ramp, Some((1., 0.6)));
        let rd = of(json!([{"acronym": "RD", "settings": {"seed": 1846047439}}]));
        assert_eq!((rd.random, rd.tag().as_str(), rd.tracked), (Some(Some(1846047439)), "RD1846047439", true));
        let unseeded = of(json!([{"acronym": "RD"}]));
        assert!(unseeded.tag().is_empty() && unseeded.without == ["RD"] && unseeded.tracked);
        let all = of(json!([{"acronym": "IN"}, {"acronym": "CS"}, {"acronym": "DA", "settings": {"overall_difficulty": 9.5}}, {"acronym": "HR"}, {"acronym": "WU"}]));
        assert_eq!(all.tag(), "IN+CS+TR1.00-1.50+OD9.5+HR");
        let back = Mods::from_tag(&all.tag());
        assert_eq!((back.invert, back.constant_speed, back.ramp, back.od, back.window < 1.), (true, true, Some((1., 1.5)), Some(9.5), true));
        assert_eq!(back.tag(), all.tag());
        assert_eq!(of(json!(["HO", "NR"])).tag(), "HO+NR", "bare acronyms too");
    }

    #[test]
    fn hold_off_keeps_each_head_and_invert_holds_to_the_next_note() {
        let mut ho = chart(4, &[(0., 400., 0), (100., 100., 1), (500., 900., 0)]);
        Mods { hold_off: true, ..Default::default() }.apply(&mut ho, &[]);
        assert_eq!((ho.t.clone(), ho.end.clone()), (vec![0., 100., 500.], vec![0., 100., 500.]));
        // column 0: notes at 0, 1000, 1200 on a 500 ms beat → holds 0–875 (gap − beat/4) and 1000–1100 (half the gap: the
        // quarter beat would leave less); the last note goes. Column 1 has one note: nothing left of it.
        let mut inv = chart(4, &[(0., 300., 0), (1000., 1000., 0), (1100., 1100., 1), (1200., 1200., 0)]);
        Mods { invert: true, ..Default::default() }.apply(&mut inv, &[(0., 500.)]);
        assert_eq!((inv.t.clone(), inv.end.clone(), inv.col.clone()), (vec![0., 1000.], vec![875., 1100.], vec![0, 0]));
    }

    #[test]
    fn random_moves_whole_columns_and_a_ramp_plays_later_notes_closer() {
        let mut rd = chart(7, &[(0., 0., 0), (100., 100., 3), (200., 200., 6)]);
        Mods { random: Some(Some(42)), ..Default::default() }.apply(&mut rd, &[]);
        let to = shuffled_columns(42, 7);
        let mut want = vec![(0f64, to[0] as u8), (100., to[3] as u8), (200., to[6] as u8)];
        want.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(rd.t.iter().copied().zip(rd.col.iter().copied()).collect::<Vec<_>>(), want);
        // 1.0 → 1.5 over the first 75 %: notes 1 s apart in chart time end up 1/1.5 s apart, the windows shrink with it
        let notes: Vec<(f64, f64, u8)> = (0..=100).map(|i| (i as f64 * 1000., i as f64 * 1000., 0)).collect();
        let mut wu = chart(4, &notes);
        Mods { ramp: Some((1., 1.5)), ..Default::default() }.apply(&mut wu, &[]);
        let gap = |i: usize| wu.t[i + 1] - wu.t[i];
        assert!((gap(0) - 1000.).abs() < 5. && (gap(99) - 1000. / 1.5).abs() < 1., "{} {}", gap(0), gap(99));
        assert!(gap(40) < gap(10) && wu.great_window() < 0.040 && wu.great_window() > 0.040 / 1.5);
        let mut wd = chart(4, &notes);
        Mods { ramp: Some((1., 0.75)), ..Default::default() }.apply(&mut wd, &[]);
        assert!((wd.t[100] - wd.t[99] - 1000. / 0.75).abs() < 1. && wd.great_window() > 0.040);
    }

    #[test]
    fn a_tagged_file_parses_as_the_modded_chart() {
        let mut osu = "osu file format v14\r\n\r\n[General]\r\nMode: 3\r\n\r\n[Difficulty]\r\nCircleSize:4\r\nOverallDifficulty:8\r\n\r\n[TimingPoints]\r\n0,500,4,1,0,60,1,0\r\n500,-50,4,1,0,60,0,0\r\n\r\n[HitObjects]\r\n".to_string();
        for i in 0..40 {
            osu.push_str(&format!("{},192,{},128,0,{}:0:0:0:0:\r\n", 64 + 128 * (i % 4), i * 250, i * 250 + 200));
        }
        let plain = crate::chart::parse(osu.as_bytes()).unwrap();
        assert!(!plain.sv.is_empty() && !plain.od_free && !plain.no_release && plain.end[0] > plain.t[0]);
        let file = tagged(osu.as_bytes(), "HO+CS+NR+OD9.5+HR");
        assert!(String::from_utf8_lossy(&file).contains("[Difficulty]\r\nManiaScopeMods:HO+CS+NR+OD9.5+HR\r\nCircleSize:4"));
        let modded = crate::chart::parse(&file).unwrap();
        assert!(modded.sv.is_empty() && modded.no_release && modded.end == modded.t);
        assert!((modded.great_window() - (64. - 3. * 9.5) / 1000. / 1.4).abs() < 1e-12);
        assert!((modded.window_shift - (64f64 - 3. * 9.5).ln() + ((64f64 - 3. * 8.).ln() + 1.4f64.ln())).abs() < 1e-12 && plain.window_shift == 0., "{}", modded.window_shift);
        // Hard Rock costs more accuracy than Easy gives; a plain chart's forecast is untouched
        let (hard, easy) = (window_law(-(1.4f64.ln())), window_law(1.4f64.ln()));
        assert!((hard - 0.5345).abs() < 1e-3 && (easy + 0.3361).abs() < 1e-3 && window_law(0.) == 0. && window_law(5.) == window_law(0.738), "{hard} {easy}");
        assert_eq!(crate::chart::parse(&tagged(b"osu file format v14\n[General]\nMode: 3\n[HitObjects]\n", "IN")).err().map(|_| ()), Some(()), "no key count: still an error, not a panic");
    }
}

//! The card (plan 2026-11-12, 1.9.0's reading on the new calc): one row per family, rated by its strongest member (the family
//! keeps its whole demand, its types are layered on it), named by its strongest type when that type reaches NAME × the
//! row's rating, else by the family's plain name. Mash and Stamina are rows of their own; Technical is a tag only. A row
//! shows at ≥ SHOW × the top row (the top always). Tags, on a row ≥ TAG_ROW × the top row: Tech (the row's demand under the
//! technical modifier keeps TAG of its rating, fitted by `calib labels --fit`) unless the name says Tech; Stamina (the row
//! keeps STAMINA_SUSTAIN of its rating sustained), which stands in for the Stamina row; the Stamina row is the strongest
//! row's sustained rating. Each tag once per card, on the highest row that qualifies; LN General never carries Tech.
use crate::demand::{Parts, P};
use crate::params::Knobs;
use crate::py;
use crate::skills::{rating, stamina_rating, Judged, Series, Sk};
use crate::taxonomy;
use crate::val::{Dict, Val};

pub fn display_name(s: Sk) -> &'static str {
    taxonomy::of(s).name
}

pub fn fitted(key: &str) -> f64 {
    let v: serde_json::Value = serde_json::from_str(include_str!("../../../calib/taxonomy.json")).expect("calib/taxonomy.json");
    v["card"][key].as_f64().unwrap_or_else(|| panic!("calib/taxonomy.json: card.{key}"))
}

/// TAG from calib/taxonomy.json.
pub fn fitted_tag() -> f64 {
    static C: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *C.get_or_init(|| fitted("tag"))
}

/// TAG at a chart's shown stars: `base` at `tag_from`, moving by `tag_rise` × ln(stars / tag_from) (stars under 1 read as 1),
/// so Tech shows on about the same share of charts at every level (user 2026-11-21: "on the hardest maps (10*+) basically
/// every map shows the tech tag", "make it show up on about 20% throughout"). Fitted by `calib labels --tech-levels`.
pub fn tag_at(base: f64, stars: f64) -> f64 {
    static C: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    let (from, rise) = *C.get_or_init(|| (fitted("tag_from"), fitted("tag_rise")));
    tag_curve(base, from, rise, stars)
}

/// `tag_at` with explicit knobs.
pub fn tag_curve(base: f64, from: f64, rise: f64, stars: f64) -> f64 {
    base + rise * (stars.max(1.0) / from).ln()
}

/// NAME from calib/taxonomy.json: a type names its family's row from this share of the row's rating.
pub fn fitted_name() -> f64 {
    static C: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *C.get_or_init(|| fitted("name"))
}

/// A type's own factor on NAME (calib/taxonomy.json card.name_scale; 1 for a type not listed).
fn name_scale(sk: Sk) -> f64 {
    static C: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    let v = C.get_or_init(|| serde_json::from_str::<serde_json::Value>(include_str!("../../../calib/taxonomy.json")).expect("calib/taxonomy.json")["card"]["name_scale"].clone());
    v[sk.name()].as_f64().unwrap_or(1.0)
}

/// Most entries a card holds: the main window's "Skills shown" goes to 6 (user 2026-11-02).
pub const CARD_MAX: usize = 6;
/// A row shows from this share of the top row (1.9.0, user 2026-11-12: "i kinda want the old way").
pub const SHOW: f64 = 0.3;
/// A tag goes on a row this share of the top row or more (1.9.0's ARCH_MATERIAL, its bar for Tech and Stamina names)…
pub const TAG_ROW: f64 = 0.6;
/// …that keeps this share of its rating sustained (1.9.0).
pub const STAMINA_SUSTAIN: f64 = 0.9;
/// The description joins the second row from this share of the first (1.9.0).
pub const DESCRIBE_PAIR: f64 = 0.85;

/// The tags a row can carry.
pub const TAGS: [&str; 2] = ["Tech", "Stamina"];
/// SV that lifts the rating by MEMORIZE_LIFT × √(the chart without it) reads as one to learn, whatever its trajectory (user
/// 2026-11-15: maps rated much higher than their no-SV version are memorization; Slowjam named one with every kind of SV).
/// Between a fixed lift and a ratio (user 2026-11-16): a 1.5× ratio let Carry me away's same SV pass on 4K (+3.34 over 5.47)
/// and fail on 7K (+2.96 over 6.27). Labelled: 7K 1.18 and the six VIPs 1.43–2.35 are; Check this out 1.13 is not.
pub const MEMORIZE_LIFT: f64 = 1.15;
pub const MEMORIZATION_SV: &str = "Memorization SV";

pub fn memorized(res: &Dict) -> bool {
    let (with, without) = (res.dict("scores").map_or(0.0, |s| s.fget("overall", 0.0)), res.fget("no_sv_overall", 0.0));
    without > 0.0 && with - without >= MEMORIZE_LIFT * without.sqrt()
}

/// A card candidate: (row skill that rates it, its name, rating).
type Cand = (Sk, &'static str, f64);

/// One candidate per family on the keymode (Technical aside: a tag; Stamina: `card_with` rates it from the rows), Mash as its
/// own, strongest first.
pub fn ranked(res: &Dict, name_at: f64) -> Vec<Cand> {
    let sc = res.dict("scores").unwrap();
    let keys = res.f("keys") as usize;
    let on = |r: &&taxonomy::Row| taxonomy::measured(r.sk, keys) && taxonomy::shown(r.sk) && sc.f(r.sk.name()) > 0.0;
    let mut out: Vec<Cand> = Vec::new();
    for &(key, label, _, _) in taxonomy::FAMILIES.iter().filter(|f| !matches!(f.0, "technical" | "stamina")) {
        let rows: Vec<&taxonomy::Row> = taxonomy::family(key).filter(on).collect();
        let Some(lead) = rows.iter().max_by(|a, b| sc.f(a.sk.name()).total_cmp(&sc.f(b.sk.name()))) else { continue };
        // the whole Stream family where compute rated it (5K+)
        let r = if key == "stream" && sc.f(STREAM_FAMILY) > 0.0 {
            sc.f(STREAM_FAMILY)
        } else if key == "chordjack" && sc.f(CHORDJACK_CARD) > 0.0 {
            sc.f(CHORDJACK_CARD)
        } else {
            sc.f(lead.sk.name())
        };
        let typed = rows.iter().filter(|x| x.child.is_some()).max_by(|a, b| sc.f(a.sk.name()).total_cmp(&sc.f(b.sk.name())));
        let plain = rows.iter().find(|x| x.child.is_none()).map_or(label, |x| x.name);
        let name = typed.filter(|t| sc.f(t.sk.name()) >= name_at * name_scale(t.sk) * r).map_or(plain, |t| t.name);
        let name = if key == "sv" && memorized(res) { MEMORIZATION_SV } else { name };
        out.push((lead.sk, name, r));
    }
    if taxonomy::measured(Sk::mash, keys) && sc.f("mash") > 0.0 {
        out.push((Sk::mash, display_name(Sk::mash), sc.f("mash")));
    }
    out.sort_by(|a, b| py::round_digits(b.2, 6).total_cmp(&py::round_digits(a.2, 6)));
    out
}

/// The score key of the card's Chordjack row: the chordjack rating × CHORDJACK_CARD (compute.rs). User 2026-11-20: Chordjack
/// led fewer cards after the jack-technique change (v1.16.0: hardest 7K 31 → 24 of 303); raised past v1.15.0's level.
pub const CHORDJACK_CARD: &str = "chordjack_card";

/// The score key of the whole Stream family on keymodes that split it (compute.rs).
pub const STREAM_FAMILY: &str = "stream_family";

/// The fast and slow Stream halves together, per bin.
pub fn stream_family(series: &Series) -> Vec<f64> {
    series.get(Sk::stream).iter().zip(series.get(Sk::delay)).map(|(a, b)| a + b).collect()
}

/// Per family row (its leading skill), each tag's share: the row's rating with its demand read under the tag (Tech:
/// min(row, technical); Stamina: the sustained rating) over the row's own rating.
pub fn tags(series: &Series, sc: &Dict, judged: &Judged, keys: usize, k: &Knobs) -> Dict {
    let min_series = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x.min(*y)).collect() };
    let mut probe = Dict::new();
    probe.set("keys", keys);
    probe.set("scores", Val::Dict(sc.clone()));
    let mut out = Dict::new();
    for (s, _, own) in ranked(&probe, 1.0).into_iter().take(CARD_MAX) {
        if own <= 0.0 || !series.has(s) || s == Sk::stamina || s == Sk::mash {
            continue;
        }
        let whole = (sc.f(STREAM_FAMILY) > 0.0 && taxonomy::of(s).family == "stream").then(|| stream_family(series));
        let row = whole.as_deref().unwrap_or(series.get(s));
        // tag shares against the row's own rating: the whole Stream family's, the unscaled chordjack rating
        let own = if whole.is_some() { rating(row, judged, keys, k) } else if s == Sk::chordjack && sc.f(CHORDJACK_CARD) > 0.0 { sc.f("chordjack") } else { own };
        let mut d = Dict::new();
        d.set("Tech", rating(&min_series(row, series.get(Sk::technical)), judged, keys, k) / own);
        d.set("Stamina", stamina_rating(row, judged, keys, k) / own);
        out.set(s.name(), d);
    }
    out
}

pub fn card(res: &Dict) -> Vec<Val> {
    card_top(res, 3)
}

/// The card with up to `n` rows. The stored `unit_card` is taken before the unit conversion: rebuilding from the converted
/// result dropped entries.
pub fn card_top(res: &Dict, n: usize) -> Vec<Val> {
    card_base(res, n, fitted_tag(), fitted_name())
}

/// The chart's stars as the player sees them (osu!'s scale, `anchor`): the Tech curve is set in those (user 2026-11-21:
/// "i meant 10* on the stars that i see").
pub fn shown_stars(res: &Dict) -> f64 {
    crate::anchor::stars(res.f("keys") as i64, res.dict("scores").map_or(0.0, |s| s.f("overall")))
}

/// The card with TAG `base` at `tag_from` shown stars, rising above (`tag_at`).
pub fn card_base(res: &Dict, n: usize, base: f64, name_at: f64) -> Vec<Val> {
    let stars = shown_stars(res);
    card_with(res, n, tag_at(base, stars), name_at)
}

/// `card_top` with an explicit TAG (at this chart's stars) and NAME (their fits, `calib labels --fit`).
pub fn card_with(res: &Dict, n: usize, tag: f64, name_at: f64) -> Vec<Val> {
    if let Some(Val::List(uc)) = res.get("unit_card") {
        return uc.iter().take(n).cloned().collect();
    }
    let sc = res.dict("scores").unwrap();
    if sc.f("overall") < 0.1 {
        return Vec::new();
    }
    let shares = res.dict("tags");
    let share = |s: Sk, m: &str| shares.and_then(|t| t.dict(s.name())).map_or(0.0, |o| o.fget(m, 0.0));
    let mut rows = ranked(res, name_at);
    // The Stamina row: the strongest row's sustained rating, on the rows' own scale (user 2026-11-12: the chart's whole
    // sustained demand put Stamina over every row of a long mixed chart, and under the rows of a dense 3:46 one).
    let sustained = rows.iter().map(|r| share(r.0, "Stamina") * r.2).fold(0.0, f64::max);
    if sustained > 0.0 {
        rows.push((Sk::stamina, display_name(Sk::stamina), sustained));
        rows.sort_by(|a, b| py::round_digits(b.2, 6).total_cmp(&py::round_digits(a.2, 6)));
    }
    let Some(top) = rows.first().map(|r| r.2) else { return Vec::new() };
    // One Tech and one Stamina per card, each on the highest row that qualifies (user 2026-11-13: "if the top skill has it
    // then the others dont"). A name saying Tech uses up nothing; LN General never carries Tech ("just hide Tech LN General").
    let (mut tech, mut stamina) = (false, false);
    let mut out: Vec<(Cand, Vec<&str>)> = rows.into_iter().filter(|r| r.2 >= SHOW * top).take(n).map(|r| {
        let mut t = Vec::new();
        let material = r.2 >= TAG_ROW * top;
        if !tech && material && share(r.0, "Tech") >= tag && !r.1.split(' ').any(|w| w == "Tech") && r.1 != taxonomy::of(Sk::ln_general).name {
            (tech, t) = (true, vec!["Tech"]);
        }
        if !stamina && material && share(r.0, "Stamina") >= STAMINA_SUSTAIN {
            stamina = true;
            t.push("Stamina");
        }
        (r, t)
    }).collect();
    // a row tagged Stamina already says it; the Stamina row would say it twice (1.9.0)
    if out.iter().any(|(_, t)| t.contains(&"Stamina")) {
        out.retain(|(r, _)| r.0 != Sk::stamina);
    }
    out.into_iter().map(|((s, name, r), t)| {
        let mut d = Dict::new();
        d.set("name", name);
        d.set("rating", r);
        d.set("parts", Val::strs(&[s.name()]));
        d.set("tags", Val::strs(&t));
        Val::Dict(d)
    }).collect()
}

/// A row in plain text: Tech before the name, Stamina after ("Tech Stream", "LN Release Stamina"; user 2026-11-12).
pub fn plain(d: &Dict) -> String {
    let tags: Vec<String> = match d.get("tags") { Some(Val::List(l) | Val::Tuple(l)) => l.iter().filter_map(|t| if let Val::Str(s) = t { Some(s.clone()) } else { None }).collect(), _ => Vec::new() };
    let n = str_of(d, "name").unwrap_or_default();
    let n = if tags.iter().any(|t| t == "Tech") { format!("Tech {n}") } else { n };
    tags.iter().filter(|t| *t != "Tech").fold(n, |n, t| format!("{n} {t}"))
}

/// The "also" line: a family off the card that still matters, at these shares of the top row (1.9.0's bars).
const ALSO: [(&str, &str, f64); 4] = [("SV", "sv", 0.4), ("LN", "ln", 0.6), ("Vibro", "vibro", 0.4), ("Mash", "mash", 0.25)];

/// (description, tags): the top row's name, joined with the second's when it is DESCRIBE_PAIR of the first; "short burst" for
/// a spike; "also X · Y" for families off the card.
pub fn describe(res: &Dict, k: &Knobs) -> (String, Vec<String>) {
    let sc = res.dict("scores").unwrap();
    let entries: Vec<Dict> = card(res).into_iter().filter_map(|e| if let Val::Dict(d) = e { Some(d) } else { None }).collect();
    let Some(first) = entries.first() else { return (String::new(), Vec::new()) };
    let mut dom = plain(first);
    if let Some(second) = entries.get(1).filter(|b| b.f("rating") >= DESCRIBE_PAIR * first.f("rating")) {
        dom = format!("{dom} / {}", plain(second));
    }
    if let Some(pd) = str_of(res, "preunit_description") {
        dom = pd;
    }
    let mut tags = Vec::new();
    let peak = res.get("timeline").map_or(0.0, |t| Val::list_f(t).into_iter().fold(0.0, f64::max));
    if peak >= k.BURST_TAG * sc.f("overall") && sc.f("overall") > 0.0 {
        tags.push("short burst".to_string());
    }
    let shown: Vec<String> = entries.iter().flat_map(parts).collect();
    tags.extend(also_line(&also_at(res, first.f("rating")), &shown));
    (dom, tags)
}

/// The families the "also" line may name, each with its member skills: the line drops a family whose member is on the
/// rows the reader sees (`also_line`), so the app filters by its own row count (user 2026-11-15: "also LN" beside LN General).
pub fn also(res: &Dict) -> Vec<(&'static str, Vec<&'static str>)> {
    card(res).first().and_then(|e| if let Val::Dict(d) = e { Some(also_at(res, d.f("rating"))) } else { None }).unwrap_or_default()
}

/// `also` against the card's top row rating `top` (the caller has the card already).
fn also_at(res: &Dict, top: f64) -> Vec<(&'static str, Vec<&'static str>)> {
    let sc = res.dict("scores").unwrap();
    let fam = |s: &str| Sk::from_name(s).map_or("", |s| if s == Sk::vibro { "vibro" } else { taxonomy::of(s).family });
    let keys = res.f("keys") as usize;
    ALSO.iter().filter_map(|(label, key, share)| {
        let members: Vec<&'static str> = taxonomy::TABLE.iter().filter(|r| fam(r.sk.name()) == *key || r.sk.name() == *key).map(|r| r.sk).filter(|s| taxonomy::measured(*s, keys)).map(|s| s.name()).collect();
        let best = members.iter().map(|s| sc.f(s)).fold(0.0, f64::max);
        (best > 0.0 && best >= share * top).then_some((*label, members))
    }).collect()
}

/// "also X · Y" for the families in `also` none of whose members is among the `shown` parts.
pub fn also_line(also: &[(&str, Vec<&str>)], shown: &[String]) -> Option<String> {
    let extra: Vec<&str> = also.iter().filter(|(_, m)| !shown.iter().any(|p| m.contains(&p.as_str()))).map(|a| a.0).collect();
    (!extra.is_empty()).then(|| format!("also {}", extra.join(" · ")))
}

/// A card row's `parts` (skill names).
pub fn parts(e: &Dict) -> Vec<String> {
    match e.get("parts") { Some(Val::List(l) | Val::Tuple(l)) => l.iter().filter_map(|x| if let Val::Str(s) = x { Some(s.clone()) } else { None }).collect(), _ => Vec::new() }
}

fn str_of(e: &Dict, k: &str) -> Option<String> {
    match e.get(k) {
        Some(Val::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

const TECH_WHY: [(P, &str); 3] = [
    (P::rodd, "rhythm: gaps off the pulse (e.g. 1/4 <-> 1/6)"),
    (P::pins, "minijacks inserted in rice"),
    (P::psw, "pattern transitions with repeated-column or irregular-hand pressure"),
];

fn window_sum(v: &[f64], i: usize, w: usize) -> f64 {
    let end = (i + w).min(v.len());
    if i >= end {
        return 0.0;
    }
    py::sum(v[i..end].iter().copied())
}

/// skill_calc._tech_where.
pub fn tech_where(series: &Series, parts: &Parts, start: f64, rate: f64, k: &Knobs) -> Vec<Val> {
    let n = 3;
    let tech = series.get(Sk::technical);
    let w = 4usize;
    let ns = 1i64.max(tech.len() as i64 - w as i64 + 1) as usize;
    let sums: Vec<f64> = (0..ns).map(|i| window_sum(tech, i, w)).collect();
    let mut order: Vec<usize> = (0..sums.len()).collect();
    order.sort_by(|&a, &b| (-sums[a]).partial_cmp(&-sums[b]).unwrap());
    let mut out = Vec::new();
    let mut taken: Vec<usize> = Vec::new();
    for i in order {
        let refv = sums[if out.is_empty() { 0 } else { taken[0] }];
        if out.len() == n || sums[i] == 0.0 || sums[i] < 0.5 * refv {
            break;
        }
        if taken.iter().any(|&j| (i as i64 - j as i64).abs() < w as i64) {
            continue;
        }
        taken.push(i);
        let pt = window_sum(series.get(Sk::patterntech), i, w);
        let rt = window_sum(series.get(Sk::rhythmtech), i, w);
        let ev: Vec<f64> = TECH_WHY.iter().map(|(p, _)| window_sum(parts.get(*p), i, w) / if *p == P::rodd { k.RT_FULL } else { k.PT_FULL }).collect();
        let denom = if pt + rt != 0.0 { pt + rt } else { 1.0 };
        let why = if ev.iter().any(|&x| x != 0.0) {
            let mut bi = 0;
            for j in 1..3 {
                if ev[j] > ev[bi] {
                    bi = j;
                }
            }
            TECH_WHY[bi].1
        } else {
            ""
        };
        let mut d = Dict::new();
        d.set("t", start + i as f64 * k.BIN * rate);
        d.set("pattern", pt / denom);
        d.set("rhythm", rt / denom);
        d.set("why", why);
        out.push(Val::Dict(d));
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The Stream family ranks by its whole rating where compute gave one, by its stronger half otherwise.
    #[test]
    fn stream_family_ranks_as_a_whole() {
        let lead = |whole: Option<f64>| {
            let (mut res, mut sc) = (Dict::new(), Dict::new());
            for (k, v) in [("stream", 5.0), ("delay", 5.0), ("chordstream", 6.0)] {
                sc.set(k, v);
            }
            if let Some(w) = whole {
                sc.set(STREAM_FAMILY, w);
            }
            res.set("keys", 7usize);
            res.set("scores", Val::Dict(sc));
            let top = ranked(&res, 1.0)[0];
            (top.1, top.2)
        };
        assert_eq!(lead(None), ("Chordstream", 6.0));
        assert_eq!(lead(Some(7.0)), ("Stream", 7.0));
    }
}

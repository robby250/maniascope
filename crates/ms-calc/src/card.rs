//! Archetypes, Technical passages, card entries and descriptions (skill_calc + archetype_names).
use crate::demand::{Parts, P};
use crate::params::Knobs;
use crate::pattern::technical_title;
use crate::py;
use crate::skills::{rating, skills, stamina_rating, Judged, Series, Sk, FIT_ONLY, LN_SKILLS, SV_KINDS, TECH_KEYS};
use crate::val::{Dict, Key, Val};

const TECH_NAMES: [(Sk, &str); 18] = [
    (Sk::stream, "Tech Stream"), (Sk::delay, "Tech Stream"), (Sk::chordstream, "Tech Chordstream"),
    (Sk::dump, "Tech Dump"), (Sk::jumptrill, "Tech Jumptrill"), (Sk::splittrill, "Tech Split Trill"),
    (Sk::jumpstream, "Tech Jumpstream"), (Sk::handstream, "Tech Handstream"), (Sk::jack, "Tech Jacks"),
    (Sk::jackspeed, "Tech Jacks"), (Sk::chordjack, "Tech Chordjack"), (Sk::minijack, "Tech Minijack"),
    (Sk::longjack, "Tech Longjack"), (Sk::bracket, "Tech Bracket"),
    (Sk::ln_inverse, "Tech Inverse"), (Sk::ln_release, "Tech LN Release"), (Sk::ln, "Tech LN"),
    (Sk::trill1h, "Tech 1H Trill"),
];
const TECH_ANCHOR: (Sk, &str) = (Sk::anchor, "Tech Anchor");
const STAMINA_NAMES: [(Sk, &str); 18] = [
    (Sk::stream, "Stream Stamina"), (Sk::delay, "Stream Stamina"), (Sk::chordstream, "Chordstream Stam"),
    (Sk::dump, "Dump Stamina"), (Sk::jumptrill, "JT Stamina"), (Sk::splittrill, "Split Stamina"),
    (Sk::jumpstream, "JS Stamina"), (Sk::handstream, "HS Stamina"), (Sk::jack, "Jack Stamina"),
    (Sk::jackspeed, "Jack Stamina"), (Sk::longjack, "Jack Stamina"), (Sk::chordjack, "CJ Stamina"),
    (Sk::bracket, "Bracket Stamina"), (Sk::ln_release, "LN Stamina"), (Sk::ln_inverse, "LN Stamina"),
    (Sk::ln_tech, "LN Stamina"), (Sk::ln_general, "LN Stamina"), (Sk::ln, "LN Stamina"),
];
const TECH_CONTROL: &str = "Tech Control";
/// Skills with types: (type skill, family, type). Every family reads "Family · Type" on the card, the family's number when
/// the family entry leads, the type's own when the type does (user 2026-11-09: one naming for LN, Jacks, Trills and SV).
const TYPES: [(&str, &str, &str); 11] = [
    ("ln_general", "LN", "General"), ("ln_tech", "LN", "Tech"), ("ln_inverse", "LN", "Inverse"), ("ln_release", "LN", "Release"),
    ("shield", "LN", "Shields"), ("minijack", "Jacks", "Minijack"), ("longjack", "Jacks", "Longjack"), ("vibro", "Jacks", "Vibro"),
    ("jumptrill", "Trills", "Jumptrill"), ("trill1h", "Trills", "1H"), ("splittrill", "Trills", "Split"),
];
const SV_TYPES: [(&str, &str); 5] = [("Fast SV", "Fast"), ("Slowjam SV", "Slowjam"), ("Accel SV", "Accel"), ("Stutter SV", "Stutters"), ("Brakes", "Brakes")];

/// The family entries a type joins: the family's own skills.
fn family_parts(family: &str) -> &'static [&'static str] {
    match family {
        "LN" => &["ln"],
        "Jacks" => &["jackspeed", "jack"],
        _ => &[],
    }
}
const REFINES: [(Sk, &[Sk]); 9] = [
    (Sk::bracket, &[Sk::chordstream]),
    (Sk::quadstream, &[Sk::handstream, Sk::minijack]),
    (Sk::longjack, &[Sk::jack, Sk::jackspeed, Sk::chordjack]),
    (Sk::vibro, &[Sk::jack, Sk::jackspeed, Sk::chordjack, Sk::longjack]),
    (Sk::ln_general, &[Sk::ln]),
    (Sk::ln_tech, &[Sk::ln]),
    (Sk::ln_inverse, &[Sk::ln]),
    (Sk::ln_release, &[Sk::ln]),
    (Sk::shield, &[Sk::ln]),
];
fn describes(s: &str) -> &'static [&'static str] {
    match s {
        "dump" => &["stream"],
        "trill1h" | "anchor" => &["stream", "delay"],
        // Both show as "Stream" (user 2026-11-02: one Stream family), so one card entry covers the other.
        "stream" => &["delay"],
        "delay" => &["stream"],
        _ => &[],
    }
}

pub fn display_name(s: Sk) -> &'static str {
    match s {
        Sk::stream => "Stream", Sk::jumpstream => "Jumpstream", Sk::handstream => "Handstream",
        Sk::quadstream => "Quadstream", Sk::sv_fast => "Fast SV", Sk::sv_slow => "Slowjam SV",
        Sk::sv_accel => "Accel SV", Sk::sv_stutter => "Stutter SV", Sk::sv_brake => "Brakes",
        Sk::jackspeed => "Jackspeed", Sk::chordjack => "Chordjack", Sk::delay => "Stream",
        Sk::chordstream => "Chordstream", Sk::bracket => "Bracket", Sk::jack => "Jack", Sk::dump => "Dump",
        Sk::jumptrill => "Jumptrill", Sk::splittrill => "Split Trill", Sk::technical => "Technical",
        Sk::patterntech => "Pattern Tech", Sk::rhythmtech => "Rhythm Tech", Sk::trill1h => "1H Trill",
        Sk::anchor => "Anchor", Sk::ln => "LN", Sk::sv => "SV", Sk::stamina => "Stamina", Sk::minijack => "Minijack",
        Sk::longjack => "Longjack", Sk::ln_release => "LN Release", Sk::ln_inverse => "Inverse", Sk::ln_general => "LN General",
        Sk::ln_tech => "LN Tech", Sk::shield => "Shield", Sk::release | Sk::inverse | Sk::hybrid => "", Sk::vibro => "Vibro", Sk::mash => "Mash", Sk::overall | Sk::switch => "",
    }
}

fn arch(name: &str, rating: f64, parts: &[&str], of: Option<&str>) -> Val {
    let mut d = Dict::new();
    d.set("name", name);
    d.set("rating", rating);
    d.set("parts", Val::strs(parts));
    if let Some(o) = of {
        d.set("of", o);
    }
    Val::Dict(d)
}

/// Scores dict lookup by skill.
fn s(sc: &Dict, k: Sk) -> f64 {
    sc.f(k.name())
}

fn min_series(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x.min(*y)).collect()
}

/// skill_calc._archetypes.
pub fn archetypes(series: &Series, sc: &Dict, judged: &Judged, keys: usize, switch: &[f64], k: &Knobs) -> Vec<Val> {
    let local = |s: &[f64]| rating(s, judged, keys, k);
    let mut out: Vec<Val> = Vec::new();
    let mut top = f64::NEG_INFINITY;
    for (key, v) in &sc.0 {
        if let Key::S(key) = key {
            if let Some(sk) = Sk::from_name(key) {
                if series.has(sk) && !TECH_KEYS.contains(&sk) && sk != Sk::ln && v.f() > top {
                    top = v.f();
                }
            }
        }
    }
    if series.has(Sk::quadstream) {
        let (q, mj) = (s(sc, Sk::quadstream), s(sc, Sk::minijack));
        if q >= 0.8 * mj && 0.8 * mj > 0.0 && q >= k.ARCH_MATERIAL * s(sc, Sk::overall) {
            out.push(arch("Quadstream", mj, &["quadstream", "minijack"], None));
        }
    }
    let tech = series.get(Sk::technical);
    let mut best = 0.0f64;
    for (key, name) in TECH_NAMES.iter().chain(std::iter::once(&TECH_ANCHOR)) {
        let sk = s(sc, *key);
        // min(skill, technical) rates no higher than either (rating rises with the series): below the bar, no search
        if !series.has(*key) || sk < k.ARCH_MATERIAL * top || sk.min(s(sc, Sk::technical)) < (k.ARCH_TECH * sk).max(0.80 * s(sc, Sk::overall)) {
            continue;
        }
        let r = local(&min_series(series.get(*key), tech));
        if r >= k.ARCH_TECH * sk && r >= 0.80 * s(sc, Sk::overall) {
            let whole = r >= k.TECH_WHOLE * sk;
            let rating = if whole { sk } else { r };
            if whole {
                out.push(arch(name, rating, &[key.name(), "technical"], Some(key.name())));
            } else {
                out.push(arch(name, rating, &["technical"], Some(key.name())));
            }
            best = best.max(rating);
        }
    }
    if !switch.is_empty() && s(sc, Sk::technical) >= k.ARCH_MATERIAL * top {
        let r = local(&min_series(switch, tech));
        if r >= k.ARCH_MATERIAL * top && best < k.REFINE_Q * r {
            out.push(arch(TECH_CONTROL, r, &["technical", "patterntech"], None));
        }
    }
    for (key, parents) in REFINES {
        if !series.has(key) {
            continue;
        }
        let sk = s(sc, key);
        let ps: Vec<Sk> = parents.iter().copied().filter(|&p| series.has(p) && sk >= k.REFINE_Q * s(sc, p) && k.REFINE_Q * s(sc, p) > 0.0).collect();
        if !ps.is_empty() {
            let mut bp = ps[0];
            for &p in &ps[1..] {
                if s(sc, p) > s(sc, bp) {
                    bp = p;
                }
            }
            out.push(arch(display_name(key), sk, &[key.name(), bp.name()], None));
        }
    }
    let teched: Vec<String> = out
        .iter()
        .filter_map(|a| {
            let Val::Dict(d) = a else { return None };
            let Some(Val::Str(of)) = d.get("of") else { return None };
            let Some(Val::Tuple(parts)) = d.get("parts") else { return None };
            parts.iter().any(|p| matches!(p, Val::Str(x) if x == of)).then(|| of.clone())
        })
        .collect();
    for (key, name) in STAMINA_NAMES {
        if !series.has(key) || s(sc, key) < k.ARCH_MATERIAL * top || teched.iter().any(|t| t == key.name()) {
            continue;
        }
        let sk = s(sc, key); // already rating(series.get(key)) (compute::series_dict_scores)
        // Sustained: the skill keeps its rating when its demand is averaged over minutes. Stamina is named after the skill it
        // sustains, never shown alone when one does (user 2026-11-08: "if the stamina is in the same section where there are
        // brackets then its bracket stamina").
        if sk > 0.0 && stamina_rating(series.get(key), judged, keys, k) >= k.STAMINA_SUSTAIN * sk {
            out.push(arch(name, sk, &[key.name(), "stamina"], None));
        }
    }
    out
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

fn parts_of(e: &Dict) -> Vec<String> {
    match e.get("parts") {
        Some(Val::Tuple(v)) | Some(Val::List(v)) => v.iter().map(|x| if let Val::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => Vec::new(),
    }
}
fn str_of(e: &Dict, k: &str) -> Option<String> {
    match e.get(k) {
        Some(Val::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

fn names_for(res: &Dict) -> impl Fn(&str) -> String + '_ {
    move |k: &str| {
        if k == "sv" {
            str_of(res, "sv_name").unwrap_or_else(|| "SV".into())
        } else {
            Sk::from_name(k).map(|s| display_name(s).to_string()).unwrap_or_default()
        }
    }
}

/// The keymode's skills, strongest first (skill_calc.ranked).
pub fn ranked(res: &Dict) -> Vec<Sk> {
    let sc = res.dict("scores").unwrap();
    let mut v = skills(res.f("keys") as usize);
    v.retain(|s| !FIT_ONLY.contains(s));
    v.sort_by(|a, b| (-py::round_digits(s(sc, *a), 2)).partial_cmp(&-py::round_digits(s(sc, *b), 2)).unwrap());
    v
}

fn tech_profile_of(res: &Dict) -> Option<[f64; 3]> {
    res.dict("tech_profile").map(|p| [p.f("technical"), p.f("patterntech"), p.f("rhythmtech")])
}

/// skill_calc.card.
pub fn card(res: &Dict) -> Vec<Val> {
    card_top(res, 3)
}

/// Most entries a card holds: the main window's "Skills shown" goes to 6 (user 2026-11-02).
pub const CARD_MAX: usize = 6;

/// The card with up to `n` entries. Picking is greedy, so the first 3 of a longer card are the 3-entry card. The stored
/// `unit_card` is taken before the unit conversion: rebuilding from the converted result dropped entries.
pub fn card_top(res: &Dict, n: usize) -> Vec<Val> {
    if let Some(Val::List(uc)) = res.get("unit_card") {
        return uc.iter().take(n).cloned().collect();
    }
    let sc = res.dict("scores").unwrap();
    if sc.f("overall") < 0.1 {
        return Vec::new();
    }
    let technical_sk = py::round_digits(sc.f("technical") / 0.05f64.max(sc.f("overall")), 4);
    let profile = tech_profile_of(res);
    let names = names_for(res);
    let scf = |p: &str| sc.f(p);
    // (score, order, entry)
    let mut cands: Vec<(f64, u8, Dict)> = Vec::new();
    if let Some(Val::List(archs)) = res.get("archetypes") {
        for a in archs {
            let Val::Dict(a) = a else { continue };
            let name = str_of(a, "name").unwrap();
            if name.starts_with("Tech ") && technical_title(&name, technical_sk, profile.as_ref()) != name {
                continue;
            }
            let parts = parts_of(a);
            let mut mx = a.f("rating");
            for p in &parts {
                if !TECH_KEYS.iter().any(|t| t.name() == p) {
                    let v = scf(p);
                    if v > mx {
                        mx = v;
                    }
                }
            }
            let of = str_of(a, "of");
            let first = of.as_deref().is_some_and(|o| matches!(o, "dump" | "trill1h" | "anchor"))
                || parts.iter().any(|p| p == "ln") && parts.iter().any(|p| LN_SKILLS.iter().chain([&Sk::shield]).any(|l| l.name() == p));
            cands.push((mx, if first { 0 } else { 1 }, a.clone()));
        }
    }
    for kk in ranked(res) {
        let v = s(sc, kk);
        if TECH_KEYS.contains(&kk) || SV_KINDS.contains(&kk) || v <= 0.0 {
            continue;
        }
        let mut d = Dict::new();
        d.set("name", names(kk.name()));
        d.set("rating", v);
        d.set("parts", Val::strs(&[kk.name()]));
        cands.push((v, 2, d));
    }
    cands.sort_by(|a, b| (-py::round_digits(a.0, 6), a.1).partial_cmp(&(-py::round_digits(b.0, 6), b.1)).unwrap());
    let mut picked: Vec<Dict> = Vec::new();
    let mut used: Vec<String> = Vec::new();
    // A family entry is named after its strongest type, even when another entry ("LN Stamina") already took it (a player
    // 2026-11-08: Azimuth, Release pack, read "LN · Tech" at 0.75× with Release 10.6 > Tech 7.1).
    // The strongest type by rating, never "Mix"; a specialty within GENERAL_YIELD of General names the chart (a player
    // 2026-11-10: Chinese Pop LN Release sets read "LN · Mix"/General; Release labels 27→36/39, dan packs 35→33/48:
    // General 7–8 read Release, as General 11–14 already did).
    const GENERAL_YIELD: f64 = 0.75;
    let weight = |key: &str| scf(key) * if key == "ln_general" { GENERAL_YIELD } else { 1. };
    let strongest = |family: &str| TYPES.iter().filter(|t| t.1 == family && scf(t.0) > 0.).max_by(|a, b| weight(a.0).total_cmp(&weight(b.0)))
        .map(|t| t.2);
    for (_, _, e) in cands {
        if picked.len() == n || (!picked.is_empty() && e.f("rating") < 0.3 * picked[0].f("rating")) {
            break;
        }
        let parts = parts_of(&e);
        let of = str_of(&e, "of");
        if parts.iter().any(|p| used.contains(p)) || of.as_ref().is_some_and(|o| used.contains(o)) {
            continue;
        }
        // A type is a slice of its family's rating, not a second skill (a player 2026-11-06: "LN 6.1 · LN General 4.6" read as two
        // skills): with the family entry on the card, the strongest type names it ("LN · General 6.1") and the rest stay off.
        if let [p] = parts.as_slice() {
            if let Some(&(_, family, _)) = TYPES.iter().find(|t| t.0 == p) {
                // Another type of the family on the card is its head too: one "LN · X" row, never two (a player 2026-11-10: "LN ·
                // General, LN · Mix" under "LN Stamina", which had taken the plain LN entry).
                let of_family = |q: &str| family_parts(family).contains(&q) || TYPES.iter().any(|t| t.0 == q && t.1 == family);
                // The family's stamina row is its head as well ("LN · Release Stamina", user 2026-11-10: stamina after the type).
                if let Some(head) = picked.iter_mut().find(|d| matches!(parts_of(d).as_slice(), [q] | [q, _] if of_family(q) && parts_of(d).get(1).is_none_or(|x| x == "stamina"))) {
                    let name = str_of(head, "name").unwrap();
                    if let (false, false, Some(t)) = (name.contains(" · "), name.ends_with("Stamina"), strongest(family)) {
                        head.set("name", format!("{family} · {t}"));
                    }
                    continue;
                }
            }
        }
        let mut d = e.clone();
        d.pop("of");
        picked.push(d);
        for p in &parts {
            if !used.contains(p) {
                used.push(p.clone());
            }
        }
        for p in parts.iter().map(|x| x.as_str()).chain(std::iter::once(of.as_deref().unwrap_or(""))) {
            for m in describes(p) {
                if !used.iter().any(|u| u == m) {
                    used.push(m.to_string());
                }
            }
        }
    }
    let lnv = sc.f("ln");
    let key = |e: &Dict| {
        let parts = parts_of(e);
        let lnk = parts.iter().any(|p| p == "ln") && parts.iter().any(|p| LN_SKILLS.iter().chain([&Sk::shield]).any(|l| l.name() == p));
        -e.f("rating").max(if lnk { lnv } else { 0.0 })
    };
    picked.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    if let Some(first) = picked.first_mut() {
        let name = str_of(first, "name").unwrap();
        let t = technical_title(&name, technical_sk, profile.as_ref());
        if t != name {
            first.set("name", t);
            let mut parts = parts_of(first);
            if !parts.iter().any(|p| p == "technical") {
                parts.push("technical".into());
            }
            first.set("parts", Val::Tuple(parts.into_iter().map(Val::Str).collect()));
        }
    }
    // A type leading its own entry reads "Family · Type" too, and SV kinds "SV · Kind". No "LN Chordstream" texture renames:
    // LN and the pattern show as separate entries (user 2026-11-08).
    for d in &mut picked {
        let name = str_of(d, "name").unwrap();
        let first = parts_of(d).into_iter().next().unwrap_or_default();
        // Stamina of a typed family reads after the type: "LN · Release Stamina", one row for LN.
        // A type's own stamina keeps its type; the family's ("ln", "jack") takes the strongest.
        let own = TYPES.iter().find(|t| t.0 == first).map(|t| (t.1, Some(t.2)));
        let family = own.or_else(|| TYPES.iter().find(|t| family_parts(t.1).contains(&first.as_str())).map(|t| (t.1, strongest(t.1))));
        if let (true, Some((family, Some(t)))) = (name.ends_with("Stamina"), family) {
            d.set("name", format!("{family} · {t} Stamina"));
            continue;
        }
        if name.contains(" · ") || name.ends_with("Stamina") || name.starts_with("Tech ") {
            continue;
        }
        if let Some(&(_, family, t)) = TYPES.iter().find(|x| x.0 == first) {
            d.set("name", format!("{family} · {}", strongest(family).unwrap_or(t)));
        } else if let Some(&(_, kind)) = SV_TYPES.iter().find(|x| x.0 == name).filter(|_| first == "sv") {
            d.set("name", format!("SV · {kind}"));
        }
    }
    picked.into_iter().map(Val::Dict).collect()
}

// ---- archetype_names ----


/// skill_calc.describe → (dominant text, tags).
pub fn describe(res: &Dict, k: &Knobs) -> (String, Vec<String>) {
    let sc = res.dict("scores").unwrap();
    let entries = card(res);
    if entries.is_empty() {
        return (String::new(), Vec::new());
    }
    let Val::Dict(a) = &entries[0] else { unreachable!() };
    let mut dom = str_of(a, "name").unwrap();
    if entries.len() > 1 {
        if let Val::Dict(b) = &entries[1] {
            if b.f("rating") >= 0.85 * a.f("rating") {
                dom = format!("{dom} / {}", str_of(b, "name").unwrap());
            }
        }
    }
    if let Some(pd) = str_of(res, "preunit_description") {
        dom = pd;
    }
    let mut tags = Vec::new();
    // A spike far above what the whole chart plays like.
    let peak = res.get("timeline").map_or(0.0, |t| Val::list_f(t).into_iter().fold(0.0, f64::max));
    if peak >= k.BURST_TAG * sc.f("overall") && sc.f("overall") > 0.0 {
        tags.push("short burst".to_string());
    }
    let on_card: Vec<String> = entries.iter().flat_map(|e| if let Val::Dict(d) = e { parts_of(d) } else { Vec::new() }).collect();
    let names = names_for(res);
    let val = |g: &str| sc.f(g);
    let mut extra = Vec::new();
    let groups: [(&[&str], f64); 4] = [
        (&["sv"], k.ALSO_SPECIAL),
        (&["ln_general", "ln_tech", "ln_inverse", "ln_release", "shield", "ln"], k.ALSO_SHARE),
        (&["vibro"], k.ALSO_SPECIAL),
        (&["mash"], 0.25),
    ];
    for (group, share) in groups {
        if on_card.iter().any(|p| group.contains(&p.as_str())) {
            continue;
        }
        let mut best: Option<&str> = None;
        for g in group {
            if sc.has(g) && best.is_none_or(|b| val(g) > val(b)) {
                best = Some(g);
            }
        }
        let Some(kk) = best else { continue };
        if kk == "mash"
            && res.dict("execution").map_or(0.0, |e| e.fget("mash", 0.0)) < 0.05
            && sc.f(kk) < 0.8 * sc.f("overall")
        {
            continue;
        }
        if val(kk) > 0. && val(kk) >= share * a.f("rating") {
            // Names only, like the skills on the card: the numbers are in Details (user 2026-11-09: "also LN 4.9" beside an LN
            // bar of 3.35 was the raw rating, and the card had too many numbers).
            extra.push(names(kk));
        }
    }
    if !extra.is_empty() {
        tags.push(format!("also {}", extra.join(" · ")));
    }
    (dom, tags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_names_fit_the_card() {
        let names = TECH_NAMES.iter().chain(STAMINA_NAMES.iter()).chain([TECH_ANCHOR].iter()).map(|&(_, n)| n);
        for n in names.chain([TECH_CONTROL]) {
            assert!(n.len() <= 16 && n.split_whitespace().count() <= 3, "{n}");
        }
    }
}

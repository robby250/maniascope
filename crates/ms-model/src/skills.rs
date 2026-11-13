//! skill_practice: the keymode-specific Skills taxonomy. Parent selections mean the whole family; selecting a child
//! narrows it; separate selections remain a union. Views of chart demand, not extra strain charges or fitted
//! coefficients. A selection is a list of signed keys: "!" means excluded.
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct Skill {
    pub key: String,
    pub name: &'static str,
    pub members: Vec<&'static str>,
    pub children: Vec<Skill>,
    pub tip: &'static str,
}

/// One keymode's tree from the skill table (ms_calc::taxonomy): each family label a parent whose members are its rows, each
/// row with a child key a child of its own.
fn tree_of(keys: usize) -> Vec<Skill> {
    use ms_calc::taxonomy::{family, FAMILIES};
    let on = |r: &&ms_calc::taxonomy::Row| (r.keys.0..=r.keys.1).contains(&keys) && ms_calc::taxonomy::shown(r.sk);
    FAMILIES.iter().filter_map(|&(key, name, _, tip)| {
        let rows: Vec<&ms_calc::taxonomy::Row> = family(key).filter(on).collect();
        (!rows.is_empty()).then(|| Skill {
            key: key.to_string(), name, tip, members: rows.iter().map(|r| r.sk.name()).collect(),
            children: rows.iter().filter_map(|r| Some(Skill { key: format!("{key}/{}", r.child?), name: r.name, members: vec![r.sk.name()], children: Vec::new(), tip: r.tip })).collect(),
        })
    }).collect()
}

struct Trees {
    four: Vec<Skill>,
    wide: Vec<Skill>,
    all: Vec<Skill>, // every node of both trees, 4K's order first, each key once
}

fn trees() -> &'static Trees {
    static T: OnceLock<Trees> = OnceLock::new();
    T.get_or_init(|| {
        let (four, wide) = (tree_of(4), tree_of(7));
        // A family in both trees with different children (Stream, Trills) holds the union, so a selection made on one
        // keymode keeps its meaning on the other.
        let mut all: Vec<Skill> = Vec::new();
        for tree in [&four, &wide] {
            for parent in tree {
                for node in std::iter::once(parent).chain(&parent.children) {
                    match all.iter_mut().find(|n| n.key == node.key) {
                        Some(n) => {
                            n.members.extend(node.members.iter().filter(|m| !n.members.contains(m)).copied().collect::<Vec<_>>());
                            n.children.extend(node.children.iter().filter(|c| !n.children.iter().any(|x| x.key == c.key)).cloned().collect::<Vec<_>>());
                        }
                        None => all.push(node.clone()),
                    }
                }
            }
        }
        Trees { four, wide, all }
    })
}

pub fn tree(keys: i64) -> &'static [Skill] {
    if keys == 4 { &trees().four } else { &trees().wide }
}

/// The tree's nodes, parents before their children.
pub fn dimensions(keys: i64) -> Vec<&'static Skill> {
    tree(keys).iter().flat_map(|p| std::iter::once(p).chain(&p.children)).collect()
}

pub fn node(key: &str) -> Option<&'static Skill> {
    trees().all.iter().find(|n| n.key == key)
}

pub fn name(key: &str) -> &'static str {
    node(key).map_or("", |n| n.name)
}

/// The strongest of a dimension's members.
pub fn member_value<S: AsRef<str>>(f: &ms_calc::val::Dict, members: &[S]) -> f64 {
    let sk = f.dict("sk");
    members.iter().filter_map(|m| Some(sk?.get(m.as_ref())?.f().max(0.))).fold(0f64, f64::max)
}

/// skill_practice.strengths and practice_strengths of one chart-rate: (key, raw, practice) per dimension.
fn strengths(f: &ms_calc::val::Dict) -> Vec<(&'static str, f64, f64)> {
    let keys = f.get("keys").map_or(7, |k| k.f() as i64);
    let profile = f.dict("tech_profile");
    dimensions(keys).into_iter().map(|node| {
        let raw = member_value(f, &node.members);
        // Technical practice needs actual technical character, not just an awkward local peak in a smooth chart.
        let leaf = node.key.rsplit('/').next().unwrap_or("");
        let technical = node.key.split('/').next() == Some("technical");
        let practice = profile.and_then(|p| p.get(leaf)).filter(|_| technical).map_or(raw, |v| raw.min(v.f().max(0.).sqrt()));
        (node.key.as_str(), raw, practice)
    }).collect()
}

/// A value against the chart's strongest card family (ms_calc::card: Technical and Stamina are tags there, not families), the
/// card's rule for "counts as X" (plan 2026-11-12; a population percentile per skill before).
pub fn relative(f: &ms_calc::val::Dict, value: f64) -> f64 {
    let keys = f.get("keys").map_or(7, |k| k.f() as i64);
    let top = tree(keys).iter().filter(|n| !matches!(n.key.as_str(), "technical" | "stamina")).map(|n| member_value(f, &n.members)).fold(0., f64::max);
    if top > 0. { (value / top).min(1.) } else { 0. }
}

/// Per dimension of the chart's keymode tree: (key, its share, its standing, the raw share's standing), standing = `relative`;
/// a technical dimension read at its practice value (actual technical character, not an awkward local peak); avoidance reads
/// the raw one (a material forbidden passage stays forbidden).
pub fn standing(f: &ms_calc::val::Dict) -> Vec<(&'static str, f64, f64, f64)> {
    strengths(f).into_iter().map(|(key, raw, practice)| (key, raw, relative(f, practice.min(raw)), relative(f, raw))).collect()
}

/// The standing a skill needs to count on a chart: the card's SHOW.
pub fn cut() -> f64 {
    ms_calc::card::SHOW
}

/// The dimensions a chart counts as (percentile ≥ `at`; skills::cut for "demanded").
pub fn counted_at(f: &ms_calc::val::Dict, at: f64) -> Vec<&'static str> {
    standing(f).into_iter().filter(|v| v.2 >= at).map(|v| v.0).collect()
}

/// The skills a chart counts as.
pub fn demanded(f: &ms_calc::val::Dict) -> Vec<&'static str> {
    counted_at(f, cut())
}

/// Per dimension of a tree, a chart's (practice, raw) standing (Tree order).
pub fn percentiles(f: &ms_calc::val::Dict) -> Vec<(f64, f64)> {
    standing(f).into_iter().map(|v| (v.2, v.3)).collect()
}

/// match(f)[1]: the top-level skills a chart demands.
pub fn matched_families(f: &ms_calc::val::Dict) -> Vec<&'static str> {
    demanded(f).into_iter().filter(|k| !k.contains('/')).collect()
}

/// session_skill: the session form skill of a Stats group.
pub fn session_skill(key: &str, keys: i64) -> Option<&'static str> {
    dimensions(keys).into_iter().find(|n| n.key == key).and_then(|n| n.members.first().copied())
}

/// dimensions() of 4K and of wider keymodes as [(key, members)], for FeatStore's match.
pub fn feat_trees() -> Vec<Vec<(String, Vec<String>)>> {
    [4, 7].iter().map(|&k| dimensions(k).iter().map(|n| (n.key.clone(), n.members.iter().map(|m| m.to_string()).collect())).collect()).collect()
}

/// Saved keys of older trees → their place now. Exclusions keep their "!".
const LEGACY: [(&str, &[&str]); 16] = [
    ("rice", &["stream", "jumpstream", "handstream"]),
    ("chords", &["jumpstream", "handstream", "chordstream"]),
    ("jack", &["jackspeed"]),
    ("trill", &["trills", "chordstream/bracket"]),
    ("jumpstream/jumptrill", &["trills/jumptrill"]),
    ("chordstream/jumptrill", &["trills/jumptrill"]),
    ("jumpstream/splittrill", &["trills/splittrill"]),
    // 2026-11-02 regroup: Delay merged into Stream; 1H Trill, Jumptrill and Split Trill under Trills; Bracket under Chordstream.
    ("delay", &["stream"]),
    ("delay/stream", &["stream"]),
    ("delay/anchor", &["stream/anchor"]),
    ("delay/trill1h", &["trills/trill1h"]),
    ("stream/trill1h", &["trills/trill1h"]),
    ("bracket", &["chordstream/bracket"]),
    ("jumptrill", &["trills/jumptrill"]),
    ("splittrill", &["trills/splittrill"]),
    // 2026-11-04 (a player): Hybrid LN deleted; its closest skill is LN Tech.
    ("ln/hybrid", &["ln/tech"]),
];

/// A saved key's current keys (settings keep weights and avoid levels under these too).
pub fn migrate(key: &str) -> Vec<&str> {
    LEGACY.iter().find(|l| l.0 == key).map_or_else(|| vec![key], |l| l.1.to_vec())
}

/// Canonical signed choices. Opposite child overrides survive; positive-only preferences migrate.
pub fn normalize(selected: &[String]) -> Vec<String> {
    let mut states: HashMap<String, i8> = HashMap::new();
    for token in selected {
        let (negative, key) = token.strip_prefix('!').map_or((false, token.as_str()), |k| (true, k));
        for k in migrate(key) {
            if node(k).is_some() {
                // An explicit exclusion wins conflicting restored preferences.
                let at = states.entry(k.to_string()).or_insert(1);
                *at = (*at).min(if negative { -1 } else { 1 });
            }
        }
    }
    // Inherited exceptions the same way as a live child click: keeping !parent + child would display an included
    // child while the match filter still rejected its whole family.
    for n in &trees().all {
        if let Some(&inherited) = states.get(&n.key) {
            // Only an excluded parent: an included one keeps its own members (plain Stream) beside an excluded child.
            if inherited < 0 && !n.children.is_empty() && n.children.iter().any(|c| states.get(&c.key) == Some(&-inherited)) {
                for child in &n.children {
                    states.entry(child.key.clone()).or_insert(inherited);
                }
                states.remove(&n.key);
            }
        }
    }
    trees().all.iter().filter(|n| {
        states.get(&n.key).is_some_and(|s| !n.key.contains('/') || states.get(n.key.split('/').next().unwrap()) != Some(s))
    }).map(|n| format!("{}{}", if states[&n.key] < 0 { "!" } else { "" }, n.key)).collect()
}

pub fn state(selected: &[String], key: &str) -> i8 {
    let values = normalize(selected);
    let has = |k: String| values.contains(&k);
    if has(format!("!{key}")) {
        return -1;
    }
    if has(key.to_string()) {
        return 1;
    }
    if let Some((parent, _)) = key.split_once('/') {
        return if has(format!("!{parent}")) { -1 } else if has(parent.to_string()) { 1 } else { 0 };
    }
    0
}

pub fn mixed(selected: &[String], key: &str) -> bool {
    let Some(n) = node(key).filter(|n| !n.children.is_empty()) else { return false };
    let states: BTreeSet<i8> = n.children.iter().map(|c| state(selected, &c.key)).collect();
    states.len() > 1 || !states.contains(&state(selected, key))
}

/// Parent operates on the whole family; a child can override or narrow it.
pub fn set_state(selected: &[String], key: &str, new: i8) -> Vec<String> {
    if node(key).is_none() || !(-1..=1).contains(&new) {
        return normalize(selected);
    }
    let mut values: BTreeSet<String> = normalize(selected).into_iter().collect();
    let as_list = |v: &BTreeSet<String>| v.iter().cloned().collect::<Vec<_>>();
    if let Some((parent, _)) = key.split_once('/') {
        // Expand inherited state before editing a child: neutral really means untargeted, not re-included by its parent.
        // An included family minus one type stays the family ("Stream" ✓ + "Anchors" ✕ kept plain stream; expanded, it
        // asked for no stream at all on 7K, user 2026-11-09).
        let keep = state(&as_list(&values), parent) == 1 && new == -1;
        if !keep && state(&as_list(&values), parent) != 0 {
            for child in &node(parent).unwrap().children {
                let s = state(&as_list(&values), &child.key);
                if s != 0 {
                    values.insert(format!("{}{}", if s < 0 { "!" } else { "" }, child.key));
                }
            }
            values.remove(parent);
            values.remove(&format!("!{parent}"));
        }
    } else {
        let prefix = format!("{key}/");
        values.retain(|v| !v.trim_start_matches('!').starts_with(&prefix));
    }
    values.remove(key);
    values.remove(&format!("!{key}"));
    if new != 0 {
        values.insert(format!("{}{key}", if new < 0 { "!" } else { "" }));
    }
    normalize(&as_list(&values))
}

/// none → ✓ → ✕ → none
pub fn cycle(selected: &[String], key: &str) -> Vec<String> {
    let current = if mixed(selected, key) { 0 } else { state(selected, key) };
    set_state(selected, key, match current { 0 => 1, 1 => -1, _ => 0 })
}

pub fn label(selected: &[String]) -> String {
    let keys = normalize(selected);
    let mut positive: Vec<&str> = keys.iter().filter(|k| !k.starts_with('!')).map(|k| name(k)).collect();
    let mut negative: Vec<&str> = keys.iter().filter_map(|k| k.strip_prefix('!')).map(name).collect();
    let once = |v: &mut Vec<&str>| {
        let mut seen = BTreeSet::new();
        v.retain(|x| seen.insert(*x));
    };
    once(&mut positive);
    once(&mut negative);
    let text = if positive.is_empty() { "Any".to_string() } else { positive.join(" + ") };
    if negative.is_empty() { text } else { format!("{text} · exclude {}", negative.join(", ")) }
}

/// The avoid steps (saved codes) and their names. A map is out when the avoided skill's percentile reaches the step's:
/// "A lot of it" = counts (skills::cut), "Led by it" halfway from there to the top, "Any of it" half the cut.
pub const AVOID: [(f64, &str); 3] = [(0.9, "Led by it"), (0.6, "A lot of it"), (0.3, "Any of it")];

/// The percentile an avoid step (saved code) keeps a map under.
pub fn avoid_percentile(hold: f64) -> f64 {
    let c = cut();
    if hold >= 0.9 { (1. + c) / 2. } else if hold >= 0.6 { c } else { c / 2. }
}

/// The sliders that change which maps are in the list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tuning {
    pub avoid: HashMap<String, f64>, // avoided skill: see AVOID
    pub other: f64,                  // slider of maps with no included skill; 0 keeps only included skills
}

impl Tuning {
    pub fn hold(&self, key: &str) -> f64 {
        self.avoid.get(key).copied().unwrap_or(0.6)
    }
}

/// {included skill: how-often slider, "": other maps} once something differs, else None (plain best-first list).
pub fn shares(selected: &[String], weights: &HashMap<String, f64>, other: f64) -> Option<HashMap<String, f64>> {
    let mut out: HashMap<String, f64> = normalize(selected).into_iter().filter(|k| !k.starts_with('!'))
        .map(|k| { let w = weights.get(&k).copied().unwrap_or(1.); (k, w) }).collect();
    if out.is_empty() {
        return None; // Any: nothing to weigh against
    }
    let first = out.values().next().copied();
    let differ = out.values().any(|v| Some(*v) != first);
    if other > 0. {
        out.insert(String::new(), other);
    }
    (differ || other > 0.).then_some(out)
}

pub fn wants_vibro(selected: &[String]) -> bool {
    normalize(selected).iter().any(|k| k == "jackspeed/vibro")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: &[&str]) -> Vec<String> {
        x.iter().map(|s| s.to_string()).collect()
    }

    /// One rule for every skill, the card's: a skill counts at SHOW of the chart's strongest family (Technical aside).
    #[test]
    fn a_skill_counts_against_the_charts_top_family() {
        let mut sk = ms_calc::val::Dict::new();
        for (k, v) in [("stream", 1.), ("jumptrill", 0.35), ("ln", 0.2), ("technical", 3.)] {
            sk.set(k, v);
        }
        let mut f = ms_calc::val::Dict::new();
        f.set("keys", 7i64);
        f.set("sk", ms_calc::val::Val::Dict(sk));
        let d = demanded(&f);
        assert!(d.contains(&"stream") && d.contains(&"trills") && !d.contains(&"ln"), "{d:?}");
        assert_eq!(relative(&f, 3.), 1.);
    }

    /// consts.json's fit families are the skill table's `fit` column (P3 generates them from it).
    #[test]
    fn the_fit_families_are_the_tables() {
        let c: serde_json::Value = serde_json::from_str(include_str!("consts.json")).unwrap();
        let set = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect::<BTreeSet<_>>();
        let consts: Vec<(String, BTreeSet<String>)> = c["members"].as_array().unwrap().iter().map(|m| (m[0].as_str().unwrap().to_string(), set(&m[1]))).collect();
        let table: Vec<(String, BTreeSet<String>)> = ms_calc::taxonomy::fit_groups().into_iter().map(|(k, m)| (k.to_string(), m.iter().map(|s| s.name().to_string()).collect())).collect();
        assert_eq!(consts.iter().cloned().collect::<HashMap<_, _>>(), table.into_iter().collect::<HashMap<_, _>>());
    }

    #[test]
    fn a_signed_cycle_covers_the_children_and_their_exceptions() {
        let sv = node("sv").unwrap();
        let on = cycle(&[], "sv");
        let off = cycle(&on, "sv");
        assert_eq!((on.clone(), off.clone()), (v(&["sv"]), v(&["!sv"])));
        for child in &sv.children {
            assert_eq!((state(&on, &child.key), state(&off, &child.key)), (1, -1));
        }
        assert!(cycle(&off, "sv").is_empty());
        let excepted = cycle(&v(&["ln"]), "ln/inverse");
        assert_eq!((state(&excepted, "ln/inverse"), state(&excepted, "ln/release")), (-1, 1));
        assert!(mixed(&excepted, "ln"));
        let neutral = cycle(&excepted, "ln/inverse");
        assert_eq!((state(&neutral, "ln/inverse"), state(&neutral, "ln/release")), (0, 1));
        assert_eq!(cycle(&neutral, "ln"), v(&["ln"]));
        assert!(label(&v(&["!sv"])).contains("exclude SV"));
    }

    #[test]
    fn a_family_minus_one_type_keeps_the_family() {
        let s = cycle(&cycle(&[], "stream"), "stream/anchor");
        assert_eq!(s, v(&["stream", "!stream/anchor"]));
        assert!(mixed(&s, "stream") && state(&s, "stream/anchor") == -1);
        assert_eq!(cycle(&s, "stream"), v(&["stream"]));
    }

    #[test]
    fn cycling_a_parent_and_a_child() {
        let one = cycle(&[], "ln");
        assert_eq!(one, v(&["ln"]));
        assert_eq!(cycle(&one, "ln"), v(&["!ln"]));
        assert!(cycle(&v(&["!ln"]), "ln").is_empty());
        // Python's result: a child cycled out of an included parent expands the parent, the child neutral.
        assert_eq!(cycle(&cycle(&one, "ln/release"), "ln/release"), v(&["ln/general", "ln/tech", "ln/inverse", "ln/shield"]));
    }

    #[test]
    fn legacy_and_conflicts() {
        assert_eq!(normalize(&v(&["jack"])), v(&["jackspeed"]));
        assert_eq!(normalize(&v(&["ln", "!ln"])), v(&["!ln"]));
        let n = normalize(&v(&["!ln", "ln/release"]));
        assert!(n.contains(&"ln/release".to_string()) && n.contains(&"!ln/inverse".to_string()) && !n.contains(&"!ln".to_string()));
    }

    #[test]
    fn the_trees_follow_the_model_families_and_old_keys_move_with_their_sign() {
        let parents = |k: i64| tree(k).iter().map(|n| n.name).collect::<Vec<_>>();
        assert_eq!(parents(7), ["Stream", "Chordstream", "Trills", "Jacks", "Chordjack", "LN", "Technical", "Stamina", "SV"]);
        assert_eq!(parents(4), ["Stream", "Jumpstream", "Handstream", "Trills", "Jacks", "Chordjack", "LN", "Technical", "Stamina", "SV"]);
        let children = |k: i64, p: &str| tree(k).iter().find(|n| n.key == p).unwrap().children.iter().map(|c| c.key.as_str()).collect::<Vec<_>>();
        assert_eq!(children(7, "stream"), ["stream/anchor"]);
        assert_eq!(children(7, "trills"), ["trills/jumptrill", "trills/trill1h"]);
        assert_eq!(children(4, "trills"), ["trills/jumptrill", "trills/splittrill", "trills/trill1h"]);
        for (old, new) in [("delay", "stream"), ("delay/stream", "stream"), ("delay/anchor", "stream/anchor"), ("delay/trill1h", "trills/trill1h"),
                           ("stream/trill1h", "trills/trill1h"), ("bracket", "chordstream/bracket"), ("jumptrill", "trills/jumptrill"), ("splittrill", "trills/splittrill")] {
            assert_eq!(normalize(&v(&[old])), v(&[new]), "{old}");
            assert_eq!(normalize(&v(&[&format!("!{old}")])), v(&[&format!("!{new}")]), "!{old}");
        }
    }

    #[test]
    fn ln_children_read_their_own_ln_skill_and_hybrid_moves_to_tech() {
        let ln = node("ln").unwrap();
        assert_eq!(ln.children.iter().map(|c| (c.key.as_str(), c.members[0])).collect::<Vec<_>>(),
                   [("ln/general", "ln_general"), ("ln/tech", "ln_tech"), ("ln/inverse", "ln_inverse"), ("ln/release", "ln_release"), ("ln/shield", "shield")]);
        assert_eq!(normalize(&v(&["ln/hybrid"])), v(&["ln/tech"]));
        assert_eq!(normalize(&v(&["!ln/hybrid"])), v(&["!ln/tech"]));
    }

    #[test]
    fn labels() {
        assert_eq!(label(&[]), "Any");
        assert_eq!(label(&v(&["ln", "!chordjack"])), "LN · exclude Chordjack");
    }
}

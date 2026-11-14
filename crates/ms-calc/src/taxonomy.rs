//! The one skill table (plan 2026-11-11 "one rule per thing"): every skill's name, family, role, keymode range, evidence
//! part and fit group. Every other list of skills (the calculator's skill set, card names, the Skills picker, the
//! predictor's groups, the dan series a skill reads) is derived from it.
use crate::demand::P;
use crate::skills::Sk;

/// What a skill is to the calculator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// A demand part of its own: a share of each bin's total.
    Base,
    /// A slice of these base skills' demand, by its evidence part.
    Slice(&'static [Sk]),
    /// What its base's slices leave of it (LN General: LN without a specialty).
    Rest(Sk),
    /// Cross-cutting: the same notes seen for one property (rhythm, sustain); a tag on the card, never a share.
    Modifier,
    /// Rated for the fit or the code, never shown.
    Internal,
}

/// The evidence a slice's ramp reads in each bin (skills::density), over ±skills::EVIDENCE_WIN bins.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Density {
    /// A count part per note row.
    PerRow(P),
    /// One part over another.
    Ratio(P, P),
    /// A demand part's share of the base parts.
    Share(P),
    /// That share × the chart's share of interleaving disjoint chord-group moves (compute::interleaved).
    Interleaved(P),
    /// The technical modifier's share of the demand.
    Technical,
    /// Vibro runs (execution: gestures and column runs).
    VibroRuns,
    /// Quad texture (pattern::quadstream).
    Quad,
}

pub struct Row {
    pub sk: Sk,
    /// The row's name (user 2026-11-11, decision 5).
    pub name: &'static str,
    /// Family label: the Stats group header and the picker's parent (key into FAMILIES); "" for none.
    pub family: &'static str,
    /// The picker's child key under its family ("ln/general" → "general"); None for the family's own members.
    pub child: Option<&'static str>,
    pub role: Role,
    /// Keymodes the calculator measures it on (inclusive).
    pub keys: (usize, usize),
    /// What decides a slice's share.
    pub density: Option<Density>,
    /// The population fit's family (ms-model consts.json `families`), "" for none.
    pub fit: &'static str,
    pub tip: &'static str,
}

/// Family labels in display order: (key, name, dan series kind, tip).
pub const FAMILIES: [(&str, &str, &str, &str); 11] = [
    ("stream", "Stream", "rice", "Single-note rice, slow or fast."),
    ("jumpstream", "Jumpstream", "rice", ""),
    ("handstream", "Handstream", "rice", ""),
    ("chordstream", "Chordstream", "rice", ""),
    ("trills", "Trills", "rice", ""),
    ("jackspeed", "Jacks", "rice", "Single-column repetition and short jack bursts; distinct from changing overlapping chords."),
    ("chordjack", "Chordjack", "rice", "Repeated or partly overlapping chords."),
    ("ln", "LN", "ln", ""),
    ("technical", "Technical", "rice", ""),
    ("stamina", "Stamina", "rice", "Sustained effort, not simply a long song."),
    ("sv", "SV", "rice", ""),
];

const FOUR: (usize, usize) = (1, 4);
const WIDE: (usize, usize) = (5, 99);
const ALL: (usize, usize) = (1, 99);
const JACKS: &[Sk] = &[Sk::jackspeed, Sk::jack, Sk::chordjack];
const STREAMS: &[Sk] = &[Sk::stream, Sk::delay];
/// Vibro is mashed repetition over whatever the hands were playing: it slices every base but trills and SV.
const VIBRO_OF: &[Sk] = &[Sk::stream, Sk::delay, Sk::jumpstream, Sk::handstream, Sk::chordstream, Sk::jackspeed, Sk::jack, Sk::chordjack, Sk::ln];

const fn row(sk: Sk, name: &'static str, family: &'static str, child: Option<&'static str>, role: Role, keys: (usize, usize), density: Option<Density>, fit: &'static str, tip: &'static str) -> Row {
    Row { sk, name, family, child, role, keys, density, fit, tip }
}

/// Every skill, in the calculator's order (ties on the card keep it).
pub const TABLE: [Row; 37] = {
    use Density::*;
    use Role::*;
    use Sk::*;
    [
        row(delay, "Stream", "stream", None, Base, WIDE, None, "rice", ""),
        row(stream, "Stream", "stream", None, Base, ALL, None, "rice", ""),
        row(jumpstream, "Jumpstream", "jumpstream", None, Base, FOUR, None, "rice", ""),
        row(handstream, "Handstream", "handstream", None, Base, FOUR, None, "rice", ""),
        row(chordstream, "Chordstream", "chordstream", None, Base, WIDE, None, "chords", ""),
        row(bracket, "Bracket", "chordstream", Some("bracket"), Slice(&[chordstream]), WIDE, Some(Interleaved(P::bracket)), "chords", "A hand alternating between disjoint finger groups, with a chord in at least one group."),
        row(dump, "Dump", "stream", Some("dump"), Slice(STREAMS), FOUR, Some(Share(P::dump)), "rice", "Very fast rice that still needs individual finger control."),
        row(jumptrill, "Jumptrill", "trills", Some("jumptrill"), Base, ALL, None, "trill", "Alternating hand impulses, including timing-feasible rolls."),
        row(splittrill, "Split Trill", "trills", Some("splittrill"), Base, FOUR, None, "trill", "Both hands alternate fingers independently; not ordinary Jumptrill."),
        row(jackspeed, "Jacks", "jackspeed", None, Base, FOUR, None, "jack", ""),
        row(jack, "Jacks", "jackspeed", None, Base, WIDE, None, "jack", ""),
        row(chordjack, "Chordjack", "chordjack", None, Base, ALL, None, "chordjack", ""),
        row(quadstream, "Quadstream", "handstream", Some("quadstream"), Slice(&[handstream]), FOUR, Some(Quad), "", "Single → quad → single stream texture with embedded minijacks; not every quad or chordjack."),
        row(minijack, "Minijacks", "jackspeed", Some("minijack"), Slice(JACKS), ALL, Some(PerRow(P::mj)), "jack", "Short repeated-column bursts inside other patterns."),
        row(longjack, "Longjacks", "jackspeed", Some("longjack"), Slice(JACKS), ALL, Some(PerRow(P::lj)), "jack", "Sustained single-column repetition."),
        row(vibro, "Vibro", "jackspeed", Some("vibro"), Slice(VIBRO_OF), ALL, Some(VibroRuns), "jack", "Specialist ultra-fast repetition. Only explicitly selecting Vibro includes pure vibro drills."),
        row(ln, "LN General", "ln", None, Base, ALL, None, "ln", ""),
        row(ln_general, "LN General", "ln", Some("general"), Rest(ln), ALL, None, "ln", "Holds and taps without one dominant LN specialty."),
        row(ln_tech, "LN Tech", "ln", Some("tech"), Slice(&[ln]), ALL, Some(Technical), "ln", "Irregular hold rhythms and awkward transitions inside LN."),
        row(ln_inverse, "LN Inverse", "ln", Some("inverse"), Slice(&[ln]), ALL, Some(Ratio(P::cover, P::used)), "ln", "Most columns held most of the time: the gaps are the notes."),
        row(ln_release, "LN Release", "ln", Some("release"), Slice(&[ln]), ALL, Some(Ratio(P::stag, P::lnend)), "ln", "Releasing one finger while others of the hand keep holding."),
        row(shield, "LN Shields", "ln", Some("shield"), Slice(&[ln]), ALL, Some(PerRow(P::shield)), "ln", "A tap immediately followed by a hold in the same column."),
        row(sv, "SV", "sv", None, Base, ALL, None, "sv", ""),
        row(sv_fast, "Fast SV", "sv", Some("sv_fast"), Slice(&[sv]), ALL, Some(Ratio(P::sv_fast, P::sv)), "", "Shortened readable approach time."),
        row(sv_slow, "Slowjam SV", "sv", Some("sv_slow"), Slice(&[sv]), ALL, Some(Ratio(P::sv_slow, P::sv)), "", "Compressed spacing and crowded notes at low scroll speed."),
        row(sv_accel, "Accel SV", "sv", Some("sv_accel"), Slice(&[sv]), ALL, Some(Ratio(P::sv_accel, P::sv)), "", "Increasing scroll speed during note approach."),
        row(sv_stutter, "Stutter SV", "sv", Some("sv_stutter"), Slice(&[sv]), ALL, Some(Ratio(P::sv_stutter, P::sv)), "", "Repeated meaningful speed-direction changes."),
        row(sv_brake, "SV Brakes", "sv", Some("sv_brake"), Slice(&[sv]), ALL, Some(Ratio(P::sv_brake, P::sv)), "", "Decreasing approach speed and timing uncertainty near judgement."),
        row(trill1h, "1H Trill", "trills", Some("trill1h"), Slice(STREAMS), ALL, Some(PerRow(P::oht)), "trill", "Alternation between fingers on one hand."),
        row(anchor, "Stream Anchors", "stream", Some("anchor"), Slice(STREAMS), ALL, Some(PerRow(P::anc)), "", "A repeated column embedded in flowing notes."),
        row(patterntech, "Pattern Tech", "technical", Some("patterntech"), Modifier, ALL, None, "technical", "Awkward finger control and genuinely difficult transitions."),
        row(rhythmtech, "Rhythm Tech", "technical", Some("rhythmtech"), Modifier, ALL, None, "technical", "Irregular execution timing, not merely changing global note gaps."),
        row(technical, "Technical", "technical", None, Modifier, ALL, None, "technical", ""),
        row(mash, "Mash", "", None, Modifier, ALL, None, "", "Dense chords hit as one smear rather than read note by note."),
        row(stamina, "Stamina", "stamina", None, Modifier, ALL, None, "", ""),
        row(overall, "", "", None, Internal, ALL, None, "", ""),
        row(switch, "", "", None, Internal, ALL, None, "", ""),
    ]
};

pub fn of(sk: Sk) -> &'static Row {
    TABLE.iter().find(|r| r.sk == sk).expect("every skill has a row")
}

/// The skills the calculator rates on a keymode, in table order (overall and the internal switch series aside).
pub fn rated(keys: usize) -> Vec<Sk> {
    TABLE.iter().filter(|r| (r.keys.0..=r.keys.1).contains(&keys) && !matches!(r.sk, Sk::overall | Sk::switch)).map(|r| r.sk).collect()
}

/// Every skill a feature row keeps (any keymode).
pub fn kept() -> impl Iterator<Item = Sk> {
    TABLE.iter().filter(|r| !matches!(r.sk, Sk::overall | Sk::switch)).map(|r| r.sk)
}

/// The calculator measures `sk` on this keymode.
pub fn measured(sk: Sk, keys: usize) -> bool {
    (of(sk).keys.0..=of(sk).keys.1).contains(&keys)
}

pub fn shown(sk: Sk) -> bool {
    of(sk).role != Role::Internal
}

pub fn modifier(sk: Sk) -> bool {
    of(sk).role == Role::Modifier
}

/// The rows of a family label.
pub fn family(key: &str) -> impl Iterator<Item = &'static Row> + '_ {
    TABLE.iter().filter(move |r| r.family == key)
}

/// (name, dan series kind, tip) of a family key.
pub fn family_label(key: &str) -> Option<(&'static str, &'static str, &'static str)> {
    FAMILIES.iter().find(|f| f.0 == key).map(|f| (f.1, f.2, f.3))
}

/// Slices in table order (the order of the RAMPS knob).
pub fn slices() -> impl Iterator<Item = Sk> {
    TABLE.iter().filter(|r| matches!(r.role, Role::Slice(_))).map(|r| r.sk)
}
pub const SLICES: usize = 17;

/// calib/taxonomy.json's ramps in `slices()` order (`calib ramps` writes them).
pub fn fitted_ramps() -> [f64; 4 * SLICES] {
    static R: std::sync::OnceLock<[f64; 4 * SLICES]> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../calib/taxonomy.json")).expect("calib/taxonomy.json");
        let mut out = [0.0; 4 * SLICES];
        for (i, s) in slices().enumerate() {
            let r = v["ramps"][s.name()].as_array().unwrap_or_else(|| panic!("calib/taxonomy.json: no ramp for {}", s.name()));
            for j in 0..4 {
                out[4 * i + j] = r.get(j).and_then(|x| x.as_f64()).unwrap_or(0.0);
            }
        }
        out
    })
}

/// The slices of a base skill (none for any other skill).
pub fn rows_of_base(base: Sk) -> impl Iterator<Item = Sk> {
    TABLE.iter().filter(move |r| matches!(r.role, Role::Slice(of) if of.contains(&base))).map(|r| r.sk)
}

/// The population fit's families in consts order, each with its members.
pub fn fit_groups() -> Vec<(&'static str, Vec<Sk>)> {
    let mut out: Vec<(&'static str, Vec<Sk>)> = Vec::new();
    for r in TABLE.iter().filter(|r| !r.fit.is_empty()) {
        match out.iter_mut().find(|g| g.0 == r.fit) {
            Some(g) => g.1.push(r.sk),
            None => out.push((r.fit, vec![r.sk])),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_skill_has_one_row_and_every_family_a_label() {
        for s in crate::skills::ALL_SK {
            assert_eq!(TABLE.iter().filter(|r| r.sk == s).count(), 1, "{s:?}");
        }
        assert_eq!(slices().count(), SLICES);
        for r in TABLE.iter().filter(|r| r.role != Role::Internal) {
            assert!(!r.name.is_empty() && family_label(r.family).is_some() || r.sk == Sk::mash, "{:?}", r.sk);
            if let Role::Rest(of) = r.role {
                assert_eq!(super::of(of).role, Role::Base, "{:?} rests on a non-base", r.sk);
            }
            if let Role::Slice(of) = r.role {
                assert!(of.iter().all(|b| super::of(*b).role == Role::Base), "{:?} slices a non-base", r.sk);
            }
        }
    }
}

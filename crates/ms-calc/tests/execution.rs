//! Compact Mash, honest Tech/SV attribution, archetype qualification.
mod common;
use common::*;
use ms_calc::card::card;
use ms_calc::features::standard_params;
use ms_calc::pattern::technical_title;
use ms_calc::scroll::reading;
use ms_calc::skills::skills;
use ms_calc::val::{Dict, Val};

#[test]
fn compact_pseudo_chords_are_not_continuous_delay_or_jumptrills() {
    let pseudo = rice((0..160).flat_map(|i| (0..7).map(move |j| (1000. + 180. * i as f64 + 8. * j as f64, j as u8))), 7, 5.);
    let precise = rice((0..1000).map(|i| (1000. + 35. * i as f64, (i % 7) as u8)), 7, 5.);
    let alternating = rice((0..400).flat_map(|i| {
        let cols: &[u8] = if i % 2 == 1 { &[0, 1, 2] } else { &[3, 4, 5, 6] };
        cols.iter().map(move |&c| (1000. + 85. * i as f64, c))
    }), 7, 5.);
    let p = run(&pseudo, 1.);
    assert!(execution(&p, "pseudo_mash") > 0.5);
    assert!(score(&p, "mash") > score(&p, "technical"));
    assert!(execution(&run(&precise, 1.), "mash") < 0.02);
    assert!(execution(&run(&alternating, 1.), "mash") < 0.02);
}

#[test]
fn changing_columns_and_independent_holds_still_require_control() {
    let changing = rice((0..300).flat_map(|i| {
        let cols: &[u8] = if i % 2 == 1 { &[0, 2, 4, 6] } else { &[0, 3, 4, 5] };
        cols.iter().map(move |&c| (1000. + 160. * i as f64, c))
    }), 7, 5.);
    let fixed = chart((0..200).flat_map(|i| (0..7).map(move |c| (1000. + 160. * i as f64, 1080. + 160. * i as f64, c as u8))).collect(), 7, 5.);
    let (a, b) = (run(&changing, 1.), run(&fixed, 1.));
    assert!(execution(&a, "mash") < 0.03);
    assert!(score(&a, "chordjack") > score(&a, "mash"));
    assert!(execution(&b, "mash") > 0.7);
    // Tail synchronization is evidence separate from head width, not a rule erasing independent LN mechanics.
    let unsynced = chart((0..50).flat_map(|i| (0..7).map(move |c| (1000. + 400. * i as f64, 1070. + 400. * i as f64 + 45. * c as f64, c as u8))).collect(), 7, 5.);
    assert!(execution(&run(&unsynced, 1.), "mash") < execution(&b, "mash"));
}

#[test]
fn monotone_brake_is_not_stutter_and_total_sv_strain_is_retained() {
    let mut c = rice((0..80).map(|i| (1000. + i as f64 * 400., (i % 7) as u8)), 7, 5.);
    c.sv = (0..80).flat_map(|i| {
        let t = i as f64 * 400.;
        [(650. + t, 1.), (800. + t, 0.7), (920. + t, 0.4)]
    }).collect();
    let (strain, kinds, _) = reading(&c, 1., &standard_params()).unwrap();
    let total: f64 = strain.iter().sum();
    assert!(total > 0.);
    assert!(strain.iter().zip(&kinds).map(|(s, k)| s * k[3]).sum::<f64>() < 0.1 * total);
    assert!(strain.iter().zip(&kinds).all(|(s, k)| *s <= 0. || (k.iter().sum::<f64>() - 1.).abs() < 1e-8));
}

#[test]
fn zero_length_two_note_chart_does_not_crash_the_catalogue_builder() {
    let r = run(&rice([(1000., 0), (1000., 6)], 7, 5.), 1.);
    assert_eq!((score(&r, "overall"), score(&r, "stamina")), (0., 0.));
}

fn scores(values: &[(&str, f64)]) -> Dict {
    let mut d = Dict::new();
    for s in skills(7) {
        d.set(s.name(), 0.);
    }
    d.set("overall", 0.);
    for (k, v) in values {
        d.set(k, *v);
    }
    d
}

#[test]
fn ln_release_refinement_precedes_generic_tech_ln() {
    let mut result = Dict::new();
    result.set("keys", 7i64);
    result.set("scores", scores(&[("overall", 10.), ("ln", 10.), ("ln_release", 9.3), ("delay", 9.5), ("technical", 8.), ("ln_general", 6.3), ("chordstream", 7.2)]));
    let entry = |name: &str, rating: f64, parts: &[&str]| {
        let mut d = Dict::new();
        d.set("name", name);
        d.set("rating", rating);
        d.set("parts", Val::strs(parts));
        Val::Dict(d)
    };
    let mut tech = entry("Tech LN", 10., &["ln", "technical"]);
    if let Val::Dict(d) = &mut tech {
        d.set("of", "ln");
    }
    result.set("archetypes", Val::List(vec![tech, entry("LN Release", 9.3, &["ln_release", "ln"])]));
    let Val::Dict(first) = &card(&result)[0] else { panic!() };
    assert!(first.get("name").is_some_and(|n| matches!(n, Val::Str(s) if s.starts_with("LN · Release"))));
}

#[test]
fn ln_tech_requires_the_original_prevalence_and_demand_without_promotion() {
    assert_eq!(technical_title("LN", 0.75, Some(&[0.43, 0., 0.])), "LN");
    assert_eq!(technical_title("LN", 0.55, Some(&[0.8, 0., 0.])), "LN");
    assert_eq!(technical_title("LN", 0.75, Some(&[0.8, 0., 0.])), "Tech LN");
}

#[test]
fn ln_types_share_one_row_when_ln_stamina_took_the_ln_entry() {
    // A player 2026-11-10, Chinese Pop LN Release chart: "LN Stamina · LN · General · LN · Mix". One LN row, the stamina one,
    // named by Release (within 0.75 of General) with Stamina after it.
    let mut result = Dict::new();
    result.set("keys", 7i64);
    result.set("scores", scores(&[("overall", 6.), ("ln", 6.4), ("stamina", 6.5), ("ln_general", 6.4), ("ln_release", 5.), ("ln_tech", 1.)]));
    let mut d = Dict::new();
    d.set("name", "LN Stamina");
    d.set("rating", 6.5);
    d.set("parts", Val::strs(&["ln", "stamina"]));
    result.set("archetypes", Val::List(vec![Val::Dict(d)]));
    let names: Vec<String> = card(&result).iter().filter_map(|e| match e { Val::Dict(d) => d.get("name").and_then(|n| match n { Val::Str(s) => Some(s.clone()), _ => None }), _ => None }).collect();
    let ln: Vec<&String> = names.iter().filter(|n| n.starts_with("LN")).collect();
    assert_eq!(ln, ["LN · Release Stamina"], "{names:?}");
}

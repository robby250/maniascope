//! Physical execution controls; no titles or hand-picked rating targets.
mod common;
use common::*;
use ms_calc::chart::Chart;
use ms_calc::compute::compute;
use ms_calc::features::standard_params;
use ms_calc::gestures::{analyse, s, Analysis, FEATURES};
use ms_calc::model::final_rating;
use ms_calc::params::Params;
use ms_calc::val::Dict;
use serde_json::{json, Value};

const BIN: f64 = 0.5;

fn rolls(pattern: &[u8], gap: f64, n: usize, keys: usize, od: f64) -> Chart {
    rice((0..n).map(|i| (1000. + i as f64 * gap, pattern[i % pattern.len()])), keys, od)
}

fn roll4(pattern: &[u8], gap: f64) -> Chart {
    rolls(pattern, gap, 1200, 4, 8.)
}

fn repeated(pattern: &[&[u8]], gap: f64, n: usize) -> Chart {
    rice((0..n).flat_map(|i| pattern[i % pattern.len()].iter().map(move |&c| (1000. + i as f64 * gap, c))), 4, 5.)
}

fn g(c: &Chart) -> Analysis {
    analyse(c, 1., BIN)
}

fn mean(a: &[f64]) -> f64 {
    a.iter().sum::<f64>() / a.len().max(1) as f64
}

fn rolled(c: &Chart) -> f64 {
    mean(&g(c).series[s::ROLLED])
}

fn feature(a: &Analysis, name: &str) -> f64 {
    a.features[FEATURES.iter().position(|f| *f == name).unwrap()]
}

fn notes(c: &Chart) -> Vec<(f64, f64, u8)> {
    (0..c.len()).map(|i| (c.t[i], c.end[i], c.col[i])).collect()
}

const FULL_ROLL: [&[u8]; 5] = [&[0, 1, 2, 3], &[0, 1, 3, 2], &[1, 0, 3, 2], &[0, 3, 1, 2], &[0, 1, 2, 3, 0, 1, 3, 2, 1, 0, 3, 2, 0, 3, 1, 2]];

#[test]
fn fast_variable_and_interleaved_rolls_are_jumptrill_execution() {
    for pattern in FULL_ROLL {
        let c = roll4(pattern, 20.);
        assert!(rolled(&c) > 0.7);
        let r = run(&c, 1.);
        assert!(score(&r, "jumptrill") > score(&r, "stream"));
    }
}

#[test]
fn hand_grouping_generalizes_and_respects_center_assignment() {
    for keys in 4..=10usize {
        // Free outer fingers still roll while a spare center finger is held.
        let cols: Vec<u8> = (0..keys / 2).chain((keys + 1) / 2..keys).map(|c| c as u8).collect();
        let c = rolls(&cols, 12., 1200, keys, 8.);
        let mirror = chart(notes(&c).into_iter().map(|(t, e, k)| (t, e, keys as u8 - 1 - k)).collect(), keys, c.od);
        let (a, b) = (g(&c), g(&mirror));
        assert!(mean(&a.series[s::ROLLED]) > 0.7);
        for (x, y) in a.features.iter().zip(&b.features) {
            assert!((x - y).abs() < 1e-8);
        }
    }
}

#[test]
fn slow_rice_and_timing_infeasible_groups_do_not_become_jumptrills() {
    let c = roll4(&[0, 1, 2, 3], 70.);
    assert_eq!(rolled(&c), 0.);
    assert_eq!(score(&run(&c, 1.), "jumptrill"), 0.);
    // The same note order can fit at OD0 but not OD10. The feasible interval is for Great, not Perfect/SS with all
    // fingers simultaneous.
    let (loose, strict) = (rolls(&[0, 3, 1, 2], 30., 1200, 4, 0.), rolls(&[0, 3, 1, 2], 30., 1200, 4, 10.));
    assert!(rolled(&loose) > rolled(&strict));
    assert_eq!(rolled(&strict), 0.);
}

#[test]
fn rate_is_applied_once_and_offsets_do_not_change_gestures() {
    let c = roll4(&[0, 1, 3, 2], 30.);
    let moved = |f: &dyn Fn(f64) -> f64| chart(notes(&c).into_iter().map(|(t, e, k)| (f(t), f(e), k)).collect(), 4, c.od);
    let (a, b, z) = (analyse(&c, 1.5, BIN), g(&moved(&|x| x / 1.5)), analyse(&moved(&|x| x + 91234.), 1.5, BIN));
    for i in 0..FEATURES.len() {
        assert!((a.features[i] - b.features[i]).abs() < 1e-8 && (a.features[i] - z.features[i]).abs() < 1e-8);
    }
}

#[test]
fn reversals_and_anchors_cost_control_instead_of_erasing_all_grouping() {
    let (plain, turns) = (g(&roll4(&[0, 1, 2, 3], 20.)), g(&roll4(&[0, 1, 2, 3, 1, 0, 3, 2], 20.)));
    assert!(feature(&turns, "roll_control") > feature(&plain, "roll_control") + 0.1);
    assert!(feature(&turns, "roll_anchor") > feature(&plain, "roll_anchor") + 0.05);
    assert!(rolled(&roll4(&[0, 1, 2, 3, 3, 2, 1, 0], 20.)) < mean(&plain.series[s::ROLLED]));
    assert!(rolled(&roll4(&[0, 1, 0, 2, 0, 3], 20.)) < 0.1);
}

#[test]
fn literal_jumptrill_split_trill_and_vibro_remain_different() {
    let (jt, split) = (run(&repeated(&[&[0, 1], &[2, 3]], 60., 400), 1.), run(&repeated(&[&[0, 2], &[1, 3]], 60., 400), 1.));
    assert!(score(&jt, "jumptrill") > 0.9 * score(&jt, "overall") && score(&jt, "vibro") == 0.);
    assert!(score(&split, "splittrill") > 0.9 * score(&split, "overall"));
    assert!(score(&split, "jumptrill") == 0. && score(&split, "vibro") == 0.);
}

const CHANGING: [&[u8]; 4] = [&[0, 1, 2], &[0, 1, 3], &[0, 1, 2, 3], &[1, 2, 3]];

#[test]
fn changing_chords_can_still_be_vibro_but_stability_is_not_control() {
    let (stable, changing) = (g(&repeated(&[&[0, 1, 2, 3]], 80., 400)), g(&repeated(&CHANGING, 80., 400)));
    assert!(feature(&stable, "vibro_easy") > 2. * feature(&changing, "vibro_easy"));
    assert!(feature(&changing, "vibro_control") > feature(&stable, "vibro_control") + 0.2);
    assert!(mean(&changing.series[s::VIBRO]) > 0.05);
    assert_eq!(mean(&g(&repeated(&[&[0, 1, 2, 3]], 170., 400)).series[s::VIBRO]), 0.);
}

#[test]
fn free_fingers_can_roll_under_a_hold_without_collapsing_the_hold() {
    let c = rolls(&[0, 1, 2, 6, 5, 4], 18., 1200, 7, 8.);
    let mut held = notes(&c);
    held.push((1000., 22600., 3));
    let r = run(&chart(held, 7, 8.), 1.);
    assert!(score(&r, "ln") > 0. && score(&r, "jumptrill") > score(&r, "delay"));
    let all_holds = chart(notes(&c).into_iter().map(|(t, _, k)| (t, t + 15., k)).collect(), 7, 8.);
    assert_eq!(rolled(&all_holds), 0.);
}

#[test]
fn sustain_is_not_reset_by_one_anchor_and_recovers_on_real_breaks() {
    let c = rolls(&[0, 1, 2, 3], 20., 2400, 4, 8.);
    let interrupted = chart(notes(&c).into_iter().enumerate().map(|(i, (t, e, k))| {
        let pause = if i >= 1200 { 10000. } else { 0. };
        (t + pause, e + pause, k)
    }).collect(), 4, 8.);
    assert!(feature(&g(&c), "jumptrill_sustain") > feature(&g(&interrupted), "jumptrill_sustain"));
    assert!(feature(&g(&repeated(&[&[0, 1, 2, 3]], 80., 800)), "vibro_sustain") > feature(&g(&repeated(&[&[0, 1, 2, 3]], 80., 40)), "vibro_sustain"));
}

fn bursts(gap_between: f64) -> Chart {
    rice((0..60).flat_map(move |b| (0..5).flat_map(move |i| (0..4).map(move |k| (1000. + b as f64 * gap_between + i as f64 * 80., k)))), 4, 5.)
}

#[test]
fn vibro_support_does_not_jump_when_a_slower_gap_crosses_the_speed_gate() {
    let c = bursts(434.);
    let (a, b) = (analyse(&c, 1.0259, BIN), analyse(&c, 1.0261, BIN));
    for name in ["vibro_easy", "vibro_control", "vibro_sustain"] {
        assert!((feature(&a, name) - feature(&b, name)).abs() < 0.01);
    }
}

#[test]
fn unfitted_execution_features_cannot_silently_discount_a_chart() {
    let mut ex = Dict::new();
    for f in FEATURES {
        ex.set(f, 1.);
    }
    assert_eq!(final_rating(&Params::new(json!({}), &Value::Null).unwrap(), 4, 9., Some(&ex)), 9.);
}

#[test]
fn long_stable_repetition_is_distinct_from_short_fixed_chords_inside_cj() {
    let long = g(&repeated(&[&[0, 1, 2, 3]], 80., 500));
    let short = g(&rice((0..50).flat_map(|b| (0..5).flat_map(move |i| (0..4).map(move |k| (1000. + b as f64 * 2000. + i as f64 * 80., k)))), 4, 5.));
    let changing = g(&repeated(&CHANGING, 80., 400));
    assert!(feature(&long, "vibro_locked") > 0.5 && feature(&short, "vibro_locked") < 0.01);
    assert!(feature(&changing, "vibro_locked") == 0. && feature(&changing, "vibro_control") > 0.2);
}

#[test]
fn degenerate_input_is_explicit() {
    assert!(g(&chart(Vec::new(), 4, 5.)).features.iter().all(|f| *f == 0.));
    assert_eq!(feature(&g(&rice([(1000., 0), (1000., 1)], 4, 5.)), "vibro_easy"), 0.);
}

#[test]
fn sv_and_constant_scroll_share_the_final_difficulty() {
    let mut c = rolls(&[0, 1, 3, 2], 27., 1600, 4, 8.);
    c.sv = vec![(0., 1.), (4000., 0.4), (5000., 1.8), (6000., 1.)];
    let prm = standard_params();
    let (full, constant) = (compute(&c, 1., true, &prm).unwrap(), compute(&c, 1., false, &prm).unwrap());
    assert!((full.f("no_sv_overall") - score(&constant, "overall")).abs() < 1e-9);
    assert!(score(&full, "overall") >= score(&constant, "overall") - 1e-9);
    let rating = final_rating(&prm, 4, full.f("raw_overall"), full.dict("execution"));
    assert!((rating / score(&full, "overall") - 1.).abs() < 1e-12);
}

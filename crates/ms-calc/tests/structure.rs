//! Chart structure, endurance and technical presence.
mod common;
use common::*;
use ms_calc::chart::Chart;
use ms_calc::features::standard_params;
use ms_calc::val::Val;


#[test]
fn density_is_rate_aware_and_vibro_needs_one_repeated_hand() {
    use ms_calc::structure::structure;
    let k = &standard_params().k;
    let s = |c: &Chart| structure(c, 1., k);
    let plain = chart(vec![(0., 0., 0), (0., 500., 1), (1000., 3000., 2)], 7, 8.);
    let st = s(&plain);
    assert_eq!((st.f("head_span"), st.f("play_span"), st.f("nps")), (1., 3., 3.));
    let vibro = rice((0..200).map(|i| (40. * i as f64, 2)), 7, 8.);
    assert!(s(&vibro).get("vibro_focused").is_some_and(Val::truthy));
    let masks: [&[u8]; 4] = [&[0, 1, 2], &[3, 4, 5], &[1, 2, 6], &[0, 3, 5]];
    let notes = |gap: f64| (0..400).flat_map(|i| masks[i % 4].iter().map(move |&c| (gap * i as f64, c))).collect::<Vec<_>>();
    let (fast, slow) = (rice(notes(70.), 7, 8.), rice(notes(70. / 0.9).iter().map(|&(t, c)| (t.round(), c)), 7, 8.));
    assert!(!s(&fast).get("vibro_focused").is_some_and(Val::truthy));
    assert_eq!(s(&fast).get("family"), s(&slow).get("family"), "a rate of a chart is the same family");
}

#[test]
fn short_repeated_drills_are_marked_without_banning_specialists() {
    use ms_calc::structure::structure;
    let k = &standard_params().k;
    let chords: [&[u8]; 2] = [&[3, 4, 5, 6], &[0, 1, 2]];
    let c = rice((0..1000).flat_map(|i: usize| chords[i % 2].iter().map(move |&col| (80. * i as f64, col))), 7, 8.);
    assert!(structure(&c, 1., k).f("drill") > 0.7);
}

#[test]
fn endurance_recovers_and_depends_on_the_rest() {
    use ms_calc::model::endurance;
    let run = |values: Vec<f64>| {
        let fine: Vec<f64> = values.iter().flat_map(|v| std::iter::repeat_n(v / 8., 8)).collect();
        let n = fine.len();
        [fine.clone(), fine, vec![0.; n]]
    };
    let ((continuous, _), _) = endurance(&[&run(vec![10.; 240])], 1., 8);
    let ((rested, _), timeline) = endurance(&[&run([vec![10.; 120], vec![0.; 60], vec![10.; 120]].concat())], 1., 8);
    assert!(continuous > rested);
    assert!(timeline[179] < timeline[119] / 5.);
}

#[test]
fn whole_map_technical_presence_tells_regular_from_irregular_hands() {
    let regular = rice((0..400).map(|i| (80. * i as f64, (i % 4) as u8)), 4, 8.);
    let mut t = 0.;
    let irregular = rice((0..400).map(|i: usize| { t += [60., 93., 47., 111., 75.][i % 5]; (t, (i % 4) as u8) }), 4, 8.);
    let profile = |c: &Chart| -> [f64; 3] {
        let r = run(c, 1.);
        let p = r.dict("tech_profile").unwrap();
        [p.f("technical"), p.f("patterntech"), p.f("rhythmtech")]
    };
    let (a, b) = (profile(&regular), profile(&irregular));
    assert!(a[0] < 0.1, "{a:?}");
    assert!(b[2] > 0.5 && b[0] > 0.5, "{b:?}");
}

#[test]
fn the_calculator_applies_the_calibration_its_rows_are_fitted_on() {
    // The fit and a carry read the chart's own OD; the calculator once read OD 8 for every chart (window column).
    let prm = standard_params();
    let masks: [&[u8]; 4] = [&[0, 1, 2], &[3, 4, 5], &[1, 2, 6], &[0, 3, 5]];
    for (keys, od) in [(7, 3.), (10, 9.5)] {
        let c = rice((0..600).map(|i| (90. * i as f64, masks[i % 4][i % 3] % keys as u8)), keys, od);
        let f = ms_calc::features::features_of(&run(&c, 1.), &prm);
        let again = ms_calc::features::recalibrated(&f, &prm, 1., &|| Some(ms_calc::flow::flow(&c, 1., &prm.k)));
        assert!(again.is_none_or(|g| (g.f("overall") / f.f("overall")).ln().abs() < 1e-9), "{keys}K OD {od}");
    }
}

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
    let s = |c: &Chart| structure(c, 1., &ms_calc::demand::rows(c, 1.), None, k);
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
    assert!(structure(&c, 1., &ms_calc::demand::rows(&c, 1.), None, k).f("drill") > 0.7);
}

#[test]
fn endurance_recovers_and_depends_on_the_rest() {
    use ms_calc::model::endurance;
    let run = |values: Vec<f64>| {
        let fine: Vec<f64> = values.iter().flat_map(|v| std::iter::repeat_n(v / 8., 8)).collect();
        let n = fine.len();
        [fine.clone(), fine, vec![0.; n]]
    };
    let ((continuous, _), _) = endurance(&[&run(vec![10.; 240])], 1., 8, 1.);
    let ((rested, _), timeline) = endurance(&[&run([vec![10.; 120], vec![0.; 60], vec![10.; 120]].concat())], 1., 8, 1.);
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

#[test]
fn the_note_model_rates_the_records_its_training_rows_hold() {
    // popfit trains on `chart_records` (a traced pass) with the calculator's own rating as an input; the analysis builds the
    // records from its pass's series and corrects that rating: both must be the same numbers.
    let Some(net) = ms_calc::notenet::Pop::shipped() else { return };
    let prm = standard_params();
    let masks: [&[u8]; 4] = [&[0, 1, 2], &[3, 4, 5], &[1, 2, 6], &[0, 3, 5]];
    for (keys, rate) in [(7usize, 1.), (4, 1.5)] {
        let c = chart((0..500).map(|i| { let t = 95. * i as f64; (t, if i % 9 == 0 { t + 300. } else { t }, masks[i % 4][i % 3] % keys as u8) }).collect(), keys, 8.);
        let res = run(&c, rate);
        let (overall, factor, note) = (res.dict("scores").unwrap().f("overall"), res.f("note_factor"), res.f("note_rating"));
        let alone = overall / factor;
        // the traced pass poprows writes its records and skill series from
        let mut shared = ms_calc::compute::Shared { trace: Some(Vec::new()), ..Default::default() };
        ms_calc::compute::compute_inner(&c, rate, true, &mut shared, &prm).unwrap();
        let trace = &shared.trace.as_ref().unwrap().iter().find(|x| x.0).unwrap().1;
        let named = |n: &str| trace.series.keys.iter().find(|p| p.name() == n).map_or(&[][..], |&p| trace.series.get(p));
        let want = net.rating(&ms_calc::notefeat::records_of(&c, rate, &trace.total, named("chordjack"), prm.k.BIN, ms_calc::notefeat::Need::ALL), alone, keys, &trace.total, &named, prm.k.BIN).0;
        assert!((note / want).ln().abs() < 1e-5, "{keys}K {rate}×: {note} against {want}");
        let w = ms_calc::model::note_weight(&prm, net.place(keys, alone).unwrap());
        assert!((factor.ln() - w * net.over(keys, note, alone, c.t.len() as f64)).abs() < 1e-9 && factor != 1., "{keys}K {rate}×: {factor}");
    }
}

#[test]
fn a_column_run_is_vibro_when_simple_and_fast_and_longjacks_when_its_chords_change() {
    // user 2026-11-13: jacks are sustained strain on one column/hand; "super hard chordjacks with constantly shifting
    // patterns should show as chordjack at 360 bpm too not vibro"; simple repetition is vibro down to ~160 BPM
    use ms_calc::structure::column_runs;
    let k = &standard_params().k;
    let runs = |notes: Vec<(f64, u8)>| column_runs(&rice(notes, 4, 8.), 1., k).into_iter().map(|r| (r.notes.len(), r.vibro)).collect::<Vec<_>>();
    let jack = |n: usize, gap: f64| (0..n).map(|i| (gap * i as f64, 0)).collect::<Vec<_>>();
    assert_eq!((runs(jack(6, 150.)), runs(jack(7, 150.))), (vec![], vec![(7, false)]), "six hits are a burst, seven slow ones Longjacks");
    assert_eq!(runs(jack(20, 80.)), [(20, true)], "one column at 80 ms is vibro");
    // column 0 every 160 ms with column 1 (same hand) between: a stream through the hand, never a run
    assert!(runs((0..80).map(|i| (80. * i as f64, (i % 2) as u8)).collect()).is_empty());
    // the same alternation at 40 ms (each column every 80 ms): the hand vibrates over both columns
    assert_eq!(runs((0..80).map(|i| (40. * i as f64, (i % 2) as u8)).collect()), [(40, true), (40, true)]);
    // an anchor at 80 ms under a changing chord of its hand: Longjacks; the same chord struck again: vibro
    let anchored = |change: bool| (0..30).flat_map(|i| [(80. * i as f64, 0)].into_iter().chain((change && i % 2 == 0 || !change).then_some((80. * i as f64, 1)))).collect::<Vec<_>>();
    assert_eq!(runs(anchored(true)).iter().find(|r| r.0 == 30), Some(&(30, false)));
    assert!(runs(anchored(false)).iter().all(|r| r.1));
}

#[test]
fn whole_hands_in_turn_have_no_shapes_and_a_stream_has_many() {
    // MikuaBot's anemone 10K: the left hand's five keys, then the right's, for the whole song
    let hands = rice((0..200).flat_map(|i| (0..5u8).map(move |c| (150. * i as f64, c + 5 * (i % 2) as u8))), 10, 8.);
    let [fill, variety] = ms_calc::features::shapes(&hands);
    assert!((fill - 0.5).abs() < 1e-9 && variety == 2. / 200., "{fill} {variety}");
    let stream = rice((0..200).map(|i| (100. * i as f64, ((i * 7) % 10) as u8)), 10, 8.);
    assert_eq!(ms_calc::features::shapes(&stream), [0.1, 10. / 200.]);
    assert_eq!(ms_calc::features::shapes(&rice([(0., 0)], 10, 8.)), [0., 1.], "too short to judge");
}

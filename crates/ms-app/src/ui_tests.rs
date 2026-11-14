//! The window against scripted worker messages. Nothing here touches the real data, clipboard or game.
use super::*;
use egui_kittest::kittest::Queryable;
use egui_kittest::Harness;
use std::sync::atomic::AtomicUsize;

fn app(prefs: Value, ctx: &egui::Context) -> App {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("ms-ui-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    let mut a = App::bare(ctx, Settings::from_json(&prefs), &dir.join("rec.db")).unwrap();
    a.clipboard = None;
    a.s.next_action = "copy".to_string(); // never the client
    a.open.profile = None;
    a.tracking = None;
    a
}

fn prefs() -> Value {
    json!({"recommendation_mode": "nps", "pp_keys": [4, 6, 9], "nps_keys": [7], "skills_keys": [4, 7], "practice_skills": ["ln", "sv"], "next_action": "copy"})
}

fn harness(a: App) -> Harness<'static, App> {
    Harness::builder().with_size(egui::vec2(520., 760.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a)
}

fn pick(sha: char) -> Pick {
    Pick { sha: Some(sha.to_string().repeat(64)), md5: Some(sha.to_string().repeat(32)), title: format!("Cosmo — Cyber Shaman {sha}"), artist: Some("Cosmo".into()), song: Some("Cyber Shaman".into()),
           version: Some("Local difficulty".into()), keys: 7, rate: 0.87, var: "HT 0.87×".into(), purpose: "nps".into(), ..Pick::default() }
}

fn pool(a: &App, mode: &str, revision: u64, shown: Vec<Pick>) -> Out {
    Out::Pool { mode: mode.into(), revision, selection_id: a.selection_id, phase: "warmup".into(), shown, note: "Synthetic UI check · target ~94%".into(), summary: "No actual prediction".into(), progress: None, key_counts: Default::default(), skill_counts: Default::default(), nps_pool: Default::default() }
}

fn tasks(a: &App) -> Vec<Task> {
    std::iter::from_fn(|| a.q.get(Duration::ZERO)).collect()
}

fn nexts(t: &[Task]) -> usize {
    t.iter().filter(|x| matches!(x, Task::Next { .. })).count()
}

#[test]
fn the_expected_accuracy_waits_for_enough_of_the_players_own_plays() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.pending = Some(crate::app::Pending { title: "Map".into(), artist: String::new(), version: String::new(), creator: String::new(), context: String::new() });
    a.card_note = Some("Waiting for osu! to report the playback rate and mods".into());
    let mut h = Harness::builder().with_size(egui::vec2(430., 240.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.main_ui(ui), a);
    h.step();
    h.get_by_label_contains("Waiting for osu!");
    h.state_mut().card = Some(Box::new(ms_rec::engine::Card { acc_mid: 0.962, sure: false, ..Default::default() }));
    h.step();
    assert!(h.query_by_label_contains("Expected").is_none() && h.query_by_label_contains("Waiting").is_none());
    h.state_mut().card.as_mut().unwrap().sure = true;
    h.step();
    h.get_by_label_contains("Expected ~96.2%");
    assert!(h.query_by_label_contains("pp").is_none(), "no stars, no pp");
    h.state_mut().card.as_mut().unwrap().pp_expected = Some(312.4); // unranked maps too (a player 2026-11-08: a pp counter)
    h.step();
    h.get_by_label_contains("~312pp");
    h.state_mut().s.display.prediction = false;
    h.step();
    assert!(h.query_by_label_contains("Expected").is_none(), "its switch hides it");
}

#[test]
fn each_main_window_row_has_its_switch() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.pending = Some(crate::app::Pending { title: "Gems Pack".into(), artist: "gems".into(), version: "Kamaboko".into(), creator: "gemboyong".into(), context: "4K · 1.00×".into() });
    a.foot = "Importing scores".into();
    let mut h = Harness::builder().with_size(egui::vec2(430., 240.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.main_ui(ui), a);
    h.step();
    let rows = ["Gems Pack", "Kamaboko", "4K · 1.00×", "Importing scores"];
    for r in rows {
        h.get_by_label_contains(r);
    }
    let d = &mut h.state_mut().s.display;
    (d.title, d.version, d.info, d.status) = (false, false, false, false);
    h.step();
    for r in rows {
        assert!(h.query_by_label_contains(r).is_none(), "{r} still shown");
    }
}

#[test]
fn each_skill_is_smaller_than_the_one_before() {
    let s = crate::mainview::skill_sizes(&[7.6, 7.5, 7.5, 2.]);
    assert_eq!(s[0], 1.);
    assert!(s.windows(2).all(|w| w[1] < w[0] || w[1] == 0.), "{s:?}");
    assert_eq!(s[3], 0.);
}

#[test]
fn the_main_window_saves_its_size_without_a_window_position() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.s.win = (400., 200.);
    let mut h = Harness::builder().with_size(egui::vec2(430., 240.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| {
        a.main_ui(ui);
        a.track_window(&ui.ctx().clone());
    }, a);
    h.step();
    assert_eq!(h.state().s.win, (430., 240.), "the harness has no inner_rect, like Wayland");
    h.get_by_label_contains("Details").click();
    h.step();
    assert!(h.state().s.details && h.state().s.win == (430., 240.));
}

#[test]
fn an_empty_pick_hints_at_next_and_a_pp_pick_keeps_its_gain() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let mut p = pick('a');
    (p.purpose, p.gain, p.p_up) = ("nps_mix".into(), Some(3.2), Some(0.4));
    let list = pool(&a, "nps", a.revision, vec![p]);
    a.on_rec(list);
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Press Next");
    h.state_mut().next();
    h.step();
    assert!(h.query_by_label_contains("Press Next").is_none());
    h.get_by_label("PP +3.2pp · 40%");
}

fn next_skips(t: &[Task]) -> Vec<bool> {
    t.iter().filter_map(|x| if let Task::Next { skip, .. } = x { Some(*skip) } else { None }).collect()
}

#[test]
fn next_skips_an_unplayed_pick_and_moves_on_from_a_played_one() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    tasks(&a);
    let list = pool(&a, "nps", a.revision, vec![pick('a'), pick('b'), pick('c')]);
    a.on_rec(list);
    a.next();
    a.next();
    assert_eq!(next_skips(&tasks(&a)), [false, true], "no pick yet, then pick a left unplayed");
    a.on_rec(Out::Result(verdict(&["nps"])));
    a.next();
    assert_eq!(next_skips(&tasks(&a)), [false], "pick b was played");
}

#[test]
fn skip_warmup_shows_only_while_the_tab_warms_up() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let list = pool(&a, "nps", a.revision, vec![pick('a')]);
    a.on_rec(list);
    let mut h = harness(a);
    h.step();
    h.get_by_label("Skip warmup").click();
    h.step();
    assert!(tasks(h.state()).iter().any(|t| matches!(t, Task::SkipWarmup)));
    let Out::Pool { mode, revision, selection_id, shown, note, summary, progress, key_counts, skill_counts, nps_pool, .. } = pool(h.state(), "nps", h.state().revision, vec![pick('a')]) else { unreachable!() };
    h.state_mut().on_rec(Out::Pool { mode, revision, selection_id, phase: "build".into(), shown, note, summary, progress, key_counts, skill_counts, nps_pool });
    h.step();
    assert!(h.query_by_label("Skip warmup").is_none());
}

#[test]
fn the_playlist_no_longer_repeats_the_selected_map() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.card = Some(Box::new(ms_rec::engine::Card { acc_mid: 0.962, sure: true, ..Default::default() }));
    let mut h = harness(a);
    h.step();
    assert!(h.query_by_label_contains("SELECTED").is_none() && h.query_by_label_contains("Expected").is_none());
}

#[test]
fn a_row_keeps_its_hover_over_the_text_inside_it() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let revision = a.revision;
    let out = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(out);
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Local difficulty").hover();
    h.step();
    h.step();
    assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::PointingHand);
}

#[test]
fn a_tab_without_its_first_list_says_it_is_loading_not_that_nothing_fits() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.s.mode = "pp".into();
    let revision = a.revision;
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Loading PP suggestions");
    let p = pool(h.state(), "pp", revision, Vec::new());
    h.state_mut().on_rec(p);
    h.step();
    h.get_by_label_contains("Nothing to suggest yet.");
}

#[test]
fn a_stale_list_is_ignored_and_the_current_one_enables_next() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let revision = a.revision;
    let stale = pool(&a, "nps", revision - 1, vec![pick('a')]);
    a.on_rec(stale);
    assert!(!a.next_enabled());
    let fresh = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(fresh);
    assert!(a.next_enabled());
    let mut h = harness(a);
    h.step();
    h.get_by_label("Next");
}

#[test]
fn one_status_line_and_bar_the_workers_progress_first_else_this_tabs_analysis() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let revision = a.revision;
    a.on_rec(Out::Status { text: "Analysing 11 new or updated maps in the background · 0 done".into(), progress: Some((30, 120)), blocking: false, learning: Some(true) });
    let mut p = pool(&a, "nps", revision, vec![pick('a')]);
    if let Out::Pool { progress, .. } = &mut p {
        *progress = Some((5, 50));
    }
    a.on_rec(p);
    assert_eq!(a.bg_progress, Some((30, 120)));
    assert!(a.stats_wait.as_ref().is_some_and(|w| w.0.contains("30 / 120")));
    a.on_rec(Out::Status { text: String::new(), progress: None, blocking: false, learning: Some(false) });
    assert!(a.stats_wait.is_none() && a.bg_progress.is_none());
    assert!(a.views["nps"].analysing.as_ref().is_some_and(|x| x.0.contains("5 / 50")));
    let done = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(done);
    assert!(a.views["nps"].analysing.is_none());
}

#[test]
fn missing_population_data_is_explained_in_the_playlist_and_only_noted_in_the_main_window() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.on_rec(Out::Status { text: "Recommendations unavailable: could not download the population data".into(), progress: None, blocking: true, learning: None });
    a.on_rec(Out::Card { c: None, note: None, path: None, rate: None });
    assert!(a.status_text.contains("could not download"), "the playlist status line carries the reason");
    assert!(a.flash.as_ref().is_some_and(|f| f.0.contains("playlist window") && f.1 > Instant::now()), "the main window notes it for a while");
    assert!(a.card_note.is_none(), "no prediction line stuck on the error");
}

#[test]
fn the_main_window_shows_the_analysis_bar_in_its_status_row_until_the_library_is_done() {
    // user 2026-11-13: the bar and its count belong in the empty status row at the bottom, never over the card's graph
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    let analysis = crate::viewer::Analysis {
        path: "x.osu".into(), rate: 1., sv: true, keys: 7, title: "Black Lotus".into(), artist: "wa.".into(), version: "Another".into(), creator: "Blocko".into(),
        overall: 7.05, notes: 2400, ln_notes: 0., nps: Some(28.1), span: Some((0., 105_000.)), timeline: (0..60).map(|i| 4. + (i as f64 / 6.).sin()).collect(),
        entries: Vec::new(), tags: Vec::new(), bars: Vec::new(), peak_nps: None, bid: None, names: Vec::new(), tech: Vec::new(), sv_ignored: false, dan_vibro: Default::default(), o2jam: None,
    };
    a.shown = Some(crate::app::Selected { path: "x.osu".into(), rate: 1., sv: true, result: Ok(std::sync::Arc::new(analysis)) });
    a.on_rec(Out::Status { text: "Analysing maps · 104,075 / 104,226".into(), progress: Some((104_075, 104_226)), blocking: false, learning: None });
    let (w, ht) = (430., 250.);
    let mut h = Harness::builder().with_size(egui::vec2(w, ht)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.main_ui(ui), a);
    h.step();
    if let Ok(path) = std::env::var("MS_SHOT") {
        h.render().expect("render").save(path).expect("save");
    }
    let r = h.get_by_label_contains("Analysing maps").rect();
    // the status row: 22 px reserved under the graph, the 8 px window margin, the 3 px bar under the text
    assert!(r.top() >= ht - 22. - 8. - 5. && r.bottom() <= ht - 8., "in the status row: {r:?}");
    h.state_mut().on_rec(Out::Status { text: String::new(), progress: None, blocking: false, learning: None });
    h.step();
    assert!(h.query_by_label_contains("Analysing").is_none(), "gone once done");
    // a few new maps later: the playlist window's bar only (user 2026-11-13: "annoying if it keeps showing up")
    h.state_mut().on_rec(Out::Status { text: "Analysing 3 new or updated maps in the background · 1 done".into(), progress: Some((1, 3)), blocking: false, learning: Some(false) });
    h.step();
    assert!(h.query_by_label_contains("Analysing 3 new").is_none(), "not in the main window");
}

#[test]
fn next_uses_the_finished_list_and_a_hover_only_defers_painting() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    tasks(&a);
    let revision = a.revision;
    let first = pool(&a, "nps", revision, vec![pick('a')]);
    a.on_rec(first);
    a.next(); // a fake worker never drains its queue: Next still navigates
    assert_eq!(nexts(&tasks(&a)), 1);
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.sha.clone()), Some(pick('a').sha));
    assert!(!a.next_enabled(), "the pick left the list");
    a.next();
    assert_eq!(nexts(&tasks(&a)), 0, "a second press with nothing left is no task");
    // A completed refresh wins even when hover deferred painting it.
    a.views.get_mut("nps").unwrap().hover = true;
    let mut b = pick('b');
    b.pred.acc_mid = 0.951;
    let fresh = pool(&a, "nps", revision, vec![b, pick('c')]);
    a.on_rec(fresh);
    a.next();
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.pred.acc_mid), Some(0.951));
    assert!(a.next_enabled());
    // A stale target can't move the selection.
    let stale = Out::Target { mode: "nps".into(), revision: revision - 1, selection_id: a.selection_id, c: Some(Box::new(pick('z'))), msg: "stale".into(), copy: Some("STALE".into()) };
    a.on_rec(stale);
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.sha.clone()), Some(pick('b').sha));
}

fn verdict(targets: &[&str]) -> Box<Verdict> {
    Box::new(Verdict { title: "t".into(), text: "x".into(), acc: 0.95, expected: None, z: None, rate: None, when: None, news: Vec::new(), targets: targets.iter().map(|m| m.to_string()).collect() })
}

#[test]
fn a_finished_play_moves_on_only_in_its_own_tab() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.s.auto_next = true;
    a.sync_rec();
    let list = pool(&a, "nps", a.revision, vec![pick('b'), pick('c')]);
    a.on_rec(list);
    a.on_rec(Out::Result(verdict(&["pp"])));
    assert!(a.views["nps"].target.is_none());
    a.on_rec(Out::Result(verdict(&["nps"])));
    assert_eq!(a.views["nps"].target.as_ref().map(|t| t.sha.clone()), Some(pick('b').sha));
}

#[test]
fn keymodes_are_per_tab_and_none_disables_next() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let list = pool(&a, "nps", a.revision, vec![pick('a')]);
    a.on_rec(list);
    a.set_keys("pp", vec![4, 5, 6, 8, 9]);
    assert_eq!(a.s.keys["nps"], [7]);
    a.set_keys("nps", Vec::new());
    assert!(!a.next_enabled());
    assert_eq!(a.views["nps"].note, "Select at least one keymode");
    a.set_keys("nps", vec![7]);
    assert_eq!(a.s.keys["pp"], [4, 5, 6, 8, 9]);
    assert!(!a.s.keys_auto.contains("nps"));
}

#[test]
fn a_row_from_an_older_skill_filter_opens_nothing_and_stale_lists_stay_out() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.set_mode("skills");
    a.sync_rec();
    let old = Pick { practice_skills: a.s.practice.clone(), purpose: "skills".into(), ..pick('a') };
    let list = pool(&a, "skills", a.revision, vec![old]);
    a.on_rec(list);
    assert!(a.next_enabled());
    a.set_skills(vec!["sv".to_string()]);
    tasks(&a);
    a.activate("skills", 0, None);
    assert!(a.views["skills"].shown.is_empty() && tasks(&a).is_empty());
    let stale = pool(&a, "skills", a.revision - 1, vec![pick('a')]);
    a.on_rec(stale);
    assert!(a.views["skills"].shown.is_empty());
    a.set_skills(Vec::new());
    assert!(a.s.practice.is_empty());
}

#[test]
fn stats_is_not_a_fourth_mode() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.set_mode("skills");
    tasks(&a);
    a.set_stats(true);
    assert!(matches!(tasks(&a).last(), Some(Task::Stats(true))));
    assert_eq!(a.mode(), "skills");
}

#[test]
fn the_convert_row_needs_a_selected_map() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.open.menu = true;
    let mut h = Harness::builder().with_size(egui::vec2(400., 620.)).allow_missing_glyphs().build_ui_state(|ui, a: &mut App| a.menu_ui(ui), a);
    h.step();
    h.get_by_label("10K").click();
    h.run();
    assert!(h.state().open.menu, "a press with no map does nothing");
    for gone in ["Next playlist", "Rate", "New session", "Follow osu!'s rate (Auto)"] {
        assert!(h.query_by_label(gone).is_none(), "{gone}");
    }
    h.get_by_label("Keys, dan, rate, length, LN %");
}

#[test]
fn one_checked_skill_gets_its_slider_and_any_clears_it() {
    let ctx = egui::Context::default();
    let mut a = app(json!({"recommendation_mode": "skills", "skills_keys": [7], "practice_skills": ["chordjack", "!sv"]}), &ctx);
    a.views.get_mut("skills").unwrap().settings = true;
    let mut h = harness(a);
    h.step();
    assert_eq!(h.get_all_by_label("✓ Chordjack").count(), 3, "the chip, and its slider row with the slider");
    assert_eq!(h.get_all_by_label("Other maps").count(), 2, "the row and its slider");
    assert_eq!(h.get_all_by_label("✕ SV").count(), 3, "the chip, and its slider row with the slider");
    h.get_by_label("Any").click();
    h.run();
    assert!(h.state().s.practice.is_empty());
    assert!(h.query_by_label("Other maps").is_none());
}

#[test]
fn the_focus_slider_names_the_side_it_leans_to() {
    use crate::playlist::focus_label;
    assert_eq!([focus_label(0.), focus_label(0.25), focus_label(0.5), focus_label(0.75)], ["100% variety", "75% variety", "Even mix", "75% NPS"]);
}

#[test]
fn a_linked_install_is_not_asked_for_its_profile() {
    let dir = std::env::temp_dir().join(format!("ms-profile-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = ms_data::store::connect(&dir.join("rec.db")).unwrap();
    assert!(super::asks_profile(&db));
    ms_data::store::kv_set(&db, "user_id", &serde_json::json!(1)).unwrap();
    assert!(!super::asks_profile(&db));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn gear_flips_the_tab_to_its_settings_and_another_tab_shows_its_list() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.views.get_mut("nps").unwrap().settings = true;
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Mix in likely PP gains");
    assert_eq!(h.get_by_label("⚙").rect().height(), h.get_by_label("Stats").rect().height(), "the cog never makes the tabs taller");
    h.state_mut().set_mode("pp");
    h.step();
    assert!(h.query_by_label_contains("Mix in likely PP gains").is_none());
}

#[test]
fn chip_rows_wrap_inside_the_width() {
    let rows = crate::playlist::wrap_rows(&[40., 50., 60., 200.], 100., 4.);
    assert_eq!(rows, [vec![0, 1], vec![2], vec![3]]);
}

#[test]
fn every_skill_chip_stays_inside_the_settings_page() {
    for width in [430., 320., 900.] {
        let ctx = egui::Context::default();
        let mut a = app(json!({"recommendation_mode": "skills", "skills_keys": [7]}), &ctx);
        a.views.get_mut("skills").unwrap().settings = true;
        let mut h = Harness::builder().with_size(egui::vec2(width, 760.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a);
        h.run(); // the second frame cuts rows from the drawn widths
        h.get_by_label("Any");
        for chip in ["Stream", "Chordstream", "Trills", "Chordjack", "Jacks", "LN", "Technical", "Stamina", "SV", "Vibro", "Mash"] {
            let right = h.get_by_label(chip).rect().right() as f32;
            assert!(right <= width, "{chip} ends at {right} of {width}");
        }
        if width > 800. {
            assert_eq!(h.get_by_label("Vibro").rect().top(), h.get_by_label("Any").rect().top(), "the first row fills a wide window");
        }
        if let Ok(path) = std::env::var("MS_SHOT") { h.render().expect("render").save(format!("{path}-{width}.png")).expect("save"); }
    }
}

#[test]
fn clicking_the_open_tab_closes_its_settings_like_the_cog() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.views.get_mut("nps").unwrap().settings = true;
    let mut h = harness(a);
    h.step();
    h.get_by_label("NPS").click();
    h.run();
    assert!(!h.state().views["nps"].settings);
    assert!(h.query_by_label_contains("Mix in likely PP gains").is_none());
}

#[test]
fn a_pp_map_in_the_nps_list_keeps_its_pp_details_on_the_row() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let row = Pick { purpose: "nps_mix".into(), gain: Some(3.1), p_up: Some(0.45), best: Some(400.), ranked: true, ..pick('a') };
    let list = pool(&a, "nps", a.revision, vec![row]);
    a.on_rec(list);
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("+3.1pp");
    assert!(h.get_all_by_label_contains("best 400pp").count() >= 1);
}

#[test]
fn unchosen_keymodes_follow_what_the_player_plays() {
    let ctx = egui::Context::default();
    let mut a = app(json!({"recommendation_mode": "nps", "skills_keys": [7]}), &ctx);
    assert_eq!(a.s.keys["nps"], [7]);
    assert!(a.s.keys_auto.contains("nps"));
    for i in 0..30 {
        let keys = if i < 24 { 4 } else if i < 28 { 7 } else { 10 }; // 80 % 4K, 13 % 7K, 7 % 10K
        a.db.execute("INSERT INTO installed(sha256,keys) VALUES(?,?)", rusqlite::params![format!("s{i}"), keys]).unwrap();
        a.db.execute("INSERT INTO scores(key,src,sha256,played) VALUES(?,?,?,?)", rusqlite::params![format!("k{i}"), "realm", format!("s{i}"), format!("2026-09-{:02}", i + 1)]).unwrap();
    }
    a.keys_checked = None;
    a.auto_keys(false);
    assert!(a.s.keys["nps"] == [4, 7] && a.s.keys["skills"] == [7] && a.s.keys_auto.is_empty());
    let saved = a.s.to_json();
    assert_eq!(saved["nps_keys"], json!([4, 7]));
}

fn osu_file(dir: &std::path::Path, name: &str) -> PathBuf {
    let mut text = "osu file format v14\n\n[General]\nMode: 3\n\n[Metadata]\nTitle:t\nVersion:v\n\n[Difficulty]\nCircleSize:7\n\n[TimingPoints]\n0,500,4,1,0,100,1,0\n\n[HitObjects]\n".to_string();
    for i in 0..200 {
        text.push_str(&format!("{},192,{},1,0,0:0:0:0:\n", (i % 7) * 73 + 36, 1000 + i * 60));
    }
    std::fs::create_dir_all(dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p
}

fn observed(path: &std::path::Path, rate: f64) -> Obs {
    Obs { source: Some("game"), path: Some(path.to_path_buf()), rate: Some(rate), ..Obs::default() }
}

#[test]
fn identical_observations_do_not_invalidate_their_pending_analysis() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    let dir = std::env::temp_dir().join(format!("ms-ui-obs-{}", std::process::id()));
    let file = osu_file(&dir, "a.osu");
    a.obs = observed(&file, 1.);
    a.refresh();
    let first = a.generation;
    for _ in 0..10 {
        a.refresh();
    }
    assert_eq!(a.generation, first);
    a.obs = observed(&file, 0.98);
    a.refresh();
    assert_eq!(a.generation, first + 1);
    assert!(a.pending.is_some(), "the header shows the new map until its analysis lands");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_late_result_for_another_chart_cannot_replace_the_settled_one() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    let dir = std::env::temp_dir().join(format!("ms-ui-late-{}", std::process::id()));
    let (first, second) = (osu_file(&dir, "a.osu"), osu_file(&dir, "b.osu"));
    a.obs = observed(&first, 1.);
    a.refresh();
    let done = |generation, path: &PathBuf| {
        let result = viewer::analyse(path, 1., true, &standard_params()).map(Arc::new);
        viewer::Done { request: viewer::Request { generation, path: path.clone(), rate: 1., sv: true }, result }
    };
    a.on_analysed(done(a.generation, &first));
    assert!(a.shown.is_some() && a.pending.is_none());
    a.obs = observed(&second, 1.);
    a.refresh();
    let old = a.generation;
    a.obs = observed(&first, 1.);
    a.refresh();
    assert!(a.generation > old && a.requested.is_none() && a.pending.is_none());
    a.on_analysed(done(old, &second));
    assert_eq!(a.shown.as_ref().map(|s| s.path.clone()), Some(first));
    let _ = std::fs::remove_dir_all(dir);
}

/// Windows has no system fallback here, so a glyph the embedded fonts lack draws as a box (`☰`, user 2026-11-01).
#[test]
fn every_character_in_the_window_is_in_the_embedded_fonts() {
    let ctx = egui::Context::default();
    crate::theme::apply(&ctx);
    ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
    let sources = [include_str!("mainview.rs"), include_str!("playlist.rs"), include_str!("windows.rs"), include_str!("fmt.rs"), include_str!("app.rs")];
    let font = egui::FontId::proportional(crate::theme::BODY);
    let missing: std::collections::BTreeSet<char> = sources.iter().flat_map(|s| s.chars()).filter(|c| !c.is_ascii()).filter(|&c| !ctx.fonts_mut(|f| f.has_glyph(&font, c))).collect();
    assert!(missing.is_empty(), "{missing:?}");
}

#[test]
fn clicking_the_title_copies_its_search_text_and_says_so() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    let dir = std::env::temp_dir().join(format!("ms-ui-title-{}", std::process::id()));
    let file = osu_file(&dir, "a.osu");
    a.obs = observed(&file, 1.);
    a.refresh();
    let result = viewer::analyse(&file, 1., true, &standard_params()).map(Arc::new);
    let generation = a.generation;
    a.on_analysed(viewer::Done { request: viewer::Request { generation, path: file.clone(), rate: 1., sv: true }, result });
    let mut h = Harness::builder().with_size(egui::vec2(430., 300.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.main_ui(ui), a);
    h.step();
    assert!(h.query_by_label_contains("Copied").is_none());
    h.get_by_label_contains("[v]").click();
    h.run();
    h.get_by_label_contains("Copied");
    assert_eq!(crate::mainview::search_text(h.state().analysis().unwrap()), " - t [v]");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn ctrl_click_copies_a_row_without_picking_or_skipping() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let list = pool(&a, "nps", a.revision, vec![pick('a'), Pick { version: Some("Second".into()), ..pick('b') }]);
    a.on_rec(list);
    tasks(&a);
    let mut h = harness(a);
    h.step();
    h.get_by_label_contains("Second").click_modifiers(egui::Modifiers::COMMAND);
    h.run();
    let t = tasks(h.state());
    assert!(!t.iter().any(|x| matches!(x, Task::Next { .. } | Task::Open { .. })), "no pick");
    assert!(h.state().flash.is_some(), "copied");
    h.get_by_label_contains("Second").click();
    h.run();
    assert!(tasks(h.state()).iter().any(|x| matches!(x, Task::Open { .. } | Task::Next { .. })), "a plain click still picks");
}

#[test]
fn right_click_next_skips_the_current_pick_for_12_hours() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let list = pool(&a, "nps", a.revision, vec![pick('a'), pick('b')]);
    a.on_rec(list);
    a.views.get_mut("nps").unwrap().target = Some(pick('c'));
    tasks(&a);
    let mut h = harness(a);
    h.step();
    h.get_by_label("Next").click_secondary();
    h.run();
    h.get_by_label("Skip for 12 h").click();
    h.run();
    let t = tasks(h.state());
    assert!(t.iter().any(|x| matches!(x, Task::Snooze { pick, mode } if mode == "nps" && pick.sha == pick_sha('c'))), "a snooze of the current pick");
    assert_eq!(nexts(&t), 0);
}

fn pick_sha(sha: char) -> Option<String> {
    pick(sha).sha
}

#[test]
fn stats_open_on_the_selected_maps_keymode_and_remember_the_last_one() {
    let ctx = egui::Context::default();
    let mut a = app(json!({"stats_keys": 4, "stats_expanded": ["7:ln"]}), &ctx);
    assert_eq!(a.s.stats_keys, Some(4));
    a.set_stats(true);
    a.set_stats(false);
    assert_eq!(a.s.stats_keys, Some(4), "no map selected: the last one viewed");
    let dir = std::env::temp_dir().join(format!("ms-ui-stats-{}", std::process::id()));
    let file = osu_file(&dir, "a.osu");
    a.obs = observed(&file, 1.);
    a.refresh();
    let result = viewer::analyse(&file, 1., true, &standard_params()).map(Arc::new);
    let generation = a.generation;
    a.on_analysed(viewer::Done { request: viewer::Request { generation, path: file.clone(), rate: 1., sv: true }, result });
    a.set_stats(true);
    assert_eq!(a.s.stats_keys, Some(7));
    let back = Settings::from_json(&a.s.to_json());
    assert_eq!((back.stats_keys, back.expanded.iter().cloned().collect::<Vec<_>>()), (Some(7), vec![(7, "ln".to_string())]));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rows_carry_website_and_convert_badges() {
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.sync_rec();
    let convert = |c, web| Pick { convert_of: Some("src".into()), source_keys: 4, source_web: web, ..pick(c) };
    let list = pool(&a, "nps", a.revision, vec![Pick { installed: true, ..pick('a') }, pick('b'), convert('c', false), convert('d', true)]);
    a.on_rec(list);
    let mut h = harness(a);
    h.step();
    assert_eq!(h.query_all_by_label("WEB").count(), 2, "the website map and the website convert");
    assert_eq!(h.query_all_by_label("4K→7K").count(), 2);
    // On the right, so the rate and title start where an unmarked row's do (user 2026-11-03).
    for b in h.query_all_by_label("WEB").chain(h.query_all_by_label("4K→7K")) {
        assert!(b.rect().left() > 400., "badge at {:?}", b.rect());
    }
}

#[test]
fn playlist_tabs_open_their_own_presets_and_the_wheel_only_scrolls_the_strip() {
    let ctx = egui::Context::default();
    let mut a = app(json!({"recommendation_mode": "skills", "skills_keys": [7], "practice_skills": ["ln"]}), &ctx);
    a.s.playlists[0].0 = "Holds".into();
    for n in 0..6 {
        a.s.add_playlist(format!("Jacks {n}"));
    }
    a.set_skills(vec!["chordjack".into()]);
    let mut h = Harness::builder().with_size(egui::vec2(300., 400.)).allow_missing_glyphs().build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a);
    h.run();
    h.get_by_label("Holds").click();
    h.run();
    assert_eq!((h.state().s.playlist, h.state().s.practice.clone()), (0, vec!["ln".to_string()]));
    let before = h.get_by_label("PP").rect().left();
    h.get_by_label("NPS").hover();
    h.run();
    for _ in 0..10 {
        h.event(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: egui::vec2(0., -40.), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel, modifiers: Default::default() });
        h.step();
    }
    h.run();
    assert!(h.get_by_label("PP").rect().left() < before, "the strip scrolled sideways");
    assert_eq!((h.state().s.playlist, h.state().mode()), (0, "skills"), "the open tab stayed");
}

#[test]
fn tabs_drag_like_browser_tabs_and_pp_moves_too() {
    let ctx = egui::Context::default();
    let mut a = app(json!({"recommendation_mode": "nps", "nps_keys": [7]}), &ctx);
    a.s.playlists[0].0 = "Holds".into();
    a.s.add_playlist("Jacks".into());
    let mut h = Harness::builder().with_size(egui::vec2(600., 400.)).allow_missing_glyphs().build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a);
    h.run();
    let (pp, nps, jacks) = (h.get_by_label("PP").rect(), h.get_by_label("NPS").rect().left(), h.get_by_label("Jacks").rect().right());
    let pointer = |h: &mut Harness<'_, App>, pos: egui::Pos2, pressed: Option<bool>| {
        h.event(egui::Event::PointerMoved(pos));
        if let Some(pressed) = pressed {
            h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() });
        }
        h.step();
    };
    pointer(&mut h, pp.center(), Some(true));
    for i in 1..=12 {
        pointer(&mut h, pp.center() + egui::vec2((jacks - pp.center().x) * i as f32 / 12., 0.), None);
    }
    h.run_steps(20); // held: it keeps repainting
    assert!(h.get_by_label("NPS").rect().left() < nps - 1., "the others slid aside while PP was held");
    let end = pp.center() + egui::vec2(jacks - pp.center().x, 0.);
    pointer(&mut h, end, Some(false));
    h.run();
    assert_eq!(h.state().s.tabs().last(), Some(&crate::settings::Tab::Mode("pp")), "PP dropped at the end");
    assert!(h.get_by_label("PP").rect().left() > h.get_by_label("Jacks").rect().left());
}

#[test]
fn a_card_row_carries_its_modifier_tags_beside_the_name() {
    // Tech/Stamina are tags beside a skill's name: Tech before it, Stamina after (user 2026-11-12).
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    let entry = |name: &str, rating: f64, tags: &[&str]| crate::viewer::Entry { name: name.into(), rating, parts: Vec::new(), tags: tags.iter().map(|t| t.to_string()).collect() };
    let analysis = crate::viewer::Analysis {
        path: "x.osu".into(), rate: 1., sv: true, keys: 7, title: "Cyber Shaman".into(), artist: "Cosmo".into(), version: "Local difficulty".into(), creator: "someone".into(),
        overall: 6.1, notes: 2400, ln_notes: 900., nps: Some(14.2), span: Some((0., 150_000.)), timeline: (0..60).map(|i| 4. + (i as f64 / 6.).sin()).collect(),
        entries: vec![entry("LN Release", 6.0, &["Tech"]), entry("Stream", 5.6, &["Stamina"]), entry("Chordjack", 4.9, &[])], tags: Vec::new(), bars: Vec::new(),
        peak_nps: None, bid: None, names: Vec::new(), tech: Vec::new(), sv_ignored: false, dan_vibro: Default::default(), o2jam: Some(84.4),
    };
    a.s.display.o2jam = true;
    a.shown = Some(crate::app::Selected { path: "x.osu".into(), rate: 1., sv: true, result: Ok(std::sync::Arc::new(analysis)) });
    a.s.display.skills = 3;
    let mut h = Harness::builder().with_size(egui::vec2(460., 260.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.main_ui(ui), a);
    h.step();
    h.get_by_label_contains("O2Jam Lv. 84"); // beside the dan, off by default (user 2026-11-13)
    h.get_by_label_contains("Tech LN Release"); // Tech before the name (user 2026-11-12); the second row is one painted galley, seen in the screenshot
    // MS_SHOT=path: a themed screenshot of this card (software Vulkan is enough).
    if let Ok(path) = std::env::var("MS_SHOT") {
        h.render().expect("render").save(path).expect("save");
    }
}

#[test]
fn slider_values_and_the_website_button_stay_inside_the_settings_page() {
    // user 2026-11-13: "80% NPS · 7899 m…" and "0.4× per ma…" ran past the right edge at ~440 px.
    for width in [440., 320.] {
        let mut h = Harness::builder().with_size(egui::vec2(width, 200.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, v: &mut f64| {
            crate::playlist::slider(ui, "Variety to high NPS", v, (0., 1., 0.05), 0.5, |x| format!("{} · 78,999 maps", crate::playlist::focus_label(x)), None);
            crate::playlist::slider(ui, "Maps you don't have", v, (0., 1., 0.05), 0.5, |_| "10.0× per map, weighted".to_string(), None);
        }, 0.8);
        h.step();
        for text in ["80% NPS · 78,999 maps", "10.0× per map, weighted"] {
            let right = h.get_by_label(text).rect().right() as f32;
            assert!(right <= width, "{text} ends at {right} of {width}");
        }
    }
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.views.get_mut("nps").unwrap().settings = true;
    let mut h = Harness::builder().with_size(egui::vec2(440., 760.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a);
    h.step();
    let right = h.get_by_label("Website charts…").rect().right() as f32;
    assert!(right <= 440., "Website charts ends at {right}");
}

#[test]
fn no_text_or_button_runs_past_any_settings_page() {
    use egui::accesskit::Role;
    use egui_kittest::kittest::NodeT;
    for mode in ["nps", "skills", "pp"] {
        for width in [440., 360.] {
            let ctx = egui::Context::default();
            let mut a = app(json!({"recommendation_mode": mode, "skills_keys": [7]}), &ctx);
            a.views.get_mut(mode).unwrap().settings = true;
            let mut h = Harness::builder().with_size(egui::vec2(width, 2400.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| a.playlist_ui(ui), a);
            h.step();
            let past: Vec<String> = h.query_all(egui_kittest::kittest::by().predicate(|n| matches!(n.role(), Role::Label | Role::Button) && n.bounding_box().is_some_and(|r| r.x1 > width as f64 + 0.5)))
                .map(|n| format!("{:?} ends at {:.0}", n.accesskit_node().label(), n.rect().right())).collect();
            assert!(past.is_empty(), "{mode} at {width}: {past:?}");
        }
    }
}

#[test]
fn closing_details_returns_the_window_to_its_compact_size() {
    // user 2026-11-14: closing Details still grew the window downwards. The compositor applies our resizes a few frames late.
    let ctx = egui::Context::default();
    let mut a = app(prefs(), &ctx);
    a.s.win = (430., 240.);
    let analysis = crate::viewer::Analysis {
        path: "x.osu".into(), rate: 1., sv: true, keys: 7, title: "T".into(), artist: "A".into(), version: "V".into(), creator: "C".into(),
        overall: 5., notes: 2400, ln_notes: 0., nps: Some(20.), span: Some((0., 105_000.)), timeline: vec![4.; 60],
        entries: Vec::new(), tags: Vec::new(), bars: (0..4).map(|i| (format!("s{i}"), format!("S{i}"), 0.5)).collect(), peak_nps: None, bid: None, names: Vec::new(), tech: Vec::new(), sv_ignored: false, dan_vibro: Default::default(), o2jam: None,
    };
    a.shown = Some(crate::app::Selected { path: "x.osu".into(), rate: 1., sv: true, result: Ok(std::sync::Arc::new(analysis)) });
    let mut h = Harness::builder().with_size(egui::vec2(430., 240.)).allow_missing_glyphs().with_context_setup(crate::theme::apply).build_ui_state(|ui, a: &mut App| {
        a.main_ui(ui);
        a.track_window(&ui.ctx().clone());
    }, a);
    let mut want = egui::vec2(430., 240.);
    let settle = |h: &mut Harness<'static, App>, want: &mut egui::Vec2| for i in 0..8 {
        h.step();
        if let Some((w, ht)) = h.state().resized { *want = egui::vec2(w, ht) }
        if i == 3 { h.set_size(*want); }
    };
    settle(&mut h, &mut want);
    let mut heights = Vec::new();
    for _ in 0..3 {
        h.get_by_label_contains("Details").click();
        settle(&mut h, &mut want);
        heights.push(want.y);
        h.get_by_label_contains("Details").click();
        settle(&mut h, &mut want);
        heights.push(want.y);
    }
    eprintln!("{heights:?} {:?}", h.state().s.win);
    assert!(heights.iter().skip(1).step_by(2).all(|&y| y == 240.), "closed back to 240: {heights:?}");
    assert!(heights.iter().step_by(2).all(|&y| y > 240. && y == heights[0]), "open grows by the rows each time: {heights:?}");
}

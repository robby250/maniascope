//! The player profile: rendering only, data comes from ms_rec::stats.
use crate::app::App;
use crate::theme::{self, ACCENT, DIM, TEXT};
use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Rect, RichText, ScrollArea, Sense, Stroke, Ui};
use ms_model::skills::{self, Skill};
use ms_rec::catalog::SUPPORTED_KEYS;
use ms_rec::dans;
use ms_rec::stats::{FormHistory, Group, Page, TARGET};

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const FORM: [(&str, Color32); 4] = [("warm-up", Color32::from_rgb(89, 168, 230)), ("peak", Color32::from_rgb(87, 194, 110)), ("bad", Color32::from_rgb(230, 84, 74)), ("fading", Color32::from_rgb(230, 161, 59))];

fn month_name(m: i64) -> String {
    format!("{} {}", MONTHS[m.rem_euclid(12) as usize], 2000 + m.div_euclid(12))
}

/// [(level, short name, full name)] of the keymode's main dan series inside lo..hi: the level matching each dan (its
/// rating / the page's dan scale).
fn dan_ticks(keys: i64, lo: f64, hi: f64, factor: f64) -> Vec<(f64, String, String)> {
    let Some(series) = dans::series(keys, "rice") else { return Vec::new() };
    let prefix = dans::short(series).trim();
    dans::tiers(series).map_or_else(Vec::new, |t| t.iter().map(|(tier, v)| (v.exp() / factor, tier)).filter(|x| lo <= x.0 && x.0 <= hi).map(|(v, tier)| {
        let name = tier.replace(prefix, "");
        (v, if name.trim().is_empty() { tier.clone() } else { name.trim().to_string() }, tier.clone())
    }).collect())
}

/// Shared stars → x mapping so every skill row and the dan axis line up.
#[derive(Clone, Copy)]
struct Scale {
    lo: f64,
    hi: f64,
    dan: f64,
}

impl Scale {
    fn x(&self, v: f64, r: Rect) -> f32 {
        r.left() + 6. + (r.width() - 12.) * (((v - self.lo) / (self.hi - self.lo)).clamp(0., 1.)) as f32
    }

    fn level(&self, x: f32, r: Rect) -> f64 {
        self.lo + (self.hi - self.lo) * (((x - r.left() - 6.) / (r.width() - 12.)).clamp(0., 1.)) as f64
    }
}

fn small_font() -> FontId {
    FontId::proportional(9.)
}

fn date(t: f64, f: &str) -> String {
    ms_data::store::local_format(t, f)
}

/// Hover: the nearest tier's full name and level, names hidden for space included.
fn dan_axis(ui: &mut Ui, keys: i64, scale: Scale, width: f32) {
    let (r, response) = ui.allocate_exact_size(vec2(width, 18.), Sense::hover());
    let p = ui.painter();
    let mut last = f32::MIN;
    let ticks = dan_ticks(keys, scale.lo, scale.hi, scale.dan);
    for (stars, name, _) in &ticks {
        let x = scale.x(*stars, r);
        let g = p.layout_no_wrap(name.clone(), small_font(), TEXT.gamma_multiply(0.55));
        let left = (x - g.size().x / 2.).max(r.left()).min(r.right() - g.size().x); // edge names stay whole
        if left < last + 4. {
            continue; // crowded tiers: keep every other name readable
        }
        last = left + g.size().x;
        p.galley(pos2(left, r.top()), g, TEXT);
        p.vline(x, (r.top() + 13.)..=r.bottom(), Stroke::new(1., TEXT.gamma_multiply(0.55)));
    }
    if let Some(at) = response.hover_pos() {
        if let Some((stars, _, full)) = ticks.iter().min_by(|a, b| (scale.x(a.0, r) - at.x).abs().total_cmp(&(scale.x(b.0, r) - at.x).abs())) {
            response.on_hover_text_at_pointer(format!("{full} · level {stars:.2}"));
        }
    }
}

/// ● your level, a faint band for its uncertainty, ○ this session (only once warm), and the line another hovered bar
/// set (one shared level across all bars).
fn skill_bar(ui: &mut Ui, keys: i64, g: Option<&Group>, scale: Scale, session: bool, width: f32, line: Option<f64>) -> egui::Response {
    let (r, response) = ui.allocate_exact_size(vec2(width, 22.), Sense::hover());
    let p = ui.painter();
    let y = r.center().y;
    let ink = TEXT.gamma_multiply(0.10);
    p.hline((r.left() + 6.)..=(r.right() - 6.), y, Stroke::new(2., ink));
    for (stars, _, _) in dan_ticks(keys, scale.lo, scale.hi, scale.dan) {
        p.vline(scale.x(stars, r), (y - 4.)..=(y + 4.), Stroke::new(1., ink));
    }
    if let Some(l) = line {
        p.vline(scale.x(l, r), r.y_range(), Stroke::new(1., TEXT.gamma_multiply(0.45)));
    }
    let Some(g) = g else { return response };
    let Some(value) = g.value else { return response };
    let (lo, hi) = (scale.x(g.low, r), scale.x(g.high, r));
    p.rect_filled(Rect::from_min_max(pos2(lo, y - 5.), pos2(hi.max(lo + 1.), y + 5.)), 0., ACCENT.gamma_multiply(0.25));
    p.circle_filled(pos2(scale.x(value, r), y), 5.5, ACCENT);
    if let (true, Some(c)) = (session, g.current) {
        p.circle_stroke(pos2(scale.x(c, r), y), 6.5, Stroke::new(2., TEXT.gamma_multiply(0.95)));
    }
    response
}

/// A chart that zooms along x: Ctrl+wheel zooms around the cursor, a drag pans once zoomed, a double-click resets.
/// Shift+drag marks a range instead (a click clears it). Returns the response, x(fraction of the whole span) for the
/// visible part, `plot` = the x-range it maps onto, and the marked range in pixels.
#[allow(clippy::type_complexity)]
fn zoomable(ui: &mut Ui, salt: impl std::hash::Hash + std::fmt::Debug, height: f32, plot: impl Fn(Rect) -> (f32, f32)) -> (Rect, egui::Response, impl Fn(f64) -> f32, Option<(f32, f32)>) {
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let response = ui.interact(r, ui.scope_id().with(salt), Sense::click_and_drag());
    let (left, right) = plot(r);
    let width = (right - left).max(1.) as f64;
    let (mut lo, mut hi) = ui.data(|d| d.get_temp::<(f64, f64)>(response.id)).unwrap_or((0., 1.));
    let marked = response.id.with("range");
    let mut range = ui.data(|d| d.get_temp::<(f64, f64)>(marked));
    if response.double_clicked() {
        (lo, hi) = (0., 1.);
    }
    if response.clicked() || response.double_clicked() {
        range = None;
    }
    let frac = |x: f32, lo: f64, hi: f64| lo + (hi - lo) * ((x - left) as f64 / width).clamp(0., 1.);
    if let Some(at) = response.hover_pos() {
        // Ctrl+wheel (egui's zoom gesture, pinch too) zooms; the plain wheel scrolls the page (user 2026-11-03: "the
        // graphs steal the scroll").
        let factor = 1. / ui.input(|i| i.zoom_delta()) as f64;
        if factor != 1. {
            let anchor = frac(at.x, lo, hi);
            let span = ((hi - lo) * factor).clamp(0.02, 1.);
            lo = anchor - (anchor - lo) * span / (hi - lo);
            hi = lo + span;
        }
    }
    if response.dragged() && ui.input(|i| i.modifiers.shift) {
        if let (Some(from), Some(to)) = (ui.input(|i| i.pointer.press_origin()), response.interact_pointer_pos()) {
            range = Some((frac(from.x, lo, hi), frac(to.x, lo, hi)));
        }
    } else if response.dragged() {
        let shift = response.drag_delta().x as f64 / width * (hi - lo);
        (lo, hi) = (lo - shift, hi - shift);
    }
    let span = hi - lo;
    lo = lo.clamp(0., 1. - span);
    ui.data_mut(|d| {
        d.insert_temp(response.id, (lo, lo + span));
        match range {
            Some(m) => { d.insert_temp(marked, m); }
            None => d.remove::<(f64, f64)>(marked),
        }
    });
    let response = if span < 1. { response.on_hover_cursor(egui::CursorIcon::Grab) } else { response };
    let fx = move |f: f64| left + (width * (f - lo) / span) as f32;
    let pixels = range.map(|(a, b)| (fx(a.min(b)), fx(a.max(b))));
    (r, response, fx, pixels)
}

/// The zoom hint, for the first few chart hovers only (user 2026-11-02: it could stop showing after the first times).
fn hint(ui: &Ui, response: &egui::Response, hints: &mut u32) -> &'static str {
    let id = response.id.with("hovering");
    let (was, now) = (ui.data(|d| d.get_temp::<bool>(id).unwrap_or(false)), response.hovered());
    if now != was {
        ui.data_mut(|d| d.insert_temp(id, now));
        if now {
            *hints += 1;
        }
    }
    if *hints <= HINTS { "\nCtrl+scroll to zoom, drag to move, Shift+drag to measure, double-click to reset" } else { "" }
}

const HINTS: u32 = 5;

/// A marked range: a band and its text along the top.
fn mark(p: &egui::Painter, r: Rect, (a, b): (f32, f32), text: String) {
    p.rect_filled(Rect::from_x_y_ranges(a..=b.max(a + 1.), r.y_range()), 0., ACCENT.gamma_multiply(0.12));
    p.text(pos2((a + b) / 2., r.top() + 2.), Align2::CENTER_TOP, text, small_font(), TEXT);
}

/// Overall level per month, dan lines behind it, peak and today marked; hover for a month's level, its change and plays.
fn history_chart(ui: &mut Ui, keys: i64, h: &[(i64, f64, f64)], peak: (i64, f64), page: &Page, hints: &mut u32) {
    let dan = page.dan_scale.get("rice").copied().unwrap_or(1.);
    let (r, response, fx, marked) = zoomable(ui, ("history", keys), 110., |r| (r.left() + 8., r.right() - 74.));
    let values: Vec<f64> = h.iter().map(|x| x.1).collect();
    let lo = values.iter().copied().fold(f64::MAX, f64::min) * 0.97;
    let hi = values.iter().copied().fold(f64::MIN, f64::max) * 1.03;
    let (m0, m1) = (h[0].0, h[h.len() - 1].0);
    let x = |m: i64| fx((m - m0) as f64 / (m1 - m0).max(1) as f64);
    let y = |v: f64| r.bottom() - 16. - (r.height() - 24.) * ((v - lo) / (hi - lo).max(1e-6)) as f32;
    let p = ui.painter();
    let plot = ui.painter_at(Rect::from_x_y_ranges((r.left() + 2.)..=(r.right() - 68.), r.y_range()));
    let mut ticks = dan_ticks(keys, lo, hi, dan);
    ticks.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut last: Option<f32> = None;
    for (stars, name, _) in ticks {
        if last.is_some_and(|l| y(stars) - l < 11.) {
            continue; // crowded low tiers: every other line stays readable
        }
        last = Some(y(stars));
        p.hline((r.left() + 8.)..=(r.right() - 68.), y(stars), Stroke::new(1., TEXT.gamma_multiply(0.12)));
        p.text(pos2(r.right() - 62., y(stars)), Align2::LEFT_CENTER, name, small_font(), TEXT.gamma_multiply(0.55));
    }
    plot.add(egui::Shape::line(h.iter().map(|&(m, v, _)| pos2(x(m), y(v))).collect(), Stroke::new(2., ACCENT)));
    plot.circle_stroke(pos2(x(peak.0), y(peak.1)), 4., Stroke::new(2., ACCENT));
    let end = h[h.len() - 1];
    plot.circle_filled(pos2(x(end.0), y(end.1)), 4., ACCENT);
    // Edge months of what is visible, once zoomed too.
    let at = |px: f32| h.iter().min_by(|a, b| (x(a.0) - px).abs().total_cmp(&(x(b.0) - px).abs())).unwrap().0;
    let (first, last) = (at(r.left() + 8.), at(r.right() - 74.));
    p.text(pos2(r.left() + 8., r.bottom() - 3.), Align2::LEFT_BOTTOM, month_name(first), small_font(), TEXT.gamma_multiply(0.55));
    p.text(pos2(r.right() - 74., r.bottom() - 3.), Align2::RIGHT_BOTTOM, month_name(last), small_font(), TEXT.gamma_multiply(0.55));
    if let Some((a, b)) = marked {
        let inside: Vec<&(i64, f64, f64)> = h.iter().filter(|e| (a..=b).contains(&x(e.0))).collect();
        if let (Some(f), Some(l)) = (inside.first(), inside.last()) {
            mark(&plot, r, (a, b), format!("{:+.2} over these {} months", l.1 - f.1, l.0 - f.0 + 1));
        }
    }
    let hint = hint(ui, &response, hints);
    if let Some(hover) = response.hover_pos().filter(|_| !response.dragged()) {
        let near_peak = (x(peak.0) - hover.x).abs() <= 6.;
        let i = h.iter().enumerate().min_by(|a, b| (x(a.1 .0) - hover.x).abs().total_cmp(&(x(b.1 .0) - hover.x).abs())).unwrap().0;
        let (m, v, plays) = if near_peak { h.iter().copied().find(|e| e.0 == peak.0).unwrap_or(h[i]) } else { h[i] };
        plot.vline(x(m), (r.top() + 4.)..=(r.bottom() - 16.), Stroke::new(1., TEXT.gamma_multiply(0.35)));
        plot.circle_filled(pos2(x(m), y(v)), 4., TEXT);
        let dan = dans::ability(keys, v, "overall", &page.dan_scale).map_or(String::new(), |d| format!(" · ≈ {d}"));
        let text = if near_peak {
            let ago = (page.now - m as f64).round() as i64;
            format!("Peak {v:.1} · {} · {}{dan}", month_name(m), match ago { 0 => "this month".to_string(), 1 => "a month ago".to_string(), n => format!("{n} months ago") })
        } else {
            let before = h.iter().rev().find(|e| e.0 < m).map_or(String::new(), |e| format!(" ({:+.2} on {})", v - e.1, month_name(e.0)));
            format!("{} · level {v:.2}{before} · {plays:.0} plays{dan}", month_name(m))
        };
        response.on_hover_text_at_pointer(format!("{text}{hint}"));
    }
}

/// The last sessions play by play, against the rested expectation; hover a dot for its session and mark: the session
/// lights up, the others dim.
fn form_chart(ui: &mut Ui, form: &FormHistory, hints: &mut u32) {
    let (mut dots, mut gaps, mut count) = (Vec::new(), Vec::new(), 0usize);
    for (s, session) in form.sessions.iter().enumerate() {
        gaps.push(count);
        let day = date(session.t, "%d %b %H:%M");
        for &(clock, value, mark) in &session.plays {
            dots.push((count, value, mark, format!("{day} · {clock:.0} min in · {value:+.1}%{}", if mark.is_empty() { String::new() } else { format!(" · {mark}") }), s));
            count += 1;
        }
        count += 2;
    }
    let top = dots.iter().map(|d| d.1.abs()).fold(0.5, f64::max);
    let (r, response, fx, marked) = zoomable(ui, "form", 90., |r| (r.left() + 6., r.right() - 6.));
    let x = |i: usize| fx(i as f64 / count.max(1) as f64);
    let y = |v: f64| r.center().y - (r.height() / 2. - 6.) * (v / top) as f32;
    let p = ui.painter_at(r);
    let ink = Stroke::new(1., TEXT.gamma_multiply(0.15));
    p.hline(r.x_range(), y(0.), ink);
    for start in gaps.iter().skip(1) {
        p.vline(x(start - 1), (r.top() + 2.)..=(r.bottom() - 2.), ink);
    }
    let hovered = response.hover_pos().filter(|_| !response.dragged()).and_then(|at| dots.iter().min_by(|a, b| (x(a.0) - at.x).abs().total_cmp(&(x(b.0) - at.x).abs())));
    for (i, value, mark, _, s) in &dots {
        let dim = if hovered.is_some_and(|h| h.4 != *s) { 0.25 } else { 1. };
        match FORM.iter().find(|f| f.0 == *mark) {
            Some(f) => p.circle_filled(pos2(x(*i), y(*value)), 3., f.1.gamma_multiply(dim)),
            None => p.circle_filled(pos2(x(*i), y(*value)), 2., TEXT.gamma_multiply(0.45 * dim)),
        };
    }
    if response.hovered() {
        p.text(pos2(r.right() - 4., y(0.) - 2.), Align2::RIGHT_BOTTOM, "rested expectation", small_font(), TEXT.gamma_multiply(0.55));
    }
    if let Some((a, b)) = marked {
        let inside: Vec<f64> = dots.iter().filter(|d| (a..=b).contains(&x(d.0))).map(|d| d.1).collect();
        if !inside.is_empty() {
            mark(&p, r, (a, b), format!("mean {:+.1}% over {} plays", inside.iter().sum::<f64>() / inside.len() as f64, inside.len()));
        }
    }
    let hint = hint(ui, &response, hints);
    if let Some(dot) = hovered {
        let s = &form.sessions[dot.4];
        let mean = s.plays.iter().map(|p| p.1).sum::<f64>() / s.plays.len().max(1) as f64;
        let length = s.plays.last().map_or(0., |p| p.0);
        response.on_hover_text_at_pointer(format!("{}\nSession: {length:.0} min · {} plays · mean {mean:+.1}%{hint}", dot.3, s.plays.len()));
    }
}

struct Layout {
    name: f32,
    tail: f32,
    bar: f32,
}

impl App {
    /// One skill: name, bar, level. Returns the level under the pointer when its bar is hovered (the shared line).
    #[allow(clippy::too_many_arguments)]
    fn skill_row(&mut self, ui: &mut Ui, keys: i64, node: &Skill, page: &Page, scale: Scale, l: &Layout, indent: bool, toggle: Option<&mut bool>, line: Option<f64>) -> Option<f64> {
        let mut pointed = None;
        let row = page.groups.get(&node.key);
        let value = row.and_then(|g| g.value);
        let dan = value.and_then(|v| dans::ability(keys, v, &node.key, &page.dan_scale));
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.;
            ui.scope(|ui| {
                ui.set_width(l.name);
                ui.add_space(if indent { 22. } else { 0. });
                let label = match &toggle {
                    Some(open) => ui.selectable_label(false, format!("{} {}", if **open { "▾" } else { "▸" }, node.name)),
                    None => ui.label(node.name),
                };
                if label.clicked() {
                    if let Some(open) = toggle {
                        *open = !*open;
                    }
                }
                if !node.tip.is_empty() {
                    label.on_hover_text(node.tip);
                }
            });
            let bar = skill_bar(ui, keys, row, scale, page.session, l.bar, line);
            let at = bar.hover_pos().map(|p| scale.level(p.x, bar.rect));
            pointed = at;
            let at = at.map_or(String::new(), |v| format!("\nAt the line: level {v:.2}{}", dans::ability(keys, v, &node.key, &page.dan_scale).map_or(String::new(), |d| format!(" ≈ {d}"))));
            if let Some(g) = row {
                if let Some(v) = g.value {
                    let examples: Vec<String> = g.examples.iter().map(|(t, r, d)| format!("{t} · {r:.2}× · difficulty {d:.1}")).collect();
                    bar.on_hover_text(format!("Level {v:.2}; likely between {:.2} and {:.2}. {} plays on {} charts.{}{at}", g.low, g.high, g.plays, g.maps, if examples.is_empty() { String::new() } else { format!("\n{}", examples.join("\n")) }));
                } else if g.extrapolated {
                    bar.on_hover_text(format!("Evidence reaches difficulty {:.1}; the fitted limit is not measured.{at}", g.tested_to));
                } else if !at.is_empty() {
                    bar.on_hover_text(at.trim_start().to_string());
                }
            } else if !at.is_empty() {
                bar.on_hover_text(at.trim_start().to_string());
            }
            ui.scope(|ui| {
                ui.set_width(l.tail);
                ui.horizontal(|ui| {
                    match value {
                        Some(v) => {
                            ui.label(RichText::new(format!("{v:.1}")).family(egui::FontFamily::Name("bold".into())));
                            if let Some(d) = &dan {
                                theme::dim_small(ui, format!("≈ {d}"));
                            }
                        }
                        None => { ui.label("—"); }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(egui::Button::new("Practice").frame(false)).on_hover_text(format!("Skills tab · {} · {keys}K · target {:.0}%", node.name, 100. * self.s.targets["skills"])).clicked() {
                            self.practice_from_stats(keys, &node.key);
                        }
                    });
                });
            });
        });
        let confidence = row.map_or("Not enough evidence", |g| g.confidence);
        let maps = row.map_or(0, |g| g.maps);
        let session = row.filter(|g| page.session && g.value.is_some()).and_then(|g| g.current).map_or(String::new(), |c| format!(" · this session {c:.1}"));
        ui.horizontal(|ui| {
            ui.add_space(l.name + 8.);
            theme::dim_small(ui, format!("{confidence} · {} charts{session}", ms_rec::engine::commas(maps)));
        });
        pointed
    }

    fn page_ui(&mut self, ui: &mut Ui, keys: i64, page: &Page) {
        let groups = &page.groups;
        let overall = groups.get("overall");
        let value = overall.and_then(|g| g.value);
        let dan_scale = &page.dan_scale;
        ui.horizontal(|ui| match value {
            Some(v) => {
                ui.label(RichText::new(format!("{v:.1}")).size(theme::BODY * 2.2).family(egui::FontFamily::Name("bold".into())));
                ui.label(RichText::new(format!("your {keys}K level")).family(egui::FontFamily::Name("bold".into())));
                if let Some(d) = dans::ability(keys, v, "overall", dan_scale) {
                    ui.label(format!("·  ≈ {d}"));
                }
            }
            None => { ui.label(RichText::new(format!("{keys}K · not enough history yet")).family(egui::FontFamily::Name("bold".into()))); }
        });
        let peak = page.peak;
        // Months with too few plays leave the history: its last month is not "now" after a break (user 2026-10-30).
        let stale = page.history.last().is_some_and(|h| page.now - h.0 as f64 > 2.);
        let mut lines = Vec::new();
        if let (Some(v), Some(p)) = (value, peak) {
            if p.1 > v + 0.05 {
                lines.push(format!("Peak {:.1} in {}{}", p.1, month_name(p.0), dans::ability(keys, p.1, "overall", dan_scale).map_or(String::new(), |d| format!(" (≈ {d})"))));
            } else if !stale {
                lines.push("This is your peak so far.".to_string());
            }
        }
        if let (Some(_), true, Some(last)) = (value, stale, page.history.last()) {
            lines.push(format!("Last measured in {}: too few {keys}K plays since to tell your form now.", month_name(last.0)));
        }
        match overall.and_then(|g| g.current).filter(|_| value.is_some() && page.session) {
            Some(c) => lines.push(format!("This session: {c:.1} (○ on the bars)")),
            None if value.is_some() => lines.push("Not warmed up: the first maps of a session play lower than this, so no session number is shown until you have played a few.".to_string()),
            None => {}
        }
        if !lines.is_empty() {
            ui.label(lines.join("\n"));
        }
        let target = format!("{:.3}", 100. * self.units.acc(TARGET));
        let target = format!("{}%", target.trim_end_matches('0').trim_end_matches('.'));
        let rice = dan_scale.get("rice").copied().unwrap_or(1.);
        if page.history.len() >= 2 {
            let end = page.history[page.history.len() - 1];
            history_chart(ui, keys, &page.history, peak.unwrap_or((end.0, end.1)), page, &mut self.s.chart_hints);
        }
        if let Some(form) = page.form.as_ref().filter(|f| !f.sessions.is_empty()) {
            ui.label("Form, play by play (last sessions)");
            form_chart(ui, form, &mut self.s.chart_hints);
            ui.horizontal(|ui| {
                for (name, colour) in FORM {
                    ui.label(RichText::new("●").color(colour).size(theme::SMALL));
                    theme::dim_small(ui, name);
                }
            });
            let fatigue = if form.fatigue != 0. { format!("you lose about {:.1}% an hour.", form.fatigue.abs()) } else { "no drop shows yet.".to_string() };
            theme::dim_small(ui, format!("The first minutes of a session cost up to {:.1}%. After 2 hours {fatigue}", form.warmup.abs()));
        }
        let (local, public, newest) = page.evidence;
        let c = ms_rec::engine::commas;
        let based = format!("{} local plays · {} website bests{}", c(local), c(public), if newest != 0. { format!(" · newest {}", date(newest, "%Y-%m-%d")) } else { String::new() });
        theme::dim(ui, format!("{based}. {} charts · {} plays. Level = the chart difficulty you play at ~{target}, from your own recent completed plays. ≈ dan: the hardest dan whose songs you would play at 95% on your better half of plays, after grinding (one song, not passing the course).", c(page.maps), c(page.plays)));
        let measured: Vec<&Group> = groups.values().filter(|g| g.value.is_some()).collect();
        let ends = peak.map(|p| p.1);
        let lo = measured.iter().map(|g| g.low).chain(ends).fold(f64::MAX, f64::min);
        let hi = measured.iter().map(|g| g.high).chain(ends).fold(f64::MIN, f64::max);
        let (lo, hi) = if lo == f64::MAX { (0.8, 10.2) } else { (lo - 0.2, hi + 0.2) };
        let scale = Scale { lo, hi: hi.max(lo + 0.5), dan: rice };
        // Every row's bar starts and ends at the same x, under the dan axis.
        let width = ui.available_width();
        let l = Layout { name: 110., tail: 150., bar: (width - 110. - 150. - 16.).max(80.) };
        theme::dim(ui, format!("● level   band: uncertainty{}", if page.session { "   ○ this session" } else { "" }));
        ui.horizontal(|ui| {
            ui.add_space(l.name + 8.);
            dan_axis(ui, keys, scale, l.bar);
        });
        let parents = skills::tree(keys);
        // One line across every bar at the hovered level: drawn from the last frame's pointer, rows above included.
        let line_id = ui.scope_id().with(("skill-line", keys));
        let line = ui.data(|d| d.get_temp::<f64>(line_id));
        let mut pointed = None;
        for parent in parents {
            if parent.children.is_empty() {
                pointed = pointed.or(self.skill_row(ui, keys, parent, page, scale, &l, false, None, line));
                continue;
            }
            let id = (keys, parent.key.clone());
            let was = self.s.expanded.contains(&id);
            let mut open = was;
            pointed = pointed.or(self.skill_row(ui, keys, parent, page, scale, &l, false, Some(&mut open), line));
            if open {
                self.s.expanded.insert(id);
                for child in &parent.children {
                    pointed = pointed.or(self.skill_row(ui, keys, child, page, scale, &l, true, None, line));
                }
            } else {
                self.s.expanded.remove(&id);
            }
            if open != was {
                self.request_save();
            }
        }
        if pointed != line {
            ui.data_mut(|d| match pointed {
                Some(v) => { d.insert_temp(line_id, v); }
                None => d.remove::<f64>(line_id),
            });
            ui.ctx().request_repaint();
        }
        let reliable: Vec<(&str, &Group)> = parents.iter().filter_map(|p| groups.get(&p.key).filter(|g| g.value.is_some() && g.maps >= 10).map(|g| (p.name, g))).collect();
        if value.is_some() {
            let by = |a: &&(&str, &Group), b: &&(&str, &Group)| a.1.value.total_cmp_opt(&b.1.value);
            if let (Some(high), Some(low)) = (reliable.iter().max_by(by), reliable.iter().min_by(by)) {
                if high.1.low > low.1.high {
                    ui.label(format!("Profile contrast: {} is better supported at higher difficulty than {}.", high.0, low.0));
                }
            }
        }
        egui::CollapsingHeader::new("Prediction strengths & blind spots").show(ui, |ui| {
            let mut messages: Vec<String> = page.diagnostics.iter().map(|(skill, delta, maps)| format!("{}: {:.1} accuracy points {} the saved pre-play predictions across {maps} charts.", skills::name(skill), self.units.points(*delta).abs(), if *delta > 0. { "above" } else { "below" })).collect();
            if messages.is_empty() {
                messages.push("No repeatable large prediction bias supported by enough distinct tracked charts yet.".to_string());
            }
            messages.push("These are model diagnostics, not proof of a personal weakness. Mixed patterns, session form and older predictor versions may contribute.".to_string());
            theme::dim(ui, messages.join("\n\n"));
        });
        if page.mash_maps > 0 {
            theme::dim(ui, format!("Special-pattern evidence: Mash appears materially in {} charts. It is an execution/difficulty correction, not a separate practice goal.", page.mash_maps));
        }
    }

    pub fn stats_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let label = self.s.stats_keys.map_or("All keymodes".to_string(), |k| format!("{k}K"));
            let before = self.s.stats_keys;
            egui::ComboBox::from_id_salt("stats-keys").selected_text(label).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.s.stats_keys, None, "All keymodes");
                for k in SUPPORTED_KEYS {
                    ui.selectable_value(&mut self.s.stats_keys, Some(k), format!("{k}K"));
                }
            });
            if self.s.stats_keys != before {
                self.request_save();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("PLAYER STATS").color(DIM));
            });
        });
        if let Some((text, done, total)) = self.stats_wait.clone() {
            theme::dim(ui, text);
            if total > 0 {
                theme::bar(ui, done as f32 / total as f32);
            }
        }
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.;
            let Some(data) = self.stats.take() else {
                ui.label("Reading your score history… (it appears here as soon as your plays are read and analysed; progress is shown above)");
                return;
            };
            match self.s.stats_keys {
                None => {
                    ui.label("Your keymodes");
                    theme::dim(ui, "Each rating uses its own keymode's difficulty scale. There is no cross-keymode average.");
                    for k in SUPPORTED_KEYS {
                        let page = data.pages.get(&k);
                        let value = page.and_then(|p| p.groups.get("overall")).and_then(|g| g.value).map_or("—".to_string(), |v| format!("{v:.1}"));
                        let (maps, plays) = page.map_or((0, 0), |p| (p.maps, p.plays));
                        if ui.add(egui::Button::new(format!("{k}K     {value}    ·    {} charts / {} plays", ms_rec::engine::commas(maps), ms_rec::engine::commas(plays))).frame(false)).clicked() {
                            self.s.stats_keys = Some(k);
                            self.request_save();
                        }
                    }
                }
                Some(k) => {
                    if let Some(page) = data.pages.get(&k) {
                        self.page_ui(ui, k, page);
                    }
                }
            }
            if self.stats.is_none() { // a practice click or a newer profile replaced it meanwhile
                self.stats = Some(data);
            }
        });
    }
}

trait Cmp {
    fn total_cmp_opt(&self, other: &Self) -> std::cmp::Ordering;
}

impl Cmp for Option<f64> {
    fn total_cmp_opt(&self, other: &Self) -> std::cmp::Ordering {
        self.unwrap_or(0.).total_cmp(&other.unwrap_or(0.))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::Harness;

    #[test]
    fn a_chart_zooms_at_the_cursor_and_pans_once_zoomed() {
        let mut h = Harness::builder().with_size(vec2(400., 120.)).build_ui_state(|ui, ends: &mut (f32, f32)| {
            let (_, _, fx, _) = zoomable(ui, "t", 100., |r| (r.left(), r.right()));
            *ends = (fx(0.), fx(1.));
        }, (0., 0.));
        h.run();
        let full = *h.state();
        h.hover_at(pos2(300., 50.));
        let wheel = |h: &mut Harness<(f32, f32)>, modifiers| {
            h.event_modifiers(egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Point, delta: vec2(0., 300.), phase: egui::TouchPhase::Move, source: egui::MouseWheelSource::Wheel, modifiers }, modifiers);
            h.run();
        };
        wheel(&mut h, egui::Modifiers::NONE);
        assert_eq!(*h.state(), full, "the plain wheel is the page's");
        wheel(&mut h, egui::Modifiers::COMMAND);
        let zoomed = *h.state();
        assert!(zoomed.1 - zoomed.0 > 1.5 * (full.1 - full.0), "{full:?} {zoomed:?}");
        let at = |e: (f32, f32), x: f32| (x - e.0) / (e.1 - e.0);
        assert!((at(zoomed, 300.) - at(full, 300.)).abs() < 0.01, "the point under the cursor stays put");
        h.drag_at(pos2(300., 50.));
        h.run();
        h.hover_at(pos2(340., 50.));
        h.run();
        h.drop_at(pos2(340., 50.));
        h.run();
        let moved = *h.state();
        assert!((moved.0 - zoomed.0 - 40.).abs() < 1., "{zoomed:?} {moved:?}");
    }

    #[test]
    fn shift_drag_marks_a_range_without_panning_and_a_click_clears_it() {
        let mut h = Harness::builder().with_size(vec2(400., 120.)).build_ui_state(|ui, out: &mut ((f32, f32), Option<(f32, f32)>)| {
            let (_, _, fx, marked) = zoomable(ui, "t", 100., |r| (r.left(), r.right()));
            *out = ((fx(0.), fx(1.)), marked);
        }, ((0., 0.), None));
        h.run();
        let ends = h.state().0;
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
        h.drag_at(pos2(100., 50.));
        h.run();
        h.hover_at(pos2(200., 50.));
        h.run();
        h.drop_at(pos2(200., 50.));
        h.run();
        let (after, marked) = *h.state();
        assert_eq!(after, ends, "no pan");
        let (a, b) = marked.expect("a marked range");
        assert!((a - 100.).abs() < 2. && (b - 200.).abs() < 2., "{a} {b}");
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
        h.drag_at(pos2(300., 50.));
        h.run();
        h.drop_at(pos2(300., 50.));
        h.run();
        assert!(h.state().1.is_none());
    }
}

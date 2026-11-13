//! The main window: glanceable difficulty of the selected map, kept small as an OBS source.
use crate::app::{open_external, App};
use crate::fmt::{self, bold, plain, small};
use crate::theme::{self, ACCENT, BODY, DIM, SMALL, TEXT};
use crate::viewer::Analysis;
use eframe::egui::{self, epaint::{text::{LayoutJob, TextFormat, TextWrapping}, Mesh, Vertex, WHITE_UV}, Align, Color32, FontId, Layout, RichText, Sense, Shape, Stroke, TextStyle, Ui, Vec2};
use ms_model::skills;
use ms_rec::catalog::SUPPORTED_KEYS;
use ms_rec::stats::Page;
use ms_rec::worker::Task;
use std::sync::Arc;
use std::time::Instant;

const BAR_FULL: f32 = 10.; // fixed scale for the skill bars: comparable across maps
const BAR_ROW: f32 = 18.;
const BAR_LABEL: f32 = 96.;  // the longest skill name
const BAR_VALUE: f32 = 40.;  // "10.00" and its gap
const HEAD: f32 = 52.; // three lines of skills, or the rating badge
const BARS_TIP: &str = "Each skill is the rating of the sections that show it, on the same scale as the overall number: a section can count toward several skills (Stream and Technical, Chordstream and Bracket); overall charges its work once. Bars: fixed 0–10, comparable across maps. Stamina is the hardest 4 minutes, a shorter map counted with rest.";

/// Rows for the skills after the first: past 3 they wrap once, and the head grows by that line.
fn rows_for(skills: usize) -> usize {
    if skills > 3 { 2 } else { 1 }
}

fn head_height(skills: usize) -> f32 {
    HEAD + (rows_for(skills) - 1) as f32 * BODY
}

fn strip(rect: egui::Rect, points: &[(f32, f32)], color: Color32) -> Shape {
    let mut m = Mesh::default();
    for (i, &(x, y)) in points.iter().enumerate() {
        m.vertices.push(Vertex { pos: egui::pos2(x, y), uv: WHITE_UV, color });
        m.vertices.push(Vertex { pos: egui::pos2(x, rect.bottom()), uv: WHITE_UV, color });
        if i > 0 {
            let n = (2 * i) as u32;
            m.indices.extend([n - 2, n - 1, n, n - 1, n + 1, n]);
        }
    }
    Shape::mesh(m)
}

/// 0–1 per skill, for type size and opacity: shrinks with distance from the top rating, and each entry is a step smaller
/// than the one before even at nearly the same rating, so the second row fits (user 2026-11-02).
pub fn skill_sizes(ratings: &[f64]) -> Vec<f32> {
    let first = ratings.first().copied().unwrap_or(1.);
    let mut sizes: Vec<f32> = Vec::new();
    for r in ratings {
        let f = ((r / first - 0.3) / 0.7).min(1.) as f32;
        sizes.push(sizes.last().map_or(f, |prev| f.min(prev - 0.25)).max(0.));
    }
    sizes
}

fn shown_bars(a: &Analysis) -> impl Iterator<Item = &(String, String, f64)> {
    a.bars.iter().filter(|b| b.2 >= 0.05)
}

/// Your Stats level on the Skills family holding this calculator skill: its subskill when measured, else the family.
fn your_skill(page: &Page, keys: i64, sk: &str) -> Option<f64> {
    let mut out = None;
    for n in skills::dimensions(keys) {
        if n.members.contains(&sk) {
            if let Some(v) = page.groups.get(&n.key).and_then(|g| g.value) {
                if out.is_none() || n.key.contains('/') {
                    out = Some(v);
                }
            }
        }
    }
    out
}

/// A chart's rating against yours, both in fitted units, read on the shown (osu!) scale.
fn versus(keys: i64, v: f64, you: Option<f64>) -> String {
    let st = |x| ms_rec::shown::stars(keys, x);
    you.map_or(String::new(), |y| format!(" · you {:.1} ({:+.1})", st(y), st(v) - st(y)))
}

/// Song time at this rate.
fn at_rate(chart_ms: f64, rate: f64) -> String {
    fmt::clock(chart_ms / 1000. / rate)
}

/// "Artist - Title [Version]": what osu!'s song search finds.
pub fn search_text(a: &Analysis) -> String {
    format!("{} - {} [{}]", a.artist, a.title, a.version)
}

impl App {
    /// Hover: a crosshair with the song time and the local difficulty; ticks at the tech passages and the peak, named
    /// when the pointer is on them.
    fn timeline(&self, ui: &mut Ui, a: &Arc<Analysis>, height: f32) {
        let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
        if a.timeline.is_empty() {
            return;
        }
        let step = (a.timeline.len() / (rect.width() as usize).max(1)).max(1); // one point per pixel at most
        let pts: Vec<f64> = a.timeline.chunks(step).map(|c| c.iter().copied().fold(f64::MIN, f64::max)).collect();
        let top = (pts.iter().copied().fold(0., f64::max) * 1.05).max(1e-9); // own peak: no dead band under the text
        let n = (pts.len() - 1).max(1) as f32;
        let points: Vec<(f32, f32)> = pts.iter().enumerate().map(|(i, v)| (rect.left() + i as f32 * rect.width() / n, rect.bottom() - rect.height() * (*v / top) as f32)).collect();
        let p = ui.painter_at(rect);
        p.add(strip(rect, &points, TEXT.gamma_multiply(0.35)));
        if let (Some(live), Some((t0, t1))) = (self.live, a.span) {
            let f = (((live - t0) / (t1 - t0).max(1.)) as f32).clamp(0., 1.);
            let x = rect.left() + rect.width() * f;
            let mut done: Vec<(f32, f32)> = points.iter().copied().take_while(|q| q.0 <= x).collect();
            if let (Some(last), Some(next)) = (done.last().copied(), points.get(done.len())) {
                let t = (x - last.0) / (next.0 - last.0).max(1e-3);
                done.push((x, last.1 + (next.1 - last.1) * t));
            }
            let blue = Color32::from_rgba_unmultiplied(54, 133, 227, 230);
            p.add(strip(rect, &done, blue));
            p.rect_filled(egui::Rect::from_min_max(egui::pos2(x - 1., rect.top()), egui::pos2(x + 1., rect.bottom())), 0., Color32::from_rgb(54, 133, 227));
        }
        let Some(hover) = response.hover_pos() else { return };
        // x ↔ chart time over the notes' span, as the live marker.
        let ms_at = |x: f32| a.span.map(|(t0, t1)| t0 + (t1 - t0) * ((x - rect.left()) / rect.width()).clamp(0., 1.) as f64);
        let x_at = |ms: f64| a.span.map(|(t0, t1)| rect.left() + rect.width() * (((ms - t0) / (t1 - t0).max(1.)) as f32).clamp(0., 1.));
        let time = |x: f32| ms_at(x).map_or(String::new(), |ms| format!("{} · ", at_rate(ms, a.rate)));
        let mut marks: Vec<(f32, String)> = a.tech.iter().filter_map(|w| {
            let why = if w.why.is_empty() { String::new() } else { format!("\n{}", w.why) };
            Some((x_at(w.t * 1000.)?, format!("Tech passage · {} · pattern {:.0}% / rhythm {:.0}%{why}", at_rate(w.t * 1000., a.rate), 100. * w.pattern, 100. * w.rhythm)))
        }).collect();
        let peak = (0..pts.len()).max_by(|&i, &j| pts[i].total_cmp(&pts[j])).unwrap_or(0);
        marks.push((points[peak].0, format!("Peak · {}difficulty {:.1}", time(points[peak].0), ms_rec::shown::stars(a.keys, pts[peak]))));
        for (x, _) in &marks {
            p.vline(*x, (rect.bottom() - 6.)..=rect.bottom(), Stroke::new(2., ACCENT));
        }
        p.vline(hover.x, rect.y_range(), Stroke::new(1., TEXT.gamma_multiply(0.5)));
        let text = match marks.into_iter().filter(|m| (m.0 - hover.x).abs() <= 4.).min_by(|a, b| (a.0 - hover.x).abs().total_cmp(&(b.0 - hover.x).abs())) {
            Some((_, name)) => name,
            None => {
                let i = (((hover.x - rect.left()) / rect.width()).clamp(0., 1.) * n).round() as usize;
                format!("{}difficulty {:.1}", time(hover.x), ms_rec::shown::stars(a.keys, pts[i.min(pts.len() - 1)]))
            }
        };
        response.on_hover_text_at_pointer(text);
    }

    /// One skill of the head: its rating against yours, what it is made of, its tech passages.
    fn skill_tip(&self, a: &Analysis, i: usize) -> String {
        let e = &a.entries[i];
        let you = self.you(a.keys).and_then(|p| e.parts.first().and_then(|s| your_skill(p, a.keys, s)));
        let mut lines = vec![format!("{} {:.1}{}", e.name, ms_rec::shown::stars(a.keys, e.rating), versus(a.keys, e.rating, you))];
        if e.parts.len() > 1 {
            lines.push(e.parts.iter().map(|p| a.names.iter().find(|n| n.0 == *p).map_or(p.as_str(), |n| n.1.as_str())).collect::<Vec<_>>().join(" + "));
        }
        if e.parts.iter().any(|p| p == "technical") && !a.tech.is_empty() {
            lines.push("Tech passages (song time):".into());
            lines.extend(a.tech.iter().map(|w| format!("  {}  pattern {:.0}% / rhythm {:.0}%{}", at_rate(w.t * 1000., a.rate), 100. * w.pattern, 100. * w.rhythm, if w.why.is_empty() { String::new() } else { format!(" — {}", w.why) })));
        }
        lines.join("\n")
    }

    /// Badge, skills, NPS, all centred on one fixed height: the timeline below keeps its size from map to map, and the NPS
    /// never pushes the skill rows apart (user 2026-11-01: the gap under the top skill). Right-click: the map menu.
    fn head(&mut self, ui: &mut Ui, a: &Arc<Analysis>) {
        let d = self.s.display.clone();
        let (rect, area) = ui.allocate_exact_size(Vec2::new(ui.available_width(), head_height(d.skills)), Sense::click());
        area.context_menu(|ui| self.map_menu(ui, a));
        let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(rect).layout(Layout::left_to_right(Align::Center)));
        if d.rating {
            let v = ms_calc::py::round_digits(ms_rec::shown::stars(a.keys, a.overall), 2);
            let (text, pill) = fmt::rating_colours(v);
            let you = self.you(a.keys).and_then(|p| Some((p.groups.get("overall")?.value?, p)));
            let versus = you.map_or(String::new(), |(y, _)| format!("\n{:+.1} vs your {}K {:.1}{}", v - ms_rec::shown::stars(a.keys, y), a.keys, ms_rec::shown::stars(a.keys, y), ms_rec::dans::ability(a.keys, y, "overall").map_or(String::new(), |d| format!(" · ≈ {d}"))));
            theme::gradient(&mut ui, &format!("{v:.2}"), BODY * 2.4, None, &text, Some(pill))
                .on_hover_text(format!("Overall difficulty: level a well-rounded player needs for controlled, accurate play. Provisional calibration — one decimal is already generous.{versus}"));
            ui.add_space(4.);
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if let (true, Some(v)) = (d.nps, a.nps) {
                let peak = a.peak_nps.map_or(String::new(), |(n, at)| format!("\nDensest 2 s: {n:.1} NPS at {}", at_rate(at, a.rate)));
                theme::gradient(ui, &format!("{v:.1}"), BODY * 1.9, Some((" NPS", BODY * 0.9)), &fmt::nps_colours(v), None).on_hover_text(format!("Notes per second, first to last note, at this rate{peak}"));
            }
            let first = a.entries.first().map_or(1., |e| e.rating);
            let sizes = skill_sizes(&a.entries.iter().map(|e| e.rating).collect::<Vec<_>>());
            let format = |i: usize| {
                let (r, f) = (a.entries[i].rating / first, sizes[i]);
                let family = if r >= 0.85 && i < 2 { egui::FontFamily::Name("bold".into()) } else { egui::FontFamily::Proportional };
                TextFormat { font_id: FontId::new(BODY * (0.70 + 0.55 * f), family), color: TEXT.gamma_multiply(0.55 + 0.45 * f), ..TextFormat::default() }
            };
            // Names only, sized by rating: the overall is the card's one number, skill stars are in Details (user 2026-11-09).
            // A comma ends every skill but the row's last: without the old numbers, same-sized names ran together (user 2026-11-09).
            // A tag sits beside its name in the accent colour, Tech before it and Stamina after ("Tech Stream", "LN Release
            // Stamina", user 2026-11-12), never under the theme's SMALL size ("the font on the modifier on the 3rd skill is way too small").
            let tagged = |i: usize| -> (String, String) {
                let t = &a.entries[i].tags;
                (t.iter().filter(|t| *t == "Tech").map(|t| format!("{t} ")).collect(), t.iter().filter(|t| *t != "Tech").map(|t| format!(" {t}")).collect())
            };
            let comma = |i: usize, end: usize| if i + 1 < end { "," } else { "" };
            let text = |i: usize, end: usize| { let (b, f) = tagged(i); format!("{b}{}{f}{}", a.entries[i].name, comma(i, end)) };
            let width = ui.available_width();
            let row = |range: std::ops::Range<usize>| {
                let mut j = LayoutJob::default();
                for i in range.clone() {
                    let f = format(i);
                    let small = TextFormat { font_id: FontId::new((f.font_id.size * 0.62).max(SMALL), egui::FontFamily::Proportional), color: ACCENT, valign: Align::TOP, ..TextFormat::default() };
                    let (b, t) = tagged(i);
                    let gap = if i > range.start { 8. } else { 0. };
                    if !b.is_empty() {
                        j.append(&b, gap, small.clone());
                    }
                    j.append(&a.entries[i].name, if b.is_empty() { gap } else { 0. }, f.clone());
                    if !t.is_empty() {
                        j.append(&t, 0., small);
                    }
                    j.append(comma(i, range.end), 0., f);
                }
                j.wrap = TextWrapping::truncate_at_width(width);
                j
            };
            let n = a.entries.len();
            let top = (d.skills > 0).then(|| a.entries.first().map(|_| row(0..1))).flatten();
            let rest_range = n.min(1)..d.skills.min(n).max(n.min(1));
            let mut rest = row(rest_range.clone());
            rest.wrap.max_rows = rows_for(d.skills); // past 3 skills the second row wraps once
            rest.wrap.break_anywhere = false;
            let tags = (!a.tags.is_empty()).then(|| a.tags.join(" · "));
            let lines = ui.fonts_mut(|f| {
                let mut h = top.clone().map_or(BODY * 1.2, |j| f.layout_job(j).size().y);
                h += f.layout_job(rest.clone()).size().y;
                h + tags.as_ref().map_or(0., |_| BODY * 1.2)
            });
            ui.allocate_ui_with_layout(ui.available_size(), Layout::top_down(Align::Min), |ui| {
                ui.spacing_mut().item_spacing.y = 0.;
                ui.add_space(((head_height(d.skills) - lines) / 2.).max(0.));
                match (top, a.entries.first()) {
                    (Some(job), Some(_)) => {
                        ui.label(job).on_hover_text(self.skill_tip(a, 0));
                    }
                    (None, Some(_)) => {} // Skills shown: 0
                    _ => { ui.label("too sparse to characterise"); }
                }
                // One galley for the row, each skill its own hover: the glyph under the pointer names it.
                let (pos, galley, r) = egui::Label::new(rest).layout_in_ui(ui);
                ui.painter().galley(pos, galley.clone(), TEXT);
                if let Some(h) = r.hover_pos() {
                    let at = galley.cursor_from_pos(h - pos).index.0;
                    let mut end = 0;
                    let hit = rest_range.clone().find(|&i| {
                        end += text(i, rest_range.end).chars().count();
                        at < end
                    });
                    if let Some(i) = hit {
                        r.on_hover_text_at_pointer(self.skill_tip(a, i));
                    }
                }
                if let Some(t) = tags {
                    theme::dim(ui, t);
                }
            });
        });
    }

    /// One row per skill the map has: zeros said nothing and pushed the footer off the window. A hovered row shows your
    /// level for that skill as a faint tick.
    fn details(&self, ui: &mut Ui, a: &Arc<Analysis>) {
        let rows: Vec<&(String, String, f64)> = shown_bars(a).collect();
        let (area, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), rows.len() as f32 * BAR_ROW), Sense::hover());
        let p = ui.painter().clone();
        let font = egui::FontId::proportional(theme::BODY);
        let page = self.you(a.keys);
        for (i, (key, name, value)) in rows.into_iter().enumerate() {
            let row = egui::Rect::from_min_size(area.min + Vec2::new(0., i as f32 * BAR_ROW), Vec2::new(area.width(), BAR_ROW));
            let hover = ui.interact(row, ui.scope_id().with(("bar", i)), Sense::hover());
            p.text(row.left_center(), egui::Align2::LEFT_CENTER, name, font.clone(), theme::TEXT);
            p.text(row.right_center(), egui::Align2::RIGHT_CENTER, format!("{:.2}", ms_rec::shown::stars(a.keys, *value)), font.clone(), theme::TEXT);
            let track = egui::Rect::from_min_max(egui::pos2(row.left() + BAR_LABEL, row.center().y - 3.), egui::pos2(row.right() - BAR_VALUE, row.center().y + 3.));
            p.rect_filled(track, 3., theme::BUTTON);
            let mut fill = track;
            fill.set_width(track.width() * (*value as f32 / BAR_FULL).clamp(0., 1.));
            p.rect_filled(fill, 3., ACCENT);
            let you = page.and_then(|pg| your_skill(pg, a.keys, key));
            if let (true, Some(y)) = (hover.hovered(), you) {
                let x = track.left() + track.width() * (y as f32 / BAR_FULL).clamp(0., 1.);
                p.vline(x, (track.top() - 3.)..=(track.bottom() + 3.), Stroke::new(2., TEXT.gamma_multiply(0.6)));
            }
            hover.on_hover_text(format!("{name} {:.2}{}{}\n\n{BARS_TIP}", ms_rec::shown::stars(a.keys, *value), versus(a.keys, *value, you), if you.is_some() { " — the tick is you" } else { "" }));
        }
    }

    /// The context line, each part with its own explanation.
    fn context_line(&self, ui: &mut Ui, a: &Arc<Analysis>) {
        let (_, _, known) = self.effective();
        let ln_share = a.ln_notes / a.notes.max(1) as f64;
        let dan = ms_rec::dans::label(a.keys, a.overall, ln_share, Some(&a.dan_vibro));
        let mut parts: Vec<(String, String)> = vec![(format!("{}K", a.keys), "Keymode".into())];
        if let Some(d) = dan {
            parts.push((d, "The dan this map's difficulty matches. Tap dans under 22% LN, LN dans from 50%, both in between; a 4K map with 10% of its notes in fast same-column runs reads on the Vibro dans too.".into()));
        }
        parts.push((format!("{:.2}×{}", a.rate, if known { "" } else { " (osu! rate unknown)" }), if known { "Rate read from osu!'s mods.".into() } else { "osu! hasn't reported a rate yet: 1.00× assumed.".into() }));
        if a.sv_ignored {
            parts.push(("SV ignored (Constant Speed)".into(), "lazer's Constant Speed is on: scroll speed changes don't count.".into()));
        }
        // Chart length, or time left (real seconds at this rate) in play.
        if let Some((t0, t1)) = a.span {
            let playing = self.live;
            let left = (t1 - playing.unwrap_or(t0).max(t0)).max(0.) / 1000. / a.rate;
            parts.push((format!("{}{}", fmt::clock(left), if playing.is_some() { " left" } else { "" }), if playing.is_some() { "Time left at this rate." } else { "First to last note at this rate." }.into()));
        }
        // Long-note heads / all note heads: each LN counts once.
        parts.push((format!("LN {:.0}%", 100. * ln_share), format!("{:.0} long notes of {} notes", a.ln_notes, a.notes)));
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 0.;
            for (i, (text, tip)) in parts.into_iter().enumerate() {
                if i > 0 {
                    theme::dim(ui, "  ·  ");
                }
                theme::dim(ui, text).on_hover_text(tip);
            }
        });
    }

    /// Right-click on the title or the head.
    fn map_menu(&mut self, ui: &mut Ui, a: &Analysis) {
        if ui.button("Copy “Artist - Title [Version]”").clicked() {
            self.copy(search_text(a));
            ui.close();
        }
        if ui.button("Open folder").on_hover_text("The folder holding this .osu file").clicked() {
            if let Some(dir) = a.path.parent() {
                open_external(&dir.to_string_lossy());
            }
            ui.close();
        }
        if ui.add_enabled(a.bid.is_some(), egui::Button::new("Open on osu.ppy.sh")).clicked() {
            open_external(&format!("https://osu.ppy.sh/b/{}", a.bid.unwrap_or(0)));
            ui.close();
        }
        ui.menu_button("Convert to", |ui| {
            for k in SUPPORTED_KEYS.filter(|k| *k != a.keys) {
                if ui.button(format!("{k}K")).on_hover_text("Adds it to osu! as its own difficulty, without the playlist").clicked() {
                    self.q.put(Task::ConvertSelected(k));
                    ui.close();
                }
            }
        });
    }

    /// Title and difficulty lines: a click copies the search text, a right-click opens the map menu.
    fn title(&mut self, ui: &mut Ui, a: &Arc<Analysis>) {
        let d = self.s.display.clone();
        let mut lines = Vec::new();
        if d.title {
            let job = theme::job(&vec![bold(if a.title.is_empty() { "?" } else { &a.title }), plain("  "), small(a.artist.clone())], 1, ui.available_width());
            lines.push(ui.add(egui::Label::new(job).sense(Sense::click())));
        }
        if d.version {
            lines.push(ui.add(egui::Label::new(RichText::new(format!("[{}]  {}", a.version, a.creator)).color(DIM)).sense(Sense::click())));
        }
        for r in lines {
            let r = r.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text("Click: copy “Artist - Title [Version]” for osu!'s search. Right-click: more.");
            if r.clicked() {
                self.copy(search_text(a));
            }
            r.context_menu(|ui| self.map_menu(ui, a));
        }
    }

    /// The selected map's prediction, or why there is none yet. Under 50 own plays it is mostly the population's guess, so
    /// none is shown (user 2026-11-02 kept that gate).
    fn prediction(&self, ui: &mut Ui) {
        if !self.s.display.prediction {
            return;
        }
        if let Some(c) = self.card.as_ref().filter(|c| c.sure) {
            let (line, tip) = fmt::prediction_line(c, self.card_is_pp(), &self.units);
            theme::line(ui, &line, 1).on_hover_text(tip);
        } else if let (None, Some(note)) = (&self.card, &self.card_note) {
            theme::dim_small(ui, note);
        }
    }

    pub fn main_ui(&mut self, ui: &mut Ui) {
        ui.spacing_mut().item_spacing.y = 2.;
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("☰  Playlist").selected(self.s.recs_open)).on_hover_text("Show or hide the playlist window (Next, Skip, Tracking, PP/NPS/Skills lists)").clicked() {
                self.s.recs_open = !self.s.recs_open;
                self.request_save();
            }
            // Next to Playlist, not beside the NPS where it read as part of it (user 2026-11-02).
            if ui.add(egui::Button::new(if self.s.details { "▴  Details" } else { "▾  Details" }).selected(self.s.details)).on_hover_text("Skill details: one bar per skill the map has").clicked() {
                self.toggle_details(ui.ctx());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("⋯").on_hover_text("Settings").clicked() {
                    self.open.menu = !self.open.menu;
                }
            });
        });
        ui.add_space(4.);
        let analysis = self.analysis().cloned();
        let d = self.s.display.clone();
        let foot = if d.status { 22. } else { 0. };
        let rows = analysis.as_ref().map_or(0, |a| shown_bars(a).count()) as f32;
        let detail_height = if self.s.details { rows * BAR_ROW + 6. } else { 0. };
        // Details add their rows under an unchanged timeline (user 2026-11-02: a separate details size grew the graph).
        // Without an analysis the window keeps what it has.
        let extra = if analysis.is_some() || !self.s.details { detail_height } else { self.extra };
        if extra != self.extra {
            self.extra = extra;
            let (w, h) = self.s.win;
            self.resized = Some((w, h + extra));
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(Vec2::new(w, h + extra)));
        }
        if let Some(a) = &analysis {
            self.title(ui, a);
            if d.info {
                self.context_line(ui, a);
            }
            self.prediction(ui);
            self.head(ui, a);
            if self.s.display.timeline {
                self.timeline(ui, a, (ui.available_height() - foot - detail_height).max(30.));
            }
            if self.s.details {
                self.details(ui, a);
            }
        } else if let Some(p) = &self.pending {
            if d.title {
                theme::line(ui, &vec![bold(p.title.clone()), plain("  "), small(p.artist.clone())], 1);
            }
            if d.version {
                theme::dim(ui, format!("[{}]  {}", p.version, p.creator));
            }
            if d.info {
                theme::dim(ui, p.context.clone());
            }
            self.prediction(ui);
            ui.horizontal(|ui| {
                ui.label(RichText::new("…").size(BODY * 3.).color(Color32::from_gray(0x55)));
                ui.label("Calculating…");
            });
        } else {
            let text = self.shown.as_ref().and_then(|s| s.result.as_ref().err().cloned()).unwrap_or_else(|| "Select a mania map in osu!".to_string());
            if let Some(s) = &self.shown {
                ui.label(s.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("–").size(BODY * 3.).color(Color32::from_gray(0x55)));
                ui.label(text);
            });
        }
        if !d.status {
            return;
        }
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.horizontal(|ui| {
                // Drawn here: the watcher replaces self.foot on every status, which erased a stored notice.
                let continuing = self.web.continued && self.web.info.as_ref().is_some_and(|i| i.running);
                let flash = self.flash.as_ref().filter(|f| f.1 > Instant::now()).map_or("", |f| f.0.as_str());
                if let Some(f) = &self.flash {
                    ui.ctx().request_repaint_after(f.1.saturating_duration_since(Instant::now()));
                }
                let foot = [flash, self.foot.as_str(), if continuing { "website charts continuing" } else { "" }].iter().filter(|t| !t.is_empty()).copied().collect::<Vec<_>>().join(" · ");
                ui.add(egui::Label::new(RichText::new(&foot).color(DIM).text_style(TextStyle::Small)).truncate()).on_hover_text(foot);
            });
        });
    }

    /// Details grow the window by their rows (main_ui resizes); collapsing returns to the compact size.
    fn toggle_details(&mut self, ctx: &egui::Context) {
        // The size being left is the one to come back to, whatever tracking caught (user 2026-11-02: it came back smaller).
        let now = ctx.input(|i| i.viewport_rect().size());
        if self.resized.is_none() && now.x > 1. {
            self.s.win = (now.x, now.y - self.extra);
        }
        self.s.details = !self.s.details;
        self.request_save();
    }

    /// Remember the sizes and places the player gave the window. `resized`: our own request has not landed yet.
    pub fn track_window(&mut self, ctx: &egui::Context) {
        // Size from the viewport itself: Wayland has no window position, so `inner_rect` is None there and G835LX never
        // saved a size (2026-11-02: 442×238 on screen, 442×214 saved).
        let (size, outer, fullscreen, maximized, minimized) = ctx.input(|i| {
            let v = i.viewport();
            (i.viewport_rect().size(), v.outer_rect, v.fullscreen, v.maximized, v.minimized)
        });
        if fullscreen == Some(true) || maximized == Some(true) || minimized == Some(true) {
            return;
        }
        {
            let size = (size.x, size.y);
            if self.resized.take_if(|w| (w.0 - size.0).abs() < 2. && (w.1 - size.1).abs() < 2.).is_none() && self.resized.is_none() && size.0 > 1. {
                let compact = (size.0, size.1 - self.extra);
                if self.s.win != compact {
                    self.s.win = compact;
                    self.request_save();
                }
            }
        }
        if let Some(o) = outer {
            let p = Some((o.left(), o.top()));
            if self.s.win_pos != p {
                self.s.win_pos = p;
                self.request_save();
            }
        }
    }
}

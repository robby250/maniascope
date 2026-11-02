//! The Breeze Dark look of the GTK app: colours from its colors.css, embedded Liberation Sans with real bold, gradient text.
use crate::fmt::{Line, Span};
use eframe::egui::{self, text::{LayoutJob, TextFormat}, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Sense, Shape, Stroke, TextWrapMode, Ui, Vec2};
use std::sync::Arc;

pub const BG: Color32 = Color32::from_rgb(0x20, 0x23, 0x26);
pub const VIEW: Color32 = Color32::from_rgb(0x14, 0x16, 0x18);
pub const BUTTON: Color32 = Color32::from_rgb(0x29, 0x2c, 0x30);
pub const ROW_HOVER: Color32 = Color32::from_rgb(0x2c, 0x3f, 0x4d);
pub const ROW_CURRENT: Color32 = Color32::from_rgb(0x23, 0x3a, 0x34); // the pick being played, apart from the list under it
pub const BORDER: Color32 = Color32::from_rgb(0x4c, 0x4e, 0x51);
pub const TEXT: Color32 = Color32::from_rgb(0xfc, 0xfc, 0xfc);
pub const DIM: Color32 = Color32::from_rgb(0xa1, 0xa9, 0xb1);
pub const ACCENT: Color32 = Color32::from_rgb(0x3d, 0xae, 0xe9);
pub const CHIP_ON: Color32 = Color32::from_rgb(0x57, 0xc2, 0x6e);
pub const CHIP_OFF: Color32 = Color32::from_rgb(0xe5, 0x53, 0x4b);
pub const BODY: f32 = 13.33; // GTK's 10 pt
pub const SMALL: f32 = 11.33; // its 0.85 scale

pub fn apply(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    // Embedded: a list of system folders missed Fedora's, and the app fell back to egui's Ubuntu (user 2026-11-01).
    let regular: &[u8] = include_bytes!("../assets/LiberationSans-Regular.ttf");
    let bold: &[u8] = include_bytes!("../assets/LiberationSans-Bold.ttf");
    for (name, data, family) in [("body", regular, FontFamily::Proportional), ("bold", bold, FontFamily::Name("bold".into()))] {
        fonts.font_data.insert(name.to_string(), Arc::new(FontData::from_static(data)));
        fonts.families.entry(family).or_default().insert(0, name.to_string());
    }
    // DejaVu Sans subset: Windows' system fallback drew `☰` as a box (user 2026-11-01). New icons: rebuild it with pyftsubset.
    fonts.font_data.insert("icons".to_string(), Arc::new(FontData::from_static(include_bytes!("../assets/Icons.ttf"))));
    fonts.families.entry(FontFamily::Proportional).or_default().insert(1, "icons".to_string());
    let fallback = fonts.families[&FontFamily::Proportional].clone();
    fonts.families.entry(FontFamily::Name("bold".into())).or_default().extend(fallback);
    ctx.set_fonts(fonts);
    // Windows keeps its light title bar over a dark app unless the window is told (user 2026-11-01).
    ctx.set_theme(egui::Theme::Dark);
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = BG;
    v.extreme_bg_color = VIEW;
    v.faint_bg_color = BUTTON;
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = Color32::from_rgb(0x1f, 0x4a, 0x61);
    v.selection.stroke = Stroke::new(1., TEXT);
    v.hyperlink_color = Color32::from_rgb(0x1d, 0x99, 0xf3);
    let radius = CornerRadius::same(4);
    for (w, fill, stroke) in [(&mut v.widgets.inactive, BUTTON, BORDER), (&mut v.widgets.hovered, Color32::from_rgb(0x33, 0x3a, 0x41), ACCENT), (&mut v.widgets.active, Color32::from_rgb(0x1f, 0x4a, 0x61), ACCENT), (&mut v.widgets.noninteractive, BG, BORDER), (&mut v.widgets.open, BUTTON, ACCENT)] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.bg_stroke = Stroke::new(1., stroke);
        w.corner_radius = radius;
        w.fg_stroke = Stroke::new(1., TEXT);
        w.expansion = 0.; // egui's hover growth shifts the outline by a pixel
    }
    ctx.set_visuals(v);
    ctx.all_styles_mut(|s| {
        s.text_styles.insert(egui::TextStyle::Body, FontId::proportional(BODY));
        s.text_styles.insert(egui::TextStyle::Button, FontId::proportional(BODY));
        s.text_styles.insert(egui::TextStyle::Small, FontId::proportional(SMALL));
        s.spacing.item_spacing = Vec2::new(8., 3.);
        s.spacing.button_padding = Vec2::new(9., 4.);
        s.spacing.interact_size.y = 24.;
        s.spacing.tooltip_width = 300.; // wraps inside the app's narrow windows: wider ran off their edge
    });
}

fn format(s: &Span, size: Option<f32>) -> TextFormat {
    let px = size.unwrap_or(if s.small { SMALL } else { BODY });
    let family = if s.bold { FontFamily::Name("bold".into()) } else { FontFamily::Proportional };
    TextFormat { font_id: FontId::new(px, family), color: s.color.unwrap_or(if s.small { DIM } else { TEXT }), ..TextFormat::default() }
}

pub fn job(line: &Line, rows: usize, width: f32) -> LayoutJob {
    let mut j = LayoutJob::default();
    for s in line {
        j.append(&s.text, 0., format(s, None));
    }
    j.wrap.max_width = width;
    j.wrap.max_rows = rows;
    j.wrap.break_anywhere = false;
    j.wrap.overflow_character = Some('…');
    j
}

/// One line capped at `rows`: longer text wraps, then ends in ….
pub fn line(ui: &mut Ui, l: &Line, rows: usize) -> egui::Response {
    let width = ui.available_width();
    ui.add(egui::Label::new(job(l, rows, width)).wrap_mode(TextWrapMode::Wrap))
}

/// FixedBlock: stacked lines, each capped; the whole block as tooltip. The playlist's top is streamed and must never grow
/// (user 2026-10-30); it may shrink, so no space is reserved.
pub fn block(ui: &mut Ui, lines: &[(Line, usize)], tip: Option<&str>) {
    let r = ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 1.;
        for (l, rows) in lines {
            if !l.is_empty() {
                line(ui, l, *rows);
            }
        }
    }).response;
    if let Some(t) = tip {
        r.on_hover_text(t);
    }
}

pub fn dim(ui: &mut Ui, text: impl Into<String>) -> egui::Response {
    ui.label(egui::RichText::new(text.into()).color(DIM))
}

pub fn dim_small(ui: &mut Ui, text: impl Into<String>) -> egui::Response {
    ui.label(egui::RichText::new(text.into()).color(DIM).size(SMALL))
}

pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(8.);
    ui.label(egui::RichText::new(title).size(SMALL).color(DIM).family(FontFamily::Name("bold".into())));
}

fn lerp(stops: &[Color32], f: f32) -> Color32 {
    if stops.len() == 1 {
        return stops[0];
    }
    let at = f.clamp(0., 1.) * (stops.len() - 1) as f32;
    let i = (at.floor() as usize).min(stops.len() - 2);
    stops[i].lerp_to_gamma(stops[i + 1], at - i as f32)
}

/// Text filled with a smooth left→right colour gradient over its first `end` characters' width, on an optional pill (lazer's
/// star badge). The rest of the text takes the last colour. Glyph quads are tinted per corner, so the gradient runs inside a
/// glyph.
pub fn gradient(ui: &mut Ui, text: &str, size: f32, tail: Option<(&str, f32)>, stops: &[Color32], pill: Option<Color32>) -> egui::Response {
    let mut j = LayoutJob::default();
    j.append(text, 0., TextFormat { font_id: FontId::new(size, FontFamily::Name("bold".into())), color: Color32::WHITE, ..TextFormat::default() });
    let galley = ui.fonts_mut(|f| f.layout_job(j));
    let number = galley.size().x;
    // The tail is an ordinary galley in the last colour: through the hand-made mesh its small glyphs showed a seam
    // ("looks cut off", user 2026-11-02).
    let tail = tail.map(|(t, px)| ui.fonts_mut(|f| f.layout_no_wrap(t.to_string(), FontId::proportional(px), *stops.last().unwrap_or(&Color32::WHITE))));
    let h = galley.size().y;
    let pad = if pill.is_some() { (h * 0.2).round() } else { 0. };
    let width = number + tail.as_ref().map_or(0., |t| t.size().x);
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width + 2. * pad, h), Sense::hover());
    if let Some(t) = tail {
        let at = egui::pos2(rect.left() + pad + number, rect.center().y - t.size().y / 2.);
        ui.painter().galley(at, t, Color32::WHITE);
    }
    // Unclipped, glyphs on whole physical pixels like the tessellator does for galleys: a clip at the row box and a
    // fractional origin both shaved the digits' tops at 1.25× (user 2026-11-02).
    let painter = ui.painter();
    if let Some(c) = pill {
        painter.rect_filled(rect, CornerRadius::same((h / 2.) as u8), c);
    }
    let origin = egui::emath::GuiRounding::round_to_pixels(rect.min + Vec2::new(pad, 0.), ui.pixels_per_point());
    // Galley meshes keep texel UVs: the tessellator normalizes them, and a hand-made mesh skips it.
    let atlas = ui.fonts_mut(|f| f.font_image_size());
    for row in &galley.rows {
        let mut mesh = row.row.visuals.mesh.clone();
        for v in &mut mesh.vertices {
            v.pos += origin.to_vec2() + row.pos.to_vec2();
            v.uv.x /= atlas[0] as f32;
            v.uv.y /= atlas[1] as f32;
            v.color = lerp(stops, (v.pos.x - origin.x) / number.max(1.));
        }
        painter.add(Shape::mesh(mesh));
    }
    response
}

pub fn bar(ui: &mut Ui, fraction: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 6.), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(2), BUTTON);
    let mut fill = rect;
    fill.set_width(rect.width() * fraction.clamp(0., 1.));
    p.rect_filled(fill, CornerRadius::same(2), ACCENT);
}

/// Two knobs on one track (user 2026-11-02: "a single slider that has 2 knobs"): low and high of a range on a `step`
/// grid inside `bounds`, low ≤ high, both may meet. A press takes the nearer knob (the one on its side when they meet)
/// and keeps it for the whole drag; double- or right-click resets to `default`.
pub fn range_slider(ui: &mut Ui, name: &str, value: &mut (f64, f64), bounds: (f64, f64), step: f64, default: (f64, f64)) -> egui::Response {
    let size = Vec2::new(ui.available_width().max(60.), ui.spacing().interact_size.y);
    let (rect, mut response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, name));
    let knob = 7.;
    let track = rect.shrink2(Vec2::new(knob, 0.));
    let x_of = |v: f64| track.left() + ((v - bounds.0) / (bounds.1 - bounds.0)) as f32 * track.width();
    let snap = |x: f32| {
        let v = bounds.0 + ((x - track.left()) / track.width()).clamp(0., 1.) as f64 * (bounds.1 - bounds.0);
        (((v - bounds.0) / step).round() * step + bounds.0).clamp(bounds.0, bounds.1)
    };
    let before = *value;
    let held = response.id.with("knob");
    if response.double_clicked() || response.secondary_clicked() {
        *value = default;
    } else if let Some(p) = response.interact_pointer_pos().filter(|_| response.dragged() || response.clicked()) {
        let nearest = || {
            let at = ui.input(|i| i.pointer.press_origin()).unwrap_or(p).x; // where the press began, not where the drag is
            let (a, b) = ((at - x_of(value.0)).abs(), (at - x_of(value.1)).abs());
            if (a - b).abs() < 0.5 { at > x_of(value.1) } else { b < a }
        };
        let high = match ui.data(|d| d.get_temp::<bool>(held)) {
            Some(h) if !response.drag_started() => h,
            _ => nearest(),
        };
        if response.dragged() {
            ui.data_mut(|d| d.insert_temp(held, high));
        }
        let v = snap(p.x);
        if high { value.1 = v.max(value.0) } else { value.0 = v.min(value.1) }
    }
    if !response.dragged() {
        ui.data_mut(|d| d.remove::<bool>(held));
    }
    if *value != before {
        response.mark_changed();
    }
    let p = ui.painter();
    let rail = egui::Rect::from_center_size(rect.center(), Vec2::new(track.width(), 4.));
    p.rect_filled(rail, CornerRadius::same(2), BUTTON);
    let (a, b) = (x_of(value.0), x_of(value.1));
    p.rect_filled(egui::Rect::from_x_y_ranges(a..=b.max(a + 1.), rail.y_range()), CornerRadius::same(2), ACCENT);
    let stroke = Stroke::new(1., if response.hovered() || response.dragged() { ACCENT } else { BORDER });
    for x in [a, b] {
        p.circle(egui::pos2(x, rect.center().y), knob, BUTTON, stroke);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2};
    use egui_kittest::kittest::Queryable;
    use egui_kittest::Harness;

    #[test]
    fn each_knob_drags_alone_low_stays_under_high_and_both_can_meet_at_one() {
        let mut h = Harness::builder().with_size(vec2(400., 40.)).build_ui_state(|ui, v: &mut (f64, f64)| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            range_slider(ui, "Range", v, (0.7, 1.5), 0.05, (0.75, 1.5));
        }, (0.75, 1.5));
        h.run();
        let r = h.get_by_label("Range").rect();
        let y = r.center().y;
        let x = move |v: f64| r.left() + 7. + ((v - 0.7) / 0.8) as f32 * (r.width() - 14.); // knob radius 7
        let drag = |h: &mut Harness<(f64, f64)>, from: f32, to: f32| {
            h.drag_at(pos2(from, y));
            h.run();
            h.hover_at(pos2(to, y));
            h.run();
            h.drop_at(pos2(to, y));
            h.run();
        };
        drag(&mut h, x(1.5), x(1.0));
        assert_eq!(*h.state(), (0.75, 1.0));
        drag(&mut h, x(0.75), x(1.3)); // the low knob stops at the high one
        assert!((h.state().0 - 1.0).abs() < 1e-9 && (h.state().1 - 1.0).abs() < 1e-9, "{:?}", h.state());
        drag(&mut h, x(1.0) + 10., x(1.2)); // met: right of them takes the high knob
        assert!((h.state().0 - 1.0).abs() < 1e-9 && (h.state().1 - 1.2).abs() < 1e-9, "{:?}", h.state());
    }
}

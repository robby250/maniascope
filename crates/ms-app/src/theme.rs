//! The osu!lazer look: its overlay palette (one hue, `OverlayColourProvider`), sheared buttons (`ShearedButton`), cards
//! over a backdrop (the selected map's image, or a fixed one), embedded Nunito, gradient text. Every colour, radius and
//! size of the app is here.
use crate::fmt::{Line, Span};
use eframe::egui::{self, text::{LayoutJob, TextFormat}, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Sense, Shape, Stroke, TextWrapMode, Ui, Vec2};
use std::sync::Arc;

/// lazer's `OverlayColourProvider.getColour`: its Purple scheme's hue, each colour a saturation and a lightness.
const fn hsl(s: f32, l: f32) -> Color32 {
    const HUE: f32 = 262. / 60.;
    let c = (1. - (2. * l - 1.).abs()) * s;
    let x = c * (1. - (HUE % 2. - 1.).abs());
    let m = l - c / 2.;
    // 240° ≤ hue < 300°: (x, 0, c)
    Color32::from_rgb(((x + m) * 255.) as u8, (m * 255.) as u8, ((c + m) * 255.) as u8)
}

const fn alpha(c: Color32, a: f32) -> Color32 {
    Color32::from_rgba_premultiplied((c.r() as f32 * a) as u8, (c.g() as f32 * a) as u8, (c.b() as f32 * a) as u8, (255. * a) as u8)
}

pub const BG: Color32 = hsl(0.3, 0.09); // the fixed backdrop's base
pub const VIEW: Color32 = hsl(0.3, 0.05);
pub const CARD_SOLID: Color32 = hsl(0.3, 0.11); // a card's own colour: the darkest a card's image gets
pub const CARD: Color32 = alpha(CARD_SOLID, 0.78); // see-through over the fixed backdrop
pub const BUTTON: Color32 = alpha(hsl(0.25, 0.24), 0.85);
pub const ROW_HOVER: Color32 = alpha(hsl(0.35, 0.34), 0.85);
pub const ROW_CURRENT: Color32 = alpha(Color32::from_rgb(0x2b, 0x5c, 0x4c), 0.85); // the pick being played, apart from the list under it
pub const BORDER: Color32 = alpha(hsl(0.3, 0.55), 0.55);
pub const TEXT: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);
pub const DIM: Color32 = hsl(0.4, 0.82);
pub const PURPLE: Color32 = hsl(1., 0.7); // lazer's Highlight1: ticked boxes, section markers
pub const ACCENT: Color32 = Color32::from_rgb(0x45, 0xcc, 0xff); // cyan: what is selected, bars, sliders
pub const SELECTED: Color32 = alpha(Color32::from_rgb(0x1f, 0x7f, 0xa6), 0.9); // a selected button's fill, ACCENT its outline
pub const PRIMARY: Color32 = Color32::from_rgb(0xff, 0x66, 0xab); // lazer's Pink1: the one main action (Next)
pub const CHIP_ON: Color32 = Color32::from_rgb(0x57, 0xc2, 0x6e);
pub const CHIP_OFF: Color32 = Color32::from_rgb(0xe5, 0x53, 0x4b);
pub const CONVERT: Color32 = Color32::from_rgb(0xff, 0xcc, 0x22); // the keymode-convert badge, beside ACCENT's website one
pub const SKILL: Color32 = Color32::from_rgb(0x4f, 0xb8, 0x9e); // the picked-skill badge
pub const INVERT: Color32 = Color32::from_rgb(0xb5, 0x8c, 0xf0); // the Invert badge, apart from the convert one
// lazer's `OsuColour.ForModType`
const MOD_REDUCTION: Color32 = Color32::from_rgb(0xb2, 0xff, 0x66);
const MOD_INCREASE: Color32 = Color32::from_rgb(0xff, 0x66, 0x66);
const MOD_CONVERSION: Color32 = Color32::from_rgb(0x8c, 0x66, 0xff);
const MOD_AUTOMATION: Color32 = Color32::from_rgb(0x66, 0xcc, 0xff);
const MOD_FUN: Color32 = PRIMARY;
const MOD_SYSTEM: Color32 = CONVERT;
pub const SHEAR: f32 = 0.2; // lazer's `OsuGame.SHEAR`
pub const RADIUS: u8 = 7; // its `ShearedButton.CORNER_RADIUS`
/// Everything 10 % bigger than the look this replaced (user 2026-11-21): egui's zoom scales every point.
pub const ZOOM: f32 = 1.1;

/// A window's size as its builder takes it: the builder alone counts in unzoomed points, everything read back from
/// a window or sent to it later is zoomed (a saved size fed back unscaled shrank the window 10 % a start).
pub fn window(size: [f32; 2]) -> [f32; 2] {
    [size[0] * ZOOM, size[1] * ZOOM]
}

/// The same for a window opened from the running app: egui sizes it with the zoom in force when it is made, which is
/// still 1 on the app's first pass and ZOOM after. Scaled always, the playlist grew 10 % each time it was closed and
/// opened (user 2026-11-22).
pub fn window_in(ctx: &egui::Context, size: [f32; 2]) -> [f32; 2] {
    let f = ZOOM / ctx.zoom_factor();
    [size[0] * f, size[1] * f]
}

thread_local! { static SCALE: std::cell::Cell<f32> = const { std::cell::Cell::new(1.) }; }

/// The window being drawn is this much bigger than its design size: every size in it multiplies by this.
pub fn scale() -> f32 {
    SCALE.with(std::cell::Cell::get)
}

/// A window's scale from its size against its design size: in 2.5 % steps (each step is a set of glyphs in the font
/// atlas), within what stays readable and sharp.
pub fn fit(ratio: f32) -> f32 {
    ((ratio / 0.025).round() * 0.025).clamp(0.6, 2.5)
}

/// A size of the design in this window.
pub fn px(v: f32) -> f32 {
    v * scale()
}

/// Draw this window `s` times its design size (user 2026-11-22: a window resized diagonally scales everything in it).
/// egui's zoom is one factor for every window, so the sizes themselves scale: the style here, the rest through `px`.
pub fn scaled(ui: &mut Ui, s: f32) {
    SCALE.with(|c| c.set(s));
    ui.ctx().all_styles_mut(|st| sizes(st, s));
    let style = ui.ctx().global_style();
    ui.set_style(style); // the window's own Ui was made before: its children take their style from it
}

fn sizes(st: &mut egui::Style, s: f32) {
    st.text_styles.insert(egui::TextStyle::Body, FontId::proportional(BODY * s));
    st.text_styles.insert(egui::TextStyle::Button, FontId::proportional(BODY * s));
    st.text_styles.insert(egui::TextStyle::Small, FontId::proportional(SMALL * s));
    st.text_styles.insert(egui::TextStyle::Monospace, FontId::monospace(12. * s));
    st.text_styles.insert(egui::TextStyle::Heading, FontId::proportional(18. * s));
    let d = egui::style::Spacing::default();
    let sp = &mut st.spacing;
    sp.item_spacing = Vec2::new(8., 3.) * s;
    sp.button_padding = Vec2::new(9., 4.) * s;
    sp.interact_size = Vec2::new(d.interact_size.x, 24.) * s;
    sp.tooltip_width = 300. * s; // wraps inside the app's narrow windows: wider ran off their edge
    (sp.indent, sp.slider_width, sp.slider_rail_height, sp.combo_width, sp.text_edit_width) = (d.indent * s, d.slider_width * s, d.slider_rail_height * s, d.combo_width * s, d.text_edit_width * s);
    (sp.icon_width, sp.icon_width_inner, sp.icon_spacing) = (d.icon_width * s, d.icon_width_inner * s, d.icon_spacing * s);
    let radius = CornerRadius::same((f32::from(RADIUS) * s) as u8);
    let w = &mut st.visuals.widgets;
    for look in [&mut w.inactive, &mut w.hovered, &mut w.active, &mut w.noninteractive, &mut w.open] {
        look.corner_radius = radius;
    }
}

const BODY: f32 = 13.33; // GTK's 10 pt
const SMALL: f32 = 11.33; // its 0.85 scale

pub fn body() -> f32 {
    BODY * scale()
}

pub fn small() -> f32 {
    SMALL * scale()
}

pub fn apply(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    // Embedded: a list of system folders missed Fedora's, and the app fell back to egui's Ubuntu (user 2026-11-01).
    // Nunito's variable font cut at two weights (fontTools varLib.instancer wght=600 / 800): lazer's Torus is not free.
    let regular: &[u8] = include_bytes!("../assets/Nunito-SemiBold.ttf");
    let bold: &[u8] = include_bytes!("../assets/Nunito-ExtraBold.ttf");
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
    ctx.set_theme(egui::Theme::Dark);
    dark_frame(ctx);
    ctx.set_zoom_factor(ZOOM);
    ctx.options_mut(|o| o.zoom_with_keyboard = false);
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = hsl(0.3, 0.12);
    v.window_stroke = Stroke::new(1., BORDER);
    v.extreme_bg_color = VIEW;
    v.faint_bg_color = BUTTON;
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = SELECTED;
    v.selection.stroke = Stroke::new(1., TEXT);
    v.slider_trailing_fill = true;
    v.hyperlink_color = ACCENT;
    v.window_corner_radius = CornerRadius::same(RADIUS + 3);
    v.menu_corner_radius = CornerRadius::same(RADIUS);
    for (w, fill, stroke) in [(&mut v.widgets.inactive, BUTTON, BORDER), (&mut v.widgets.hovered, ROW_HOVER, ACCENT), (&mut v.widgets.active, SELECTED, ACCENT), (&mut v.widgets.noninteractive, BG, BORDER), (&mut v.widgets.open, BUTTON, ACCENT)] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.bg_stroke = Stroke::new(1., stroke);
        w.fg_stroke = Stroke::new(1., TEXT);
        w.expansion = 0.; // egui's hover growth shifts the outline by a pixel
    }
    ctx.set_visuals(v);
    ctx.all_styles_mut(|st| sizes(st, 1.));
}

fn format(s: &Span, size: Option<f32>) -> TextFormat {
    let px = size.unwrap_or(if s.small { small() } else { body() });
    let family = if s.bold { FontFamily::Name("bold".into()) } else { FontFamily::Proportional };
    TextFormat { font_id: FontId::new(px, family), color: s.color.unwrap_or(if s.small { DIM } else { TEXT }), ..TextFormat::default() }
}

/// Windows keeps a light title bar over a dark app unless each window is told: set_theme colours only egui (user 2026-11-01,
/// back with eframe 0.37). Sent on a window's first pass, so a reopened window gets it again.
pub fn dark_frame(ctx: &egui::Context) {
    if ctx.cumulative_pass_nr() == 0 {
        ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(egui::SystemTheme::Dark));
    }
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

/// The map's title, a size up from the body (user 2026-11-22: "a bit bigger"), then its artist; one row.
pub fn title_job(title: &str, artist: &str, width: f32) -> LayoutJob {
    let mut j = LayoutJob::default();
    j.append(title, 0., TextFormat { font_id: FontId::new(body() * 1.1, FontFamily::Name("bold".into())), color: TEXT, ..TextFormat::default() });
    j.append(artist, body() * 0.6, TextFormat { font_id: FontId::proportional(small()), color: DIM, valign: egui::Align::BOTTOM, ..TextFormat::default() });
    j.wrap.max_width = width;
    j.wrap.max_rows = 1;
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
        ui.spacing_mut().item_spacing.y = px(1.);
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
    ui.label(egui::RichText::new(text.into()).color(DIM).size(small()))
}

/// A settings group's title: lazer's marker pill, then the name.
pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(px(8.));
    ui.horizontal(|ui| {
        let (mark, _) = ui.allocate_exact_size(Vec2::new(px(5.), body()), Sense::hover());
        ui.painter().rect_filled(mark, px(2.5), PURPLE);
        ui.label(egui::RichText::new(title).family(FontFamily::Name("bold".into())));
    });
}

/// A panel over the backdrop: title block, list row, settings group.
pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> egui::InnerResponse<R> {
    card_in(ui, card_frame(CARD), |ui| {
        ui.spacing_mut().item_spacing.y = px(3.);
        add(ui)
    })
}

/// A card in `frame`. Over a map's image it is tinted glass: the card's colour, nearly solid at its left where the
/// text starts and half see-through at its right, where the image shows (user 2026-11-22), inside a light rim.
pub fn card_in<R>(ui: &mut Ui, frame: egui::Frame, add: impl FnOnce(&mut Ui) -> R) -> egui::InnerResponse<R> {
    let id = frost_id(ui.ctx()); // before the lock `data` takes
    let frost: Option<(egui::Rect, Vec<Layer>, bool)> = ui.data(|d| d.get_temp(id));
    let Some((window, layers, black)) = frost.filter(|f| !f.1.is_empty()) else { return frame.show(ui, add) };
    let glass = ui.painter().add(Shape::Noop); // under the content, drawn once the card's size is known
    let rim = Stroke::new(px(1.), alpha(hsl(0.5, 0.82), 0.3));
    let out = frame.fill(Color32::TRANSPARENT).stroke(if frame.stroke.color == card_frame(CARD).stroke.color { rim } else { frame.stroke }).show(ui, add);
    let rect = out.response.rect;
    let mut shapes: Vec<Shape> = Vec::new();
    if black {
        // a black window shows no image: the card shows its own part of it
        shapes.extend(layers.iter().map(|l| {
            let all = cover(l.size, window);
            let at = |p: egui::Pos2| all.min + (p - window.min) / window.size() * all.size();
            let t = (255. * dimming(l.brightness(all))) as u8;
            egui::epaint::RectShape::filled(rect, frame.corner_radius, Color32::from_rgb(t, t, t).gamma_multiply(l.fade)).with_texture(l.window, egui::Rect::from_min_max(at(rect.min), at(rect.max))).into()
        }));
    }
    let tint = if frame.fill == CARD { CARD_SOLID } else { frame.fill.to_opaque() };
    let uv = egui::Rect::from_min_max(egui::pos2(0.5 / 64., 0.), egui::pos2(63.5 / 64., 1.));
    shapes.push(egui::epaint::RectShape::filled(rect, frame.corner_radius, tint).with_texture(card_glass(ui.ctx()), uv).into());
    ui.painter().set(glass, Shape::Vec(shapes));
    out
}

/// The cards' opacity, left to right, as a texture: a rounded shape takes one colour, or a texture.
fn card_glass(ctx: &egui::Context) -> egui::TextureId {
    let id = egui::Id::unique("card-glass");
    if let Some(t) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return t.id();
    }
    let (left, right) = (0.94f32, 0.5f32);
    let pixels: Vec<u8> = (0..64).flat_map(|x| { let f = x as f32 / 63.; [255, 255, 255, (255. * (left + (right - left) * f * f)) as u8] }).collect();
    let t = ctx.load_texture("card-glass", egui::ColorImage::from_rgba_unmultiplied([64, 1], &pixels), egui::TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, t.clone()));
    t.id()
}

/// A map's image as a window shows it, with its opacity while it fades in or out.
#[derive(Clone)]
pub struct Layer {
    pub window: egui::TextureId,
    pub size: Vec2,
    pub lum: Arc<Vec<f32>>, // the image's brightness, `backdrop::GRID` cells a side
    pub fade: f32,
}

impl Layer {
    /// How bright the part `uv` of the image is, 0–1.
    fn brightness(&self, uv: egui::Rect) -> f32 {
        let n = crate::backdrop::GRID;
        let cell = |v: f32| ((v.clamp(0., 1.) * n as f32) as usize).min(n - 1);
        let (mut sum, mut count) = (0., 0);
        for y in cell(uv.top())..=cell(uv.bottom()) {
            for x in cell(uv.left())..=cell(uv.right()) {
                sum += self.lum.get(y * n + x).copied().unwrap_or(0.5);
                count += 1;
            }
        }
        sum / count.max(1) as f32
    }
}

/// How much of an image as bright as `l` the window shows: a bright one is dimmed more, and still ends brighter than
/// a dark one (user 2026-11-22: "should be brighter with a bright background than a dark background but should be
/// much dimmer than it is currently", by "the bg part that is used in the window").
fn dimming(l: f32) -> f32 {
    ((0.09 + 0.11 * l) / l.max(0.01)).min(0.6)
}

fn frost_id(ctx: &egui::Context) -> egui::Id {
    egui::Id::unique(("frost", ctx.viewport_id()))
}

/// `c` as a window with the black background may paint it: a stream's colour key takes black out, so nothing in the window
/// is darker than a card. Lifted onto the card's colour as the cards' images are (user 2026-11-22: the 9 ★ pill's black
/// went see-through in OBS). Elsewhere `c` itself.
pub fn keyed(ui: &Ui, c: Color32) -> Color32 {
    let id = frost_id(ui.ctx()); // before the lock `data` takes
    if !ui.data(|d| d.get_temp::<(egui::Rect, Vec<Layer>, bool)>(id)).is_some_and(|f| f.2) {
        return c;
    }
    let lift = |x: u8, floor: u8| floor + (u16::from(x) * u16::from(255 - floor) / 255) as u8;
    Color32::from_rgb(lift(c.r(), CARD_SOLID.r()), lift(c.g(), CARD_SOLID.g()), lift(c.b(), CARD_SOLID.b()))
}

/// The part of an image of `size` that fills `rect` without stretching, as uv.
fn cover(size: Vec2, rect: egui::Rect) -> egui::Rect {
    let want = rect.aspect_ratio();
    let (w, h) = if size.x / size.y > want { (want * size.y / size.x, 1.) } else { (1., size.x / size.y / want) };
    egui::Rect::from_center_size(egui::pos2(0.5, 0.5), Vec2::new(w, h))
}

pub fn card_frame(fill: Color32) -> egui::Frame {
    egui::Frame::new().fill(fill).stroke(Stroke::new(1., BORDER.gamma_multiply(0.6))).corner_radius(px(f32::from(RADIUS) + 3.)).inner_margin(egui::Margin::symmetric(px(9.) as i8, px(6.) as i8))
}

/// The outline of a lazer button in `rect`: a parallelogram leaning right by SHEAR, its corners rounded.
fn sheared(rect: egui::Rect) -> Vec<egui::Pos2> {
    let lean = rect.height() * SHEAR;
    let corners = [egui::pos2(rect.left() + lean, rect.top()), rect.right_top(), egui::pos2(rect.right() - lean, rect.bottom()), rect.left_bottom()];
    let r = px(f32::from(RADIUS)).min(rect.height() / 2.).min((rect.width() - lean) / 2.);
    let mut out = Vec::with_capacity(4 * 6);
    for i in 0..4 {
        let (at, before, after) = (corners[i], corners[(i + 3) % 4], corners[(i + 1) % 4]);
        let (a, b) = (at + (before - at).normalized() * r, at + (after - at).normalized() * r);
        // a quadratic curve through the corner
        out.extend((0..=5).map(|k| { let t = k as f32 / 5.; let (u, v) = (a.lerp(at, t), at.lerp(b, t)); u.lerp(v, t) }));
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Kind { Plain, Selected, Primary }

/// Every button and tab: lazer's sheared button. `gap`: room kept clear inside its right end (a tab's cog).
pub struct Btn {
    text: egui::WidgetText,
    kind: Kind,
    sense: Sense,
    gap: f32,
}

impl Btn {
    pub fn new(text: impl Into<egui::WidgetText>) -> Self {
        Btn { text: text.into(), kind: Kind::Plain, sense: Sense::click(), gap: 0. }
    }

    pub fn selected(mut self, on: bool) -> Self {
        if on {
            self.kind = Kind::Selected;
        }
        self
    }

    /// The window's one main action.
    pub fn primary(mut self) -> Self {
        self.kind = Kind::Primary;
        self
    }

    pub fn sense(mut self, sense: Sense) -> Self {
        self.sense = sense;
        self
    }

    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }
}

impl egui::Widget for Btn {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        let pad = ui.spacing().button_padding;
        let height = ui.spacing().interact_size.y + px(2.);
        let lean = height * SHEAR;
        let galley = self.text.into_galley(ui, Some(TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Button);
        let size = Vec2::new(galley.size().x + 2. * pad.x + lean + self.gap, height.max(galley.size().y + 2. * pad.y));
        let (rect, response) = ui.allocate_exact_size(size, self.sense);
        let (text, selected) = (galley.text().to_string(), self.kind == Kind::Selected);
        response.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, ui.is_enabled(), selected, &text));
        if ui.is_rect_visible(rect) {
            let hover = ui.ctx().animate_bool_with_time(response.id, response.hovered() || response.is_pointer_button_down_on(), 0.1);
            let (fill, edge, ink) = match self.kind {
                Kind::Plain => (BUTTON.lerp_to_gamma(ROW_HOVER, hover), BORDER.lerp_to_gamma(ACCENT, hover), TEXT),
                Kind::Selected => (SELECTED, ACCENT, TEXT),
                Kind::Primary => (PRIMARY.lerp_to_gamma(Color32::WHITE, 0.18 * hover), PRIMARY.lerp_to_gamma(Color32::WHITE, 0.35), TEXT),
            };
            let fade = if ui.is_enabled() { 1. } else { 0.45 };
            ui.painter().add(Shape::convex_polygon(sheared(rect.shrink(0.5)), fill.gamma_multiply(fade), Stroke::new(px(1.5), edge.gamma_multiply(fade))));
            let at = egui::pos2(rect.left() + lean / 2. + pad.x, rect.center().y - galley.size().y / 2.);
            ui.painter().galley(at, galley, ink.gamma_multiply(fade));
        }
        response
    }
}

pub fn button(ui: &mut Ui, text: impl Into<egui::WidgetText>) -> egui::Response {
    ui.add(Btn::new(text))
}

/// An on/off setting: a box filled and ticked when on (one control for every switch, user 2026-11-21).
pub fn checkbox(ui: &mut Ui, value: &mut bool, text: impl Into<egui::WidgetText>) -> egui::Response {
    let side = body() + px(3.);
    // the name wraps in what the row has left (user 2026-11-22: a narrowed Settings window ran off its edge)
    // (one word never does: "10K" broke in two in a table cell)
    let text = text.into();
    let room = if text.text().contains(' ') { (ui.available_width() - side - ui.spacing().icon_spacing).max(side) } else { f32::INFINITY };
    let galley = text.into_galley(ui, Some(TextWrapMode::Wrap), room, egui::TextStyle::Body);
    let gap = if galley.size().x > 0. { ui.spacing().icon_spacing } else { 0. };
    let row = galley.size().y / galley.rows.len().max(1) as f32;
    let size = Vec2::new(side + gap + galley.size().x, side.max(galley.size().y).max(ui.spacing().interact_size.y - px(4.)));
    let (rect, mut response) = ui.allocate_exact_size(size, Sense::click());
    if response.clicked() {
        *value = !*value;
        response.mark_changed();
    }
    let (on, label) = (*value, galley.text().to_string());
    // a bare box (a table cell under its column's title) still needs a name for screen readers
    response.widget_info(|| egui::WidgetInfo::selected(egui::Role::CheckBox, ui.is_enabled(), on, if label.is_empty() { "on" } else { &label }));
    if ui.is_rect_visible(rect) {
        let hover = ui.ctx().animate_bool_with_time(response.id, response.hovered(), 0.1);
        let fill = ui.ctx().animate_bool_with_time(response.id.with("on"), on, 0.1);
        // the box sits on the name's first line
        let mid = rect.top() + if galley.rows.len() > 1 { side.max(row) } else { rect.height() } / 2.;
        let mark = egui::Rect::from_center_size(egui::pos2(rect.left() + side / 2., mid), Vec2::splat(side));
        let p = ui.painter();
        p.rect(mark, px(4.), VIEW.lerp_to_gamma(PURPLE, fill), Stroke::new(px(1.5), BORDER.lerp_to_gamma(PURPLE, hover.max(fill))), egui::StrokeKind::Inside);
        if on {
            let (c, u) = (mark.center(), side / 10.);
            p.line(vec![c + Vec2::new(-3. * u, 0.), c + Vec2::new(-u, 2. * u), c + Vec2::new(3. * u, -2.2 * u)], Stroke::new(px(2.), VIEW));
        }
        p.galley(egui::pos2(mark.right() + gap, mid - row / 2.), galley, TEXT);
    }
    response
}

/// A row's small mark (WEB, 7K→10K, a skill): a capsule in its colour's dark shade, outlined and written in the
/// colour, as lazer's pills are (user 2026-11-22: "prettier in the same design as the rest of the window").
pub fn badge(ui: &mut Ui, text: &str, colour: Color32) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(text.to_string(), FontId::new(px(9.5), FontFamily::Name("bold".into())), colour);
    let size = Vec2::new(galley.size().x + px(12.), galley.size().y + px(4.));
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Label, true, text));
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        p.rect(rect, size.y / 2., CARD_SOLID.lerp_to_gamma(colour, 0.22), Stroke::new(px(1.), colour.gamma_multiply(0.85)), egui::StrokeKind::Inside);
        p.galley(rect.center() - galley.size() / 2., galley, colour);
    }
    response
}

macro_rules! mods {
    ($(($acronym:literal, $file:literal, $name:literal, $colour:ident)),* $(,)?) => {
        /// lazer's mania mods (`ManiaRuleset.GetModsFor`): acronym, icon, name, its type's colour. The icons are
        /// lazer's own (ppy/osu-resources, CC BY-NC: assets/mods/LICENSE).
        pub const MODS: &[(&str, &[u8], &str, Color32)] = &[$(($acronym, include_bytes!(concat!("../assets/mods/mod-", $file, ".png")), $name, $colour)),*];
    };
}
mods![("EZ", "easy", "Easy", MOD_REDUCTION), ("NF", "no-fail", "No Fail", MOD_REDUCTION), ("HT", "half-time", "Half Time", MOD_REDUCTION), ("DC", "daycore", "Daycore", MOD_REDUCTION),
      ("NR", "no-release", "No Release", MOD_REDUCTION), ("HR", "hard-rock", "Hard Rock", MOD_INCREASE), ("SD", "sudden-death", "Sudden Death", MOD_INCREASE), ("PF", "perfect", "Perfect", MOD_INCREASE),
      ("DT", "double-time", "Double Time", MOD_INCREASE), ("NC", "nightcore", "Nightcore", MOD_INCREASE), ("FI", "fade-in", "Fade In", MOD_INCREASE), ("HD", "hidden", "Hidden", MOD_INCREASE),
      ("CO", "cover", "Cover", MOD_INCREASE), ("FL", "flashlight", "Flashlight", MOD_INCREASE), ("AC", "accuracy-challenge", "Accuracy Challenge", MOD_INCREASE),
      ("RD", "random", "Random", MOD_CONVERSION), ("DS", "dual-stages", "Dual Stages", MOD_CONVERSION), ("MR", "mirror", "Mirror", MOD_CONVERSION),
      ("DA", "difficulty-adjust", "Difficulty Adjust", MOD_CONVERSION), ("CL", "classic", "Classic", MOD_CONVERSION), ("IN", "invert", "Invert", MOD_CONVERSION),
      ("CS", "constant-speed", "Constant Speed", MOD_CONVERSION), ("HO", "hold-off", "Hold Off", MOD_CONVERSION),
      ("1K", "one-key", "One Key", MOD_CONVERSION), ("2K", "two-keys", "Two Keys", MOD_CONVERSION), ("3K", "three-keys", "Three Keys", MOD_CONVERSION), ("4K", "four-keys", "Four Keys", MOD_CONVERSION),
      ("5K", "five-keys", "Five Keys", MOD_CONVERSION), ("6K", "six-keys", "Six Keys", MOD_CONVERSION), ("7K", "seven-keys", "Seven Keys", MOD_CONVERSION), ("8K", "eight-keys", "Eight Keys", MOD_CONVERSION),
      ("9K", "nine-keys", "Nine Keys", MOD_CONVERSION), ("10K", "ten-keys", "Ten Keys", MOD_CONVERSION),
      ("AT", "autoplay", "Autoplay", MOD_AUTOMATION), ("CN", "cinema", "Cinema", MOD_AUTOMATION),
      ("WU", "wind-up", "Wind Up", MOD_FUN), ("WD", "wind-down", "Wind Down", MOD_FUN), ("MU", "muted", "Muted", MOD_FUN), ("AS", "adaptive-speed", "Adaptive Speed", MOD_FUN),
      ("SV2", "score-v2", "Score V2", MOD_SYSTEM), ("TD", "touch-device", "Touch Device", MOD_SYSTEM)];

fn sprite(ctx: &egui::Context, name: &str, png: &[u8]) -> Option<egui::TextureHandle> {
    let i = image::load_from_memory(png).ok()?.to_rgba8();
    Some(ctx.load_texture(name, egui::ColorImage::from_rgba_unmultiplied([i.width() as usize, i.height() as usize], i.as_raw()), egui::TextureOptions::LINEAR))
}

/// The mod icons' textures, decoded on first use: the hexagon, its extender, then one per `MODS` row.
fn mod_sprites(ctx: &egui::Context) -> Arc<Vec<Option<egui::TextureHandle>>> {
    let id = egui::Id::unique("mod-sprites");
    if let Some(s) = ctx.data(|d| d.get_temp::<Arc<Vec<Option<egui::TextureHandle>>>>(id)) {
        return s;
    }
    let base: [(&str, &[u8]); 2] = [("mod-icon", include_bytes!("../assets/mods/mod-icon.png")), ("mod-icon-extender", include_bytes!("../assets/mods/mod-icon-extender.png"))];
    let all = Arc::new(base.into_iter().chain(MODS.iter().map(|m| (m.0, m.1))).map(|(name, png)| sprite(ctx, name, png)).collect::<Vec<_>>());
    ctx.data_mut(|d| d.insert_temp(id, all.clone()));
    all
}

/// A mod to draw: acronym, the extender's text, tooltip, whether its settings are not the mod's own.
pub type ModShown = (String, Option<String>, String, bool);

/// The active mods as lazer shows them (`ModDisplay`, contracted: each icon slides under the next; `ModIcon`: the
/// hexagon in its type's colour, the mod's glyph, what it is set to on the extender behind). `mods`: acronym, the
/// extender's text, tooltip. An acronym lazer has no icon for is written on the hexagon, as lazer does.
pub fn mod_row(ui: &mut Ui, mods: &[ModShown]) {
    let sprites = mod_sprites(ui.ctx());
    // lazer's order (`ModExtensions.AsOrdered`): by type, then by acronym
    const TYPES: [Color32; 6] = [MOD_REDUCTION, MOD_INCREASE, MOD_CONVERSION, MOD_AUTOMATION, MOD_FUN, MOD_SYSTEM];
    let known = |m: &ModShown| MODS.iter().position(|k| k.0 == m.0);
    let mut mods: Vec<&ModShown> = mods.iter().collect();
    mods.sort_by_key(|m| (known(m).map_or(TYPES.len(), |a| TYPES.iter().position(|t| *t == MODS[a].3).unwrap_or(TYPES.len())), m.0.clone()));
    let height = body() * 1.1 + px(9.);
    let width = height * 135. / 100.; // the hexagon sprite's shape
    let wide = |m: &ModShown| if m.1.is_some() { width * (58. + 116.) / 80. } else { width };
    // each tucked under the one before as deep as lazer's contracted `ModDisplay`: spacing −25 between icons of 80 drawn at
    // scale 0.6 (48 wide), so 25/48 of an icon (user 2026-11-22: 0.36 "STILL overlap less than the ones in osu")
    let under = width * 25. / 48.;
    let total = mods.iter().map(|m| wide(m) - under).sum::<f32>() + if mods.is_empty() { 0. } else { under };
    let (rect, _) = ui.allocate_exact_size(Vec2::new(total, height), Sense::hover());
    let starts: Vec<f32> = mods.iter().scan(rect.left(), |x, m| { let at = *x; *x += wide(m) - under; Some(at) }).collect();
    // the last first: the left icon lies on top
    for (i, m) in mods.iter().enumerate().rev() {
        let (x, at) = (starts[i], known(m));
        let colour = at.map_or(MOD_SYSTEM, |a| MODS[a].3);
        let ink = keyed(ui, Color32::BLACK.lerp_to_gamma(colour, 0.1));
        let hexagon = egui::Rect::from_min_size(egui::pos2(x, rect.top()), Vec2::new(width, height));
        let all = egui::Rect::from_min_size(hexagon.min, Vec2::new(wide(m), height));
        let p = ui.painter();
        let uv = egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(1., 1.));
        if let Some(text) = &m.1 {
            let extender = egui::Rect::from_min_size(egui::pos2(x + width * 58. / 80., rect.top()), Vec2::new(width * 116. / 80., height));
            if let Some(t) = &sprites[1] {
                p.image(t.id(), extender, uv, Color32::from_rgb((f32::from(colour.r()) / 3.8) as u8, (f32::from(colour.g()) / 3.8) as u8, (f32::from(colour.b()) / 3.8) as u8)); // lazer: Darken(2.8)
            }
            p.text(extender.center() + Vec2::new(width * 6. / 80., 0.), egui::Align2::CENTER_CENTER, text, FontId::new(width * 0.3, FontFamily::Name("bold".into())), colour);
        }
        if let Some(t) = &sprites[0] {
            p.image(t.id(), hexagon, uv, colour);
        }
        match at.and_then(|a| sprites[2 + a].as_ref()) {
            // the glyphs are drawn flush with the hexagon inside its sprite (131 × 92 of 135 × 100)
            Some(t) => { p.image(t.id(), egui::Rect::from_center_size(hexagon.center(), Vec2::new(width * 131. / 135., height * 92. / 100.)), uv, ink); }
            None => { p.text(hexagon.center(), egui::Align2::CENTER_CENTER, &m.0, FontId::new(width * 0.3, FontFamily::Name("bold".into())), ink); }
        }
        if m.3 {
            // lazer's adjustment marker (`ModIcon`: a circle of 20 centred at (64, 14) of the icon's 80 × 80 box, the hexagon
            // in the box's middle): top right, its top a little outside the hexagon (user 2026-11-22)
            let at = egui::pos2(hexagon.left() + width * 64. / 80., hexagon.center().y - width / 2. + width * 14. / 80.);
            p.circle_filled(at, width * 10. / 80., ink);
            p.text(at, egui::Align2::CENTER_CENTER, "⚙", FontId::proportional(width * 0.2), colour);
        }
        let name = at.map_or(m.0.as_str(), |a| MODS[a].2);
        // the part not under its neighbour answers the pointer
        let own = if i > 0 { all.with_min_x(all.left() + under) } else { all };
        ui.interact(own, ui.scope_id().with(("mod", i)), Sense::hover()).on_hover_text(if m.2.is_empty() { name.to_string() } else { format!("{name}\n{}", m.2) });
    }
}

/// What is behind a window's cards: the selected map's image (cover-fitted; each layer with its opacity while one
/// fades into the next) over the fixed backdrop, a dark purple fall with a soft glow (user 2026-11-21: settings keep
/// the fixed one). `black`: nothing but black between the cards, for a stream's colour key (user 2026-11-22).
pub fn backdrop(ui: &Ui, images: &[Layer], black: bool) {
    let rect = ui.ctx().input(|i| i.content_rect());
    let id = frost_id(ui.ctx());
    ui.data_mut(|d| d.insert_temp(id, (rect, images.to_vec(), black)));
    let p = ui.ctx().layer_painter(egui::LayerId::background());
    if black {
        p.rect_filled(rect, 0., Color32::BLACK);
        return;
    }
    let mut mesh = egui::Mesh::default();
    let (top, bottom) = (hsl(0.42, 0.15), hsl(0.3, 0.06));
    for (at, c) in [(rect.left_top(), top), (rect.right_top(), hsl(0.5, 0.2)), (rect.left_bottom(), bottom), (rect.right_bottom(), bottom)] {
        mesh.colored_vertex(at, c);
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 2, 3);
    // the glow: a fan fading out from a point right of the middle
    let (centre, reach) = (rect.left_top() + rect.size() * Vec2::new(0.72, 0.38), rect.size().max_elem() * 0.6);
    let first = mesh.vertices.len() as u32;
    mesh.colored_vertex(centre, alpha(hsl(0.6, 0.35), 0.5));
    const RIM: u32 = 24;
    for k in 0..RIM {
        let a = k as f32 / RIM as f32 * std::f32::consts::TAU;
        mesh.colored_vertex(centre + reach * Vec2::new(a.cos(), a.sin()), Color32::TRANSPARENT);
        mesh.add_triangle(first, first + 1 + k, first + 1 + (k + 1) % RIM);
    }
    p.add(Shape::mesh(mesh));
    for l in images {
        let uv = cover(l.size, rect);
        let t = (255. * dimming(l.brightness(uv))) as u8;
        p.image(l.window, rect, uv, Color32::from_rgb(t, t, t).gamma_multiply(l.fade));
    }
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
        let shown = keyed(ui, c);
        painter.rect_filled(rect, CornerRadius::same((h / 2.) as u8), shown);
        if shown != c {
            // lifted, the darkest pill is the card's own colour: a rim keeps its shape
            painter.rect_stroke(rect, CornerRadius::same((h / 2.) as u8), Stroke::new(px(1.), alpha(hsl(0.5, 0.82), 0.3)), egui::StrokeKind::Inside);
        }
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
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), px(6.)), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(3), BUTTON);
    let mut fill = rect;
    fill.set_width(rect.width() * fraction.clamp(0., 1.));
    p.rect_filled(fill, CornerRadius::same(2), ACCENT);
}

/// Two knobs on one track (user 2026-11-02: "a single slider that has 2 knobs"): low and high of a range on a `step`
/// grid inside `bounds`, low ≤ high, both may meet. A press takes the nearer knob (the one on its side when they meet)
/// and keeps it for the whole drag; double- or right-click resets to `default`.
pub fn range_slider(ui: &mut Ui, name: &str, value: &mut (f64, f64), bounds: (f64, f64), step: f64, default: (f64, f64)) -> egui::Response {
    let size = Vec2::new(ui.available_width().max(px(60.)), ui.spacing().interact_size.y);
    let (rect, mut response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Slider, true, name));
    let knob = px(7.);
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
    let rail = egui::Rect::from_center_size(rect.center(), Vec2::new(track.width(), px(4.)));
    p.rect_filled(rail, CornerRadius::same(2), BUTTON);
    let (a, b) = (x_of(value.0), x_of(value.1));
    // rail and knobs as egui's own sliders draw theirs: one look for every slider
    p.rect_filled(egui::Rect::from_x_y_ranges(a..=b.max(a + 1.), rail.y_range()), CornerRadius::same(2), ui.visuals().selection.bg_fill);
    let look = ui.style().interact(&response);
    for x in [a, b] {
        p.circle(egui::pos2(x, rect.center().y), knob, look.bg_fill, look.fg_stroke);
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
    fn a_button_is_found_by_its_text_and_leans_right_inside_its_box() {
        let mut h = Harness::builder().with_size(vec2(300., 80.)).with_context_setup(apply).build_ui_state(|ui: &mut Ui, n: &mut i32| {
            if ui.add(Btn::new("Go").selected(*n == 1)).clicked() {
                *n += 1;
            }
        }, 0);
        h.step();
        h.get_by_label("Go").click();
        h.run();
        assert_eq!(*h.state(), 1);
        let rect = egui::Rect::from_min_size(pos2(10., 10.), vec2(100., 30.));
        let shape = sheared(rect);
        assert!(shape.iter().all(|p| rect.expand(0.01).contains(*p)), "inside its box");
        let (top, bottom) = (shape.iter().filter(|p| p.y < 12.).map(|p| p.x).fold(f32::MAX, f32::min), shape.iter().filter(|p| p.y > 38.).map(|p| p.x).fold(f32::MAX, f32::min));
        assert!(top - bottom > 4., "the top edge starts right of the bottom one (6 px at this height): {top} {bottom}");
    }

    #[test]
    fn every_mod_has_its_icon_and_one_lazer_adds_later_still_draws() {
        for m in MODS {
            assert!(image::load_from_memory(m.1).is_ok_and(|i| i.height() == 64), "{}", m.0);
        }
        assert!(["EZ", "HD", "MR", "DT", "IN", "10K", "SV2"].iter().all(|a| MODS.iter().any(|m| m.0 == *a)));
        let mods = vec![("DT".to_string(), Some("1.10x".to_string()), String::new(), true), ("ZZ".to_string(), None, String::new(), false), ("HD".to_string(), None, String::new(), false)];
        let mut h = Harness::builder().with_size(vec2(300., 80.)).with_context_setup(apply).build_ui(move |ui| {
            let before = ui.cursor().left();
            mod_row(ui, &mods);
            let width = ui.min_rect().right() - before;
            let (icon, plain) = ((body() * 1.1 + 9.) * 1.35, 3. * (body() * 1.1 + 9.) * 1.35);
            assert!(width < plain + icon * 1.45 && width > plain, "the icons overlap (lazer's contracted row), the rate's extender adds its width: {width}");
        });
        h.run();
    }

    #[test]
    fn a_black_window_paints_nothing_darker_than_a_card() {
        // user 2026-11-22: OBS's black colour key took the 9 ★ pill's black background out with the window's
        let seen = |black: bool| {
            let mut out = (Color32::RED, Color32::RED);
            let mut h = Harness::builder().with_size(vec2(200., 80.)).with_context_setup(apply).build_ui(|ui| {
                backdrop(ui, &[], black);
                out = (keyed(ui, Color32::BLACK), keyed(ui, Color32::WHITE));
            });
            h.run();
            drop(h);
            out
        };
        assert_eq!(seen(true), (CARD_SOLID, Color32::WHITE), "black lifts to the card's colour, white stays");
        assert_eq!(seen(false), (Color32::BLACK, Color32::WHITE), "lazer's black pill without the option");
    }

    #[test]
    fn a_scaled_window_draws_every_size_that_much_bigger() {
        let height = |s: f32| {
            let mut out = (0., 0.);
            let mut h = Harness::builder().with_size(vec2(600., 200.)).with_context_setup(apply).build_ui(|ui| {
                scaled(ui, s);
                let r = button(ui, "Go").rect;
                out = (r.height(), card(ui, |ui| ui.label("x").rect.height()).inner);
            });
            h.run();
            drop(h);
            out
        };
        let (one, two) = (height(1.), height(2.));
        assert!((two.0 / one.0 - 2.).abs() < 0.05 && (two.1 / one.1 - 2.).abs() < 0.1, "{one:?} {two:?}");
        assert_eq!((fit(2.004), fit(0.1), fit(9.), fit(0.9886)), (2., 0.6, 2.5, 1.));
    }

    #[test]
    fn a_bright_image_is_dimmed_more_and_still_ends_brighter_than_a_dark_one() {
        let shown = |l: f32| l * dimming(l);
        assert!(dimming(0.85) < dimming(0.25) && shown(0.85) > shown(0.25), "{} {} {} {}", dimming(0.85), dimming(0.25), shown(0.85), shown(0.25));
        assert!(shown(1.) <= 0.2 && shown(0.05) > 0.02, "dark overall, a dark image not black");
        let half = Layer { window: egui::TextureId::default(), size: vec2(16., 16.), fade: 1.,
                           lum: Arc::new((0..crate::backdrop::GRID * crate::backdrop::GRID).map(|i| if i % crate::backdrop::GRID < 8 { 1. } else { 0. }).collect()) };
        let part = |x0: f32, x1: f32| half.brightness(egui::Rect::from_min_max(pos2(x0, 0.), pos2(x1, 1.)));
        assert!(part(0., 0.45) > 0.99 && part(0.55, 1.) < 0.01, "only the part the window shows counts");
    }

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

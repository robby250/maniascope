//! The playlist window, its own so the main one can stay an OBS source: selected map, last play, the tabs and their settings.
use crate::app::App;
use crate::fmt::{self, keys_label};
use crate::settings::{mode_name, Tab};
use crate::theme::{self, ACCENT, BORDER, CHIP_OFF, CHIP_ON, DIM};
use eframe::egui::{self, containers::menu::{MenuButton, MenuConfig}, Color32, PopupCloseBehavior, RichText, ScrollArea, Sense, Stroke, Ui, UiBuilder};
use ms_data::webmaps::web_dir;
use ms_model::skills::{self, Skill, AVOID};
use ms_model::predict::{MAX_RATE, MIN_RATE};
use ms_rec::builder::{normalize_focus, normalize_rates, normalize_target, web_factor, DEFAULT_RATES, DEFAULT_TARGET};
use ms_rec::catalog::{Pick, SUPPORTED_KEYS};
use ms_rec::converts::Setting;
use ms_rec::worker::{rate_text, Task};

/// The side the slider leans to, so the left half doesn't read "20% NPS" (user 2026-11-01).
pub fn focus_label(v: f64) -> String {
    match (100. * v).round() as i64 {
        50 => "Even mix".to_string(),
        n if n < 50 => format!("{}% variety", 100 - n),
        n => format!("{n}% NPS"),
    }
}

const GEAR: f32 = 19.; // the tab cog's glyph; 11 was "way too small" (user 2026-11-01)
const NAME: f32 = 134.; // the longest row name fits
const VALUE: f32 = 106.; // "95.0% (default)" fits
const WEIGHTS: [f64; 6] = [0.25, 0.5, 0.75, 1., 1.5, 2.]; // multiplies a skill's share of the list (user 2026-11-01: "0.25x-2x")
const OTHER: [f64; 7] = [0., 0.25, 0.5, 0.75, 1., 1.5, 2.];
const CHIP_GAP: f32 = 4.;

fn row(ui: &mut Ui, name: &str, tip: Option<&str>, add: impl FnOnce(&mut Ui) -> bool) -> bool {
    let r = ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(egui::vec2(NAME, 16.), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            ui.set_min_width(NAME);
            ui.add(egui::Label::new(name).truncate());
        });
        add(ui)
    });
    if let Some(t) = tip {
        r.response.on_hover_text(t);
    }
    r.inner
}

/// Double- or right-click: back to `default`; the wheel over it: one step, the page stays put (user 2026-11-02).
fn nudge(ui: &Ui, r: &egui::Response, value: &mut f64, (lo, hi, step): (f64, f64, f64), default: f64) -> bool {
    let before = *value;
    if r.double_clicked() || r.secondary_clicked() {
        *value = default;
    } else if r.hovered() {
        let wheel: f32 = ui.input(|i| i.events.iter().filter_map(|e| if let egui::Event::MouseWheel { delta, .. } = e { Some(delta.y) } else { None }).sum());
        if wheel != 0. {
            ui.ctx().input_mut(|i| i.smooth_scroll_delta.y = 0.);
            *value = (*value + step * f64::from(wheel.signum())).clamp(lo, hi);
        }
    }
    *value != before
}

/// A settings row: name, slider, current value.
#[allow(clippy::too_many_arguments)]
fn slider(ui: &mut Ui, name: &str, value: &mut f64, range: (f64, f64, f64), default: f64, show: impl Fn(f64) -> String, tip: Option<&str>) -> bool {
    let tip = format!("{}Double-click: default. Wheel over it: one step.", tip.map_or(String::new(), |t| format!("{t}\n")));
    row(ui, name, Some(&tip), |ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - VALUE).max(60.);
        let r = ui.add(egui::Slider::new(value, range.0..=range.1).step_by(range.2).show_value(false)).on_hover_text(name);
        let changed = r.changed() | nudge(ui, &r, value, range, default);
        ui.label(show(*value));
        changed
    })
}

/// The slider steps through a list.
fn stepped(ui: &mut Ui, name: &str, value: &mut f64, values: &[f64], default: f64, show: impl Fn(f64) -> String, tip: Option<&str>) -> bool {
    let tip = format!("{}Double-click: default. Wheel over it: one step.", tip.map_or(String::new(), |t| format!("{t}\n")));
    let index = |v: f64| values.iter().enumerate().min_by(|a, b| (a.1 - v).abs().total_cmp(&(b.1 - v).abs())).map_or(0, |x| x.0);
    let mut at = index(*value);
    let changed = row(ui, name, Some(&tip), |ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - VALUE).max(60.);
        let r = ui.add(egui::Slider::new(&mut at, 0..=values.len() - 1).show_value(false)).on_hover_text(name);
        let mut i = at as f64;
        let nudged = nudge(ui, &r, &mut i, (0., (values.len() - 1) as f64, 1.), index(default) as f64);
        at = i as usize;
        ui.label(show(values[at]));
        r.changed() | nudged
    });
    if changed {
        *value = values[at];
    }
    changed
}

/// A chip's width, before it exists: rows are cut from this. A `ui.horizontal` inside `horizontal_wrapped` has no size when
/// the wrap is decided, so the row ran off the window (user 2026-11-01: "the buttons go offscreen").
fn chip_width(ui: &Ui, text: &str) -> f32 {
    let mark = ui.painter().layout_no_wrap("✓ ".to_string(), egui::FontId::proportional(theme::BODY), theme::TEXT).size().x;
    let text = ui.painter().layout_no_wrap(text.to_string(), egui::FontId::proportional(theme::BODY), theme::TEXT).size().x;
    text + mark + 2. * ui.spacing().button_padding.x + 2.
}

/// The indices of each row, in order: a unit wider than the whole width still gets a row of its own.
pub fn wrap_rows(widths: &[f32], available: f32, gap: f32) -> Vec<Vec<usize>> {
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut used = 0.;
    for (i, w) in widths.iter().enumerate() {
        match rows.last_mut() {
            Some(row) if used + gap + w <= available => {
                row.push(i);
                used += gap + w;
            }
            _ => {
                rows.push(vec![i]);
                used = *w;
            }
        }
    }
    rows
}

fn weight_text(value: f64, rest: f64) -> String {
    format!("{value}× · {:.0}%", 100. * value / (value + rest))
}

fn switch(ui: &mut Ui, name: &str, value: &mut bool, tip: Option<&str>) -> bool {
    let r = ui.checkbox(value, name);
    let changed = r.changed();
    if let Some(t) = tip {
        r.on_hover_text(t);
    }
    changed
}

const WEIGHT_TIP: &str = "On: the slider sets the chance per map, so at the middle every map is equally likely and more maps mean more picks. Off: the slider is a plain share of your picks.";

fn how_often(value: f64, weighted: bool) -> String {
    let f = web_factor(value);
    if f <= 0. {
        "never".to_string()
    } else if weighted {
        if (f - 1.).abs() < 1e-9 { "same per map".to_string() } else { format!("{f:.1}× per map") }
    } else {
        pct(f / (1. + f)).replace('%', "% of picks")
    }
}

fn pct(v: f64) -> String {
    format!("{:.0}%", 100. * v)
}

fn pp_text(v: f64) -> String {
    format!("{v}pp")
}

/// A pick read for its pp: the PP tab's own (not warm-ups) and PP maps mixed into NPS.
fn pp_row(c: &Pick, mode: &str) -> bool {
    c.purpose == "nps_mix" || (mode == "pp" && c.purpose != "warmup")
}

/// A row's hover: why it was picked, its accuracy range and pp, how popular it is.
fn row_tip(c: &Pick, shown: &ms_rec::history::Shown) -> String {
    let p = &c.pred;
    let mut facts = vec![format!("~{:.1}% (likely {:.1}–{:.1}%)", 100. * shown.acc(p.acc_mid), 100. * shown.acc(p.acc_lo), 100. * shown.acc(p.acc_hi))];
    if let (Some(lo), Some(hi)) = (c.pp_lo, c.pp_hi) {
        facts.push(format!("{lo:.0}–{hi:.0}pp"));
    }
    if let Some(b) = c.best.filter(|b| *b > 0.) {
        facts.push(format!("best {b:.0}pp"));
    }
    if let Some(n) = c.playcount.filter(|n| *n > 0) {
        facts.push(format!("{} website plays", ms_rec::engine::commas(n as usize)));
    }
    let why = if c.why.is_empty() { String::new() } else { format!("{}\n", c.why) };
    let from: Vec<&str> = badges(c).iter().map(|b| b.2).collect();
    let why = if from.is_empty() { why } else { format!("{why}{}\n", from.join(" · ")) };
    format!("{why}{}\n\nCtrl+click: copy its search. Right-click: skip for 12 h, copy, website.", facts.join(" · "))
}

/// "+pp · %" hovered: from your best to what this play should give.
fn pp_tip(c: &Pick) -> String {
    let best = c.best.filter(|b| *b > 0.).map_or("no score yet".to_string(), |b| format!("best {b:.0}"));
    let range = match (c.pp_lo, c.pp_hi) { (Some(lo), Some(hi)) => format!(" ({lo:.0}–{hi:.0})"), _ => String::new() };
    format!("{best} → expected {:.0}{range}, {:.0}% to {}", c.pp_mid.unwrap_or(0.), 100. * c.p_up.unwrap_or(0.), if c.best.is_some_and(|b| b > 0.) { "beat" } else { "count" })
}

/// Where a row comes from, as small filled badges on the right under the number (user 2026-11-03: "website or convert or both … easy to
/// read"): a website map not installed yet, a keymode convert, or a convert of a website map.
fn badges(c: &Pick) -> Vec<(String, Color32, &'static str)> {
    let mut out = Vec::new();
    if c.convert_of.as_ref().map_or(!c.installed, |_| c.source_web) {
        out.push(("WEB".to_string(), theme::ACCENT, "From the website, not installed yet"));
    }
    if c.convert_of.is_some() {
        out.push((format!("{}K→{}K", c.source_keys, c.keys), theme::CONVERT, "A keymode convert, written and imported when picked"));
    }
    out
}

/// Small badges right-aligned in a right-to-left layout, so the title never moves (user 2026-11-03: "somewhere in the right
/// where they don't shift the text and make them smaller").
fn draw_badges(ui: &mut Ui, c: &Pick) {
    ui.spacing_mut().item_spacing.x = 3.;
    for (text, fill, _) in badges(c).into_iter().rev() {
        egui::Frame::new().fill(fill).corner_radius(2).inner_margin(egui::Margin::symmetric(3, 0)).show(ui, |ui| {
            ui.label(RichText::new(text).size(9.5).color(theme::VIEW));
        });
    }
}

fn row_text(c: &Pick, mode: &str, shown: &ms_rec::history::Shown) -> (String, Vec<String>) {
    let length = fmt::clock(c.length);
    let acc = 100. * shown.acc(c.pred.acc_mid);
    let (number, sub) = if pp_row(c, mode) {
        let n = format!("{}+{:.1}pp · {:.0}%", if mode != "pp" { "PP " } else { "" }, c.gain.unwrap_or(0.), 100. * c.p_up.unwrap_or(0.));
        (n, vec![c.var.clone(), c.stars.map_or(String::new(), |s| format!("{:.1}★", ms_rec::shown::stars(c.keys, s))), c.best.filter(|b| *b != 0.).map_or("new".to_string(), |b| format!("best {b:.0}pp")), length])
    } else if mode == "pp" {
        (format!("warm-up ~{acc:.1}%"), vec![length])
    } else {
        let mut sub = if mode == "nps" { vec![format!("{:.1} NPS", c.nps.unwrap_or(0.))] } else { Vec::new() };
        sub.extend([length, if c.description.is_empty() { c.skill.clone().unwrap_or_default() } else { c.description.clone() }]);
        (format!("~{acc:.1}%"), sub)
    };
    (number, sub.into_iter().filter(|s| !s.is_empty()).collect())
}

impl App {
    fn key_picker(&mut self, ui: &mut Ui, mode: &str) {
        let keys = self.s.keys[mode].clone();
        let mut next = None;
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(egui::vec2(NAME, 16.), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.set_min_width(NAME);
                ui.label("Keymodes");
            });
            let config = MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside);
            MenuButton::new(keys_label(&keys)).config(config).ui(ui, |ui| {
                if ui.button("All supported modes (4K–10K)").clicked() {
                    next = Some(SUPPORTED_KEYS.collect::<Vec<_>>());
                }
                for k in SUPPORTED_KEYS {
                    let mut on = keys.contains(&k);
                    if ui.checkbox(&mut on, format!("{k}K")).changed() {
                        let mut v: Vec<i64> = SUPPORTED_KEYS.filter(|x| if *x == k { on } else { keys.contains(x) }).collect();
                        v.sort_unstable();
                        next = Some(v);
                    }
                }
            }).0.on_hover_text("Choose any combination independently for this tab. All covers 4K–10K; empty selection stays empty.");
        });
        if let Some(v) = next {
            self.set_keys(mode, v);
        }
    }

    fn acc_target(&mut self, ui: &mut Ui, mode: &str) {
        let mut v = 100. * self.s.targets[mode];
        if slider(ui, "Accuracy target", &mut v, (88., 98., 0.5), 100. * DEFAULT_TARGET, |v| format!("{v:.1}%{}", if v / 100. == DEFAULT_TARGET { " (default)" } else { "" }), Some("Maps and rates are chosen so the predicted accuracy lands near this.")) {
            self.s.targets.insert(mode.to_string(), normalize_target(v / 100.));
            self.sync_rec();
        }
    }

    /// The rates this tab may offer: 1.00–1.00 turns HT/DT off without another switch.
    fn rate_range(&mut self, ui: &mut Ui, mode: &str) {
        let mut v = self.s.rates[mode];
        let changed = row(ui, "HT/DT range", Some("Rates this tab may offer. Both knobs on 1.00: no HT or DT. Double-click: back to 0.75–1.50×."), |ui| {
            let width = (ui.available_width() - VALUE).max(60.);
            let c = ui.scope(|ui| {
                ui.set_max_width(width);
                theme::range_slider(ui, "HT/DT range", &mut v, (MIN_RATE, MAX_RATE), 0.05, DEFAULT_RATES).changed()
            }).inner;
            ui.label(rate_text(v));
            c
        });
        if changed {
            self.s.rates.insert(mode.to_string(), normalize_rates(v));
            self.sync_rec();
        }
    }

    /// How often this tab suggests website charts (maps you don't have) next to your own.
    fn web_bias(&mut self, ui: &mut Ui, mode: &str) {
        // Without website charts it can change nothing (a player, 2026-10-31: set "Never", saw no difference).
        let have = web_dir().join("web.msp").is_file() || web_dir().join("web.pkl").is_file();
        let tip = if have { "Website charts (Website charts…): their share of your picks. Middle: half. Right end: 80 %. Left end: never, only maps you own." } else { "Nothing to mix yet: add charts with Website charts… on the NPS tab. Until then every pick is a map you own." };
        let mut v = self.s.web_bias[mode];
        let weighted = self.s.web_weighted[mode];
        if ui.add_enabled_ui(have, |ui| slider(ui, "Maps you don't have", &mut v, (0., 1., 0.05), 0.5, |v| how_often(v, weighted), Some(tip))).inner {
            self.s.web_bias.insert(mode.to_string(), normalize_focus(v));
            self.sync_rec();
        }
        let mut on = weighted;
        if ui.add_enabled_ui(have, |ui| switch(ui, "Weight website maps by map count", &mut on, Some(WEIGHT_TIP))).inner {
            self.s.web_weighted.insert(mode.to_string(), on);
            self.sync_rec();
        }
    }

    fn web_button(&mut self, ui: &mut Ui) {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            let label = self.web_label();
            if ui.button(label).on_hover_text("Add maps you don't have to the playlists (chart text only, no audio).").clicked() {
                self.open.web = true;
            }
        });
    }

    /// Keymode converts for each of the tab's keymodes: which keymodes feed it and how often they come up.
    fn converts(&mut self, ui: &mut Ui, mode: &'static str) {
        theme::section(ui, "CONVERTS");
        for target in self.s.keys[mode].clone() {
            let mut s = self.s.converts[mode].get(&target).cloned().unwrap_or_else(|| Setting::new(target));
            let mut changed = switch(ui, &format!("Convert other keymodes to {target}K"), &mut s.on, Some("Maps of other keymodes join this list, converted when their turn comes. Their difficulty is known ahead."));
            if s.on {
                changed |= row(ui, "From", Some("The keymodes to convert from."), |ui| {
                    let mut changed = false;
                    ui.spacing_mut().item_spacing.x = 3.;
                    for k in SUPPORTED_KEYS.filter(|k| *k != target) {
                        let mut on = s.sources.contains(&k);
                        if ui.toggle_value(&mut on, format!("{k}K")).changed() {
                            s.sources.retain(|x| *x != k);
                            if on {
                                s.sources.push(k);
                                s.sources.sort_unstable();
                            }
                            changed = true;
                        }
                    }
                    changed
                });
                let weighted = s.weighted;
                changed |= slider(ui, "How often", &mut s.often, (0., 1., 0.05), Setting::new(target).often, |v| how_often(v, weighted), Some("Converts against real maps of this keymode."));
                changed |= switch(ui, "Weight converts by map count", &mut s.weighted, Some(WEIGHT_TIP));
            }
            if changed {
                s.often = normalize_focus(s.often);
                self.s.converts.get_mut(mode).unwrap().insert(target, s);
                self.sync_rec();
            }
        }
    }

    fn chip(&mut self, ui: &mut Ui, key: &str, name: &str, tip: &str) {
        let (state, mixed) = (skills::state(&self.s.practice, key), skills::mixed(&self.s.practice, key));
        let word = if mixed { "mixed" } else { match state { 1 => "included", -1 => "excluded", _ => "off" } };
        let (mark, colour) = if mixed { ("– ", DIM) } else { match state { 1 => ("✓ ", CHIP_ON), -1 => ("✕ ", CHIP_OFF), _ => ("", BORDER) } };
        let mut text = egui::text::LayoutJob::default();
        let plain = egui::TextFormat { font_id: egui::FontId::proportional(theme::BODY), color: theme::TEXT, ..Default::default() };
        text.append(mark, 0., egui::TextFormat { color: colour, ..plain.clone() });
        text.append(name, 0., plain);
        let mut b = egui::Button::new(text).stroke(Stroke::new(1., colour));
        if !mixed && state != 0 {
            b = b.fill(colour.gamma_multiply(0.16));
        }
        let tip = if tip.is_empty() { "Broad practice for this family; ▾ opens its subskills.".to_string() } else { tip.to_string() };
        let r = ui.add(b).on_hover_text(format!("{name}: {word}\n{tip}\nRight-click: avoid. Shift+click: only this."));
        if r.clicked() && ui.input(|i| i.modifiers.shift) {
            self.set_skills(vec![key.to_string()]);
        } else if r.clicked() {
            let next = skills::cycle(&self.s.practice, key);
            self.set_skills(next);
        } else if r.secondary_clicked() {
            let next = skills::set_state(&self.s.practice, key, -1);
            self.set_skills(next);
        }
    }

    /// The wrapped units of a chip row: a unit is drawn by `add`.
    fn chip_rows<T>(&mut self, ui: &mut Ui, units: &[T], width: impl Fn(&Ui, &T) -> f32, mut add: impl FnMut(&mut Self, &mut Ui, &T)) {
        let widths: Vec<f32> = units.iter().map(|u| width(ui, u)).collect();
        for row in wrap_rows(&widths, ui.available_width(), CHIP_GAP) {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(CHIP_GAP, CHIP_GAP);
                for i in row {
                    add(self, ui, &units[i]);
                }
            });
        }
    }

    fn any_chip(&mut self, ui: &mut Ui) {
        let on = self.s.practice.is_empty();
        let mut b = egui::Button::new("Any").stroke(Stroke::new(1., if on { CHIP_ON } else { BORDER }));
        if on {
            b = b.fill(CHIP_ON.gamma_multiply(0.16));
        }
        if ui.add(b).on_hover_text("Regular practice: no skill chosen or avoided.").clicked() {
            self.set_skills(Vec::new());
        }
    }

    /// Skills chips, over the same taxonomy as player Stats. Replaced a popover of checkboxes whose third state read as dots
    /// (user 2026-10-31: "kinda scuffed").
    fn skill_picker(&mut self, ui: &mut Ui) {
        theme::dim_small(ui, "Click: include → avoid → off. ▾ opens subskills.");
        let keys = self.s.keys["skills"].clone();
        let modes: Vec<i64> = [keys.contains(&4).then_some(4), keys.iter().any(|k| *k >= 5).then_some(7)].into_iter().flatten().collect();
        if modes.is_empty() {
            ui.label("Select a keymode first.");
        }
        for (at, mode) in modes.iter().copied().enumerate() {
            if modes.len() > 1 {
                ui.label(RichText::new(if mode == 4 { "4K" } else { "5K–10K" }).family(egui::FontFamily::Name("bold".into())));
            }
            let mut units: Vec<Option<&Skill>> = if at == 0 { vec![None] } else { Vec::new() };
            units.extend(skills::tree(mode).iter().map(Some));
            self.chip_rows(ui, &units, |ui, u| u.map_or(chip_width(ui, "Any"), |p| chip_width(ui, p.name) + if p.children.is_empty() { 0. } else { chip_width(ui, "▾") }), |app, ui, u| {
                let Some(parent) = u else { return app.any_chip(ui) };
                // Together: a chip and its ▾ wrap as one.
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.;
                    app.chip(ui, &parent.key, parent.name, parent.tip);
                    if !parent.children.is_empty() {
                        let id = (mode, parent.key.clone());
                        let open = app.chips_open.contains(&id);
                        if ui.selectable_label(open, "▾").on_hover_text(format!("{} subskills", parent.name)).clicked() {
                            if open { app.chips_open.remove(&id); } else { app.chips_open.insert(id); }
                        }
                    }
                });
            });
            for parent in skills::tree(mode) {
                if !parent.children.is_empty() && self.chips_open.contains(&(mode, parent.key.clone())) {
                    theme::dim(ui, parent.name);
                    let children: Vec<&Skill> = parent.children.iter().collect();
                    self.chip_rows(ui, &children, |ui, c| chip_width(ui, c.name), |app, ui, c| app.chip(ui, &c.key, c.name, c.tip));
                }
            }
        }
        self.skill_sliders(ui);
    }

    /// How much of each included skill, and how strongly each avoided one is kept out.
    fn skill_sliders(&mut self, ui: &mut Ui) {
        let (included, avoided): (Vec<String>, Vec<String>) = self.s.practice.iter().cloned().partition(|k| !k.starts_with('!'));
        if !included.is_empty() {
            theme::section(ui, "HOW MUCH OF EACH");
            let total = included.iter().map(|k| self.s.skill_weights.get(k).copied().unwrap_or(1.)).sum::<f64>() + self.s.skill_other;
            for key in included {
                let mut v = self.s.skill_weights.get(&key).copied().unwrap_or(1.);
                let before = v;
                if stepped(ui, &format!("✓ {}", skills::name(&key)), &mut v, &WEIGHTS, 1., |x| weight_text(x, total - before), Some("How much of the list this skill gets, against the others here.")) {
                    self.s.skill_weights.insert(key, v);
                    self.sync_rec();
                }
            }
            let mut v = self.s.skill_other;
            if stepped(ui, "Other maps", &mut v, &OTHER, 0., |x| if x == 0. { "Off".to_string() } else { weight_text(x, total - self.s.skill_other) }, Some("Off: only the skills above. On: maps without them join the list at this share.")) {
                self.s.skill_other = v;
                self.sync_rec();
            }
        }
        if !avoided.is_empty() {
            theme::section(ui, "AVOID");
            for token in avoided {
                let key = token.trim_start_matches('!').to_string();
                let hold = self.s.skill_tuning().hold(&key);
                let mut v = AVOID.iter().position(|a| a.0 == hold).unwrap_or(1) as f64;
                if stepped(ui, &format!("✕ {}", skills::name(&key)), &mut v, &[0., 1., 2.], 1., |x| AVOID[x as usize].1.to_string(), Some("How much of it a map may hold before it leaves the list.")) {
                    self.s.skill_avoid.insert(key, AVOID[v as usize].0);
                    self.sync_rec();
                }
            }
        }
    }

    fn settings_page(&mut self, ui: &mut Ui, mode: &'static str) {
        let d = crate::settings::Ui::from_json(&serde_json::Value::Null); // the defaults a double-click returns to
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.set_max_width(ui.available_width() - 10.); // the scroll bar's place
            ui.spacing_mut().item_spacing.y = 6.;
            match mode {
                "nps" => {
                    theme::section(ui, "MAPS");
                    self.key_picker(ui, mode);
                    self.acc_target(ui, mode);
                    self.rate_range(ui, mode);
                    theme::section(ui, "MIX");
                    let mut focus = self.s.nps_focus;
                    if slider(ui, "Variety to high NPS", &mut focus, (0., 1., 0.05), 0.5, focus_label, Some("Left: pattern types (jacks, streams, LN…) get an even share of the list. Right: the highest NPS maps, mostly chordstreams and brackets. Both aim at the accuracy target.")) {
                        self.s.nps_focus = normalize_focus(focus);
                        self.sync_rec();
                    }
                    self.web_bias(ui, mode);
                    self.web_button(ui);
                    self.converts(ui, mode);
                    theme::section(ui, "PP MAPS IN THIS LIST");
                    let mut changed = false;
                    let m = &mut self.s.mix;
                    changed |= switch(ui, "Mix in likely PP gains", &mut m.on, Some("PP maps you are likely to beat now, between NPS maps. Off: no PP scoring while you play NPS."));
                    if m.on {
                        let mut every = m.every as f64;
                        if slider(ui, "One PP map every", &mut every, (2., 8., 1.), d.mix.every as f64, |v| format!("{v:.0} picks"), None) {
                            m.every = every as usize;
                            changed = true;
                        }
                        changed |= stepped(ui, "Min gain", &mut m.min_gain, &[1., 2., 3., 5., 8., 10., 15.], d.mix.min_gain, pp_text, None);
                        changed |= slider(ui, "Min chance", &mut m.chance, (0.2, 0.9, 0.05), d.mix.chance, pct, Some("Chance to beat your best, NM and DT. It already includes this session's form."));
                        changed |= slider(ui, "Min chance (HT)", &mut m.ht_chance, (0.4, 0.95, 0.05), d.mix.ht_chance, pct, None);
                    }
                    if changed {
                        self.sync_rec();
                    }
                }
                "skills" => {
                    let i = self.s.playlist;
                    ui.horizontal(|ui| {
                        let label = ui.add_sized(egui::vec2(NAME, 16.), egui::Label::new("Name"));
                        let mut name = self.s.playlists[i].0.clone();
                        if ui.text_edit_singleline(&mut name).labelled_by(label.id).changed() && !name.trim().is_empty() {
                            self.s.playlists[i].0 = name;
                            self.request_save();
                        }
                    });
                    theme::section(ui, "SKILLS");
                    self.skill_picker(ui);
                    theme::section(ui, "MAPS");
                    self.key_picker(ui, mode);
                    self.acc_target(ui, mode);
                    self.rate_range(ui, mode);
                    theme::section(ui, "MIX");
                    let mut focus = self.s.skills_focus;
                    let show = |v: f64| if v < 0. { "off".to_string() } else { focus_label(v) };
                    if slider(ui, "Variety to high NPS", &mut focus, (crate::settings::FOCUS_OFF, 1., 0.05), 0., show, Some("Leftmost: off, the best skill match at your accuracy target first (with any skill: accuracy alone). Then from an even share of pattern types to the highest NPS maps.")) {
                        self.s.skills_focus = if focus < 0. { crate::settings::FOCUS_OFF } else { focus.min(1.) };
                        self.sync_rec();
                    }
                    self.web_bias(ui, mode);
                    self.web_button(ui);
                    self.converts(ui, mode);
                    theme::section(ui, "PP MAPS IN THIS LIST");
                    let mut changed = false;
                    let m = &mut self.s.skills_mix;
                    changed |= switch(ui, "Mix in likely PP gains", &mut m.on, Some("PP maps you are likely to beat now, between this playlist's maps."));
                    if m.on {
                        let mut every = m.every as f64;
                        if slider(ui, "One PP map every", &mut every, (2., 8., 1.), d.mix.every as f64, |v| format!("{v:.0} picks"), None) {
                            m.every = every as usize;
                            changed = true;
                        }
                        changed |= stepped(ui, "Min gain", &mut m.min_gain, &[1., 2., 3., 5., 8., 10., 15.], d.mix.min_gain, pp_text, None);
                        changed |= slider(ui, "Min chance", &mut m.chance, (0.2, 0.9, 0.05), d.mix.chance, pct, Some("Chance to beat your best, NM and DT. It already includes this session's form."));
                        changed |= slider(ui, "Min chance (HT)", &mut m.ht_chance, (0.4, 0.95, 0.05), d.mix.ht_chance, pct, None);
                    }
                    if changed {
                        self.sync_rec();
                    }
                    theme::section(ui, "PLAYLIST");
                    self.delete_button(ui, i);
                }
                _ => {
                    theme::section(ui, "CHOICE");
                    self.key_picker(ui, mode);
                    let mut changed = false;
                    let o = &mut self.s.pp;
                    changed |= stepped(ui, "Min gain", &mut o.min_gain, &[0.01, 0.1, 0.25, 0.5, 1., 2., 3., 5., 10.], d.pp.min_gain, pp_text, Some("Expected account pp gain a map needs."));
                    changed |= slider(ui, "Min chance", &mut o.min_chance, (0.05, 0.6, 0.01), d.pp.min_chance, pct, Some("Chance to beat your best. Maps above your usual level keep at least 25%."));
                    changed |= switch(ui, "Top-100 maps only", &mut o.top100, Some("Only maps that are, or would enter, your top 100."));
                    theme::section(ui, "MODS");
                    changed |= switch(ui, "DT", &mut o.dt, None);
                    changed |= switch(ui, "HT", &mut o.ht, Some("HT only as a likely, real improvement."));
                    if o.ht {
                        changed |= slider(ui, "HT min chance", &mut o.ht_chance, (0.3, 0.95, 0.05), d.pp.ht_chance, pct, None);
                        changed |= stepped(ui, "HT min gain", &mut o.ht_gain, &[1., 2., 3., 5., 8., 10., 15., 20.], d.pp.ht_gain, pp_text, None);
                    }
                    theme::section(ui, "STYLE");
                    changed |= slider(ui, "Safe to Push", &mut o.push, (-0.5, 0.5, 0.05), d.pp.push, |v| if v.abs() < 0.01 { "auto".to_string() } else { format!("{} {:.2}", if v > 0. { "push" } else { "safe" }, v.abs()) }, Some("Middle: follows your form. Push: higher-pp, lower-accuracy maps. Safe: comfortable ones."));
                    theme::section(ui, "REPEATS");
                    changed |= slider(ui, "Repeat cooldown", &mut o.cooldown_days, (1., 14., 1.), d.pp.cooldown_days, |v| format!("{v:.0} days"), Some("Offers and plays older than this no longer hold a map back."));
                    if changed {
                        self.sync_rec();
                    }
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                if ui.button("Done").clicked() {
                    self.views.get_mut(mode).expect("a tab").settings = false;
                }
            });
        });
    }

    fn list_page(&mut self, ui: &mut Ui, mode: &'static str) {
        let v = &self.views[mode];
        if let Some(c) = &v.target {
            let lines = fmt::target_lines(c, &self.obs, &self.units);
            // A PP pick keeps its "+pp · chance" on the right, like the rows under it (user 2026-11-02).
            let number = pp_row(c, mode).then(|| row_text(c, mode, &self.units).0);
            // A red SET says what osu! still shows differently.
            let off = fmt::set_check(c, &self.obs).2;
            let tip = [v.target_tip.clone(), (!off.is_empty()).then(|| format!("Not set yet: {}.", off.join("; ")))].into_iter().flatten().reduce(|a, b| format!("{a}\n{b}"));
            egui::Frame::new().fill(theme::ROW_CURRENT).inner_margin(egui::Margin::symmetric(6, 5)).show(ui, |ui| {
                ui.set_width(ui.available_width());
                let marked = !badges(c).is_empty();
                ui.horizontal(|ui| {
                    ui.scope(|ui| {
                        ui.set_max_width((ui.available_width() - if number.is_some() || marked { 104. } else { 0. }).max(60.));
                        theme::block(ui, &[(lines[0].clone(), 1), (lines[1].clone(), 2)], tip.as_deref());
                    });
                    ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                        if let Some(n) = &number {
                            ui.label(RichText::new(n).family(egui::FontFamily::Name("bold".into())));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| draw_badges(ui, c));
                    });
                });
            });
        } else if !v.target_msg.is_empty() {
            theme::block(ui, &[(vec![fmt::plain(v.target_msg.clone())], 1)], v.target_tip.as_deref());
        }
        let note_height = if v.note.is_empty() { 0. } else { 18. };
        let (mut clicked, mut copied, mut snoozed) = (None, None, None);
        let height = (ui.available_height() - note_height).max(40.);
        let (shown, loaded, status) = (v.shown.clone(), v.loaded, self.status_text.clone());
        let units = (self.units.client.clone(), self.units.miss_ratio);
        let area = ScrollArea::vertical().id_salt(mode).max_height(height).auto_shrink([false, false]).show(ui, |ui| {
            if shown.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(24.);
                    let head = if loaded { "Nothing to suggest yet.".to_string() } else { format!("Loading {} suggestions…", if mode == "skills" { self.s.playlists[self.s.playlist].0.as_str() } else { mode_name(mode) }) };
                    theme::dim(ui, if status.is_empty() { head } else { format!("{head}\n\n{status}") });
                });
            }
            let units = ms_rec::history::Shown { client: units.0.clone(), miss_ratio: units.1 };
            ui.spacing_mut().item_spacing.y = 0.;
            for (i, c) in shown.iter().enumerate() {
                let (number, sub) = row_text(c, mode, &units);
                let row = ui.scope_builder(UiBuilder::new().id_salt(i).sense(Sense::click()), |ui| {
                    ui.style_mut().interaction.selectable_labels = false; // selectable text took the hover from its row (user 2026-11-01)
                    ui.spacing_mut().item_spacing.y = 2.;
                    let fill = if ui.response().hovered() { theme::ROW_HOVER } else if i % 2 == 1 { theme::BUTTON } else { Color32::TRANSPARENT };
                    let frame = egui::Frame::new().inner_margin(egui::Margin::symmetric(6, 7)).fill(fill);
                    frame.show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let number_rect = ui.horizontal(|ui| {
                            let width = ui.available_width() - 104.;
                            let mut line = vec![fmt::bold(format!("{:.2}×", c.rate)), fmt::plain("  ")];
                            line.extend(fmt::pick_line(c));
                            ui.scope(|ui| {
                                ui.set_max_width(width.max(60.));
                                theme::line(ui, &line, 3);
                            });
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                                ui.label(RichText::new(&number).family(egui::FontFamily::Name("bold".into()))).rect
                            }).inner
                        }).inner;
                        ui.horizontal(|ui| {
                            theme::dim_small(ui, sub.join(" · "));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| draw_badges(ui, c));
                        });
                        number_rect
                    }).inner
                });
                let on_number = row.response.hover_pos().is_some_and(|p| row.inner.contains(p));
                let tip = if pp_row(c, mode) && on_number { pp_tip(c) } else { row_tip(c, &units) };
                let r = row.response.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tip);
                // Ctrl+click or middle-click: its search text only, no pick and no skip.
                if r.middle_clicked() || (r.clicked() && ui.input(|i| i.modifiers.command)) {
                    copied = Some(i);
                } else if r.clicked() {
                    clicked = Some(i);
                }
                r.context_menu(|ui| {
                    // No "never show again": dangerous without an undo (user 2026-11-02).
                    if ui.button("Skip for 12 h").on_hover_text("Out of every list until then").clicked() {
                        snoozed = Some(i);
                        ui.close();
                    }
                    if ui.button("Copy its search").clicked() {
                        copied = Some(i);
                        ui.close();
                    }
                    if ui.add_enabled(c.bid.is_some(), egui::Button::new("Open on osu.ppy.sh")).clicked() {
                        crate::app::open_external(&format!("https://osu.ppy.sh/b/{}", c.bid.unwrap_or(0)));
                        ui.close();
                    }
                });
            }
        });
        let hover = ui.rect_contains_pointer(area.inner_rect);
        let view = self.views.get_mut(mode).expect("a tab");
        let left = view.hover && !hover;
        view.hover = hover;
        if left {
            self.flush_pool(mode); // a list under the pointer never moves
        }
        let v = &self.views[mode];
        if !v.note.is_empty() {
            let (note, tip) = (v.note.clone(), v.note_tip.clone());
            let r = theme::dim_small(ui, note);
            if !tip.is_empty() {
                r.on_hover_text(tip);
            }
        }
        if let Some(text) = copied.and_then(|i| self.views[mode].shown.get(i)).map(|c| ms_rec::navigation::song_search(c, &self.units.client)) {
            self.copy(text);
        }
        if let Some(c) = snoozed.and_then(|i| self.views[mode].shown.get(i)).cloned() {
            self.snooze(c, mode);
        }
        if let Some(i) = clicked {
            self.activate(mode, i, None);
        }
    }

    /// "Skip for 12 h": out of every list at once; the worker logs it and moves the current pick on.
    fn snooze(&mut self, c: Pick, mode: &str) {
        for v in self.views.values_mut() {
            v.shown.retain(|p| p.identity() != c.identity());
        }
        self.q.put(Task::Snooze { pick: Box::new(c), mode: mode.to_string() });
    }

    /// A tab's hover: what it draws from.
    fn tab_tip(&self, m: &str, playlist: Option<usize>) -> String {
        if playlist.is_some_and(|i| i != self.s.playlist) {
            return "Right-click: rename, duplicate, delete · drag to move".to_string(); // its settings are stored, not live
        }
        let keys = keys_label(&self.s.keys[m]);
        if m == "pp" {
            return format!("{keys} · maps likely to raise your pp\nDrag to move");
        }
        let more = if playlist.is_some() { "\nRight-click: rename, duplicate, delete · drag to move" } else { "\nDrag to move" };
        format!("{keys} · target {:.1}% · {} · {} shown{more}", 100. * self.s.targets[m], rate_text(self.s.rates[m]), self.views[m].shown.len())
    }

    /// One tab; its cog when open. `playlist`: a custom playlist's index.
    fn tab(&mut self, ui: &mut Ui, m: &'static str, name: &str, playlist: Option<usize>, mode: &str) -> egui::Response {
        let active = !self.stats_open && m == mode && playlist.is_none_or(|i| i == self.s.playlist);
        let tab = ui.add(egui::Button::selectable(active, name).sense(egui::Sense::click_and_drag())).on_hover_text(self.tab_tip(m, playlist));
        if tab.clicked() {
            self.set_stats(false);
            if let Some(i) = playlist.filter(|i| *i != self.s.playlist) {
                self.s.open_playlist(i);
                self.clear_custom(format!("Loading {name}…"));
                self.views.get_mut(m).expect("a tab").settings = false;
                self.s.mode = m.to_string();
                self.sync_rec();
            } else {
                self.set_mode(m);
            }
            if active {
                self.views.get_mut(m).expect("a tab").settings = false; // back to the list, like the cog
            }
        }
        if active {
            let view = self.views.get_mut(m).expect("a tab");
            // Painted, not laid out: a glyph this big as a button would make the row taller (user 2026-11-01).
            let (rect, gear) = ui.allocate_exact_size(egui::vec2(GEAR + 4., tab.rect.height()), egui::Sense::click());
            gear.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "⚙"));
            let color = ui.style().interact_selectable(&gear, view.settings).text_color();
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "⚙", egui::FontId::proportional(GEAR), color);
            if gear.on_hover_text("Settings for this tab").clicked() {
                view.settings = !view.settings;
            }
        }
        tab
    }

    /// PP, NPS and the custom playlists (in place of Skills, user 2026-11-06) in their saved order, then "+" and Stats. Every
    /// tab drags like a browser tab (user 2026-11-08: "feel exactly like chrome tabs", PP and NPS too): the held tab follows
    /// the pointer, the others slide aside to show where it lands, and on release it settles there.
    fn tab_strip(&mut self, ui: &mut Ui, mode: &str) {
        const SLIDE: f32 = 0.12; // seconds a tab takes to slide into its place
        let tabs = self.s.tabs();
        let names: Vec<String> = tabs.iter().map(|t| match t { Tab::Mode(m) => mode_name(m).to_string(), Tab::List(i) => self.s.playlists[*i].0.clone() }).collect();
        // Stable across moves: kind and name, numbered among equal names.
        let ids: Vec<egui::Id> = (0..tabs.len()).map(|k| {
            let same = names[..k].iter().filter(|n| **n == names[k]).count();
            ui.scope_id().with(("strip-tab", matches!(tabs[k], Tab::Mode(_)), names[k].as_str(), same))
        }).collect();
        let font = egui::TextStyle::Button.resolve(ui.style());
        let pad = 2. * ui.spacing().button_padding.x;
        let widths: Vec<f32> = ids.iter().zip(&names).map(|(id, name)| ui.ctx().data(|d| d.get_temp::<f32>(*id))
            .unwrap_or_else(|| ui.painter().layout_no_wrap(name.clone(), font.clone(), Color32::WHITE).size().x + pad)).collect();
        let gap = ui.spacing().item_spacing.x;
        let total = widths.iter().sum::<f32>() + gap * tabs.len().saturating_sub(1) as f32;
        let (strip, _) = ui.allocate_exact_size(egui::vec2(total, ui.spacing().interact_size.y), Sense::hover());
        if self.open.drag.is_some_and(|d| d.0 >= tabs.len()) {
            self.open.drag = None;
        }
        // Where each tab goes: with one held, the order it would drop into.
        let mut order: Vec<usize> = (0..tabs.len()).collect();
        let pointer = ui.input(|i| i.pointer.interact_pos());
        let mut held = None;
        if let (Some((from, grab)), Some(p)) = (self.open.drag, pointer) {
            let x = (p.x - grab).clamp(strip.left(), strip.right() - widths[from]);
            held = Some((from, x));
            order.remove(from);
            let (centre, mut cursor) = (x + widths[from] / 2., strip.left());
            let at = order.iter().position(|&t| {
                let before = centre < cursor + widths[t] / 2.;
                cursor += widths[t] + gap;
                before
            }).unwrap_or(order.len());
            order.insert(at, from);
        }
        let mut place = vec![0f32; tabs.len()];
        let mut cursor = strip.left();
        for &t in &order {
            place[t] = cursor;
            cursor += widths[t] + gap;
        }
        let mut sliding = false;
        let mut drawn = order.clone();
        if let Some((from, _)) = held {
            drawn.retain(|&t| t != from);
            drawn.push(from); // the held tab over the others
        }
        for k in drawn {
            let x_id = ids[k].with("x");
            let x = match held {
                Some((from, x)) if from == k => ui.ctx().animate_value_with_time(x_id, x, 0.),
                _ => ui.ctx().animate_value_with_time(x_id, place[k], SLIDE),
            };
            sliding |= (x - place[k]).abs() > 0.5;
            let rect = egui::Rect::from_min_size(egui::pos2(x, strip.top()), egui::vec2(widths[k], strip.height()));
            let mut child = ui.new_child(UiBuilder::new().max_rect(rect).layout(egui::Layout::left_to_right(egui::Align::Center)));
            let (m, playlist) = match tabs[k] { Tab::Mode(m) => (m, None), Tab::List(i) => ("skills", Some(i)) };
            let tab = self.tab(&mut child, m, &names[k], playlist, mode);
            let width = child.min_rect().width();
            if (width - widths[k]).abs() > 0.5 {
                ui.ctx().data_mut(|d| d.insert_temp(ids[k], width));
                ui.ctx().request_repaint();
            }
            if tab.drag_started() {
                if let Some(p) = pointer {
                    self.open.drag = Some((k, p.x - x));
                }
            }
            if let Some(i) = playlist {
                let name = names[k].clone();
                tab.context_menu(|ui| {
                    if ui.button("Rename").clicked() {
                        self.open.rename = Some((i, name.clone()));
                        ui.close();
                    }
                    if ui.button("Duplicate").clicked() {
                        if i != self.s.playlist {
                            self.s.open_playlist(i);
                            self.clear_custom(String::new());
                        }
                        self.s.add_playlist(format!("{name} copy"));
                        self.s.mode = "skills".into();
                        self.sync_rec();
                        ui.close();
                    }
                    self.delete_button(ui, i);
                });
            }
        }
        if let Some((from, _)) = self.open.drag {
            if ui.input(|i| i.pointer.any_released()) {
                self.open.drag = None;
                let to = order.iter().position(|&t| t == from).unwrap_or(from);
                if to != from {
                    self.s.move_tab(from, to);
                    self.request_save();
                }
            } else {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            }
        }
        if sliding || self.open.drag.is_some() {
            ui.ctx().request_repaint();
        }
        if ui.button("+").on_hover_text("A new playlist with the open playlist's settings").clicked() {
            if self.mode() != "skills" {
                let i = self.s.playlist;
                self.s.open_playlist(i);
            }
            self.s.add_playlist(format!("Playlist {}", self.s.playlists.len() + 1));
            self.s.mode = "skills".into();
            self.set_stats(false);
            self.views.get_mut("skills").expect("a tab").settings = true; // a new one starts at its settings
            self.sync_rec();
        }
        if ui.selectable_label(self.stats_open, "Stats").clicked() {
            self.set_stats(true);
        }
        if let Some((i, mut text)) = self.open.rename.take() {
            let mut keep = true;
            egui::Window::new("Rename playlist").collapsible(false).resizable(false).show(ui.ctx(), |ui| {
                let label = ui.label("Name");
                let edit = ui.text_edit_singleline(&mut text).labelled_by(label.id);
                edit.request_focus();
                ui.horizontal(|ui| {
                    if ui.button("OK").clicked() || (edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                        if let Some(p) = self.s.playlists.get_mut(i).filter(|_| !text.trim().is_empty()) {
                            p.0 = text.trim().to_string();
                            self.request_save();
                        }
                        keep = false;
                    }
                    if ui.button("Cancel").clicked() {
                        keep = false;
                    }
                });
            });
            if keep {
                self.open.rename = Some((i, text));
            }
        }
    }

    /// Delete, asking once; the last playlist stays.
    fn delete_button(&mut self, ui: &mut Ui, i: usize) {
        let last = self.s.playlists.len() == 1;
        let asked = self.open.delete == Some(i);
        let label = if asked { "Click again to delete" } else { "Delete" };
        if ui.add_enabled(!last, egui::Button::new(label)).on_disabled_hover_text("Keep at least one playlist").clicked() {
            if asked {
                self.open.delete = None;
                self.s.remove_playlist(i);
                self.clear_custom(String::new());
                self.sync_rec();
            } else {
                self.open.delete = Some(i);
            }
        }
    }

    pub fn playlist_ui(&mut self, ui: &mut Ui) {
        let mode = self.mode();
        let enabled = self.next_enabled();
        ui.horizontal(|ui| {
            let next = egui::Button::new(RichText::new("Next").color(Color32::WHITE)).fill(ACCENT.gamma_multiply(0.7));
            let tip = if enabled { "Next pick from the open tab. A pick you haven't played is skipped: it stays away for about 12 h.".to_string() } else { self.status_text.clone() };
            let r = ui.add_enabled(enabled, next).on_hover_text(format!("{tip}\nRight-click: skip this pick for 12 h."));
            if r.clicked() {
                self.next();
            }
            let current = self.views[mode].target.clone();
            r.context_menu(|ui| {
                if ui.add_enabled(current.is_some(), egui::Button::new("Skip for 12 h")).on_hover_text("The current pick leaves every list until then; the next one comes").clicked() {
                    if let Some(c) = current.clone() {
                        self.snooze(c, mode);
                    }
                    ui.close();
                }
            });
            if self.views[mode].warmup && ui.button("Skip warmup").on_hover_text("Already warm: drop the warm-up maps and the cold-session prior for PP, NPS and Skills").clicked() {
                self.q.put(Task::SkipWarmup);
            }
            if let Some(on) = self.tracking {
                if ui.selectable_label(on, if on { "Tracking on" } else { "Tracking off" }).on_hover_text("Paused plays stay excluded after imports and restarts. Resume for the next play.").clicked() {
                    self.tracking_toggled(!on);
                }
            }
        });
        ui.add_space(6.);
        // The selected map lives in the main window only (user 2026-11-02: "too much repeating information").
        if self.views[mode].target.is_none() {
            theme::dim(ui, "Press Next for a map: its title is copied to your clipboard, paste it into osu!'s song search (Ctrl+V).");
        }
        if let Some((lines, tip)) = &self.result {
            theme::block(ui, &lines.iter().cloned().zip([1, 1, 2]).collect::<Vec<_>>(), Some(tip));
        }
        ui.add_space(6.);
        // More tabs than room: the strip scrolls sideways under the mouse wheel; the open tab stays (user 2026-11-06).
        ui.style_mut().always_scroll_the_only_direction = true;
        egui::ScrollArea::horizontal().id_salt("tab-strip").scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden).show(ui, |ui| {
            ui.horizontal(|ui| self.tab_strip(ui, mode));
        });
        ui.style_mut().always_scroll_the_only_direction = false;
        ui.separator();
        // The one status line and bar for the whole window, at the bottom (user 2026-10-30: "Analysing" three times).
        let analysing = if self.stats_open || self.bg_progress.is_some() { None } else { self.views[mode].analysing.clone() };
        let paused = analysing.as_ref().filter(|_| self.live.is_some()).map_or("", |_| "paused while you play");
        let flash = self.flash.as_ref().filter(|f| f.1 > std::time::Instant::now()).map_or(String::new(), |f| f.0.clone());
        let text: Vec<&str> = [flash.as_str(), self.status_text.as_str(), analysing.as_ref().map_or("", |a| a.0.as_str()), paused].into_iter().filter(|t| !t.is_empty()).collect();
        let text = text.join(" · ").replace('\n', " · ");
        let (done, total) = self.bg_progress.or(analysing.as_ref().map(|a| (a.1, a.2))).unwrap_or((0, 0));
        egui::Panel::bottom("rec-status").show_separator_line(false).show(ui, |ui| {
            if total > 0 {
                theme::bar(ui, done as f32 / total as f32);
            }
            if !text.is_empty() {
                theme::dim_small(ui, text);
            }
        });
        egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| {
            if self.stats_open {
                self.stats_ui(ui);
            } else if self.views[mode].settings {
                self.settings_page(ui, mode);
            } else {
                self.list_page(ui, mode);
            }
        });
    }

    /// Only the player's resize (and the saved size) sets the window: content never does (user 2026-10-30, twice).
    pub fn track_playlist(&mut self, ctx: &egui::Context) {
        // Size from the viewport: on Wayland `inner_rect` is None (see `track_window`).
        let (size, outer, fullscreen, maximized, minimized) = ctx.input(|i| {
            let v = i.viewport();
            (i.viewport_rect().size(), v.outer_rect, v.fullscreen, v.maximized, v.minimized)
        });
        if fullscreen == Some(true) || maximized == Some(true) || minimized == Some(true) {
            return;
        }
        let (size, at) = ((size.x, size.y), outer.map(|o| (o.left(), o.top())));
        if size.0 > 1. && size != self.s.recs_win {
            self.s.recs_win = size;
            self.request_save();
        }
        if at.is_some() && at != self.s.recs_pos {
            self.s.recs_pos = at;
            self.request_save();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn badges_say_website_convert_or_both() {
        let text = |c: Pick| badges(&c).into_iter().map(|b| b.0).collect::<Vec<_>>().join(" ");
        assert_eq!(text(Pick { installed: true, ..Pick::default() }), "");
        assert_eq!(text(Pick { installed: false, ..Pick::default() }), "WEB");
        let convert = Pick { convert_of: Some("src".into()), source_keys: 7, keys: 10, ..Pick::default() };
        assert_eq!(text(Pick { installed: true, ..convert.clone() }), "7K→10K", "an imported convert stays one");
        assert_eq!(text(Pick { source_web: true, ..convert }), "WEB 7K→10K");
    }
}

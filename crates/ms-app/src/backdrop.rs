//! The selected map's background image behind the main and playlist windows (user 2026-11-21: "the background image
//! changes with the selected map"): found, decoded, blurred and dimmed off the UI thread; the newest selection wins.
use crate::app::{Mail, Msg};
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use ms_data::paths;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::time::{Instant, SystemTime};

const FADE: f64 = 0.3; // seconds a new image takes to come in
/// One map's image, blurred, at its own brightness (the window dims it by how bright the part it shows is);
/// `lum`: its brightness on a GRID × GRID raster.
pub struct Images {
    pub light: ColorImage,
    pub lum: Vec<f32>,
}

pub const GRID: usize = 16;

/// The background file a chart names: its `[Events]` line `0,0,"file",…`.
pub fn named(chart: &str) -> Option<String> {
    let events = chart.split("[Events]").nth(1)?;
    events.lines().take_while(|l| !l.starts_with('[')).find_map(|l| {
        let mut parts = l.splitn(3, ',');
        (parts.next()?.trim() == "0" && parts.next().is_some()).then(|| parts.next())?.map(|rest| rest.split(',').next().unwrap_or("").trim().trim_matches('"').replace('\\', "/"))
    }).filter(|n| !n.is_empty())
}

/// lazer keeps files by hash: every chart's image hash, read from its database when a chart is new to us.
#[derive(Default)]
struct Index {
    by_chart: HashMap<String, String>,
    read: Option<(Option<SystemTime>, Instant)>,
}

impl Index {
    fn file(&mut self, chart: &str) -> Option<PathBuf> {
        let root = paths::lazer_data();
        let realm = root.join("client.realm");
        let stamp = std::fs::metadata(&realm).ok().and_then(|m| m.modified().ok());
        // A chart without an image misses every time: one read per change of the database, a minute apart at most.
        let stale = self.read.is_none_or(|(at, when)| at != stamp && when.elapsed().as_secs() >= 60);
        if !self.by_chart.contains_key(chart) && stale {
            if let Ok(found) = ms_data::imports::with_copy(&realm, ms_data::realm::backgrounds) {
                self.by_chart = found;
            }
            self.read = Some((stamp, Instant::now()));
        }
        paths::local_file(self.by_chart.get(chart)?, &root).filter(|p| p.is_file())
    }
}

fn find(chart: &Path, index: &mut Index) -> Option<PathBuf> {
    // beside the chart: stable's Songs folder
    let beside = std::fs::read(chart).ok().and_then(|d| named(&String::from_utf8_lossy(&d))).and_then(|n| Some(chart.parent()?.join(n))).filter(|p| p.is_file());
    beside.or_else(|| index.file(chart.file_name()?.to_str()?))
}

fn colour(img: &image::RgbaImage) -> ColorImage {
    ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw())
}

pub fn decode(data: &[u8]) -> Option<Images> {
    let full = image::load_from_memory(data).ok()?;
    // Small, then stretched by the GPU's linear filter: a wide blur for the cost of a thumbnail.
    let light = image::imageops::blur(&full.thumbnail(320, 320).to_rgba8(), 2.5);
    let (w, h) = (light.width() as usize, light.height() as usize);
    let mut lum = vec![(0f32, 0u32); GRID * GRID];
    for (x, y, p) in light.enumerate_pixels() {
        let cell = &mut lum[(y as usize * GRID / h.max(1)).min(GRID - 1) * GRID + (x as usize * GRID / w.max(1)).min(GRID - 1)];
        *cell = (cell.0 + (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])) / 255., cell.1 + 1);
    }
    // an image under 16 px a side leaves cells empty: they take the whole image's brightness
    let all = lum.iter().map(|c| c.0).sum::<f32>() / lum.iter().map(|c| c.1).sum::<u32>().max(1) as f32;
    Some(Images { light: colour(&light), lum: lum.into_iter().map(|(s, n)| if n == 0 { all } else { s / n as f32 }).collect() })
}

type Shown = (TextureHandle, std::sync::Arc<Vec<f32>>);

pub struct Backdrop {
    tx: Sender<PathBuf>,
    wanted: Option<PathBuf>,
    shown: Option<Shown>,
    before: Option<Shown>, // fades out under the new one
    since: f64,
}

impl Backdrop {
    pub fn new(mail: Mail) -> Self {
        let (tx, rx) = channel::<PathBuf>();
        std::thread::Builder::new().name("backdrop".into()).spawn(move || {
            let mut index = Index::default();
            while let Ok(mut chart) = rx.recv() {
                while let Ok(newer) = rx.try_recv() {
                    chart = newer;
                }
                let images = find(&chart, &mut index).and_then(|p| std::fs::read(p).ok()).and_then(|d| decode(&d));
                mail.send(Msg::Backdrop(chart, images.map(Box::new)));
            }
        }).expect("a thread");
        Backdrop { tx, wanted: None, shown: None, before: None, since: 0. }
    }

    /// The selected chart changed.
    pub fn want(&mut self, chart: &Path) {
        if self.wanted.as_deref() != Some(chart) {
            self.wanted = Some(chart.to_path_buf());
            let _ = self.tx.send(chart.to_path_buf());
        }
    }

    pub fn arrived(&mut self, ctx: &egui::Context, chart: PathBuf, images: Option<Box<Images>>) {
        if self.wanted.as_ref() != Some(&chart) {
            return;
        }
        self.before = self.shown.take();
        self.shown = images.map(|i| (ctx.load_texture("backdrop", i.light, TextureOptions::LINEAR), std::sync::Arc::new(i.lum)));
        self.since = ctx.input(|i| i.time);
    }

    /// The image's layers, oldest first, each with its opacity; repaints only while one fades.
    pub fn layers(&mut self, ctx: &egui::Context) -> Vec<crate::theme::Layer> {
        let fade = ((ctx.input(|i| i.time) - self.since) / FADE).clamp(0., 1.) as f32;
        if fade < 1. {
            ctx.request_repaint();
        } else {
            self.before = None;
        }
        let layer = |t: &Shown, fade| crate::theme::Layer { window: t.0.id(), size: t.0.size_vec2(), lum: t.1.clone(), fade };
        let old = self.before.as_ref().map(|t| layer(t, if self.shown.is_some() { 1. } else { 1. - fade }));
        old.into_iter().chain(self.shown.as_ref().map(|s| layer(s, fade))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chart_names_its_background_and_a_video_line_is_not_one() {
        let chart = "[General]\nAudioFilename: a.mp3\n[Events]\n//Background and Video events\nVideo,0,\"v.mp4\"\n0,0,\"sub\\bg image.jpg\",0,0\n[TimingPoints]\n0,0,\"no.png\"\n";
        assert_eq!(named(chart).as_deref(), Some("sub/bg image.jpg"));
        assert_eq!(named("[Events]\n2,100,200\n[HitObjects]\n"), None);
    }

    #[test]
    fn a_file_that_is_no_image_gives_none_and_an_image_gives_both_sizes() {
        assert!(decode(b"not an image").is_none());
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(1280, 720, image::Rgba([200, 100, 50, 255]))).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let i = decode(&png).unwrap();
        assert_eq!((i.light.size, i.lum.len()), ([320, 180], GRID * GRID));
        assert!(i.lum.iter().all(|l| (l - 0.46).abs() < 0.02), "its brightness everywhere: {:?}", &i.lum[..3]);
    }
}

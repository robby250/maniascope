//! A window resizing itself. KWin keeps a Wayland window's own resizes centred (xdgshellwindow.cpp:
//! preferredSpontaneousGravity = Center), so closing Details moved the main window down (user 2026-11-14). On KDE
//! Wayland the compositor resizes it instead, which keeps the top-left corner; elsewhere the app's own resize does.
use eframe::egui::{Context, Vec2, ViewportCommand};

fn kde_wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some() && std::env::var("XDG_CURRENT_DESKTOP").is_ok_and(|d| d.to_uppercase().contains("KDE"))
}

/// Resize this viewport (titled `title`) to the inner size `to`.
pub fn inner(ctx: &Context, title: &str, to: Vec2) {
    let now = ctx.input(|i| i.viewport_rect().size());
    if !kde_wayland() || now.x < 1. {
        return ctx.send_viewport_cmd(ViewportCommand::InnerSize(to));
    }
    let (d, title, ctx) = (to - now, title.to_string(), ctx.clone());
    std::thread::spawn(move || {
        if kwin(&title, d).is_err() {
            ctx.send_viewport_cmd(ViewportCommand::InnerSize(to));
            ctx.request_repaint();
        }
    });
}

/// A one-shot KWin script growing our window's frame by `d`: the frame and its contents change by the same amount.
fn kwin(title: &str, d: Vec2) -> std::io::Result<()> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let name = format!("maniascope-resize-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed));
    let path = std::env::temp_dir().join(format!("{name}.js"));
    let title = serde_json::to_string(title).unwrap_or_default();
    std::fs::write(&path, format!("for (const w of workspace.windowList()) if (w.pid === {} && w.caption === {title}) {{ const g = w.frameGeometry; \
        w.frameGeometry = {{x: g.x, y: g.y, width: Math.round(g.width + {}), height: Math.round(g.height + {})}}; }}", std::process::id(), d.x, d.y))?;
    let call = |path: &str, method: &str, args: &[&str]| -> std::io::Result<String> {
        let out = std::process::Command::new("gdbus").args(["call", "--session", "--dest", "org.kde.KWin", "--object-path", path, "--method", method]).args(args).output()?;
        if out.status.success() { Ok(String::from_utf8_lossy(&out.stdout).into_owned()) } else { Err(std::io::Error::other(String::from_utf8_lossy(&out.stderr).into_owned())) }
    };
    let file = path.to_string_lossy().into_owned();
    let run = call("/Scripting", "org.kde.kwin.Scripting.loadScript", &[&file, &name]).and_then(|id| {
        let id: String = id.chars().filter(char::is_ascii_digit).collect();
        call(&format!("/Scripting/Script{id}"), "org.kde.kwin.Script.run", &[])
    });
    std::thread::sleep(std::time::Duration::from_millis(200)); // the script runs on KWin's loop
    let _ = call("/Scripting", "org.kde.kwin.Scripting.unloadScript", &[&name]);
    let _ = std::fs::remove_file(&path);
    run.map(drop)
}

//! Explicit Next actions. Search syntax checked against ppy/osu FilterQueryParser.cs and FilterCriteria.cs: quoted
//! values followed by ! match the complete field; backslash-escaped quotes are NOT supported. A local SHA/MD5 is an
//! identity, not a song-search keyword.
use crate::catalog::Pick;
use ms_data::paths;
use ms_osumem::process::exes;
use std::path::PathBuf;
use std::process::{Command, Stdio};

pub const ACTIONS: [&str; 3] = ["auto", "copy", "link"];

/// A submitted chart can update its bytes without changing its map ID.
pub fn same_map(p: &Pick, sha: Option<&str>, md5: Option<&str>, bid: Option<i64>) -> bool {
    (sha.is_some() && sha == p.sha.as_deref()) || (md5.is_some() && md5 == p.md5.as_deref()) || (bid.is_some_and(|b| b > 0) && bid == p.bid)
}

fn field(name: &str, value: &str) -> String {
    let value = value.trim();
    // The lazer parser has no quoted-string escape convention. A literal safe fragment is preferable to a
    // syntactically broken 'escaped' exact match.
    let fragments: Vec<&str> = value.split(['"', '\r', '\n', '\t']).collect();
    let fragment = fragments.iter().copied().max_by_key(|f| f.chars().count()).unwrap_or("").trim();
    if fragment.is_empty() {
        return String::new();
    }
    format!("{name}=\"{fragment}\"{}", if fragments.len() == 1 { "!" } else { "" })
}

/// Search the local difficulty; intended playback rate is deliberately absent.
pub fn song_search(p: &Pick, client: &str) -> String {
    let (artist, title, version, creator) = (p.artist.as_deref().unwrap_or(""), p.song.as_deref().unwrap_or(""), p.version.as_deref().unwrap_or(""), p.creator.as_deref().unwrap_or(""));
    if client == "stable" {
        // stable has no quoted field values: its words match artist/title/difficulty/creator/tags, keys= filters
        // (osu! wiki, Beatmap search)
        let words = format!("{artist} {title} {version}").replace(['"', '=', '<', '>', '!'], " ");
        let query = words.split_whitespace().collect::<Vec<_>>().join(" ");
        return if query.is_empty() { query } else if p.keys != 0 { format!("{query} keys={}", p.keys) } else { query };
    }
    let fields = [field("artist", artist), field("title", title), field("diff", version), field("creator", creator)];
    let query = fields.iter().filter(|f| !f.is_empty()).cloned().collect::<Vec<_>>().join(" ");
    if !query.is_empty() {
        return if p.keys != 0 { format!("{query} cs={}", p.keys) } else { query };
    }
    p.bid.filter(|&b| b > 0).map_or_else(|| p.title.trim().to_string(), |b| b.to_string())
}

/// The tosu the player runs themselves (Windows: its in-game overlay is Windows only).
pub fn running_tosu() -> Option<PathBuf> {
    exes("tosu.exe").into_iter().next().map(|e| e.1)
}

/// The active client's running osu!.exe (Windows; lazer and stable share the name); stable's sits beside its osu!.db.
fn client_exe(client: &str) -> Option<PathBuf> {
    exes("osu!.exe").into_iter().map(|e| e.1).find(|e| e.with_file_name("osu!.db").is_file() == (client == "stable"))
}

/// The active client is running (stable is osu!.exe under Wine).
pub fn game_running(client: &str) -> bool {
    if paths::WINDOWS {
        return client_exe(client).is_some();
    }
    let want = if client == "stable" { "osu!.exe" } else { "osu!" };
    std::fs::read_dir("/proc").into_iter().flatten().flatten().any(|e| {
        e.file_name().to_string_lossy().bytes().all(|b| b.is_ascii_digit()) && std::fs::read_to_string(e.path().join("comm")).is_ok_and(|c| c.trim() == want)
    })
}

/// Hand an osu:// link to the running client: a second process of the same exe only forwards it over IPC. On Windows
/// never through the URL handler: osu:// belongs to whichever client registered it, so a lazer player got stable
/// launched (user 2026-10-31).
fn open_url(url: &str, client: &str) -> std::io::Result<()> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut command = if paths::WINDOWS {
        let mut c = Command::new(client_exe(client).ok_or_else(|| std::io::Error::other("osu! has closed"))?);
        c.arg(url);
        c
    } else if client == "lazer" && std::path::Path::new(&home).join(".var/app/sh.ppy.osu").is_dir() {
        let mut c = Command::new("flatpak");
        c.args(["run", "sh.ppy.osu", url]);
        c
    } else {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    command.stdout(Stdio::null()).stderr(Stdio::null()).spawn().map(|_| ())
}

/// Hand a file to the running client: a second process of the same exe forwards it over IPC (Program.cs), and lazer
/// imports it. Linux lazer: the running binary itself, or Flatpak's; xdg-open only knows what's registered.
pub fn hand_over(path: &std::path::Path, client: &str) -> Result<(), String> {
    if client == "stable" && !paths::WINDOWS {
        return Ok(()); // the .osz is in Songs and stable imports it on entering song select (Wine, 2026-11-01); xdg-open would run whatever is registered, lazer included
    }
    #[cfg(unix)]
    if client == "lazer" {
        match lazer_ipc(path) {
            Ok(()) => return Ok(()),
            Err(e) => eprintln!("lazer IPC hand-over failed, starting osu! with the file: {e}"),
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let mut command = if paths::WINDOWS {
        let mut c = Command::new(client_exe(client).ok_or("osu! has closed")?);
        c.arg(path);
        c
    } else if client == "lazer" && std::path::Path::new(&home).join(".var/app/sh.ppy.osu").is_dir() {
        let mut c = Command::new("flatpak");
        c.args(["run", "sh.ppy.osu"]).arg(path);
        c
    } else if let Some(exe) = running_lazer().filter(|_| client == "lazer") {
        let mut c = Command::new(exe);
        c.arg(path);
        c
    } else {
        let mut c = Command::new("xdg-open");
        c.arg(path);
        c
    };
    command.stdout(Stdio::null()).stderr(Stdio::null()).spawn().map(|_| ()).map_err(|e| e.to_string())
}

fn lazer_proc() -> Option<PathBuf> {
    std::fs::read_dir("/proc").ok()?.flatten().find(|e| std::fs::read_to_string(e.path().join("comm")).is_ok_and(|c| c.trim() == "osu!")).map(|e| e.path())
}

fn running_lazer() -> Option<PathBuf> {
    std::fs::read_link(lazer_proc()?.join("exe")).ok()
}

/// lazer's import channel (osu-framework NamedPipeIpcProvider, osu.Game ArchiveImportIPCChannel): the message a second
/// `osu! <file>` sends, without starting that process (flatpak run + .NET startup).
/// ponytail: Linux/macOS only; Windows still starts osu!.exe with the file (its pipe is `\\.\pipe\osu-framework-<name>`).
#[cfg(unix)]
fn lazer_ipc(path: &std::path::Path) -> Result<(), String> {
    use std::io::Write;
    let proc = lazer_proc().ok_or("osu! has closed")?;
    let root = proc.join("root"); // Flatpak: the sandbox's own /tmp and /app
    let exe = std::fs::read_link(proc.join("exe")).map_err(|e| e.to_string())?;
    let dll = root.join(exe.with_file_name("osu.Game.dll").strip_prefix("/").map_err(|e| e.to_string())?);
    let version = assembly_version(&std::fs::read(&dll).map_err(|e| format!("{}: {e}", dll.display()))?).ok_or("osu.Game.dll has no version")?;
    let environ = std::fs::read(proc.join("environ")).unwrap_or_default();
    let tmp = environ.split(|b| *b == 0).find_map(|v| v.strip_prefix(b"TMPDIR=")).map_or("/tmp".into(), String::from_utf8_lossy);
    // .NET's named pipe on Unix; the game names it ("osu-lazer" on 2026.804.2), so take the one osu-framework pipe there.
    let socket = std::fs::read_dir(root.join(tmp.trim_start_matches('/'))).map_err(|e| e.to_string())?.flatten()
        .find(|e| e.file_name().to_string_lossy().starts_with("CoreFxPipe_osu-framework-")).ok_or("osu! has no import pipe")?.path();
    let mut stream = std::os::unix::net::UnixStream::connect(&socket).map_err(|e| format!("{}: {e}", socket.display()))?;
    stream.write_all(&ipc_frame(path, &version)).map_err(|e| e.to_string())
}

/// One IpcMessage: a little-endian i32 length (BitConverter on every lazer platform), then its JSON. lazer drops a
/// message whose Type isn't its exact AssemblyQualifiedName, version included.
fn ipc_frame(path: &std::path::Path, version: &str) -> Vec<u8> {
    let kind = format!("osu.Game.IPC.ArchiveImportMessage, osu.Game, Version={version}, Culture=neutral, PublicKeyToken=null");
    let json = serde_json::json!({"Type": kind, "Value": {"Path": path.to_string_lossy()}}).to_string();
    let mut frame = (json.len() as u32).to_le_bytes().to_vec();
    frame.extend(json.as_bytes());
    frame
}

/// A .NET DLL's "Assembly Version" from its version resource (UTF-16, 4-byte aligned).
fn assembly_version(dll: &[u8]) -> Option<String> {
    let key: Vec<u8> = "Assembly Version".encode_utf16().flat_map(u16::to_le_bytes).collect();
    let at = dll.windows(key.len()).position(|w| w == key.as_slice())? + key.len();
    let units: Vec<u16> = dll[at..].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).skip_while(|u| *u == 0).take_while(|u| *u != 0).collect();
    String::from_utf16(&units).ok().filter(|v| !v.is_empty())
}

#[derive(Default)]
pub struct Performed {
    pub copy: Option<String>,
    pub msg: String,
    pub sent: bool,
    pub unavailable: bool,
}

/// The worker writes or downloads this pick into the client. `auto` off (Website charts: "look first", user 2026-11-06):
/// nothing is downloaded; a website map and a website convert's original open in osu! instead. A convert of a map you
/// have still arrives by itself: nothing to download.
pub fn delivers(p: &Pick, auto: bool) -> bool {
    !p.installed && if p.convert_of.is_some() { auto || !p.source_web } else { auto && p.set_id.is_some_and(|s| s > 0) }
}

/// copy/msg/sent; never starts the game or automatically changes mods.
pub fn perform(p: &mut Pick, action: &str, client: &str, auto: bool) -> Performed {
    if delivers(p, auto) {
        // the worker delivers it
        return Performed { msg: if p.convert_of.is_some() { format!("Converting to {}K…", p.keys) } else { "Downloading the map…".to_string() }, ..Performed::default() };
    }
    let original = p.convert_of.is_some() && !p.installed; // a website convert, looked at first: its original opens
    if p.installed {
        let path = p.sha.as_deref().and_then(|s| crate::pp::chart_path(Some(s)));
        if path.is_none() {
            if p.mode == "nps" || p.mode == "skills" {
                return Performed { msg: "Chart is no longer installed — refresh the library".to_string(), unavailable: true, ..Performed::default() };
            }
            p.installed = false;
        }
    }
    let action = if ACTIONS.contains(&action) { action } else { "auto" };
    let bid = p.bid.filter(|&b| b > 0);
    // A known edited revision must never open the online original instead.
    let trusted = bid.is_some() && (p.online_match || original);
    if action == "copy" || (action == "auto" && p.installed) || !trusted {
        let query = song_search(p, client);
        if query.is_empty() {
            return Performed { msg: "No searchable metadata for this chart".to_string(), ..Performed::default() };
        }
        let msg = if !p.installed {
            "Search copied, but you don't have this map — download it, then Ctrl+V in song select"
        } else if action == "link" && !trusted {
            "No matching online map — search copied, press Ctrl+V in song select"
        } else {
            "Search copied — press Ctrl+V in song select"
        };
        return Performed { copy: Some(query), msg: msg.to_string(), ..Performed::default() };
    }
    let url = format!("osu://b/{}", bid.unwrap());
    if !game_running(client) {
        return Performed { copy: Some(url), msg: "osu! is not running — copied beatmap link".to_string(), ..Performed::default() };
    }
    if let Err(e) = open_url(&url, client) {
        return Performed { copy: Some(url), msg: format!("Could not open beatmap panel ({e}) — copied link"), ..Performed::default() };
    }
    // Auto still copies the search, so after downloading it can be pasted in song select (user 2026-10-09).
    let query = (action == "auto").then(|| song_search(p, client)).filter(|q| !q.is_empty());
    let msg = if original {
        format!("Opened the original in osu! — download it, then ⋯ → Convert this map to {}K", p.keys)
    } else {
        format!("Sent beatmap link to osu! — choose Go to beatmap; mods unchanged{}", if query.is_some() { " · search copied, Ctrl+V in song select" } else { "" })
    };
    Performed { copy: query, msg, sent: true, unavailable: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_import_message_is_a_length_then_lazers_json() {
        let u16s = |t: &str| t.encode_utf16().flat_map(u16::to_le_bytes).collect::<Vec<u8>>();
        let dll = [b"MZ..".to_vec(), u16s("Assembly Version"), vec![0; 4], u16s("2026.804.2.0"), vec![0; 2], u16s("junk")].concat();
        let version = assembly_version(&dll).unwrap();
        assert_eq!(version, "2026.804.2.0");
        let frame = ipc_frame(std::path::Path::new("/d/maniascope/a.osz"), &version);
        let json = &frame[4..];
        assert_eq!(u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize, json.len());
        assert_eq!(std::str::from_utf8(json).unwrap(), r#"{"Type":"osu.Game.IPC.ArchiveImportMessage, osu.Game, Version=2026.804.2.0, Culture=neutral, PublicKeyToken=null","Value":{"Path":"/d/maniascope/a.osz"}}"#);
    }

    #[test]
    fn search_fragments_a_quoted_title_and_never_escapes() {
        assert_eq!(field("title", "plain"), "title=\"plain\"!");
        assert_eq!(field("title", "a \"long part\" b"), "title=\"long part\"");
        assert_eq!(field("title", "  "), "");
    }

    #[test]
    fn look_first_downloads_nothing_but_a_convert_of_a_map_you_have_still_arrives() {
        let web = Pick { set_id: Some(5), ..Pick::default() };
        let web_convert = Pick { convert_of: Some("s".into()), source_web: true, ..web.clone() };
        let own_convert = Pick { convert_of: Some("s".into()), ..Pick::default() };
        assert!(delivers(&web, true) && delivers(&web_convert, true) && delivers(&own_convert, true));
        assert!(!delivers(&web, false) && !delivers(&web_convert, false) && delivers(&own_convert, false));
        assert!(!delivers(&Pick { installed: true, ..web }, true));
    }

    #[test]
    fn stable_search_has_no_field_syntax() {
        let p = Pick { artist: Some("A=B".into()), song: Some("T!".into()), version: Some("V".into()), keys: 7, ..Pick::default() };
        assert_eq!(song_search(&p, "stable"), "A B T V keys=7");
        assert_eq!(song_search(&p, "lazer"), "artist=\"A=B\"! title=\"T!\"! diff=\"V\"! cs=7");
    }
}

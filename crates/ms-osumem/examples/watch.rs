//! Live check of the reader against a running game: `watch [lazer|stable]` prints the frame whenever it changes
//! (state, selected chart, the mods with their settings). Run it where it may read the game's memory.
fn main() {
    let prefer = std::env::args().nth(1).unwrap_or_else(|| "lazer".into());
    let (mut reader, mut last) = (ms_osumem::Reader::new(), String::new());
    loop {
        let frame = reader.poll(&prefer).map_or_else(|| "no game".to_string(), |v| v.to_string());
        if frame != last {
            println!("{frame}");
            last = frame;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

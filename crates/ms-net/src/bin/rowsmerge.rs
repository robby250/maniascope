//! `rowsmerge OUT DIR …`: one training set from several players' `calib rows` folders. Each play carries its folder's index as
//! `player` (0 = the first folder): notefit and thetafc key every time-local part (sessions, the θ walk, the 15-day θ average,
//! the previous play) by player, while every note trains the shared chart model (user 2026-11-15: train on my replays and
//! another player's). Plays keep their times and order; the `play` column is renumbered.
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = std::path::Path::new(&args[1]);
    std::fs::create_dir_all(out).unwrap();
    let mut bin = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(out.join("rows.bin")).unwrap());
    let (mut columns, mut plays, mut rows): (Option<serde_json::Value>, Vec<serde_json::Value>, u64) = (None, Vec::new(), 0);
    for (player, spec) in args[2..].iter().enumerate() {
        let dir = spec.as_str();
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/rows.json")).unwrap()).unwrap();
        let cols: Vec<String> = meta["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
        match &columns {
            None => columns = Some(meta["columns"].clone()),
            Some(c) => assert_eq!(c, &meta["columns"], "{dir}: other columns"),
        }
        let (w, ci) = (cols.len(), cols.iter().position(|c| c == "play").unwrap());
        let offset = plays.len() as f32;
        for p in meta["plays"].as_array().unwrap() {
            let mut p = p.clone();
            p["player"] = player.into();
            plays.push(p);
        }
        let file = std::fs::File::open(format!("{dir}/rows.bin")).unwrap();
        let data = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let f: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };
        for r in f.chunks_exact(w) {
            let mut r = r.to_vec();
            r[ci] += offset;
            for x in &r {
                bin.write_all(&x.to_le_bytes()).unwrap();
            }
            rows += 1;
        }
        eprintln!("{dir}: {} plays, {} notes", meta["plays"].as_array().unwrap().len(), f.len() / w);
    }
    bin.flush().unwrap();
    let meta = serde_json::json!({"columns": columns, "rows": rows, "plays": plays});
    std::fs::write(out.join("rows.json"), serde_json::to_string(&meta).unwrap()).unwrap();
    eprintln!("{} plays, {rows} notes", meta["plays"].as_array().unwrap().len());
}

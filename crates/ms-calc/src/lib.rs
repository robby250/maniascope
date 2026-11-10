//! ManiaScope difficulty calculator. calc_id is a constant: moving source files must not re-analyse a library.
pub mod background;
pub mod card;
pub mod chart;
pub mod compute;
pub mod demand;
pub mod execution;
pub mod features;
pub mod flow;
pub mod gestures;
pub mod kern;
pub mod model;
pub mod params;
pub mod pattern;
pub mod py;
pub mod scroll;
pub mod skills;
pub mod structure;
pub mod val;

// Every binary links the calculator: its many short-lived vectors made the system allocator ~9 % of a chart's analysis.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

// A refit takes the sha256 of calib/difficulty.json; a rules-only change (labels, same stars) that of it plus the rule commit.
pub const CALC_ID: &str = "f92680b7b136";

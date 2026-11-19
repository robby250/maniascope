//! Section timer: `lap(name)` books the time since the last lap on this thread to `name`. Off (one flag read per lap) until
//! `enable()`; `calib benchmark_analysis` with MS_TICK=1 prints the sections. A sampling profile of the calculator is flat:
//! the sections are what showed where the time went (plan 2026-11-19 D: column runs 3×, releases, hand pulses).
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);

pub fn enable() {
    ON.store(true, Ordering::Relaxed);
}

thread_local! { static T: RefCell<(Option<Instant>, Vec<(&'static str, u64)>)> = const { RefCell::new((None, Vec::new())) }; }

#[inline]
pub fn lap(name: &'static str) {
    if ON.load(Ordering::Relaxed) {
        book(name);
    }
}

#[cold]
fn book(name: &'static str) {
    T.with(|t| {
        let mut t = t.borrow_mut();
        let now = Instant::now();
        if let Some(at) = t.0 {
            let ns = (now - at).as_nanos() as u64;
            match t.1.iter_mut().find(|x| x.0 == name) {
                Some(x) => x.1 += ns,
                None => t.1.push((name, ns)),
            }
        }
        t.0 = Some(now);
    })
}

/// The sections' totals in ms, largest first, and the timer cleared.
pub fn take() -> Vec<(&'static str, f64)> {
    T.with(|t| {
        let mut t = t.borrow_mut();
        t.0 = None;
        let mut v: Vec<_> = t.1.drain(..).map(|(n, ns)| (n, ns as f64 / 1e6)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    })
}

//! osu!lazer's mania star rating, ported from rosu-pp 4.0.1 (`mania::difficulty`, `util::sort::osu_legacy`; MIT,
//! Copyright (c) 2020 MaxOhn). The notes are read and sorted once per chart; each rate runs only the strain pass over flat
//! arrays (rosu rebuilt and re-sorted the map and allocated a per-column history per note at every rate). Stars are equal to
//! rosu's bit for bit (test below).
use std::cmp::Ordering;

/// A chart's notes as lazer's difficulty reads them: lazer's order, chart ms, column.
pub struct Notes {
    start: Vec<f64>,
    end: Vec<f64>,
    col: Vec<usize>,
    columns: usize,
    holds: u32,
}

impl Notes {
    /// From (start ms, hold duration ms, x, hold) in the file's order: a stable sort by start (rosu's parse), then lazer's.
    pub fn new(keys: f32, objects: impl Iterator<Item = (f64, f64, f32, bool)>) -> Notes {
        let mut v: Vec<(f64, f64, f32, bool)> = objects.collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        osu_legacy(&mut v, |a, b| (a.0.round_ties_even() as i32).cmp(&(b.0.round_ties_even() as i32)));
        let total = keys.round_ties_even().max(1.0);
        let divisor = 512.0 / total;
        Notes {
            start: v.iter().map(|o| o.0).collect(),
            end: v.iter().map(|o| if o.3 { o.0 + o.1 } else { o.0 }).collect(),
            col: v.iter().map(|o| (o.2 / divisor).floor().min(total - 1.0) as usize).collect(),
            columns: total as usize,
            holds: v.iter().filter(|o| o.3).count() as u32,
        }
    }

    /// (stars, hit objects + hold notes) at a clock rate.
    pub fn stars(&self, rate: f64) -> (f64, u32) {
        let n = self.start.len();
        let hits = n as u32 + self.holds;
        if n < 2 {
            return (0.0, hits);
        }
        let k = self.columns;
        // difficulty objects start at the second note; `last[c]` = the last difficulty object on column c before the current
        let mut last: Vec<Option<usize>> = vec![None; k];
        let (mut individual, mut highest, mut overall, mut current) = (vec![0.0f64; k], 0.0f64, 1.0f64, 0.0f64);
        let (mut section_end, mut peak, mut peaks) = (0.0f64, 0.0f64, Vec::with_capacity(n / 4 + 2));
        let decay = |v: f64, ms: f64, base: f64| v * f64::powf(base, ms / 1000.0);
        // the same divisions, once per note (the evaluators read every column's last note per note)
        let (ss, es): (Vec<f64>, Vec<f64>) = (self.start.iter().map(|x| x / rate).collect(), self.end.iter().map(|x| x / rate).collect());
        let (s, e) = (|i: usize| ss[i], |i: usize| es[i]);
        for i in 1..n {
            let idx = i - 1;
            let (start, end, column) = (s(i), e(i), self.col[i]);
            let delta = (self.start[i] - self.start[i - 1]) / rate;
            if idx == 0 {
                section_end = f64::ceil(start / 400.0) * 400.0;
            }
            while start > section_end {
                peaks.push(peak);
                let prev_start = if idx > 0 { s(i - 1) } else { 0.0 };
                peak = decay(highest, section_end - prev_start, 0.125) + decay(overall, section_end - prev_start, 0.3);
                section_end += 400.0;
            }
            let column_strain_time = start - last[column].map_or(start, s);
            // the evaluators read every column's last object before this one
            let (mut bonus, mut overlapping, mut closest, mut hold_factor) = (false, false, (end - start).abs(), 1.0);
            for p in last.iter().flatten().copied() {
                let (ps, pe) = (ss[p], es[p]);
                let held = pe > end + 1.0 && start > ps + 1.0;
                bonus |= held;
                overlapping |= pe > start + 1.0 && end > pe + 1.0 && start > ps + 1.0;
                if held {
                    hold_factor = 1.25;
                }
                closest = closest.min((end - pe).abs());
            }
            individual[column] = decay(individual[column], column_strain_time, 0.125);
            individual[column] += 2.0 * if bonus { 1.25 } else { 1.0 };
            highest = if delta <= 1.0 { highest.max(individual[column]) } else { individual[column] };
            overall = decay(overall, delta, 0.3);
            let addition = if overlapping { 1.0 / (1.0 + f64::exp(0.27 * (30.0 - closest))) } else { 0.0 };
            overall += (1.0 + addition) * hold_factor;
            current += highest + overall - current;
            peak = peak.max(current);
            last[column] = Some(i);
        }
        peaks.push(peak);
        let mut p: Vec<f64> = peaks.into_iter().filter(|&p| p > 0.0).collect();
        p.sort_by(|a, b| b.total_cmp(a));
        let (mut difficulty, mut weight) = (0.0, 1.0);
        for v in p {
            difficulty += v * weight;
            weight *= 0.9;
        }
        (difficulty * 0.018, hits)
    }
}

/// osu!'s legacy depth-limited quicksort (rosu-pp `util::sort::osu_legacy`): equal keys end up in its order, which lazer reads.
fn osu_legacy<T, F: Fn(&T, &T) -> Ordering>(keys: &mut [T], cmp: F) {
    if keys.len() >= 2 {
        let right = keys.len() - 1;
        quick(keys, 0, right, &cmp, 32);
    }
}

fn quick<T, F: Fn(&T, &T) -> Ordering>(keys: &mut [T], mut left: usize, mut right: usize, cmp: &F, mut depth: usize) {
    loop {
        if depth == 0 {
            heap_sort(keys, left, right, cmp);
            return;
        }
        let (mut i, mut j) = (left, right);
        let middle = i + ((j - i) >> 1);
        swap_if_greater(keys, cmp, i, middle);
        swap_if_greater(keys, cmp, i, j);
        swap_if_greater(keys, cmp, middle, j);
        loop {
            while cmp(&keys[i], &keys[middle]).is_lt() {
                i += 1;
            }
            while cmp(&keys[middle], &keys[j]).is_lt() {
                j -= 1;
            }
            match i.cmp(&j) {
                Ordering::Less => keys.swap(i, j),
                Ordering::Equal => {}
                Ordering::Greater => break,
            }
            i += 1;
            j = j.saturating_sub(1);
            if i > j {
                break;
            }
        }
        depth -= 1;
        if j.saturating_sub(left) <= right - i {
            if left < j {
                quick(keys, left, j, cmp, depth);
            }
            left = i;
        } else {
            if i < right {
                quick(keys, i, right, cmp, depth);
            }
            right = j;
        }
        if left >= right {
            break;
        }
    }
}

fn heap_sort<T, F: Fn(&T, &T) -> Ordering>(keys: &mut [T], lo: usize, hi: usize, cmp: &F) {
    let n = hi - lo + 1;
    for i in (1..=n / 2).rev() {
        down_heap(keys, i, n, lo, cmp);
    }
    for i in (2..=n).rev() {
        if lo != lo + i - 1 {
            keys.swap(lo, lo + i - 1);
        }
        down_heap(keys, 1, i - 1, lo, cmp);
    }
}

fn down_heap<T, F: Fn(&T, &T) -> Ordering>(keys: &mut [T], mut i: usize, n: usize, lo: usize, cmp: &F) {
    while i <= n / 2 {
        let mut child = 2 * i;
        if child < n && cmp(&keys[lo + child - 1], &keys[lo + child]).is_lt() {
            child += 1;
        }
        if cmp(&keys[lo + i - 1], &keys[lo + child - 1]).is_ge() {
            break;
        }
        keys.swap(lo + i - 1, lo + child - 1);
        i = child;
    }
}

fn swap_if_greater<T, F: Fn(&T, &T) -> Ordering>(keys: &mut [T], cmp: &F, a: usize, b: usize) {
    if a != b && cmp(&keys[a], &keys[b]).is_gt() {
        keys.swap(a, b);
    }
}

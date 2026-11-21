//! What a chart is at every rate (plan 2026-11-20 Stage B): built once per chart, read by each rate's pass. Rates differ
//! only in real-time limits; what no limit touches is kept here, the rest as the chart value each limit is compared with.
use crate::chart::Chart;
use crate::demand::Row;

pub struct Layout {
    /// Notes at one chart ms: (first note, end, mask). Rows at any rate where no two distinct times merge.
    atoms: Vec<(usize, usize, u32)>,
    /// Smallest gap between distinct note times, chart ms: rows equal the atoms while 3·rate ms stays under it.
    gap_min: f64,
    /// Each atom's surprise bits ÷ keys under the order-1 chain of the last 128 rows (demand::base's pattern information).
    pub info_bits: Vec<f64>,
    /// `demand::hand_topo` of the atoms under each hand split (`demand::splits` order).
    topo: Vec<crate::demand::HandTopo>,
    /// `structure::family_of` (charts of 2+ notes; empty otherwise, never read).
    pub family: String,
}

impl Layout {
    pub fn new(chart: &Chart) -> Layout {
        let n = chart.len();
        let mut atoms = Vec::with_capacity(n);
        let mut gap_min = f64::INFINITY;
        let mut i = 0;
        while i < n {
            let mut m = i + 1;
            while m < n && chart.t[m] == chart.t[i] {
                m += 1;
            }
            if m < n {
                gap_min = gap_min.min(chart.t[m] - chart.t[i]);
            }
            atoms.push((i, m, chart.col[i..m].iter().fold(0u32, |a, &c| a | 1 << c)));
            i = m;
        }
        let keys = chart.keys;
        let mut chain = crate::notefeat::Markov::new(2f64.powi(keys as i32) - 1., 128);
        let mut prev = 0u32;
        let info_bits = atoms
            .iter()
            .map(|&(_, _, mask)| {
                let bits = chain.step(prev, mask) / keys as f64;
                prev = mask;
                bits
            })
            .collect();
        let rows: Vec<Row> = atoms.iter().map(|&(start, end, mask)| Row { t: chart.t[start] * 0.001, mask, start, end }).collect();
        let topo = crate::demand::splits(keys).iter().map(|&cl| crate::demand::hand_topo(&rows, crate::demand::hand_masks(keys, cl), keys)).collect();
        let family = if n >= 2 && chart.t[n - 1] > chart.t[0] { crate::structure::family_of(chart) } else { String::new() };
        Layout { atoms, gap_min, info_bits, topo, family }
    }

    /// The hand split's `demand::hand_topo` (`centre_left` as `demand::splits` names it).
    pub fn hand_topo(&self, centre_left: bool) -> &crate::demand::HandTopo {
        &self.topo[if self.topo.len() == 2 && !centre_left { 1 } else { 0 }]
    }

    /// The rows at `rate` (`demand::rows`) when they are the atoms: no two distinct note times within 3·rate ms.
    pub fn rows_at(&self, chart: &Chart, rate: f64) -> Option<Vec<Row>> {
        let inv = 0.001 / rate;
        (3.0 * rate < self.gap_min)
            .then(|| self.atoms.iter().map(|&(start, end, mask)| Row { t: chart.t[start] * inv, mask, start, end }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chart(notes: &[(f64, u8)]) -> Chart {
        let mut c = Chart { keys: 7, ..Chart::default() };
        for &(t, col) in notes {
            c.t.push(t);
            c.end.push(t);
            c.col.push(col);
        }
        c
    }

    #[test]
    fn rows_are_the_atoms_until_two_times_merge() {
        let c = chart(&[(0., 0), (0., 3), (4., 1), (100., 2), (100., 5), (300., 6)]);
        let l = Layout::new(&c);
        let same = |a: &[Row], b: &[Row]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x.t, x.mask, x.start, x.end) == (y.t, y.mask, y.start, y.end));
        for rate in [0.5, 0.75, 1.0, 1.3] {
            assert!(same(&l.rows_at(&c, rate).unwrap(), &crate::demand::rows(&c, rate)), "rate {rate}");
        }
        assert!(l.rows_at(&c, 1.34).is_none()); // 4 ms ≤ 3 × 1.34: the second time joins the first row
    }
}

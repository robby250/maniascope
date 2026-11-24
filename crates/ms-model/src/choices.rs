//! The NPS/Skills choice list from the Builder's scoring to the shown list: one record per chart (its best exact
//! rate) and the worker's prepare → weighting → shuffle → web mix over all of them.
use crate::predict::{HashMap, Prediction};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::Arc;

/// What the worker's ranking and the shown list read of one chart's choice.
#[derive(Clone)]
pub struct Choice {
    pub sha: String,
    pub md5: String, // "" when the chart has none
    pub keys: i64,
    pub rate: f64,
    pub value: f64, // base_value
    pub installed: bool,
    pub convert: bool, // a keymode convert, imported or not
    pub source_web: bool, // a convert of a website chart
    pub playcount: f64,
    pub family: String,
    pub description: String,
    pub group: String, // nps.group
    pub nps: f64,
    pub length: f64, // play span at this rate, s
    pub profile: Vec<f64>,
    pub acc_mid: f64,
    pub matched: Vec<String>, // included Skills the chart matches, for the how-often sliders
    pub practice: String,     // skill_practice.describe_match, Skills lists only
    pub pred: Prediction,
    pub generated_at: f64,
    /// Under the target at every rate of the range, offered at the nearest (predict::LEARN_FLOOR): a choice only while its
    /// skill has fewer than LEARN_POOL charts at the target in the keymode (`Choices::learners`).
    pub learning: bool,
}

/// Charts a learning slot keeps (`Choices::learners`): enough for a session's turns of one skill in one keymode.
pub const LEARN_POOL: usize = 20;

pub const REAL: u8 = 0;
pub const WEB: u8 = 1;
pub const CONVERT: u8 = 2;
/// A convert of a website chart: it answers to both sliders (user 2026-11-18: at web 0.3 half the convert offers were website
/// sourced, as half the convert pool is).
pub const WEB_CONVERT: u8 = 3;

impl Choice {
    /// Which how-often slider the chart answers to. An imported convert is still a convert.
    pub fn stream(&self) -> u8 {
        if self.convert { if self.source_web { WEB_CONVERT } else { CONVERT } } else if self.installed { REAL } else { WEB }
    }

    /// Offered under the Invert mod (the Inverse mod skill: such a chart counts as nothing else).
    pub fn inverted(&self) -> bool {
        self.matched.iter().any(|m| m == crate::skills::INVERT_MOD)
    }

    /// recommend.event_key
    pub fn identity(&self) -> String {
        if self.md5.is_empty() { format!("sha:{}", self.sha) } else { format!("md5:{}", self.md5) }
    }
}

/// A rate proposal: (quality, rate, (keys, group)).
#[derive(Clone)]
pub struct Proposal {
    pub quality: f64,
    pub rate: f64,
    pub keys: i64,
    pub group: String,
    pub skill: bool, // the chart counts as a picked skill
}

/// One request's scored charts (the Builder's memo across its passes): per catalogue chart, unscored or
/// (choice, proposal). The Builder drops a chart whose analyses change.
pub struct Choices {
    order: Vec<String>,
    at: HashMap<String, usize>,
    pub slots: Vec<Option<(Option<Arc<Choice>>, Option<Proposal>)>>,
    forecast: Arc<Forecast>,
}

/// accuracy_forecast provenance of a request: calculator, predictor, BAND_Z.
#[derive(Clone, Default)]
pub struct Forecast {
    pub calc: String,
    pub predictor: String,
    pub band_z: f64,
}

impl Choices {
    pub fn slot(&self, sha: &str) -> Option<usize> {
        self.at.get(sha).copied()
    }

    /// The catalogue's charts in order, and the request's accuracy_forecast provenance.
    pub fn new(order: Vec<String>, forecast: Forecast) -> Self {
        let at = order.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let slots = vec![None; order.len()];
        Choices { order, at, slots, forecast: Arc::new(forecast) }
    }

    /// Charts not scored yet, in catalogue order.
    pub fn unscored(&self) -> Vec<String> {
        self.order.iter().zip(&self.slots).filter(|(_, s)| s.is_none()).map(|(o, _)| o.clone()).collect()
    }

    /// A chart whose analyses changed is scored again.
    pub fn drop_chart(&mut self, sha: &str) {
        if let Some(i) = self.slot(sha) {
            self.slots[i] = None;
        }
    }

    /// candidates_from's frontier: the proposals not refined or blocked, the best 24 overall and 4 of every
    /// (keys, group) (retention, not quotas: the eligible population keeps its proportions) → [(quality, sha,
    /// rate)], best first.
    pub fn proposals(&self, refined: &[(String, f64)], blocked: &HashSet<String>) -> Vec<(f64, String, f64)> {
        let refined: HashSet<(&str, u64)> = refined.iter().map(|(s, r)| (s.as_str(), r.to_bits())).collect();
        let mut all: Vec<(f64, &str, f64, &Proposal)> = Vec::new();
        for (sha, slot) in self.order.iter().zip(&self.slots) {
            if let Some((_, Some(p))) = slot {
                if !blocked.contains(sha) && !refined.contains(&(sha.as_str(), p.rate.to_bits())) {
                    all.push((p.quality, sha, p.rate, p));
                }
            }
        }
        // heapq.nlargest's tuple order: quality, then sha, rate and group.
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(a.1)).then(b.2.total_cmp(&a.2))
            .then((b.3.keys, &b.3.group).cmp(&(a.3.keys, &a.3.group))));
        let mut kept: Vec<(f64, &str, f64)> = all.iter().take(24).map(|x| (x.0, x.1, x.2)).collect();
        let mut per: HashMap<(i64, &str), usize> = HashMap::default();
        for x in &all {
            let n = per.entry((x.3.keys, x.3.group.as_str())).or_insert(0);
            if *n < 4 {
                *n += 1;
                kept.push((x.0, x.1, x.2));
            }
        }
        kept.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(a.1)).then(b.2.total_cmp(&a.2)));
        kept.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
        kept.into_iter().map(|(q, s, r)| (q, s.to_string(), r)).collect()
    }

    /// The charts of a picked skill that no analysed rate fits yet, with the rate that would (inside `rates`: at an
    /// end of the range it is too far off), in catalogue order. A skill's slot takes its turns from these few charts,
    /// and a chart indexed at 1× only (Invert, converts) is a choice only when 1× happens to fit: 1,611 of 23,882 7K Invert
    /// charts, 12 of 5,209 on 10K (2026-11-21), so the skill landed on the keymode with the most such luck.
    /// A learning slot's charts are also analysed at the range's end, where their forecast is nearest the target: only they,
    /// a keymode's whole too-hard tail (6,312 of 6,485 10K Invert charts, 2026-11-22) would be analysed for nothing.
    pub fn slot_proposals(&self, refined: &HashSet<&str>, rates: (f64, f64)) -> Vec<(String, f64)> {
        let learners = self.learners();
        self.order.iter().zip(&self.slots).filter_map(|(sha, slot)| match slot {
            Some((c, Some(p))) if p.skill && !refined.contains(sha.as_str()) && c.as_ref().is_none_or(|c| c.learning)
                && ((p.rate > rates.0 && p.rate < rates.1) || learners.contains(sha.as_str())) => Some((sha.clone(), p.rate)),
            _ => None,
        }).collect()
    }

    /// The learning choices that stand: a (keymode, leading picked skill) with fewer than LEARN_POOL choices at the target
    /// is filled up to that many with the nearest to it (user 2026-11-22: a picked skill with nothing on target in a
    /// keymode offers its nearest charts; 10K Invert had 18 at the target and 1,359 near it, and no turn in 300 presses:
    /// `deal` gives such a slot its slider's turns). A skill with LEARN_POOL charts at the target has none.
    // ponytail: a slot counts every scored chart at the target, a website one too while its slider is at 0; split by stream if that shows.
    fn learners(&self) -> HashSet<&str> {
        fn slot(c: &Choice) -> (i64, &str) {
            (c.keys, c.matched.first().map_or("", String::as_str))
        }
        let choices = || self.slots.iter().filter_map(|s| s.as_ref()?.0.as_deref());
        let mut filled: HashMap<(i64, &str), usize> = HashMap::default();
        choices().filter(|c| !c.learning).for_each(|c| *filled.entry(slot(c)).or_default() += 1);
        let mut pools: HashMap<(i64, &str), Vec<&Choice>> = HashMap::default();
        choices().filter(|c| c.learning).for_each(|c| pools.entry(slot(c)).or_default().push(c));
        pools.into_iter().flat_map(|(slot, mut pool)| {
            pool.sort_by(|a, b| b.acc_mid.total_cmp(&a.acc_mid).then(a.sha.cmp(&b.sha)));
            pool.into_iter().take(LEARN_POOL.saturating_sub(filled.get(&slot).copied().unwrap_or(0))).map(|c| c.sha.as_str())
        }).collect()
    }

    /// The choices as they stand, in catalogue order (what a publication carries; later passes don't touch it).
    pub fn snapshot(&self) -> ChoiceList {
        let learners = self.learners();
        ChoiceList {
            items: self.slots.iter().filter_map(|s| s.as_ref().and_then(|(c, _)| c.clone())).filter(|c| !c.learning || learners.contains(c.sha.as_str())).collect(),
            forecast: self.forecast.clone(),
        }
    }
}

/// One publication's choices (immutable): what the worker prepares, ranks and shows.
#[derive(Clone, Default)]
pub struct ChoiceList {
    pub items: Vec<Arc<Choice>>,
    pub forecast: Arc<Forecast>,
}

/// Newer exact features than the Builder had (Worker.t_nps_update).
pub struct Metadata {
    pub family: String,
    pub group: String,
    pub nps: f64,
    pub length: f64,
    pub profile: Vec<f64>,
    pub description: String,
    pub matched: Vec<String>,
    pub practice: String,
}

impl ChoiceList {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The `n` charts of the highest base_value: what the next request re-predicts first.
    pub fn lead(&self, n: usize) -> Vec<String> {
        let mut order: Vec<&Arc<Choice>> = self.items.iter().collect();
        order.sort_by(|a, b| b.value.total_cmp(&a.value));
        order.into_iter().take(n).map(|c| c.sha.clone()).collect()
    }

    /// (sha, rate) of every choice.
    pub fn refs(&self) -> Vec<(String, f64)> {
        self.items.iter().map(|c| (c.sha.clone(), c.rate)).collect()
    }

    /// The restart cache's references: the 512 best by base_value and 2048 spread by sha256(sha), sorted.
    pub fn restart_refs(&self) -> Vec<(String, f64)> {
        let mut best: Vec<&Arc<Choice>> = self.items.iter().collect();
        best.sort_by(|a, b| b.value.total_cmp(&a.value));
        let mut spread: Vec<(Vec<u8>, &Arc<Choice>)> = self.items.iter().map(|c| (Sha256::digest(c.sha.as_bytes()).to_vec(), c)).collect();
        spread.sort_by(|a, b| a.0.cmp(&b.0));
        let mut refs: Vec<(String, f64)> = best.into_iter().take(512).chain(spread.into_iter().take(2048).map(|x| x.1))
            .map(|c| (c.sha.clone(), c.rate)).collect();
        refs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        refs.dedup();
        refs
    }

    /// This list with refreshed metadata over the choices of {(event_key, rate as bits): metadata}.
    pub fn patched(&self, patches: &HashMap<(String, u64), Metadata>) -> ChoiceList {
        let mut items = self.items.clone();
        for c in items.iter_mut() {
            let Some(m) = patches.get(&(c.identity(), c.rate.to_bits())) else { continue };
            *c = Arc::new(Choice { family: m.family.clone(), group: m.group.clone(), nps: m.nps, length: m.length, profile: m.profile.clone(), description: m.description.clone(),
                                  matched: m.matched.clone(), practice: m.practice.clone(), ..(**c).clone() });
        }
        ChoiceList { items, forecast: self.forecast.clone() }
    }

    /// The worker's prepared pool: choices of `keys` at an offered rate whose chart is still here (a website
    /// chart, or installed and in `local_shas`).
    pub fn prepare(self: &Arc<Self>, keys: &[i64], rates: (f64, f64), local_shas: &dyn Fn(&str) -> bool) -> Prepared {
        let kept: Vec<usize> = self.items.iter().enumerate()
            .filter(|(_, c)| !(c.installed && !local_shas(&c.sha)) && keys.contains(&c.keys) && (rates.0 - 1e-6..=rates.1 + 1e-6).contains(&c.rate))
            .map(|x| x.0).collect();
        let by_identity = kept.iter().map(|&i| (self.items[i].identity(), i)).collect();
        let thin = kept.iter().map(|&i| &self.items[i]).filter(|c| c.learning).map(|c| (c.keys, c.matched.first().cloned().unwrap_or_default())).collect();
        Prepared { list: self.clone(), kept, by_identity, thin }
    }
}

/// A ChoiceList prepared for one worker state (keymodes, local charts).
pub struct Prepared {
    pub list: Arc<ChoiceList>,
    kept: Vec<usize>,
    by_identity: HashMap<String, usize>,
    thin: HashSet<(i64, String)>, // the (keymode, skill) slots filled up with learning choices: `Tags::learning` of all their rows
}

/// The worker's ranking inputs besides the pool (Worker._pool).
pub struct Ranking<'a> {
    pub keys: Vec<i64>,
    pub nps: bool, // NPS (density floor, shuffle) or Skills
    pub web: f64,
    pub convert: f64, // odds of converts against your own maps, as web
    pub taste: &'a HashMap<(i64, String), f64>, // {(keys, group): preference}
    pub eventful: &'a dyn Fn(&str) -> bool,     // identities with session events: the others are fresh
    pub freshness: &'a dyn Fn(&str, f64, f64) -> f64, // (identity, length, pool) → Session.freshness
    pub recent_families: Vec<String>,
    pub recent_profiles: Vec<Vec<f64>>,
    pub current_keys: Option<i64>,
    pub waits: &'a dyn Fn(&Choice) -> bool, // offered or played a moment ago: back only once nothing else is left
    pub shuffle: Option<(String, f64)>, // pp_playlist.preview's (str(seed), focus)
    pub skills: bool, // the list is dealt by skill slots (`Shares::skills`): the pattern groups are left as they fall
}

impl Prepared {
    pub fn len(&self) -> usize {
        self.kept.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kept.is_empty()
    }

    /// The prepared choice of a chart revision (recommend.event_key), as an entry index.
    pub fn find(&self, identity: &str) -> Option<usize> {
        self.by_identity.get(identity).copied()
    }

    /// Each chart family's best rate among the tab's keymodes and streams; equally good maps then reproduce library proportions.
    pub fn family_best(&self, keys: &[i64], web: f64, convert: f64) -> Vec<usize> {
        let items = &self.list.items;
        let mut best: Vec<usize> = Vec::new();
        let mut family_at: HashMap<(i64, &str), usize> = HashMap::default();
        for &i in &self.kept {
            let c = &items[i];
            if !keys.contains(&c.keys) || (web <= 0. && matches!(c.stream(), WEB | WEB_CONVERT)) || (convert <= 0. && matches!(c.stream(), CONVERT | WEB_CONVERT)) {
                continue;
            }
            match family_at.get(&(c.keys, c.family.as_str())) {
                Some(&at) => {
                    if c.value > items[best[at]].value {
                        best[at] = i;
                    }
                }
                None => {
                    family_at.insert((c.keys, c.family.as_str()), best.len());
                    best.push(i);
                }
            }
        }
        best
    }

    /// Each keymode's NPS over `rows` (entry indices), sorted: what the NPS floor reads. Invert maps are apart (`nps_slots`).
    pub fn nps_pool(&self, rows: &[usize]) -> std::collections::BTreeMap<i64, Vec<f64>> {
        self.nps_slots(rows).into_iter().filter(|(k, _)| !k.1).map(|(k, ns)| (k.0, ns)).collect()
    }

    /// `nps_pool` per (keymode, under Invert): maps that fit under Invert are easy ones with few notes a second, so against
    /// the keymode's floor nearly all of them left (7K: 1,582 → 104, 10K: 12 → 0; user 2026-11-21: "floor among Invert
    /// maps only"). They are measured against each other; the others' floor is read without them.
    pub fn nps_slots(&self, rows: &[usize]) -> std::collections::BTreeMap<(i64, bool), Vec<f64>> {
        let mut pool: std::collections::BTreeMap<(i64, bool), Vec<f64>> = Default::default();
        for c in rows.iter().map(|&i| &self.list.items[i]).filter(|c| c.nps > 0.) {
            pool.entry((c.keys, c.inverted())).or_default().push(c.nps);
        }
        pool.values_mut().for_each(|ns| ns.sort_by(f64::total_cmp));
        pool
    }

    /// The rows a list may take, best first: each chart family's best choice above the NPS floor, weighed by value, freshness
    /// and taste, in the session's shuffle → ([(entry index, value, weight, category_share)], freshness relaxed). Which row
    /// takes which place is `deal`'s.
    pub fn rank(&self, r: Ranking<'_>) -> (Vec<(usize, f64, f64, f64)>, bool) {
        let items = &self.list.items;
        let mut rows = self.family_best(&r.keys, r.web, r.convert);
        // The floor follows each keymode's eligible pool and the Variety ↔ NPS slider: far below it is not NPS practice
        // (user 2026-10-22: a 6 NPS map "should never appear"; 2026-11-13: "minimum nps should move with the nps/variety slider").
        let mut floor: HashMap<(i64, bool), f64> = HashMap::default();
        // One floor for both tabs' slider (user 2026-11-14: the custom playlist's shows its map count too); off has none.
        if r.nps || r.shuffle.is_some() {
            let focus = r.shuffle.as_ref().map_or(0.5, |s| s.1);
            for (k, ns) in self.nps_slots(&rows) {
                floor.insert(k, nps_floor(&ns, focus));
            }
            rows.retain(|&i| {
                let c = &items[i];
                !(0. < c.nps && c.nps < NPS_CUT * floor.get(&(c.keys, c.inverted())).copied().unwrap_or(0.))
            });
        }
        if rows.is_empty() {
            return (Vec::new(), false);
        }
        let mut counts: HashMap<(i64, &str), usize> = HashMap::default();
        rows.iter().for_each(|&i| *counts.entry((items[i].keys, items[i].group.as_str())).or_insert(0) += 1);
        let top = rows.iter().map(|&i| items[i].value).fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = rows.iter().map(|&i| (items[i].value - top).exp()).collect();
        let pool = w.iter().sum::<f64>().powi(2) / w.iter().map(|x| x * x).sum::<f64>();
        let mut fresh = Vec::with_capacity(rows.len());
        for &i in &rows {
            let identity = items[i].identity();
            fresh.push(if (r.eventful)(&identity) { (r.freshness)(&identity, items[i].length, pool) } else { 1. });
        }
        let relax = fresh.iter().sum::<f64>() < 3.;
        let several = r.keys.len() > 1 && r.current_keys.is_some_and(|k| r.keys.contains(&k));
        let mut values = Vec::with_capacity(rows.len());
        for (&i, fr) in rows.iter().zip(&fresh) {
            let c = &items[i];
            let mut value = c.value + 0.65 * fr.powf(if relax { 0.3 } else { 1. }).max(0.001).ln();
            let f = floor.get(&(c.keys, c.inverted())).copied().unwrap_or(0.);
            if f != 0. && c.nps > 0. {
                value -= 24. * (f / c.nps).ln().max(0.);
            }
            if r.recent_families.contains(&c.family) {
                value -= 1.2;
            }
            if c.stream() == WEB {
                // Mild popularity prior for website charts only: ×0.7 near 10 plays → ×1 from 10k.
                value += (0.6 + 0.1 * 4f64.min((1. + c.playcount).log10())).ln();
            }
            let similar = r.recent_profiles.iter().filter(|a| a.len() == c.profile.len())
                .map(|a| a.iter().zip(&c.profile).map(|(x, y)| x * y).sum::<f64>()).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))));
            if let Some(s) = similar {
                value -= 0.18 * s;
            }
            if several && Some(c.keys) != r.current_keys {
                value -= 0.12;
            }
            values.push(value);
        }
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut ranked: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(rows.len());
        for (&i, &value) in rows.iter().zip(&values) {
            let c = &items[i];
            let preference = r.taste.get(&(c.keys, c.group.clone())).map_or(1., |v| (0.65 * v).exp());
            ranked.push((i, value, (value - maximum).max(-20.).exp() * preference, counts[&(c.keys, c.group.as_str())] as f64 / rows.len() as f64));
        }
        ranked.sort_by(|a, b| b.2.total_cmp(&a.2));
        let mut available: Vec<(usize, f64, f64, f64)> = ranked.iter().filter(|x| !(r.waits)(&items[x.0])).copied().collect();
        if available.is_empty() {
            available = ranked;
        }
        if let Some((seed, focus)) = &r.shuffle {
            // Each pattern group's mass is divided out as focus falls: at 0 every group has an equal part of the sampled list,
            // at 1 the weights are untouched (squared: the slider's middle keeps most of the library's mix). Library proportions
            // alone made low variety "doar chordstreams si brackets" (a player, 2026-10-31): those own the top of NPS.
            let mut mass: HashMap<(i64, &str), f64> = HashMap::default();
            available.iter().for_each(|x| *mass.entry((items[x.0].keys, items[x.0].group.as_str())).or_insert(0.) += x.2);
            let evening = if r.skills { 0. } else { (1. - focus).powi(2) };
            let weights: Vec<f64> = available.iter().map(|x| x.2 / mass[&(items[x.0].keys, items[x.0].group.as_str())].max(1e-300).powf(evening)).collect();
            let ids: Vec<(String, f64)> = available.iter().zip(weights).map(|(x, w)| (if items[x.0].md5.is_empty() { items[x.0].sha.clone() } else { items[x.0].md5.clone() }, w)).collect();
            available = preview_order(&ids, seed, *focus).into_iter().map(|i| available[i]).collect();
        }
        (available, relax)
    }

    /// What `deal` reads of an entry: its skill slot is its leading picked skill when the list has skill slots.
    pub fn tags(&self, i: usize, skills: bool) -> Tags {
        let c = &self.list.items[i];
        let skill = if skills { c.matched.first().cloned().unwrap_or_default() } else { String::new() };
        Tags { keys: c.keys, learning: skills && self.thin.contains(&(c.keys, skill.clone())), skill, stream: c.stream() }
    }
}

/// np.quantile (linear)
/// Below `NPS_CUT` × the floor a map leaves the NPS list; between that and the floor it is pushed down.
pub const NPS_CUT: f64 = 0.75;

/// A keymode's NPS floor at slider `focus` over its sorted pool: the pool's 50th percentile at the variety end, 90th at the
/// NPS end, 70th (the fixed floor before 2026-11-13) in the middle.
pub fn nps_floor(sorted: &[f64], focus: f64) -> f64 {
    0.85 * quantile(&mut sorted.to_vec(), 0.5 + 0.4 * focus.clamp(0., 1.))
}

/// Maps a keymode keeps in the NPS list at slider `focus`.
pub fn nps_eligible(sorted: &[f64], focus: f64) -> usize {
    let cut = NPS_CUT * nps_floor(sorted, focus);
    sorted.len() - sorted.partition_point(|&n| n < cut)
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = q * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    let t = pos - lo as f64;
    let d = v[hi] - v[lo];
    if t >= 0.5 { v[hi] - d * (1. - t) } else { v[lo] + d * t }
}

/// The odds of each stream against your own maps. Weighted, the sliders set a chance per map: at the middle every map
/// is equally likely, so twice as many converts the list can offer now means twice as many convert picks (user
/// 2026-11-01). Off, they are plain shares of the picks.
pub fn stream_odds(sliders: [f64; 2], weighted: [bool; 2], streams: &[u8]) -> [f64; 4] {
    let count = |k: u8| streams.iter().filter(|&&s| s == k).count() as f64;
    let per_map = |k: usize, maps: f64| if weighted[k] { sliders[k] * maps / count(REAL).max(1.) } else { sliders[k] };
    // a convert of a website chart counts as the website slider's part of a map: the converts' odds split by that
    let (own, web) = (count(CONVERT), sliders[0] * count(WEB_CONVERT));
    let converts = per_map(1, own + web);
    let of_web = if own + web > 0. { web / (own + web) } else { 0. };
    [1., per_map(0, count(WEB)), converts * (1. - of_web), converts * of_web]
}

/// Each keymode's share of a list over several keymodes: its slider, × its maps and converts here when `by_count` (user
/// 2026-11-09: even turns by default, map counts as a switch). Shares sum to 1 over `modes`.
pub fn key_shares(modes: &[i64], weights: &std::collections::BTreeMap<i64, f64>, counts: &std::collections::BTreeMap<i64, usize>, by_count: bool) -> std::collections::BTreeMap<i64, f64> {
    mix_shares(&modes.iter().map(|k| (*k, weights.get(k).copied().unwrap_or(1.), counts.get(k).copied().unwrap_or(0))).collect::<Vec<_>>(), by_count)
}

/// One mixing rule for keymodes and skills (user 2026-11-14): each entry's share of the list is its slider × (its eligible
/// maps when shares follow map counts, else 1), over all entries. A slider at 0 (Off) gets none; all Off: even.
pub fn mix_shares<K: Ord + Clone>(entries: &[(K, f64, usize)], by_count: bool) -> std::collections::BTreeMap<K, f64> {
    let weight = |e: &(K, f64, usize)| e.1.max(0.) * if by_count { e.2.max(1) as f64 } else { 1. };
    let total: f64 = entries.iter().map(weight).sum();
    entries.iter().map(|e| (e.0.clone(), if total > 0. { weight(e) / total } else { 1. / entries.len() as f64 })).collect()
}

/// A PP map dealt into another tab's list: its own stream, one turn in `Shares::pp_every`.
pub const PP: u8 = 4;

/// What a turn is dealt by: a row's keymode, its skill slot (the leading picked skill, "" for none) and its stream.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tags {
    pub keys: i64,
    pub skill: String,
    pub stream: u8,
    pub learning: bool, // of a slot filled up with learning offers (`Choice::learning`): dealt by the skill's slider, not by its map count
}

/// The sliders a list is dealt by.
#[derive(Clone, Debug, Default)]
pub struct Shares {
    pub keys: std::collections::BTreeMap<i64, f64>, // `key_shares`; a keymode without one: even
    pub skills: Option<std::collections::HashMap<String, f64>>, // `mix_shares` over the picked skills and "" (other maps); None: no skill slots
    pub sliders: [f64; 2], // website, converts (`stream_odds`)
    pub weighted: [bool; 2],
    pub pp_every: usize, // one PP turn in this many; 0: none
}

/// A keymode, or a skill or stream inside one.
#[derive(Clone, Copy)]
struct Slot {
    share: f64,
    owed: f64,   // turns
    on: bool,    // its slider isn't at Off
    last: usize, // the turn it last took
}

impl Slot {
    /// One turn went by, `mine`: this slot's. A slot never owes or is owed more than one turn: a long run of one kind (a
    /// slot that had no rows) must not be paid back in a run of the other.
    fn step(&mut self, mine: bool, turn: usize) {
        self.owed = (self.owed + self.share - f64::from(u8::from(mine))).clamp(-1., 1.);
        if mine {
            self.last = turn;
        }
    }

    /// (turns owed once this one is added, turns since its last): the most owed goes first, the longest waiting on a tie.
    fn due(&self, turn: usize) -> (i64, usize) {
        (((self.owed + self.share) * 1e9).round() as i64, turn - self.last)
    }
}

struct Mode {
    turn: Slot,
    skills: Vec<(String, Slot)>,
    streams: [Slot; 4],
}

/// The playlist's one allocator (user 2026-11-18: "rewrite the playlist shuffling system … into a more unified system"; the
/// stacked stages each dealt their own share and each fix of one broke another's). `pool`: the rows a list may still take, best
/// first. `ledger`: the tab's recent offers, oldest first. `queue`: the shown list, Some for a row that stays, None for a
/// place to fill. → the pool row for each place to fill, in order (fewer once the pool runs out).
///
/// A slot's share is the product of the sliders that apply: its keymode's share, × its skill's share there (a picked skill
/// goes where its maps are, user 2026-11-15), × its stream's share there (`stream_odds` over that keymode's rows). Every offer
/// and every row adds each slot's share to what it is owed and takes one turn from its own keymode, skill and stream; a
/// place goes to the keymode owed most, and there to the skill × stream owed most that still has a row. A slot without rows
/// gets no share (it goes to its keymode's other skills; a keymode without rows gives its turns to the others); a slot at Off
/// only fills places nothing else can. A PP turn sits on top: one in `pp_every` places since the last, and it moves no other slot.
pub fn deal(pool: &[Tags], shares: &Shares, ledger: &[Tags], queue: &[Option<Tags>]) -> Vec<usize> {
    let all = || pool.iter().chain(queue.iter().flatten()).filter(|t| t.stream != PP);
    let mut keymodes: Vec<i64> = all().map(|t| t.keys).collect();
    keymodes.sort_unstable();
    keymodes.dedup();
    let key_weight = |k: &i64| shares.keys.get(k).copied().unwrap_or(1. / keymodes.len().max(1) as f64).max(0.);
    let key_total: f64 = keymodes.iter().map(key_weight).sum();
    let mut maps: HashMap<&str, f64> = HashMap::default();
    all().filter(|t| !t.learning).for_each(|t| *maps.entry(t.skill.as_str()).or_insert(0.) += 1.);
    let slots = |raw: Vec<f64>| -> Vec<Slot> {
        let total: f64 = raw.iter().sum();
        raw.iter().map(|r| Slot { share: if total > 0. { r / total } else { 1. / raw.len() as f64 }, owed: 0., on: true, last: 0 }).collect()
    };
    let mut modes: Vec<Mode> = keymodes.iter().map(|k| {
        let key_share = if key_total > 0. { key_weight(k) / key_total } else { 1. / keymodes.len() as f64 };
        let mut names: Vec<&str> = Vec::new();
        let (mut counts, mut streams): (Vec<f64>, Vec<u8>) = Default::default();
        let mut learning: Vec<&str> = Vec::new(); // the skills with nothing at the target here: their nearest charts
        for t in all().filter(|t| t.keys == *k) {
            streams.push(t.stream);
            if t.learning && !learning.contains(&t.skill.as_str()) {
                learning.push(&t.skill);
            }
            match names.iter().position(|n| *n == t.skill) {
                Some(i) => counts[i] += 1.,
                None => {
                    names.push(&t.skill);
                    counts.push(1.);
                }
            }
        }
        // a picked skill: its share × this keymode's part of its maps ÷ this keymode's part of the list; other maps: the rest.
        // A skill filled up with learning offers here has too few maps to go by: it comes as often as its slider says (user
        // 2026-11-22: "The skill's normal turns"; by its 20 charts among 4K's thousands 10K Invert got no turn in 300).
        let slider = |n: &str| shares.skills.as_ref().map_or(1., |s| s.get(n).copied().unwrap_or(0.));
        // The Inverse mod skill too, in every keymode (user 2026-11-22: "i want the share per keymode to be equal to that
        // keymode's share … not whichever keymode has more available maps"): any chart can be played under Invert, so its
        // map counts say where charts are, not where the skill is.
        let even = |n: &str| learning.contains(&n) || n == crate::skills::INVERT_MOD;
        let raw: Vec<f64> = names.iter().zip(&counts).map(|(n, c)| if n.is_empty() || key_share <= 0. || even(n) { slider(n) } else { slider(n) * c / maps[n] / key_share }).collect();
        let picked: f64 = names.iter().zip(&raw).filter(|(n, _)| !n.is_empty()).map(|(_, r)| r).sum();
        let raw = names.iter().zip(&raw).map(|(n, r)| if n.is_empty() && shares.skills.is_some() { (1. - picked).max(0.) } else { *r }).collect();
        let mut skills = slots(raw);
        skills.iter_mut().zip(&names).for_each(|(s, n)| s.on = slider(n) > 0.);
        let odds = stream_odds(shares.sliders, shares.weighted, &streams);
        let mut by_stream = slots((0..4).map(|s| if streams.contains(&(s as u8)) { odds[s] } else { 0. }).collect());
        by_stream.iter_mut().zip(odds).for_each(|(s, o)| s.on = o > 0.);
        Mode { turn: Slot { share: key_share, owed: 0., on: key_weight(k) > 0., last: 0 }, skills: names.iter().map(|n| n.to_string()).zip(skills).collect(), streams: std::array::from_fn(|s| by_stream[s]) }
    }).collect();
    let pp_share = if shares.pp_every > 0 { 1. / shares.pp_every as f64 } else { 0. };
    let (mut pp_owed, mut turn) = (0f64, 0);
    // one turn taken by `t`
    let advance = |modes: &mut Vec<Mode>, pp_owed: &mut f64, turn: &mut usize, t: &Tags| {
        if t.stream == PP {
            *pp_owed = 0.; // counted on from the last PP map
            return;
        }
        *pp_owed += pp_share;
        *turn += 1;
        let Some(at) = keymodes.iter().position(|k| *k == t.keys) else { return }; // a keymode that left the tab moves nothing
        modes.iter_mut().enumerate().for_each(|(i, m)| m.turn.step(i == at, *turn));
        modes[at].skills.iter_mut().for_each(|s| s.1.step(s.0 == t.skill, *turn));
        modes[at].streams.iter_mut().enumerate().for_each(|(i, s)| s.step(i == t.stream as usize, *turn));
    };
    ledger.iter().for_each(|t| advance(&mut modes, &mut pp_owed, &mut turn, t));
    // the rows left of every slot, best first
    let mut left: std::collections::BTreeMap<(usize, usize, usize), std::collections::VecDeque<usize>> = Default::default();
    let mut pp: std::collections::VecDeque<usize> = Default::default();
    for (i, t) in pool.iter().enumerate() {
        if t.stream == PP {
            pp.push_back(i);
        } else if let Some(k) = keymodes.iter().position(|k| *k == t.keys) {
            let skill = modes[k].skills.iter().position(|s| s.0 == t.skill).expect("counted above");
            left.entry((k, skill, t.stream as usize)).or_default().push_back(i);
        }
    }
    let mut out = Vec::new();
    for place in queue {
        if let Some(t) = place {
            advance(&mut modes, &mut pp_owed, &mut turn, t);
            continue;
        }
        let row = if pp_share > 0. && !pp.is_empty() && pp_owed + pp_share >= 1. - 1e-9 { pp.pop_front() } else {
            // of the slots that still have a row: none of its sliders at Off, then its keymode's turn, then its skill's and stream's
            let best = left.iter().filter(|(_, rows)| !rows.is_empty()).map(|(&(k, skill, stream), _)| {
                let (m, s, st) = (&modes[k], &modes[k].skills[skill].1, &modes[k].streams[stream]);
                let (key, a, b) = (m.turn.due(turn), s.due(turn), st.due(turn));
                ((m.turn.on && s.on && st.on, m.turn.on, key, (a.0 + b.0, a.1 + b.1)), (k, skill, stream))
            }).max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            best.and_then(|(_, slot)| left.get_mut(&slot)?.pop_front()).or_else(|| pp.pop_front())
        };
        let Some(row) = row else { break };
        advance(&mut modes, &mut pp_owed, &mut turn, &pool[row]);
        out.push(row);
    }
    out
}

/// A list's order of [(identity, weight)]: a stable weighted shuffle keyed by `seed` and each identity (focus 0 samples by
/// weight, 1 keeps the highest weights in order).
pub fn preview_order(rows: &[(String, f64)], seed: &str, focus: f64) -> Vec<usize> {
    let priority: Vec<f64> = rows.iter().map(|(id, w)| {
        let h = blake2b8(format!("{seed}:{id}").as_bytes());
        let u = (u64::from_be_bytes(h) as f64 + 1.) / (2f64.powi(64) + 1.);
        (1. - focus) * (-u.ln()).max(1e-300).ln() - w.max(1e-12).ln()
    }).collect();
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| priority[a].total_cmp(&priority[b]));
    order
}

/// BLAKE2b (RFC 7693) with an 8-byte digest, unkeyed: hashlib.blake2b(data, digest_size=8).digest().
fn blake2b8(data: &[u8]) -> [u8; 8] {
    const IV: [u64; 8] = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                          0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179];
    const SIGMA: [[usize; 16]; 10] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4], [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13], [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11], [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5], [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    ];
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 8;
    let compress = |h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool| {
        let m: Vec<u64> = block.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= t as u64;
        v[13] ^= (t >> 64) as u64;
        if last {
            v[14] = !v[14];
        }
        let g = |v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64| {
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
            v[d] = (v[d] ^ v[a]).rotate_right(32);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(24);
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
            v[d] = (v[d] ^ v[a]).rotate_right(16);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(63);
        };
        for round in 0..12 {
            let s = &SIGMA[round % 10];
            g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for i in 0..8 {
            h[i] ^= v[i] ^ v[i + 8];
        }
    };
    let mut t: u128 = 0;
    let mut chunks = data.chunks(128).peekable();
    if data.is_empty() {
        compress(&mut h, &[0; 128], 0, true);
    }
    while let Some(c) = chunks.next() {
        let mut block = [0u8; 128];
        block[..c.len()].copy_from_slice(c);
        t += c.len() as u128;
        compress(&mut h, &block, t, chunks.peek().is_none());
    }
    h[0].to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake2b_matches_hashlib() {
        // hashlib.blake2b(data, digest_size=8).hexdigest()
        let hex = |b: [u8; 8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert_eq!(hex(blake2b8(b"9:abc")), "813f29f73b38f7d5");
        assert_eq!(hex(blake2b8(b"")), "e4a6a0577479b2b4");
        assert_eq!(hex(blake2b8(&[b'x'; 300])), "12b539c1cad94615");
        assert_eq!(hex(blake2b8(b"123456789:0123456789abcdef0123456789abcdef")), "a9633b2b960a21fb");
    }
}

/// The allocator on synthetic rows: every share it deals, and one test per playlist bug reported.
#[cfg(test)]
mod deal_tests {
    use super::*;

    fn row(keys: i64, skill: &str, stream: u8) -> Tags {
        Tags { keys, skill: skill.into(), stream, learning: false }
    }

    fn rows(n: usize, keys: i64, skill: &str, stream: u8) -> Vec<Tags> {
        vec![row(keys, skill, stream); n]
    }

    /// Next pressed `n` times on a pool that never runs out: each offer is the first place of a list dealt after the last 400.
    fn presses(pool: &[Tags], shares: &Shares, n: usize) -> Vec<Tags> {
        let mut offers: Vec<Tags> = Vec::new();
        for _ in 0..n {
            let first = deal(pool, shares, &offers[offers.len().saturating_sub(400)..], &[None])[0];
            offers.push(pool[first].clone());
        }
        offers
    }

    fn count(offers: &[Tags], of: impl Fn(&Tags) -> bool) -> usize {
        offers.iter().filter(|t| of(t)).count()
    }

    fn skills(shares: &[(&str, f64)]) -> Option<std::collections::HashMap<String, f64>> {
        Some(shares.iter().map(|(k, v)| (k.to_string(), *v)).collect())
    }

    /// User 2026-11-01: "if there's 5000 10k beatmaps and 10000 7k beatmaps then if you put it in the middle ... 7k converts would be twice as likely".
    #[test]
    fn the_middle_of_a_weighted_slider_is_the_same_chance_per_map() {
        let pool = [rows(5000, 10, "", REAL), rows(10000, 10, "", CONVERT)].concat();
        let streams: Vec<u8> = pool.iter().map(|t| t.stream).collect();
        assert_eq!(stream_odds([0., 1.], [true, true], &streams), [1., 0., 2., 0.]);
        assert_eq!(stream_odds([0., 1.], [false, false], &streams), [1., 0., 1., 0.]);
        let got = presses(&pool, &Shares { sliders: [0., 1.], weighted: [true, true], ..Shares::default() }, 300);
        assert!(count(&got, |t| t.stream == CONVERT).abs_diff(200) <= 2, "{}", count(&got, |t| t.stream == CONVERT));
        // User 2026-11-18: converts of website charts answer to the website slider too (at web 0.3 half the convert offers
        // were website sourced, as half the convert pool is): a third of a map each at 1/3
        let mixed = [rows(100, 10, "", REAL), rows(100, 10, "", CONVERT), rows(300, 10, "", WEB_CONVERT)].concat();
        let streams: Vec<u8> = mixed.iter().map(|t| t.stream).collect();
        let odds = stream_odds([1. / 3., 1.], [true, true], &streams);
        assert!((odds[2] - 1.).abs() < 1e-9 && (odds[3] - 1.).abs() < 1e-9, "{odds:?}");
        let got = presses(&mixed, &Shares { sliders: [1. / 3., 1.], weighted: [true, true], ..Shares::default() }, 300);
        let (own, web) = (count(&got, |t| t.stream == CONVERT), count(&got, |t| t.stream == WEB_CONVERT));
        assert!(own.abs_diff(100) <= 2 && own.abs_diff(web) <= 2, "{own} {web}"); // equal odds: a third each
        let off = presses(&mixed, &Shares { sliders: [0., 1.], weighted: [true, true], ..Shares::default() }, 100);
        assert_eq!(count(&off, |t| t.stream == WEB_CONVERT), 0, "website off: none of them");
    }

    #[test]
    fn each_stream_keeps_its_share_across_picks() {
        let pool = [rows(100, 7, "", REAL), rows(100, 7, "", CONVERT)].concat();
        for share in [0.15, 0.5, 0.8] {
            let got = presses(&pool, &Shares { sliders: [0., share / (1. - share)], ..Shares::default() }, 40);
            assert!((count(&got, |t| t.stream == CONVERT) as f64 - 40. * share).abs() <= 1., "{share}");
        }
        let three = [rows(50, 7, "", REAL), rows(50, 7, "", WEB), rows(50, 7, "", CONVERT)].concat();
        let got = presses(&three, &Shares { sliders: [1., 2.], ..Shares::default() }, 40);
        assert!(count(&got, |t| t.stream == WEB).abs_diff(10) <= 1 && count(&got, |t| t.stream == CONVERT).abs_diff(20) <= 1);
        // one list dealt at once keeps the shares too, and goes on with what is left
        let pool = [rows(10, 7, "", REAL), rows(30, 7, "", WEB)].concat();
        let list = |web: f64| deal(&pool, &Shares { sliders: [web, 0.], ..Shares::default() }, &[], &vec![None; 40]).into_iter().map(|i| pool[i].stream).collect::<Vec<_>>();
        assert_eq!(list(1.).len(), 40);
        assert_eq!(list(1.)[..20].iter().filter(|s| **s == WEB).count(), 10);
        assert_eq!(list(4.)[..20].iter().filter(|s| **s == WEB).count(), 16);
        assert!(list(1.)[20..].iter().all(|s| *s == WEB), "locals ran out");
        assert_eq!(list(1. / 19.)[..20].iter().position(|s| *s == WEB), Some(9), "5 %: one in 20, the first halfway");
    }

    #[test]
    fn keymodes_take_turns_by_their_sliders_after_recent_offers() {
        let pool = [rows(6, 10, "", REAL), rows(3, 7, "", REAL)].concat();
        let counts = [(7, 3), (10, 6)].into_iter().collect();
        let order = |w: &[(i64, f64)], by_count: bool, recent: &[Tags]| -> Vec<i64> {
            let shares = Shares { keys: key_shares(&[7, 10], &w.iter().copied().collect(), &counts, by_count), ..Shares::default() };
            deal(&pool, &shares, recent, &vec![None; 9]).into_iter().map(|i| pool[i].keys).collect()
        };
        assert_eq!(&order(&[], true, &[])[..6], [10, 7, 10, 10, 7, 10]); // 6 of 9 maps are 10K: two thirds of the picks
        assert_eq!(&order(&[(7, 2.)], true, &[])[..4], [7, 10, 7, 10]); // 7K's slider at 2× evens 3 maps against 6
        assert_eq!(order(&[], true, &rows(2, 10, "", REAL))[0], 7); // 10K was just offered twice: 7K is owed
        assert_eq!(&order(&[], false, &[])[..4], [7, 10, 7, 10]); // the toggle off: even turns
        assert_eq!(order(&[], true, &[]).len(), 9, "a keymode that ran out leaves its turns to the other");
    }

    /// User 2026-11-13: in a 4K–10K list "converts really far in the list … never end up in the front": one convert share
    /// was taken over every keymode's maps (0.3 against all 7K maps) instead of 10K's own.
    #[test]
    fn each_keymode_mixes_in_its_converts_by_its_own_maps() {
        let pool = [rows(30, 7, "", REAL), rows(12, 10, "", REAL), rows(24, 10, "", CONVERT)].concat();
        let list = deal(&pool, &Shares { sliders: [0., 1.], weighted: [false, true], ..Shares::default() }, &[], &vec![None; 20]);
        let wide: Vec<u8> = list.iter().filter(|&&i| pool[i].keys == 10).map(|&i| pool[i].stream).collect();
        assert_eq!((wide.len(), wide.iter().filter(|s| **s == CONVERT).count()), (10, 7), "twice as many converts as 10K maps: two in three 10K rows");
        assert!(wide[..3].contains(&CONVERT));
    }

    /// User 2026-11-18: "10 of 11 4K offers Vibro at a 7 % share" (22 of 37 in four days): the skill turns ran over every
    /// stream, the website rows that took the other maps' turns were then held back, and the installed rows left alternated.
    #[test]
    fn a_small_skill_gets_its_share_of_its_keymode_whatever_the_streams() {
        let pool = [rows(40, 4, "vibro", REAL), rows(160, 4, "", REAL), rows(400, 4, "", WEB)].concat();
        let shares = Shares { skills: skills(&[("vibro", 0.07), ("", 0.93)]), sliders: [0.2, 0.], ..Shares::default() };
        let got = presses(&pool, &shares, 300);
        let (vibro, web) = (count(&got, |t| t.skill == "vibro"), count(&got, |t| t.stream == WEB));
        assert!(vibro.abs_diff(21) <= 3 && web.abs_diff(50) <= 3, "{vibro} vibro, {web} website of 300");
        // the first list too: no run of one skill at its top (user 2026-11-14: Vibro at 0.15 of 1.15 filled the list)
        let list = deal(&pool, &shares, &[], &vec![None; 20]);
        assert!(list.iter().filter(|&&i| pool[i].skill == "vibro").count() <= 2);
    }

    /// User 2026-11-15: Vibro landed on 10K's few chance-vibro maps as often as on 4K's packs.
    #[test]
    fn a_skill_goes_where_its_maps_are() {
        let pool = [rows(40, 4, "vibro", REAL), rows(160, 4, "", REAL), rows(2, 10, "vibro", REAL), rows(198, 10, "", REAL)].concat();
        let got = presses(&pool, &Shares { skills: skills(&[("vibro", 0.13), ("", 0.87)]), ..Shares::default() }, 120);
        let (vibro, wide) = (count(&got, |t| t.skill == "vibro"), count(&got, |t| t.skill == "vibro" && t.keys == 10));
        assert!((13..=18).contains(&vibro) && wide <= 2, "{vibro} of 120 vibro, {wide} of them 10K");
        assert!(count(&got, |t| t.keys == 10).abs_diff(60) <= 1, "the keymodes still take even turns");
    }

    /// User 2026-11-22: 10K Invert reaches the target at no rate; its nearest charts come "the skill's normal turns".
    #[test]
    fn a_skill_with_only_learning_offers_in_a_keymode_comes_as_often_as_its_slider_says() {
        let pool = |learning: bool| [rows(200, 4, "ln", REAL), rows(200, 4, "", REAL), rows(200, 10, "", REAL), vec![Tags { learning, ..row(10, "ln", REAL) }; 20]].concat();
        let shares = Shares { skills: skills(&[("ln", 0.15), ("", 0.85)]), ..Shares::default() };
        let turns = |got: &[Tags], keys: i64| count(got, |t| t.keys == keys && t.skill == "ln");
        let got = presses(&pool(true), &shares, 120);
        assert!(turns(&got, 10).abs_diff(9) <= 2 && turns(&got, 4).abs_diff(16) <= 2, "10K {} (0.15 of its 60), 4K {} (where the skill's maps are)", turns(&got, 10), turns(&got, 4));
        // 20 charts at the target go by their count, as every skill does
        let by_count = presses(&pool(false), &shares, 120);
        assert!(turns(&by_count, 10) <= 3 && turns(&by_count, 4).abs_diff(16) <= 2, "{} {}", turns(&by_count, 10), turns(&by_count, 4));
    }

    #[test]
    fn the_inverse_mod_skill_takes_its_slider_of_every_keymodes_turns() {
        // user 2026-11-22: 7K has thousands of charts that fit under Invert, 10K a few: 10K still gets its keymode's share
        let pool = [rows(400, 7, "invert_mod", REAL), rows(200, 7, "", REAL), rows(30, 10, "invert_mod", REAL), rows(200, 10, "", REAL)].concat();
        let shares = Shares { keys: [(7, 0.4), (10, 0.6)].into(), skills: skills(&[("invert_mod", 0.5), ("", 0.5)]), ..Shares::default() };
        let got = presses(&pool, &shares, 100);
        let turns = |keys: i64| count(&got, |t| t.keys == keys && t.skill == "invert_mod");
        assert!(turns(7).abs_diff(20) <= 2 && turns(10).abs_diff(30) <= 2, "7K {} of its 40, 10K {} of its 60", turns(7), turns(10));
    }

    /// User 2026-11-18 (fixed in 1.15.0, kept here): 4K Vibro picked alone filled the list with other keymodes' maps without
    /// it; wide charts never fill a Vibro turn, they come only as "other maps".
    #[test]
    fn a_skill_picked_for_one_keymode_leaves_the_others_to_other_maps() {
        let pool = [rows(40, 4, "vibro", REAL), rows(160, 4, "", REAL), rows(200, 10, "", REAL)].concat();
        let only = presses(&pool, &Shares { skills: skills(&[("vibro", 1.), ("", 0.)]), ..Shares::default() }, 60);
        assert_eq!(count(&only, |t| t.keys == 4 && t.skill == "vibro"), 60, "other maps at Off: no 10K row while a Vibro map is left");
        let with_other = presses(&pool, &Shares { skills: skills(&[("vibro", 0.5), ("", 0.5)]), ..Shares::default() }, 60);
        assert_eq!(count(&with_other, |t| t.skill == "vibro"), 30);
        assert_eq!(count(&with_other, |t| t.keys == 10 && t.skill.is_empty()), 30, "10K comes as other maps");
        // Off rows fill a list only once nothing else is left
        let few = [rows(2, 4, "vibro", REAL), rows(5, 4, "", REAL)].concat();
        let list: Vec<&str> = deal(&few, &Shares { skills: skills(&[("vibro", 1.), ("", 0.)]), ..Shares::default() }, &[], &vec![None; 4]).into_iter().map(|i| few[i].skill.as_str()).collect();
        assert_eq!(list, ["vibro", "vibro", "", ""]);
    }

    #[test]
    fn other_maps_take_their_slider_share_of_the_list() {
        let pool = [rows(50, 7, "chordjack", REAL), rows(50, 7, "", REAL)].concat();
        let sliders = crate::skills::sliders(&["chordjack".to_string()], &[("chordjack".to_string(), 2.)].into(), 0.5).unwrap();
        let entries: Vec<(String, f64, usize)> = sliders.into_iter().map(|(k, w)| (k, w, 50)).collect();
        assert_eq!(mix_shares(&entries, true), mix_shares(&entries, false), "equal counts: the same shares either way");
        let got = presses(&pool, &Shares { skills: Some(mix_shares(&entries, false).into_iter().collect()), ..Shares::default() }, 100);
        assert_eq!(count(&got, |t| t.skill == "chordjack"), 80);
        assert!(crate::skills::sliders(&[], &Default::default(), 0.).is_none(), "no settings ask for no shares");
        assert!(crate::skills::sliders(&["chordjack".to_string()], &Default::default(), 0.).is_none(), "one skill alone neither");
    }

    /// A keymode at 0.4 % of the list still comes, about once in 250 offers (5K–9K on the user's settings never did while
    /// the offers were counted over the last 40 only).
    #[test]
    fn a_small_share_still_gets_its_turn() {
        let pool = [rows(500, 7, "", REAL), rows(20, 5, "", REAL)].concat();
        let shares = Shares { keys: [(7, 0.996), (5, 0.004)].into_iter().collect(), ..Shares::default() };
        let got = count(&presses(&pool, &shares, 1000), |t| t.keys == 5);
        assert!((3..=5).contains(&got), "{got} of 1000");
    }

    /// A slot without rows gives its share to its keymode's other skills, a keymode without rows to the other keymodes.
    #[test]
    fn an_empty_slot_leaves_its_turns_to_its_keymode_first() {
        let pool = [rows(2, 4, "vibro", REAL), rows(50, 4, "", REAL), rows(50, 7, "", REAL)].concat();
        let shares = Shares { skills: skills(&[("vibro", 0.5), ("", 0.5)]), ..Shares::default() };
        let list: Vec<&Tags> = deal(&pool, &shares, &[], &vec![None; 20]).into_iter().map(|i| &pool[i]).collect();
        assert_eq!((list.iter().filter(|t| t.keys == 4).count(), list.iter().filter(|t| t.skill == "vibro").count()), (10, 2), "4K keeps its half after its Vibro maps ran out");
        let short = [rows(3, 4, "", REAL), rows(50, 7, "", REAL)].concat();
        let list = deal(&short, &Shares::default(), &[], &vec![None; 20]);
        assert_eq!((list.len(), list.iter().filter(|&&i| short[i].keys == 4).count()), (20, 3));
        assert_eq!(deal(&short[..3], &Shares::default(), &[], &vec![None; 20]).len(), 3, "a pool that ran out leaves the list short");
    }

    /// User 2026-11-15 (Aegleseeker): a due PP map stayed second after every play. PP turns come every `pp_every`-th place,
    /// counted on from the tab's offers, and take no keymode's or skill's turn.
    #[test]
    fn a_due_pp_map_comes_first_and_then_every_nth() {
        let pool = [rows(50, 7, "", REAL), rows(50, 4, "", REAL), rows(5, 7, "", PP)].concat();
        let shares = Shares { pp_every: 4, ..Shares::default() };
        let at = |ledger: &[Tags]| deal(&pool, &shares, ledger, &vec![None; 12]).into_iter().enumerate().filter(|(_, i)| pool[*i].stream == PP).map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(at(&[]), [3, 7, 11]);
        assert_eq!(at(&rows(3, 7, "", REAL)), [0, 4, 8], "three offers since the last PP map: it is due now");
        assert_eq!(at(&[rows(9, 7, "", REAL), rows(1, 7, "", PP), rows(1, 7, "", REAL)].concat()), [2, 6, 10]);
        let list = deal(&pool, &shares, &[], &vec![None; 12]);
        assert_eq!(list.iter().filter(|&&i| pool[i].keys == 4 && pool[i].stream != PP).count(), 5, "the 9 other places go on in even turns");
        assert!(deal(&pool, &Shares::default(), &[], &vec![None; 12]).iter().all(|&i| pool[i].stream != PP), "the mix off: PP maps only once nothing else is left");
    }

    /// User 2026-11-14: "why does the playlist update when i press next … the order of songs changes". The shown list is the
    /// allocator's own output: rows that stay are counted as dealt, a place that opened is filled where it is, and after a
    /// pick the list is the old one moved up plus one row.
    #[test]
    fn the_shown_list_stays_as_dealt() {
        let mut pool = [rows(40, 4, "vibro", REAL), rows(160, 4, "", REAL), rows(100, 4, "", WEB), rows(200, 7, "", REAL), rows(60, 7, "", CONVERT)].concat();
        let shares = Shares { skills: skills(&[("vibro", 0.2), ("", 0.8)]), sliders: [0.3, 0.5], ..Shares::default() };
        let ledger = rows(3, 4, "", REAL);
        let long: Vec<Tags> = deal(&pool, &shares, &ledger, &vec![None; 21]).into_iter().map(|i| pool[i].clone()).collect();
        let first = deal(&pool, &shares, &ledger, &vec![None; 20]);
        assert_eq!(first.iter().map(|&i| pool[i].clone()).collect::<Vec<_>>(), long[..20]);
        // Next took the first row: the rest stay, and the new last row is the one a longer deal had there
        let mut taken = first;
        taken.sort_unstable();
        taken.iter().rev().for_each(|&i| { pool.remove(i); }); // rows on the list are no longer in the pool
        let offered = [ledger, vec![long[0].clone()]].concat();
        let queue: Vec<Option<Tags>> = long[1..20].iter().cloned().map(Some).chain([None]).collect();
        let new = deal(&pool, &shares, &offered, &queue);
        assert_eq!(new.iter().map(|&i| pool[i].clone()).collect::<Vec<_>>(), long[20..]);
        // a row that left the middle is replaced there by a row of the slot owed most at that place: the same kind
        let mut holed = queue;
        (holed[5], holed[19]) = (None, Some(long[20].clone()));
        let fill = deal(&pool, &shares, &offered, &holed);
        assert_eq!(fill.iter().map(|&i| pool[i].clone()).collect::<Vec<_>>(), [long[6].clone()]);
    }
}

/// Family best, density floor, focus, popularity prior and sliders on synthetic choices.
#[cfg(test)]
mod list_tests {
    use super::*;

    fn choice(sha: &str, nps: f64, group: &str) -> Choice {
        Choice { sha: sha.into(), md5: sha.into(), keys: 7, rate: 1., value: 0., installed: true, convert: false, source_web: false, playcount: -1., family: sha.into(), description: "Mixed".into(),
                 group: group.into(), nps, length: 180., profile: vec![1.], acc_mid: 0.94, matched: Vec::new(), practice: String::new(), pred: Prediction::default(), generated_at: 0., learning: false }
    }

    fn prepared(rows: Vec<Choice>, keys: &[i64]) -> Prepared {
        let order: Vec<String> = rows.iter().map(|c| c.sha.clone()).collect();
        let mut memo = Choices::new(order, Forecast::default());
        for (slot, c) in memo.slots.iter_mut().zip(rows) {
            *slot = Some((Some(Arc::new(c)), None));
        }
        Arc::new(memo.snapshot()).prepare(keys, (crate::predict::MIN_RATE, crate::predict::MAX_RATE), &|_| true)
    }

    struct Ask {
        keys: Vec<i64>,
        nps: bool,
        web: f64,
        shuffle: Option<(String, f64)>,
    }

    impl Default for Ask {
        fn default() -> Ask {
            Ask { keys: vec![7], nps: true, web: 1., shuffle: None }
        }
    }

    /// (choice, weight), best first.
    fn ranked(p: &Prepared, a: Ask) -> Vec<(Arc<Choice>, f64)> {
        let taste = HashMap::default();
        let (rows, _) = p.rank(Ranking { keys: a.keys, nps: a.nps, web: a.web, convert: 0., taste: &taste, eventful: &|_| false, freshness: &|_, _, _| 1.,
                                         recent_families: Vec::new(), recent_profiles: Vec::new(), current_keys: None, waits: &|_| false, shuffle: a.shuffle, skills: !a.nps });
        rows.into_iter().map(|(i, _, weight, _)| (p.list.items[i].clone(), weight)).collect()
    }

    #[test]
    fn the_nps_floor_follows_the_slider() {
        let ns: Vec<f64> = (1..=101).map(f64::from).collect(); // NPS 1…101: the q-th percentile is 1 + 100 q
        assert!((nps_floor(&ns, 0.5) - 0.85 * 71.).abs() < 1e-9, "the middle keeps the fixed 70th percentile");
        assert!((nps_floor(&ns, 0.) - 0.85 * 51.).abs() < 1e-9 && (nps_floor(&ns, 1.) - 0.85 * 91.).abs() < 1e-9);
        assert!(nps_eligible(&ns, 0.) > nps_eligible(&ns, 0.5) && nps_eligible(&ns, 0.5) > nps_eligible(&ns, 1.));
        // the list drops what the count drops
        let p = prepared((0..101).map(|i| choice(&i.to_string(), f64::from(i + 1), "g")).collect(), &[7]);
        for focus in [0., 0.5, 1.] {
            let got = ranked(&p, Ask { shuffle: Some(("1".into(), focus)), ..Ask::default() });
            assert_eq!(got.len(), nps_eligible(&ns, focus));
        }
    }

    #[test]
    fn ranking_keeps_one_choice_per_chart_family_and_keymode() {
        let mut rows: Vec<Choice> = (0..35).map(|i| {
            let g = if i < 25 { 0 } else if i < 30 { 1 } else { 2 };
            let mut c = choice(&i.to_string(), if g == 0 { 60. } else { 40. }, &g.to_string());
            c.profile = vec![0., 0., 0.];
            c.profile[g] = 1.;
            c.value = c.nps.ln();
            c.length = 240.;
            c
        }).collect();
        let mut copy = rows[30].clone();
        copy.sha = "copy30".into();
        copy.md5 = "c".repeat(32);
        rows.push(copy);
        let p = prepared(rows, &[7]);
        let got = ranked(&p, Ask::default());
        let families: HashSet<&str> = got.iter().map(|c| c.0.family.as_str()).collect();
        assert!(got.len() == 35 && families.len() == 35);
        assert!(ranked(&p, Ask { keys: vec![4], ..Ask::default() }).is_empty());
    }

    #[test]
    fn nps_never_offers_a_chart_far_below_the_density_floor() {
        let mut rows = vec![choice("floor", 6.2, "sv")];
        rows.extend((0..20).map(|i| choice(&i.to_string(), 25., "chords")));
        let got = ranked(&prepared(rows, &[7]), Ask::default());
        assert!(got.len() == 20 && got.iter().all(|c| c.0.nps == 25.));
    }

    #[test]
    fn a_4k_floor_follows_the_4k_pool() {
        let four = |sha: &str, nps: f64| Choice { keys: 4, ..choice(sha, nps, "chords") };
        let mut rows = vec![four("low", 12.)];
        rows.extend((0..20).map(|i| four(&i.to_string(), 18.)));
        let got = ranked(&prepared(rows, &[4]), Ask { keys: vec![4], ..Ask::default() });
        assert_eq!(got.last().map(|c| c.0.nps), Some(12.), "12 NPS ranks under a pool of 18 (it led at 11.4 floor)");
        assert!(got.last().unwrap().1 < got[0].1 * (-3f64).exp(), "its weight falls by more than e³");
    }

    #[test]
    fn a_picked_skills_chart_without_a_fitting_rate_asks_for_the_one_that_fits() {
        let order: Vec<String> = ["fits", "skill", "other", "edge", "done"].iter().map(|s| s.to_string()).collect();
        let mut memo = Choices::new(order, Forecast::default());
        let want = |rate: f64, skill: bool| Some(Proposal { quality: 0., rate, keys: 10, group: "rice".into(), skill });
        memo.slots[0] = Some((Some(Arc::new(choice("fits", 9., "rice"))), want(0.9, true))); // has a choice already
        memo.slots[1] = Some((None, want(0.85, true)));
        memo.slots[2] = Some((None, want(0.85, false))); // not the picked skill's: the frontier's business
        memo.slots[3] = Some((None, want(0.75, true))); // the range's end: still too hard there
        memo.slots[4] = Some((None, want(1.2, true)));
        let done: HashSet<&str> = ["done"].into_iter().collect();
        assert_eq!(memo.slot_proposals(&done, (0.75, 1.5)), vec![("skill".to_string(), 0.85)]);
    }

    #[test]
    fn a_skill_with_nothing_at_the_target_in_a_keymode_offers_its_nearest_charts() {
        // user 2026-11-22: 10K Invert reached the target at no rate and was never offered: "offer the easiest anyway"
        let invert = |sha: String, acc: f64, learning: bool| Choice { keys: 10, acc_mid: acc, learning, matched: vec![crate::skills::INVERT_MOD.to_string()], ..choice(&sha, 3., "ln") };
        let end = |skill: bool| Some(Proposal { quality: 0., rate: 0.75, keys: 10, group: "ln".into(), skill });
        let order: Vec<String> = (0..25).map(|i| format!("l{i:02}")).chain(["other".to_string(), "fits".to_string()]).collect();
        let mut memo = Choices::new(order, Forecast::default());
        for i in 0..25 {
            memo.slots[i] = Some((Some(Arc::new(invert(format!("l{i:02}"), 0.86 + 0.002 * i as f64, true))), end(true)));
        }
        // another skill of the keymode, with nothing at the target either: a slot of its own
        memo.slots[25] = Some((Some(Arc::new(Choice { keys: 10, learning: true, matched: vec!["ln".to_string()], ..choice("other", 3., "ln") })), end(true)));
        let shas = |m: &Choices| m.snapshot().items.iter().map(|c| c.sha.clone()).collect::<Vec<_>>();
        let nearest: Vec<String> = (5..25).map(|i| format!("l{i:02}")).chain(["other".to_string()]).collect();
        assert_eq!(shas(&memo), nearest, "the 20 nearest of 25, and the other skill's one");
        assert_eq!(memo.slot_proposals(&HashSet::default(), (0.75, 1.5)).len(), 21, "they are analysed at the range's end, the other five are not");
        // one chart at the target takes one of the 20 places
        memo.slots[26] = Some((Some(Arc::new(invert("fits".to_string(), 0.96, false))), None));
        assert_eq!(shas(&memo), (6..25).map(|i| format!("l{i:02}")).chain(["other".to_string(), "fits".to_string()]).collect::<Vec<_>>());
        assert_eq!(memo.slot_proposals(&HashSet::default(), (0.75, 1.5)).len(), 20);
    }

    #[test]
    fn invert_maps_have_a_floor_of_their_own() {
        let invert = |sha: &str, nps: f64| Choice { matched: vec![crate::skills::INVERT_MOD.to_string()], ..choice(sha, nps, "ln") };
        let mut rows: Vec<Choice> = (0..20).map(|i| choice(&i.to_string(), 25., "chords")).collect();
        rows.extend((0..20).map(|i| invert(&format!("i{i}"), 5.)));
        rows.push(invert("thin", 1.));
        rows.push(choice("low", 6.2, "sv"));
        let p = prepared(rows, &[7]);
        let got = ranked(&p, Ask::default());
        let kept = |sha: &str| got.iter().any(|c| c.0.sha == sha);
        assert!(kept("i0") && !kept("thin") && !kept("low"), "5 NPS under Invert stays beside 25 NPS maps; 1 NPS is thin among Invert maps, 6.2 among the others");
        assert_eq!(p.nps_pool(&(0..42).collect::<Vec<_>>())[&7].len(), 21, "the keymode's own floor is read without them");
    }

    #[test]
    fn nps_focus_changes_selection_without_changing_rate_targets() {
        let rows: Vec<Choice> = (0..100).map(|i| {
            let mut c = choice(&i.to_string(), if i < 80 { 22. } else { 34. }, "chords");
            c.rate = 1.1;
            c.value = c.nps.ln();
            c
        }).collect();
        let p = prepared(rows, &[7]);
        let list = |focus: f64| ranked(&p, Ask { shuffle: Some(("9".into(), focus)), ..Ask::default() }).into_iter().take(20).collect::<Vec<_>>();
        assert!(list(1.).iter().all(|c| c.0.nps == 34.));
        let variety = list(0.);
        assert!(variety.iter().any(|c| c.0.nps == 22.) && variety.iter().any(|c| c.0.nps == 34.));
        let shas = |rows: &[(Arc<Choice>, f64)]| rows.iter().map(|c| c.0.sha.clone()).collect::<Vec<_>>();
        assert_eq!(shas(&variety), shas(&list(0.)));
        assert!(variety.iter().all(|c| c.0.rate == 1.1 && c.0.acc_mid == 0.94));
        assert_eq!(variety.iter().map(|c| &c.0.family).collect::<HashSet<_>>().len(), 20);
    }

    #[test]
    fn website_charts_get_a_mild_popularity_prior_and_the_left_end_drops_them() {
        let row = |sha: &str, installed: bool, playcount: f64| {
            let mut c = choice(sha, 30., "chords");
            (c.installed, c.playcount) = (installed, playcount);
            c
        };
        let p = prepared(vec![row("local", true, -1.), row("rare", false, 10.), row("popular", false, 20000.)], &[7]);
        let got = ranked(&p, Ask::default());
        let w = |s: &str| got.iter().find(|c| c.0.sha == s).unwrap().1;
        assert!((w("popular") / w("local") - 1.).abs() < 1e-9);
        assert!(0.65 < w("rare") / w("local") && w("rare") / w("local") < 0.75);
        assert_eq!(ranked(&p, Ask { web: 0., ..Ask::default() }).iter().map(|c| c.0.sha.as_str()).collect::<Vec<_>>(), ["local"]);
    }

    #[test]
    fn low_variety_evens_out_pattern_groups() {
        let mut rows: Vec<Choice> = (0..90).map(|i| choice(&format!("c{i}"), 30., "chords")).collect();
        rows.extend((0..10).map(|i| choice(&format!("j{i}"), 30., "jack")));
        let p = prepared(rows, &[7]);
        let jacks = |focus: f64| (0..40).map(|seed| ranked(&p, Ask { shuffle: Some((seed.to_string(), focus)), ..Ask::default() }).iter().take(20).filter(|c| c.0.group == "jack").count()).sum::<usize>() as f64 / 800.;
        assert!(jacks(1.) < 0.1 && jacks(0.) > 0.3, "10 % of the pool; the 10 jack maps cap a list of 20 at half");
    }

    #[test]
    fn a_row_is_dealt_by_its_leading_picked_skill_only_on_a_list_with_skill_slots() {
        let mut c = choice("a", 30., "chords");
        (c.matched, c.installed, c.convert, c.source_web) = (vec!["vibro".to_string(), "jackspeed".to_string()], false, true, true);
        let p = prepared(vec![c, choice("b", 30., "chords")], &[7]);
        assert_eq!(p.tags(0, true), Tags { keys: 7, skill: "vibro".into(), stream: WEB_CONVERT, learning: false });
        assert_eq!((p.tags(0, false).skill.as_str(), p.tags(1, true)), ("", Tags { keys: 7, skill: String::new(), stream: REAL, learning: false }));
    }

    #[test]
    fn mix_shares_weigh_sliders_by_counts_and_drop_off() {
        let e = [("a", 1., 30), ("b", 2., 10), ("c", 0., 50)];
        let by = mix_shares(&e, true);
        assert!((by["a"] - 0.6).abs() < 1e-9 && (by["b"] - 0.4).abs() < 1e-9 && by["c"] == 0.);
        let even = mix_shares(&e, false);
        assert!((even["a"] - 1. / 3.).abs() < 1e-9 && even["c"] == 0.);
    }
}

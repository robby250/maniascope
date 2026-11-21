//! Calculator knobs: module constants (overridable by name for calibration trials) and the fitted
//! stages from calib/difficulty.json.
use serde_json::Value;

pub trait Knob: Sized {
    fn from_value(v: &Value) -> Option<Self>;
}
impl Knob for f64 {
    fn from_value(v: &Value) -> Option<Self> {
        v.as_f64()
    }
}
impl Knob for usize {
    fn from_value(v: &Value) -> Option<Self> {
        v.as_u64().map(|x| x as usize)
    }
}
impl<const N: usize> Knob for [f64; N] {
    fn from_value(v: &Value) -> Option<Self> {
        let a = v.as_array()?;
        if a.len() != N {
            return None;
        }
        let mut out = [0.0; N];
        for (o, x) in out.iter_mut().zip(a) {
            *o = x.as_f64()?;
        }
        Some(out)
    }
}
impl<const N: usize> Knob for [(f64, f64); N] {
    fn from_value(v: &Value) -> Option<Self> {
        let a = v.as_array()?;
        if a.len() != N {
            return None;
        }
        let mut out = [(0.0, 0.0); N];
        for (o, x) in out.iter_mut().zip(a) {
            let p = x.as_array()?;
            *o = (p.first()?.as_f64()?, p.get(1)?.as_f64()?);
        }
        Some(out)
    }
}

pub trait Scalar {
    fn scalar(&self) -> Option<f64>;
    fn json(&self) -> Value;
}
impl Scalar for f64 {
    fn scalar(&self) -> Option<f64> {
        Some(*self)
    }
    fn json(&self) -> Value {
        Value::from(*self)
    }
}
impl Scalar for usize {
    fn scalar(&self) -> Option<f64> {
        Some(*self as f64)
    }
    fn json(&self) -> Value {
        Value::from(*self)
    }
}
impl<const N: usize> Scalar for [f64; N] {
    fn scalar(&self) -> Option<f64> {
        None
    }
    fn json(&self) -> Value {
        Value::from(self.to_vec())
    }
}
impl<const N: usize> Scalar for [(f64, f64); N] {
    fn scalar(&self) -> Option<f64> {
        None
    }
    fn json(&self) -> Value {
        Value::from(self.iter().map(|&(a, b)| Value::from(vec![a, b])).collect::<Vec<Value>>())
    }
}

macro_rules! knobs {
    ($($name:ident : $ty:ty = $val:expr),* $(,)?) => {
        #[allow(non_snake_case)]
        #[derive(Clone, Debug)]
        pub struct Knobs { $(pub $name: $ty,)* }
        impl Default for Knobs {
            fn default() -> Self { Knobs { $($name: $val,)* } }
        }
        impl Knobs {
            pub fn set(&mut self, name: &str, v: &Value) -> Result<(), String> {
                match name {
                    $(stringify!($name) => {
                        self.$name = <$ty as Knob>::from_value(v)
                            .ok_or_else(|| format!("bad value for {}: {}", name, v))?;
                        Ok(())
                    })*
                    _ => Err(format!("unknown knob: {}", name)),
                }
            }
            /// Every knob as JSON (name order of the definition).
            pub fn dump(&self) -> Vec<(&'static str, Value)> {
                vec![$((stringify!($name), self.$name.json()),)*]
            }
            /// A scalar knob's value (None: unknown or not scalar).
            pub fn get(&self, name: &str) -> Option<f64> {
                match name {
                    $(stringify!($name) => self.$name.scalar(),)*
                    _ => None,
                }
            }
        }
    };
}

knobs! {
    // skill_calc
    SCALE: f64 = 2.2,
    STAR_A: f64 = 0.1743,
    STAR_B: f64 = 1.095,
    W_HAND: f64 = 1.0,
    V_FAST_A: f64 = 0.6107, // the fast-hand onset × e^(A·x + B·x²), x = ln(keys/7): started at the old hinge's 4K/7K/10K values (×1, ×1, ×10/7)
    V_FAST_B: f64 = 1.0914,
    V_FAST: f64 = 12.0,
    SPEED_P: f64 = 1.5,
    W_JACK: f64 = 3.3,      // on 7K; other keymodes as (7/keys)^W_JACK_KEYS: 4.37 on 4K, 2.76 on 10K (2026-11-19: 7K residual rises with a map's jack share; searched with the keymode power)
    W_JACK_KEYS: f64 = 0.67, // flat 0.57–0.77 (thin-keymode retune 2026-11-19)
    JACK_P: f64 = 0.5,      // on 4K; wider keymodes as (keys/4)^JACK_P_KEYS (open loop p8/p12: wide keys want a steeper jack speed curve)
    JACK_P_KEYS: f64 = 1.44, // flat 1.3–1.55 (thin-keymode retune 2026-11-19: 8K–10K confirm −0.3 … −1 %, 4K/7K ±0.1 %)
    CHORDJACK_CARD: f64 = 1.2, // card only: the Chordjack row's rating × this (the model keeps the rating). User 2026-11-20: "bring back to the previous level and then a bit more": hardest 303 7K charts lead with Chordjack 24 → 33 (v1.15.0: 31; 1.16 matches it), on the card 42 → 73 (57)
    STREAM_FAMILY: f64 = 0.888, // 5K+ card: the Stream row = this × the fast and slow halves rated together. One flat scale (user 2026-11-20: "just outright lowering the stream occurrence on all levels equally") fitted so the hardest 7K charts lead with Stream as often as the stronger half did (top 1 %: 20.3 vs 23.7 %, 95–99 %: 20.0 vs 23.0, 90–95 %: 20.4 vs 19.4); easy charts lower than before (54.0 → 45.5 %). Was 0.933 (all-chart count)
    // A re-hit is played by the cheapest of two techniques (user 2026-11-19: finger, wrist and arm jack with different speed,
    // stamina and precision). Neither cost falls with speed, so a rating never falls when the rate rises (the old knee made a
    // re-hit under 70 ms cheaper than one at 70 ms). Searched together 2026-11-19 (31 trials): the values below.
    JACK_RUN_P: f64 = 0.195, // finger: effort × ((1 + re-hits before it in the column's run) / JACK_RUN_REF)^this (top replays: loss over charge doubles from a lone re-hit to the 17th, every speed, 4K and 7K); flat .15–.25
    JACK_RUN_REF: f64 = 4.0,
    JACK_ARM: f64 = 1.0,     // arm: its effort at JACK_ARM_V re-hits/s over the finger's there; ±0.2 costs 0.2–0.35 %
    JACK_ARM_V: f64 = 14.0,  // on 4K; other keymodes as (keys/4)^JACK_ARM_KEYS
    JACK_ARM_KEYS: f64 = -0.25, // flat from −0.45 to −0.05
    JACK_ARM_Q: f64 = 0.0,   // arm effort ∝ speed^this, floored at 0: the search asks for a falling cost, flat costs 0.02 %
    JACK_ARM_RUN_P: f64 = 0.054, // arm: × ((1 + re-hits before) / JACK_RUN_REF)^this (it lasts longer than the finger)
    W_IJACK: f64 = 0.4,
    JACK_SHARE: f64 = 1.4,
    MIRROR: f64 = 0.25,
    W_SPLIT: f64 = 1.5,
    W_ALT: f64 = 0.75,
    W_CROSS: f64 = 0.15,
    W_CROSS_WIDE: f64 = 0.05,
    W_TECH: f64 = 0.3,
    W_SV: f64 = 2.8,
    W_REL: f64 = 0.50,
    W_HOLD_SAME: f64 = 0.30,
    W_HOLD_OTHER: f64 = 0.08,
    W_REPRESS: f64 = 0.30,
    LN_BASE: f64 = 30.0,    // own replays: LN sections played harder than charged on every keymode (open loop p9/p13; 10 before)
    ROW_SHARE: f64 = 0.3,
    ROW_FAST: f64 = 0.0,   // fast chords merge into one gesture: a row's share exponent + this·(1 − smooth(gap, ROW_FAST_LO, ROW_FAST_HI)) (top-50 panels: big chords play easier at the top, r −.55 4K / −.31 7K)
    ROW_FAST_LO: f64 = 0.08,
    ROW_FAST_HI: f64 = 0.2,
    FLAM: f64 = 0.045,
    MASH_SPAN: f64 = 1.0, // Chart::mashed: rows on other columns within this share of the great window, one gesture (0 = off)
    MASH_EASE: f64 = 0.5, // share of a bin's demand eased where its written rows merged into one gesture, × great window / OD 8's (compute.rs)
    MASH_RAMP: f64 = 0.4, // a merged roll eases in full from this share of the span inside it, in proportion before (0 = at once)
    MASH_COVER: f64 = 0.8, // Chart::mashed: share of the keys a gesture covers to merge (a rolled chord, not a flam)
    VF_JACK: f64 = 0.5,     // 0 → 1: a jacked note's finger-repeat factor fades out (its jack effort already scales with that speed); 4K value
    VF_JACK_KEYS: f64 = 2.0, // VF_JACK scales as (4/keys)^this: the 4K gain, flat it cost 7K/10K (open loop p7/p11)
    V_FINGER: f64 = 6.0,    // a finger's own sustainable tap rate (finger-tapping studies: 5–7/s); replays: cost starts at 120–160 ms a column
    RHD_W: f64 = 0.0,       // hardest-7K study: a column re-hit past V_FINGER costs × (1 + this · (rate/V_FINGER − 1) · hand excess) (0 = off)
    RHD_N: f64 = 10.0,      // the hand excess: its notes in the last RHD_SPAN s / (this · (keys/7)^RHD_X) − 1, floored at 0
    RHD_X: f64 = 1.0,
    RHD_SPAN: f64 = 0.5,
    // Chord switching, measured on 5.2M notes of own replays at equal speed, width and play (docs/REPORT_2026-11-01_CHORD_SWITCH.md).
    W_GROUP: f64 = 0.5,     // each extra separate group of fingers in a hand's chord: `10110`, not `11100`
    GROUP_A: f64 = 0.7785,  // W_GROUP × e^(A·x + B·x²): started at the old (min(keys,7)/7)^2 at 4K/7K/10K
    GROUP_B: f64 = -2.1827,
    GROUP_SIZE: f64 = 3.0,  // rows smaller than this pay nothing: a split jump inside a stream isn't a chord to place
    W_MIX: f64 = 0.6,       // a hand row where some fingers repeat and others are new
    MIX_V: f64 = 12.0,      // hand rows/s where the mix starts to cost on 7K
    MIX_KEYS: f64 = 3.0,    // that speed falls as (7/keys)^MIX_KEYS: about 4/s on 10K
    W_SHAPES: f64 = 1.0,    // per doubling of the chord shapes a hand plays in 2 s past SHAPES_T
    SHAPES_T: f64 = 4.0,    // on 10K; narrower keymodes start later, as (10/keys)^SHAPES_KEYS
    SHAPES_KEYS: f64 = 2.0,
    JACK_SHARE_A: f64 = 2.2215, // the chordjack discount × e^(A·x + B·x²): started at the old hinge's 4K/7K/10K values (own 10K replays: chordjacks overcharged; round 18: 4K/5K keep less)
    JACK_SHARE_B: f64 = 2.1827,
    LOCAL_WINDOW: f64 = 2.0, // seconds of demand a note's local level reads (skills::local_levels)
    LOCAL_TOL: f64 = 1.75,   // the star scale of that level (it was the 2 s horizon's tolerance)
    BIN: f64 = 0.5,
    PHASES: usize = 8,
    W_CENTRE: f64 = 0.3,    // an odd keymode's middle column (the thumb) costs × (1 + this): round 18 residual mining, centre share
    ACC_TARGET: f64 = 0.10, // the rating is the skill whose average note loses this much (skills::accuracy_rating)
    SUSTAIN_W: f64 = 0.15,     // sustained load: every bin × (1 + this·(keys/7)^SUSTAIN_X·e^(SUSTAIN_L·L)·max(0, long-window rating / overall − SUSTAIN_R)) (0 = off)
    SUSTAIN_X: f64 = 0.0,
    SUSTAIN_L: f64 = 0.0,
    SUSTAIN_R: f64 = 0.61,     // the ratio where sustained load starts to count
    SUSTAIN_WINDOW: f64 = 180.0, // seconds of demand the sustained rating averages
    END_REF_Q: f64 = 0.65,   // endurance loads are relative to this quantile of the map's 30 s loads (model::endurance_of; 1 = its peak)
    FATIGUE_W: f64 = 0.2,   // a note's level × (1 + this·fatigue), fatigue from model::endurance (skills::powered)
    FATIGUE_W_KEYS: f64 = 3.0, // that weight as (keys/7)^this: wide keymodes are finger-limited, so fatigue weighs more
    ACC_POWER: f64 = 3.0,   // a note's loss grows as (local level / skill)^this (skills::rating)
    // The rating's own constants as curves over keymode × level (plan 2026-11-16 curves; skills::rating, second pass): each ×
    // e^(_L·L + _XL·x·L), L = ln(first-pass rating / LEVEL_REF), x = ln(keys/7). Zero = the constant.
    ACC_POWER_LVL: f64 = 0.0,  // ACC_POWER (top-50 panels: density spikes play harder at the top, r +.33 4K / +.28 7K)
    ACC_POWER_XL: f64 = 0.0,
    ACC_TARGET_L: f64 = 0.0,   // ACC_TARGET (top players score 98–99 %: the loss a rating aims at may shrink with level)
    ACC_TARGET_XL: f64 = 0.0,
    LOCAL_WINDOW_L: f64 = 0.0, // LOCAL_WINDOW (how long a burst a note's level reads)
    LOCAL_WINDOW_XL: f64 = 0.0,
    FATIGUE_W_L: f64 = 0.0,    // the fatigue weight over level
    FATIGUE_W_XL: f64 = 0.0,
    LEVEL_REF: f64 = 12.0,     // the pivot: raw 12, ~8★
    // Pattern strength surfaces (plan 2026-11-16 curves): demand part p × e^(Σ_j SURF[5p+j]·basis_j), basis = (x, x², L, x·L, L²),
    // x = ln(keys/7), L = ln(first-pass overall / SURF_REF); parts in demand::P order. Zero = off.
    SURF: [f64; 200] = [0.0; 200],
    SURF_REF: f64 = 12.0,
    // compute.rs level_terms (plan B6/B7), default 0 = off
    OD_POW: f64 = 0.0,   // bin × (great window / OD 8's)^−(OD_POW + OD_POW_L·L)
    OD_POW_L: f64 = 0.0,
    VIB_L: f64 = 0.0,    // vibro bins × e^(VIB_L·L·run share·vibro share): top players' sustained vibro is far easier than rated (panel 2026-11-17)
    OD_LN: f64 = 0.8,    // LN parts × (great window / OD 8's)^−(OD_LN + OD_LN_L·L): wide windows ease releases (7K panel, OD<5 LN −.24)
    OD_LN_L: f64 = 0.0,
    LN_TOP: f64 = 0.0,     // LN parts × e^(−LN_TOP·max(0, L − LN_TOP_L0)·LN share of the last LN_TOP_WIN s): top-only LN ease (compute.rs level_terms)
    LN_TOP_L0: f64 = 0.0,  // the hinge, L = ln(first-pass overall / LEVEL_REF)
    LN_TOP_WIN: f64 = 30.0, // s of LN share the ease follows (0 = the whole map, share 1)
    LN_TOP_P: f64 = 1.0,   // the ease × share^this: only LN-heavy stretches
    LN_TOP_MIX: f64 = 0.1, // every part × e^(this·max(0, L − LN_TOP_L0)·max(0, 4s(C − s)/C²)): rice with few LNs harder at the top
    LN_TOP_MIX_X: f64 = -3.0, // LN_TOP_MIX × (keys/7)^this (own audit: 4K plays gain, 7K lose at 0)
    LN_TOP_MIX_C: f64 = 0.5, // C: the LN share where that stops (7K panel ≥ 10★: share .05–.4 +.03…+.14, .55–.85 −.06…−.08)
    SWITCH_W: f64 = 0.0, // bin × (1 + SWITCH_W·e^(SWITCH_L·L)·(1 − cos(pattern mix, previous bin's)))
    SWITCH_L: f64 = 0.0,
    TECH_FULL: f64 = 0.09,
    RT_WIN: usize = 4,
    RT_FULL: f64 = 0.2,
    OHT_MIX: f64 = 1.0, // a 1H trill also counts while the other hand plays sparse notes between (round 19: 5–7K, own 7K/10K better)
    PT_FULL: f64 = 0.05,
    PT_CHAIN: f64 = 1.5,
    ROLL_MIN: usize = 6,
    R_EVENT: f64 = 0.025,
    DELAY_GAP: f64 = 0.07,
    TRILL_GAP: f64 = 0.2,
    MJ_BURST: usize = 2,
    MJ_FULL: f64 = 0.11, // demand.rs bursts: a re-hit this fast is a full minijack, fading to none at MJ_NONE
    MJ_NONE: f64 = 0.18, // demand.rs bursts: no minijack from this re-hit gap on (user 2026-11-16: KORO-KO-LOID 1/2 at ~235 ms, DUN DUN 1/4 chords at ~171 ms read Minijacks first, none seen)
    MJ_WIDE_LO: f64 = 0.5, // demand.rs bursts: a re-hit sharing up to this share of the keys is a full minijack,
    MJ_WIDE_HI: f64 = 0.75, // none from this share on (most keys re-hit = chordjack/mash, user 2026-11-17)
    VIBRO_GAP: f64 = 0.098, // structure::vibro_gap: a VIBRO_LEN_REF-hit run is vibro under this re-hit gap (≈153 BPM 1/4; 16 hits ≈165, user 2026-11-14)
    VIBRO_LEN_REF: f64 = 32.0, // structure::vibro_gap: the run length VIBRO_GAP is stated at
    VIBRO_GAP_LONG: f64 = 0.1, // structure::vibro_gap: never slower than this however long the run (150 BPM 1/4: Windows Vs.AGK's long quads and a 157 BPM chord spam are vibro, user 2026-11-14)
    VIBRO_LEN_P: f64 = 0.12, // structure::vibro_gap: × (hits / VIBRO_LEN_REF)^this (short bursts need more speed, long runs less)
    VIBRO_MIN: usize = 7, // column_runs: 7 hits (2026-11-13: 5–6 read DESTRUCTION 3,2,1 stream bursts as vibro)
    VIBRO_P: f64 = 2.0, // column_runs: a run is vibro under VIBRO_GAP × (chord repeat share)^VIBRO_P
    VIBRO_RUN: f64 = 11.0, // gestures::vibro: a fast same-chord run counts as vibro from 4 rows, fully at this many
    VIBRO_SHIFT: f64 = 0.45, // gestures::vibro: share of a fast run's vibro kept while its chords keep changing (anchored jacks)
    HS_SAT: f64 = 5.0, // demand: 4K stream work is all handstream once hands are 1/HS_SAT of its rows
    LJ_GAP: f64 = 0.3, // Longjacks: column runs (structure::column_runs) slower than VIBRO_GAP up to this
    RAMPS: [f64; 4 * crate::taxonomy::SLICES] = crate::taxonomy::fitted_ramps(), // each slice's [lo, hi, e, g] (skills::ramp), calib/taxonomy.json
    STAMINA_WINDOW: f64 = 60.0, // seconds of demand the stamina rating averages (skills::local_levels)
    BURST_TAG: f64 = 1.5,   // card tag "short burst": the 2 s timeline peaks this far above the overall
    SV_STUTTER: f64 = 0.5,  // scale on the fitted stutter weight of the scroll reading (scroll::reading)
    PATTERN_INFO: f64 = 0.2, // demand: a bin's total × e^(PATTERN_INFO · max(0, chord-sequence surprise bits/keys in the bin − PATTERN_INFO_B))
    PATTERN_INFO_B: f64 = 10.0, // surprise (rows' −log2 ÷ keys) per 0.5 s bin before it counts (~20 novel rows/s)
    PATTERN_INFO_X: f64 = 3.0, // PATTERN_INFO × (keys/7)^this (plan 2026-11-18 H7: one curve over keymodes)
    SV_Q: f64 = 1.416,      // a note's reading strain to this power (scroll::reading; < 1: the cost saturates). Top replays of 10 SV charts
                            // (plan 2026-11-20 C2) hinted concave; the joint search on every keymode chose convex (1.0 → 1.42)
    SV_SPEED: f64 = 0.184,  // the SV reading cost grows as row speed to this power (demand::base; pivot 10 rows/s). Searched with SV_Q,
                            // SV_READ (plan 2026-11-20 C2): 0.5 → 0.18, flat 0–0.25; reading is a cost per row more than per second
    SV_READ: f64 = 1.079,   // scale on every fitted scroll-reading weight (scroll::reading)
    // flow descriptors (flow.rs)
    FLOW_LINK: f64 = 0.25,   // s: a hand's next note further than this starts a new phrase
    FLOW_FLOOR: f64 = 0.04,  // s: faster than this counts the same
    FLOW_MEMORY: usize = 32, // rows: a pair of rows seen this recently is no new pattern
    FAST_LN: f64 = 0.05,     // s: LN heads this close on adjacent columns roll
}

/// Components an ablation can switch off (overrides {"off": [...]}): each then contributes its neutral
/// value and every stage reading it still runs. "fit:<section>" drops a fitted stage of difficulty.json.
pub const COMPONENTS: [&str; 7] =
    ["sv_reading", "gestures", "execution", "quadstream", "tech_profile", "archetype_profile", "endurance"];
pub const FITTED: [&str; 3] = ["modes", "scroll_reading", "joint"];

pub struct Params {
    pub k: Knobs,
    off: Vec<&'static str>,
    /// calib/difficulty.json (or a calibration candidate).
    pub model: Value,
    overrides: String,
    id: std::sync::OnceLock<u64>,
    /// `features::rules_stamp`, made once (it hashed the formatted knobs per chart-rate).
    pub(crate) rules: std::sync::OnceLock<String>,
}

impl Params {
    /// Equal for equal calculators, whichever instance holds them.
    pub fn id(&self) -> u64 {
        *self.id.get_or_init(|| {
            use std::hash::{Hash, Hasher};
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            (self.model.to_string(), &self.overrides).hash(&mut hash);
            hash.finish()
        })
    }

    pub fn new(model: Value, overrides: &Value) -> Result<Params, String> {
        let mut k = Knobs::default();
        let mut model = model;
        let mut off = Vec::new();
        if let Some(o) = overrides.as_object() {
            for (name, v) in o {
                if name != "off" {
                    k.set(name, v)?;
                    continue;
                }
                for c in v.as_array().ok_or("off: a list of component names")? {
                    let c = c.as_str().unwrap_or("");
                    if let Some(&known) = COMPONENTS.iter().find(|&&x| x == c) {
                        off.push(known);
                    } else if let Some(section) = c.strip_prefix("fit:").filter(|s| FITTED.contains(s)) {
                        if let Some(m) = model.as_object_mut() {
                            m.remove(section);
                        }
                    } else {
                        return Err(format!("unknown component: {c} (known: {COMPONENTS:?}, fit:{FITTED:?})"));
                    }
                }
            }
        }
        Ok(Params { k, off, model, overrides: overrides.to_string(), id: std::sync::OnceLock::new(), rules: std::sync::OnceLock::new() })
    }

    pub fn off(&self, component: &str) -> bool {
        self.off.contains(&component)
    }
}

impl Knobs {
    /// A slice's ramp [lo, hi, e, g] (taxonomy::TABLE order of the slices).
    pub fn ramp(&self, sk: crate::skills::Sk) -> [f64; 4] {
        let i = crate::taxonomy::slices().position(|s| s == sk).expect("a slice");
        [self.RAMPS[4 * i], self.RAMPS[4 * i + 1], self.RAMPS[4 * i + 2], self.RAMPS[4 * i + 3]]
    }
}

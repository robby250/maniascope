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
    V_FAST_KEYS: f64 = 1.0, // above 7K the fast-hand onset grows with the hand's fingers; ^2 put 10K's past v()'s 22.2/s cap, so it never fired
    V_FAST: f64 = 12.0,
    SPEED_P: f64 = 1.5,
    W_JACK: f64 = 3.0,      // on 7K; other keymodes as (7/keys)^W_JACK_KEYS: 3.3 on 4K as before, 2.82 on 10K (was 3.3 everywhere but 7K)
    W_JACK_KEYS: f64 = 0.17,
    W_JACK_7K: f64 = 0.0,   // > 0: 7K alone takes this and the rest W_JACK (calculators before the revamp: 3.0 and 3.3)
    MIRROR_ALL: f64 = 1.0,  // 1: the partial-mirror jack rule on every keymode; 0: 7K only (before the revamp)
    CENTRE_SPLIT: f64 = 1.0, // 1: an odd keymode's middle key with each hand in turn, averaged; 0: execution gave it to the right hand, the bracket gate to neither
    JACK_P: f64 = 0.5,
    W_IJACK: f64 = 0.4,
    JACK_SHARE: f64 = 0.7,
    MIRROR: f64 = 0.25,
    W_SPLIT: f64 = 1.5,
    W_ALT: f64 = 0.75,
    W_CROSS: f64 = 0.15,
    W_CROSS_WIDE: f64 = 0.05,
    W_TECH: f64 = 0.3,
    W_SV: f64 = 2.8,
    W_REL: f64 = 0.50,
    W_STAG: f64 = 0.30,
    W_HOLD_SAME: f64 = 0.30,
    W_HOLD_OTHER: f64 = 0.08,
    W_REPRESS: f64 = 0.30,
    LN_BASE: f64 = 10.0,
    ROW_SHARE: f64 = 0.3,
    FLAM: f64 = 0.045,
    V_FINGER: f64 = 6.0,    // a finger's own sustainable tap rate (finger-tapping studies: 5–7/s); replays: cost starts at 120–160 ms a column
    // Chord switching, measured on 5.2M notes of own replays at equal speed, width and play (docs/REPORT_2026-11-01_CHORD_SWITCH.md).
    W_GROUP: f64 = 0.5,     // each extra separate group of fingers in a hand's chord: `10110`, not `11100`
    GROUP_KEYS: f64 = 2.0,  // below 7K it fades as (keys/7)^this: 5K and 6K public maps allow about half
    GROUP_SIZE: f64 = 3.0,  // rows smaller than this pay nothing: a split jump inside a stream isn't a chord to place
    W_MIX: f64 = 0.6,       // a hand row where some fingers repeat and others are new
    MIX_V: f64 = 12.0,      // hand rows/s where the mix starts to cost on 7K
    MIX_KEYS: f64 = 3.0,    // that speed falls as (7/keys)^MIX_KEYS: about 4/s on 10K
    W_SHAPES: f64 = 1.0,    // per doubling of the chord shapes a hand plays in 2 s past SHAPES_T
    SHAPES_T: f64 = 4.0,    // on 10K; narrower keymodes start later, as (10/keys)^SHAPES_KEYS
    SHAPES_KEYS: f64 = 2.0,
    JACK_SHARE_KEYS: f64 = 2.0, // above 7K the chordjack discount falls as (7/keys)^this
    LOCAL_WINDOW: f64 = 2.0, // seconds of demand a note's local level reads (skills::local_levels)
    LOCAL_TOL: f64 = 1.75,   // the star scale of that level (it was the 2 s horizon's tolerance)
    BIN: f64 = 0.5,
    PHASES: usize = 8,
    RATE_G: f64 = 0.12,
    ACC_TARGET: f64 = 0.10, // the rating is the skill whose average note loses this much (skills::accuracy_rating)
    FATIGUE_W: f64 = 0.2,   // a note's level × (1 + this·fatigue), fatigue from model::endurance (skills::powered)
    FATIGUE_W_KEYS: f64 = 3.0, // that weight as (keys/7)^this: wide keymodes are finger-limited, so fatigue weighs more
    ACC_POWER: f64 = 3.0,   // a note's loss grows as (local level / skill)^this (skills::rating)
    TECH_FULL: f64 = 0.09,
    RT_WIN: usize = 4,
    RT_FULL: f64 = 0.2,
    COLUMN_ORIGIN: f64 = 0.0, // 1: column-run bins count from the first note, as the demand does
    OHT_FULL: f64 = 0.5,
    ANC_FULL: f64 = 0.5,
    PT_FULL: f64 = 0.05,
    PT_CHAIN: f64 = 1.5,
    ROLL_MIN: usize = 6,
    R_EVENT: f64 = 0.025,
    DELAY_GAP: f64 = 0.07,
    TRILL_GAP: f64 = 0.2,
    BRACKET_FULL: f64 = 0.4,
    BRACKET_IL_FULL: f64 = 0.45,
    DUMP_FULL: f64 = 0.5,
    MJ_BURST: usize = 2,
    MJ_FULL: f64 = 0.2,
    VIBRO_GAP: f64 = 0.095,
    VIBRO_MIN: usize = 5,
    VIB_FULL: f64 = 0.3,
    LJ_FULL: f64 = 0.5,
    SHIELD_FULL: f64 = 0.5,
    INV_FULL: f64 = 1.5,
    HYB_FULL: f64 = 0.8,
    REL_FULL: f64 = 0.6,
    // The LN skills' shares (skills::attribute), set on the four 7K LN Dan practice packs (Less 2026-11-04) so each pack's maps
    // rate their own skill top: held share of column-time where Inverse starts and is full, staggered releases per release
    // where Release is full, technical share where Tech is full.
    LN_INV_LO: f64 = 0.5,
    LN_INV_HI: f64 = 0.6,
    LN_REL_FULL: f64 = 1.0,
    LN_TECH_FULL: f64 = 1.0,
    STAMINA_WINDOW: f64 = 60.0, // seconds of demand the stamina rating averages (skills::local_levels)
    SV_KIND_SHARE: f64 = 0.5,
    REFINE_Q: f64 = 0.9,
    ALSO_SHARE: f64 = 0.6,
    ALSO_SPECIAL: f64 = 0.4,
    ARCH_MATERIAL: f64 = 0.6,
    ARCH_TECH: f64 = 0.75,
    TECH_WHOLE: f64 = 0.85,
    STAMINA_SUSTAIN: f64 = 0.9,
    BURST_TAG: f64 = 1.5,   // card tag "short burst": the 2 s timeline peaks this far above the overall
    SV_PIECES: usize = 3,
    W_WOB: f64 = 2.0,
    SV_VIS: f64 = 450.0,
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
        Ok(Params { k, off, model, overrides: overrides.to_string(), id: std::sync::OnceLock::new() })
    }

    pub fn off(&self, component: &str) -> bool {
        self.off.contains(&component)
    }
}

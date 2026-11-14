//! O2Jam level of a 7K chart, read off our own calc's outputs: boosted regression trees fitted by `calib o2jam_fit`
//! on charts with known O2Jam levels (calib/o2jam.json). Costs a few thousand comparisons per chart.
use serde_json::Value;
use std::sync::OnceLock;

const MODEL: &str = include_str!("../../../calib/o2jam.json");

/// Inputs beside the calc's scores: LN share of notes, log of played seconds, note count.
pub const EXTRA: [&str; 3] = ["_ln_share", "_log_seconds", "_notes"];

/// One tree, flattened: node i splits on `feat[i]` at `thr[i]` (≤ goes left), or is a leaf (`feat[i] < 0`) worth `value[i]`.
#[derive(Default, Debug, PartialEq)]
pub struct Tree {
    pub feat: Vec<i32>,
    pub thr: Vec<f64>,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    pub value: Vec<f64>,
}

impl Tree {
    pub fn eval(&self, x: &[f64]) -> f64 {
        let mut i = 0;
        while self.feat[i] >= 0 {
            i = if x[self.feat[i] as usize] <= self.thr[i] { self.left[i] } else { self.right[i] } as usize;
        }
        self.value[i]
    }
}

#[derive(Default, Debug)]
pub struct Model {
    pub features: Vec<String>,
    pub base: f64,
    pub rate: f64,
    pub trees: Vec<Tree>,
}

impl Model {
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.base + self.rate * self.trees.iter().map(|t| t.eval(x)).sum::<f64>()
    }

    pub fn to_json(&self) -> Value {
        let trees: Vec<Value> = self.trees.iter().map(|t| serde_json::json!({"feat": t.feat, "thr": t.thr, "left": t.left, "right": t.right, "value": t.value})).collect();
        serde_json::json!({"features": self.features, "base": self.base, "rate": self.rate, "trees": trees})
    }

    pub fn from_json(v: &Value) -> Option<Model> {
        let nums = |t: &Value, k: &str| -> Option<Vec<f64>> { t[k].as_array()?.iter().map(Value::as_f64).collect() };
        let trees = v["trees"].as_array()?.iter().map(|t| Some(Tree {
            feat: nums(t, "feat")?.into_iter().map(|x| x as i32).collect(),
            thr: nums(t, "thr")?,
            left: nums(t, "left")?.into_iter().map(|x| x as u32).collect(),
            right: nums(t, "right")?.into_iter().map(|x| x as u32).collect(),
            value: nums(t, "value")?,
        })).collect::<Option<Vec<_>>>()?;
        Some(Model { features: v["features"].as_array()?.iter().map(|f| f.as_str().map(str::to_string)).collect::<Option<_>>()?, base: v["base"].as_f64()?, rate: v["rate"].as_f64()?, trees })
    }

    /// The feature row: each named calc score (0 when the calc doesn't report it), then EXTRA.
    pub fn row(&self, score: impl Fn(&str) -> f64, ln_share: f64, seconds: f64, notes: f64) -> Vec<f64> {
        self.features.iter().map(|f| match f.as_str() {
            "_ln_share" => ln_share,
            "_log_seconds" => seconds.max(5.).ln(),
            "_notes" => notes,
            k => score(k),
        }).collect()
    }
}

fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| Model::from_json(&serde_json::from_str(MODEL).expect("calib/o2jam.json")).expect("calib/o2jam.json: model"))
}

/// O2Jam level of a 7K chart from its calc scores at the played rate (None for other keymodes).
pub fn level(keys: i64, score: impl Fn(&str) -> f64, ln_share: f64, seconds: f64, notes: f64) -> Option<f64> {
    (keys == 7).then(|| model().predict(&model().row(score, ln_share, seconds, notes)).max(1.))
}

#[cfg(test)]
mod tests {
    #[test]
    fn shipped_model_reads_harder_charts_higher() {
        let m = super::model();
        assert!(!m.trees.is_empty() && m.features.iter().any(|f| f == "overall"));
        let at = |o: f64| super::level(7, |k| if k == "overall" { o } else { o * 0.6 }, 0.5, 150., 2000.).unwrap();
        assert!(at(6.) < at(11.) && at(11.) < at(16.), "{} {} {}", at(6.), at(11.), at(16.));
        assert_eq!(super::level(4, |_| 10., 0.5, 150., 2000.), None);
    }
}

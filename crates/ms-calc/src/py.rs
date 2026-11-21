//! Python numeric semantics the calculator was written against.

/// `round(x)`: half to even.
#[inline]
pub fn round(x: f64) -> f64 {
    x.round_ties_even()
}

/// `round(x, nd)`: correctly rounded decimal (half to even on the exact binary value), back to the nearest double.
pub fn round_digits(x: f64, nd: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    // x·10^nd then a division of an exact integer by an exact power of ten is the nearest double to the rounded decimal,
    // unless x·10^nd sits within its own rounding error of a half (then the exact decimal decides, as Python does)
    if nd <= 15 {
        let p = 10f64.powi(nd as i32);
        let y = x * p;
        if y.abs() < 4.5e15 && ((y - y.trunc()).abs() - 0.5).abs() > y.abs() * 4e-16 {
            return y.round() / p;
        }
    }
    format!("{:.*}", nd, x).parse().unwrap()
}

/// `float(s)`: surrounding whitespace, inf/nan, single underscores between digits.
pub fn float(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.contains('_') {
        let b = s.as_bytes();
        for (i, &c) in b.iter().enumerate() {
            if c == b'_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
                return None;
            }
        }
        return s.replace('_', "").parse().ok();
    }
    s.parse().ok()
}

/// `int(s)` for a decimal integer string.
pub fn int(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.contains('_') {
        return float(s).filter(|_| !s.contains(['.', 'e', 'E'])).map(|v| v as i64);
    }
    s.parse().ok()
}

/// `int(float(s))`: NaN is a ValueError (Err(false)), ±inf an OverflowError (Err(true)).
pub fn int_float(s: &str) -> Result<i64, bool> {
    // nearly every field is a plain integer: below 2^53 the float round trip is exact
    let b = s.as_bytes();
    if (1..=15).contains(&b.len()) && b.iter().all(u8::is_ascii_digit) {
        return Ok(b.iter().fold(0i64, |v, d| v * 10 + i64::from(d - b'0'))); // digits only, under 10^15: the same value, no trim or sign handling
    }
    if let Ok(v) = s.trim().parse::<i64>() {
        if v.unsigned_abs() < 1 << 53 {
            return Ok(v);
        }
    }
    match float(s) {
        None => Err(false),
        Some(v) if v.is_nan() => Err(false),
        Some(v) if v.is_infinite() => Err(true),
        Some(v) => Ok(v.trunc() as i64),
    }
}

/// `builtins.sum` of floats (CPython ≥3.12: Neumaier-compensated).
pub fn sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut total = 0.0f64;
    let mut c = 0.0f64;
    for v in values {
        let t = total + v;
        if total.abs() >= v.abs() {
            c += (total - t) + v;
        } else {
            c += (v - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total + c
    } else {
        total
    }
}

/// `statistics.median`.
pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Floor division of Python ints, b > 0.
#[inline]
pub fn floordiv(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounding() {
        assert_eq!(round(2.5), 2.0);
        assert_eq!(round(3.5), 4.0);
        assert_eq!(round(-2.5), -2.0);
        assert_eq!(round_digits(0.0625, 3), 0.062);
        assert_eq!(round_digits(0.0635, 3), 0.064);
        assert_eq!(round_digits(333.3335, 3), 333.334);
        assert_eq!(round_digits(461.53846153846155, 3), 461.538);
    }
    #[test]
    fn floats() {
        assert_eq!(float(" 1_000 "), Some(1000.0));
        assert_eq!(float("1__0"), None);
        assert_eq!(int_float("12.9"), Ok(12));
        assert_eq!(int_float("-12.9"), Ok(-12));
        assert_eq!(int_float("nan"), Err(false));
        assert_eq!(floordiv(-1, 512), -1);
        assert_eq!(floordiv(511, 512), 0);
    }
    #[test]
    fn neumaier() {
        assert_eq!(sum([0.1; 10]), 1.0);
        assert_eq!(sum([1e100, 1.0, -1e100]), 1.0);
    }
}

#[cfg(test)]
mod round_tests {
    #[test]
    fn round_digits_matches_the_decimal_rounding() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for i in 0..400_000 {
            let r = next();
            let x = match i % 4 {
                0 => (r as f64 / u64::MAX as f64 - 0.5) * 40.0,
                1 => (r % 2_000_000) as f64 / 1e6 + 0.5e-4 * ((r >> 40) % 3) as f64, // near halves
                2 => f64::from_bits(r) ,
                _ => ((r % 100_000) as f64 + 0.5) / 10f64.powi((r >> 50) as i32 % 7),
            };
            if !x.is_finite() {
                continue;
            }
            for nd in [0, 1, 2, 4, 6, 9] {
                let want: f64 = format!("{:.*}", nd, x).parse().unwrap();
                assert_eq!(super::round_digits(x, nd).to_bits(), want.to_bits(), "{x} {nd}");
            }
        }
    }
}

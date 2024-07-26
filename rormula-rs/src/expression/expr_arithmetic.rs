use exmex::BinOp;
use exmex::Express;
use exmex::FlatEx;
use exmex::MakeOperators;
use exmex::Operator;
use std::mem;

use super::ops_common;
use super::Value;
use crate::array::Array2d;
use crate::array::DefaultOrder;
use crate::array::MemOrder;
fn apply_op<M: MemOrder>(
    mut a: Value<M>,
    mut b: Value<M>,
    op: &impl Fn(f64, f64) -> f64,
) -> Value<M> {
    // We take mutable references because otherwise in second (scalar) arm they would already have
    // been moved and not available anymore
    let res = match (&mut a, &mut b) {
        (Value::Array(a), Value::Array(b)) => {
            ops_common::op_componentwise_array(mem::take(a), mem::take(b), op).map(Value::Array)
        }
        (_, Value::Error(e)) => Ok(Value::Error(mem::take(e))),
        (Value::Error(e), _) => Ok(Value::Error(mem::take(e))),
        _ => Ok(ops_common::op_scalar(a, b, op)),
    };
    match res {
        Ok(res) => res,
        Err(e) => Value::Error(e.to_string()),
    }
}

pub fn op_add<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    apply_op(a, b, &|x, y| x + y)
}
pub fn op_sub<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    apply_op(a, b, &|x, y| x - y)
}
pub fn op_mul<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    apply_op(a, b, &|x, y| x * y)
}
pub fn op_div<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    apply_op(a, b, &|x, y| x / y)
}

pub fn op_unary<M: MemOrder>(a: Value<M>, op: &impl Fn(f64) -> f64) -> Value<M> {
    match a {
        Value::Array(mut arr) => {
            arr.elt_mutate(op);
            Value::Array(arr)
        }
        Value::Scalar(s) => Value::Scalar(s),
        _ => Value::Error("can only apply unary operator to numerical values".to_string()),
    }
}

fn compare_slices<T>(v1: &[T], v2: &[T], f: impl Fn(&T, &T) -> bool) -> Vec<usize>
where
    T: PartialEq,
{
    v1.iter()
        .zip(v2.iter())
        .enumerate()
        .filter(|(_, (v1i, v2i))| f(*v1i, *v2i))
        .map(|(i, _)| i)
        .collect()
}

/// compare floating point numbers including edge cases
fn floats_almost_equals(a: f64, b: f64, epsilon: f64) -> bool {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();
    if a == b {
        // handles infinities
        true
    } else if a == 0.0 || b == 0.0 || diff < f64::MIN_POSITIVE {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        diff < (epsilon * f64::MIN_POSITIVE)
    } else {
        // use relative error
        (diff / f64::min(abs_a + abs_b, f64::MAX)) < epsilon
    }
}

fn floats_ge(a: f64, b: f64, epsilon: f64) -> bool {
    floats_almost_equals(a, b, epsilon) || a > b
}

fn floats_gt(a: f64, b: f64, epsilon: f64) -> bool {
    !floats_almost_equals(a, b, epsilon) && a > b
}
fn floats_le(a: f64, b: f64, epsilon: f64) -> bool {
    floats_almost_equals(a, b, epsilon) || a < b
}
fn floats_lt(a: f64, b: f64, epsilon: f64) -> bool {
    !floats_almost_equals(a, b, epsilon) && a < b
}

macro_rules! op_compare {
    ($a:expr, $b:expr, $comp_exact:expr, $comp_float:expr) => {
        match ($a, $b) {
            (Value::Scalar(s), Value::Array(a)) => Value::RowInds(
                a.iter()
                    .enumerate()
                    .filter(|(_, ai)| $comp_float(*ai, s, 1e-8))
                    .map(|(i, _)| i)
                    .collect(),
            ),
            (Value::Array(a), Value::Scalar(s)) => Value::RowInds(
                a.iter()
                    .enumerate()
                    .filter(|(_, ai)| $comp_float(*ai, s, 1e-8))
                    .map(|(i, _)| i)
                    .collect(),
            ),
            (Value::Array(a), Value::Array(b)) => Value::RowInds(
                a.iter()
                    .zip(b.iter())
                    .enumerate()
                    .filter(|(_, (ai, bi))| $comp_float(*ai, *bi, 1e-8))
                    .map(|(i, _)| i)
                    .collect(),
            ),
            (Value::Cats(c1), Value::Cats(c2)) => {
                Value::RowInds(compare_slices(&c1, &c2, $comp_exact))
            }
            (Value::RowInds(ri1), Value::RowInds(ri2)) => {
                Value::RowInds(compare_slices(&ri1, &ri2, $comp_exact))
            }
            (Value::Error(e), _) => Value::Error(e),
            (_, Value::Error(e)) => Value::Error(e),
            _ => Value::Error("cannot compare values".to_string()),
        }
    };
}

pub fn op_compare_ge<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    op_compare!(a, b, |v1, v2| v1 >= v2, floats_ge)
}
pub fn op_compare_le<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    op_compare!(a, b, |v1, v2| v1 <= v2, floats_le)
}
pub fn op_compare_gt<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    op_compare!(a, b, |v1, v2| v1 > v2, floats_gt)
}
pub fn op_compare_lt<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    op_compare!(a, b, |v1, v2| v1 < v2, floats_lt)
}
pub fn op_compare_equals<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    op_compare!(a, b, |v1, v2| v1 == v2, floats_almost_equals)
}

pub fn op_restrict<M: MemOrder>(a: Value<M>, b: Value<M>) -> Value<M> {
    match (a, b) {
        (Value::Array(a), Value::RowInds(ris)) => {
            let max = ris.iter().max();
            if let Some(max) = max {
                if *max >= a.n_rows() {
                    Value::Error(format!(
                        "row index out of bounds: {} >= {}",
                        max,
                        a.n_rows()
                    ))
                } else {
                    let data = ris.iter().map(|ri| a.get(*ri, 0)).collect::<Vec<f64>>();
                    let n_rows = data.len();
                    let res = Array2d::new(data, n_rows, 1);
                    match res {
                        Ok(res) => Value::Array(res),
                        Err(e) => Value::Error(e.to_string()),
                    }
                }
            } else {
                Value::Array(Array2d::ones(0, a.n_cols()))
            }
        }
        (Value::Cats(mut c), Value::RowInds(ris)) => {
            Value::Cats(ris.iter().map(|i| mem::take(&mut c[*i])).collect())
        }
        (Value::RowInds(a), Value::RowInds(ris)) => {
            Value::RowInds(ris.iter().map(|i| a[*i]).collect())
        }
        (Value::Error(e), _) => Value::Error(e),
        (_, Value::Error(e)) => Value::Error(e),
        _ => Value::Error("can only restrict arrays, categories or row indices".to_string()),
    }
}

#[derive(Clone, Debug)]
pub struct ArithmeticOpsFactory;
impl<M> MakeOperators<Value<M>> for ArithmeticOpsFactory
where
    M: Clone + MemOrder,
{
    fn make<'b>() -> Vec<Operator<'b, Value<M>>> {
        vec![
            Operator::make_bin(
                "^",
                BinOp {
                    apply: ops_common::op_power,
                    prio: 6,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "*",
                BinOp {
                    apply: op_mul,
                    prio: 4,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "+",
                BinOp {
                    apply: op_add,
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "/",
                BinOp {
                    apply: op_div,
                    prio: 5,
                    is_commutative: false,
                },
            ),
            Operator::make_bin_unary(
                "-",
                BinOp {
                    apply: op_sub,
                    prio: 3,
                    is_commutative: false,
                },
                |a| op_unary(a, &|a| -a),
            ),
            Operator::make_bin(
                "==",
                BinOp {
                    apply: op_compare_equals,
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "|",
                BinOp {
                    apply: op_restrict,
                    prio: 0,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "<",
                BinOp {
                    apply: op_compare_lt,
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "<=",
                BinOp {
                    apply: op_compare_le,
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                ">",
                BinOp {
                    apply: op_compare_gt,
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                ">=",
                BinOp {
                    apply: op_compare_ge,
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_unary("abs", |a| op_unary(a, &|x| x.abs())),
            Operator::make_unary("sqrt", |a| op_unary(a, &|x| x.sqrt())),
            Operator::make_unary("round", |a| op_unary(a, &|x| x.round())),
            Operator::make_unary("floor", |a| op_unary(a, &|x| x.floor())),
            Operator::make_unary("ceil", |a| op_unary(a, &|x| x.ceil())),
            Operator::make_unary("trunc", |a| op_unary(a, &|x| x.trunc())),
            Operator::make_unary("fract", |a| op_unary(a, &|x| x.fract())),
            Operator::make_unary("sign", |a| op_unary(a, &|x| x.signum())),
            Operator::make_unary("sin", |a| op_unary(a, &|x| x.sin())),
            Operator::make_unary("cos", |a| op_unary(a, &|x| x.cos())),
            Operator::make_unary("tan", |a| op_unary(a, &|x| x.tan())),
            Operator::make_unary("asin", |a| op_unary(a, &|x| x.asin())),
            Operator::make_unary("acos", |a| op_unary(a, &|x| x.acos())),
            Operator::make_unary("atan", |a| op_unary(a, &|x| x.atan())),
            Operator::make_unary("exp", |a| op_unary(a, &|x| x.exp())),
            Operator::make_unary("ln", |a| op_unary(a, &|x| x.ln())),
            Operator::make_unary("log", |a| op_unary(a, &|x| x.ln())),
            Operator::make_unary("log2", |a| op_unary(a, &|x| x.log2())),
            Operator::make_unary("log10", |a| op_unary(a, &|x| x.log10())),
        ]
    }
}

const ROW_CHANGE_OPS: [&str; 1] = ["|"];

pub fn has_row_change_op(expr: &ExprArithmetic) -> bool {
    expr.operator_reprs()
        .iter()
        .any(|o| ROW_CHANGE_OPS.contains(&o.as_str()))
}

pub type ExprArithmetic<M = DefaultOrder> = FlatEx<Value<M>, ArithmeticOpsFactory>;

#[cfg(test)]
use crate::array::ColMajor;
#[test]
fn keep_or_change_ops() {
    let x = ExprArithmetic::parse("x + 1").unwrap();
    assert!(!has_row_change_op(&x));
    let x = ExprArithmetic::parse("x + y - 1 == 4").unwrap();
    assert!(!has_row_change_op(&x));
    let x = ExprArithmetic::parse("sin(x) + y - 2").unwrap();
    assert!(!has_row_change_op(&x));
    let x = ExprArithmetic::parse("sin(x)|y==2").unwrap();
    assert!(has_row_change_op(&x));
}
#[test]
fn test() {
    let a = Array2d::<ColMajor>::from_iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0].iter(), 3, 2).unwrap();
    let a_ref = Array2d::from_iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter(), 3, 2).unwrap();
    let res = op_add(Value::Array(a.clone()), Value::Scalar(1.0));
    match res {
        Value::Array(a) => assert_eq!(a, a_ref.clone()),
        _ => assert!(false),
    }
    let res = op_sub(Value::Scalar(1.0), Value::Array(a.clone()));
    let a_ref = Array2d::from_iter([1.0, 0.0, -1.0, -2.0, -3.0, -4.0].iter(), 3, 2).unwrap();
    match res {
        Value::Array(a) => assert_eq!(a, a_ref.clone()),
        _ => assert!(false),
    }
    let a = Array2d::<ColMajor>::from_iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0].iter(), 6, 1).unwrap();
    let b = Array2d::from_iter([2.0, 1.0, 3.0, 5.0, 10.0, 9.0].iter(), 6, 1).unwrap();
    let res = op_mul(Value::Array(b.clone()), Value::Array(a.clone()));
    let a_ref = Array2d::from_iter([0.0, 1.0, 6.0, 15.0, 40.0, 45.0].iter(), 6, 1).unwrap();
    match res {
        Value::Array(a) => assert_eq!(a, a_ref.clone()),
        _ => assert!(false),
    }
    let res = op_div(Value::Array(a.clone()), Value::Array(b.clone()));
    let a_ref = Array2d::from_iter(
        [0.0, 1.0, 2.0 / 3.0, 3.0 / 5.0, 2.0 / 5.0, 5.0 / 9.0].iter(),
        6,
        1,
    )
    .unwrap();
    match res {
        Value::Array(a) => assert_eq!(a, a_ref.clone()),
        _ => assert!(false),
    }

    let res = op_compare_ge(Value::Array(a.clone()), Value::Array(b.clone()));
    let a_ref = Value::RowInds(vec![1]);
    assert_eq!(res, a_ref);
    let res = op_compare_gt(Value::Array(a.clone()), Value::Array(b.clone()));
    let a_ref = Value::RowInds(vec![]);
    assert_eq!(res, a_ref);
    let res = op_compare_le(Value::Array(a.clone()), Value::Array(b.clone()));
    let a_ref = Value::RowInds(vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(res, a_ref);
    let res = op_compare_lt(Value::Array(a.clone()), Value::Array(b.clone()));
    let a_ref = Value::RowInds(vec![0, 2, 3, 4, 5]);
    assert_eq!(res, a_ref);

    let res = op_compare_equals(Value::Scalar(1.0), Value::Array(b.clone()));
    let a_ref = Value::RowInds(vec![1]);
    assert_eq!(res, a_ref);
    let res = op_compare_equals(Value::Array(a.clone()), Value::Array(b.clone()));
    let a_ref = Value::RowInds(vec![1]);
    assert_eq!(res, a_ref);
    let res = op_compare_equals(
        Value::<ColMajor>::Cats(vec!["a".to_string(), "b".to_string()]),
        Value::Cats(vec!["a".to_string(), "c".to_string()]),
    );
    let a_ref = Value::RowInds(vec![0]);
    assert_eq!(res, a_ref);
    let res: Value<ColMajor> =
        op_compare_equals(Value::RowInds(vec![4, 3, 2]), Value::RowInds(vec![1, 3, 7]));
    let a_ref = Value::RowInds(vec![1]);
    assert_eq!(res, a_ref);

    let res = op_restrict(Value::Array(a.clone()), Value::RowInds(vec![0, 2, 4]));
    let a_ref = Value::Array(Array2d::from_iter([0.0, 2.0, 4.0].iter(), 3, 1).unwrap());
    assert_eq!(res, a_ref);
    let res: Value<ColMajor> = op_restrict(
        Value::Cats(vec!["a".to_string(), "b".to_string()]),
        Value::RowInds(vec![1]),
    );
    let c_ref = Value::Cats(vec!["b".to_string()]);
    assert_eq!(res, c_ref);
    let res: Value<ColMajor> = op_restrict(
        Value::RowInds(vec![1, 2, 3, 4]),
        Value::RowInds(vec![1, 2, 3]),
    );
    let r_ref = Value::RowInds(vec![2, 3, 4]);
    assert_eq!(res, r_ref);
}

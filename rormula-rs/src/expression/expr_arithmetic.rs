use exmex::BinOp;
use exmex::FlatEx;
use exmex::MakeOperators;
use exmex::Operator;
use std::mem;

use super::ops_common;
use super::Value;

fn apply_op(mut a: Value, mut b: Value, op: &impl Fn(f64, f64) -> f64) -> Value {
    // We take mutable references because otherwise in second (scalar) arm they would already have
    // been moved and not available anymore
    let res = match (&mut a, &mut b) {
        (Value::Array(a), Value::Array(b)) => {
            ops_common::op_componentwise_array(mem::take(a), mem::take(b), op).map(Value::Array)
        }
        _ => Ok(ops_common::op_scalar(a, b, op)),
    };
    match res {
        Ok(res) => res,
        Err(e) => Value::Error(e.to_string()),
    }
}

pub fn op_add(a: Value, b: Value) -> Value {
    apply_op(a, b, &|x, y| x + y)
}
pub fn op_sub(a: Value, b: Value) -> Value {
    apply_op(a, b, &|x, y| x - y)
}
pub fn op_mul(a: Value, b: Value) -> Value {
    apply_op(a, b, &|x, y| x * y)
}
pub fn op_div(a: Value, b: Value) -> Value {
    apply_op(a, b, &|x, y| x / y)
}

pub fn op_unary(a: Value, op: &impl Fn(f64) -> f64) -> Value {
    match a {
        Value::Array(mut arr) => {
            arr.elt_mutate(op);
            Value::Array(arr)
        }
        Value::Scalar(s) => Value::Scalar(s),
        _ => Value::Error("can only apply unary operator to numerical values".to_string()),
    }
}

#[derive(Clone, Debug)]
pub struct ArithmeticOpsFactory;
impl MakeOperators<Value> for ArithmeticOpsFactory {
    fn make<'b>() -> Vec<Operator<'b, Value>> {
        vec![
            Operator::make_bin(
                "^",
                BinOp {
                    apply: ops_common::op_power,
                    prio: 4,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "*",
                BinOp {
                    apply: op_mul,
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "+",
                BinOp {
                    apply: op_add,
                    prio: 0,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "/",
                BinOp {
                    apply: op_div,
                    prio: 3,
                    is_commutative: false,
                },
            ),
            Operator::make_bin_unary(
                "-",
                BinOp {
                    apply: op_sub,
                    prio: 1,
                    is_commutative: false,
                },
                |a| op_unary(a, &|a| -a),
            ),
        ]
    }
}

pub type ExprArithmetic = FlatEx<Value, ArithmeticOpsFactory>;

#[cfg(test)]
use crate::array::Array2d;

#[test]
fn test() {
    let a = Array2d::from_iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0].iter(), 3, 2).unwrap();
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
    let a = Array2d::from_iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0].iter(), 6, 1).unwrap();
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
}

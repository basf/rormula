use std::fmt::Debug;

use exmex::{ops_factory, BinOp, FlatEx, MakeOperators, Operator};

use crate::array::Array2d;
use crate::expression::{ops_common, value::Value};
use crate::result::RoResult;

use super::value::NameValue;

fn cat_to_dummy_name(c: NameValue) -> RoResult<NameValue> {
    if let NameValue::Cats((feature_name, cats)) = c {
        let (unique, _) = ops_common::unique_cats(&cats)?;
        let names = unique
            .into_iter()
            .map(|cat| format!("{feature_name}_{cat}"))
            .collect::<Vec<_>>();
        Ok(NameValue::Array(names))
    } else {
        Ok(c)
    }
}

fn apply_op(a: Value, b: Value, op: &impl Fn(Array2d, Array2d) -> RoResult<Array2d>) -> Value {
    let a = match ops_common::cat_to_dummy(a) {
        Ok(arr) => arr,
        Err(e) => Value::Error(e.msg().to_string()),
    };
    let b = match ops_common::cat_to_dummy(b) {
        Ok(arr) => arr,
        Err(e) => Value::Error(e.msg().to_string()),
    };
    match (a, b) {
        (Value::Array(a), Value::Array(b)) => {
            let new_val = op(a, b);
            match new_val {
                Ok(a) => Value::Array(a),
                Err(e) => Value::Error(e.to_string()),
            }
        }
        (Value::Error(e), _) => Value::Error(e),
        (_, Value::Error(e)) => Value::Error(e),
        _ => Value::Error("some error during operation".to_string()),
    }
}

pub fn op_concat(a: Value, b: Value) -> Value {
    apply_op(a, b, &|a, b| {
        #[cfg(feature = "print_timings")]
        let now = std::time::Instant::now();

        let res = a.concatenate_cols(b);

        #[cfg(feature = "print_timings")]
        eprintln!("plus op {}", now.elapsed().as_nanos());

        res
    })
}

pub fn op_multiply(a: Value, b: Value) -> Value {
    apply_op(a, b, &|a: Array2d, b: Array2d| {
        ops_common::op_componentwise_array(a, b, &|x, y| x * y)
    })
}

#[derive(Clone, Debug)]
pub struct WilkinsonOpsFactory;
impl MakeOperators<Value> for WilkinsonOpsFactory {
    fn make<'b>() -> Vec<Operator<'b, Value>> {
        vec![
            Operator::make_bin(
                "^",
                BinOp {
                    apply: ops_common::op_power,
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                ":",
                BinOp {
                    apply: op_multiply,
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "+",
                BinOp {
                    apply: op_concat,
                    prio: 0,
                    is_commutative: false,
                },
            ),
        ]
    }
}

pub type ExprWilkinson = FlatEx<Value, WilkinsonOpsFactory>;

fn apply_name_op(
    a: NameValue,
    b: NameValue,
    op: fn(Vec<String>, Vec<String>) -> Vec<String>,
) -> NameValue {
    let a = match cat_to_dummy_name(a) {
        Ok(arr) => arr,
        Err(e) => NameValue::Error(e.msg().to_string()),
    };
    let b = match cat_to_dummy_name(b) {
        Ok(arr) => arr,
        Err(e) => NameValue::Error(e.msg().to_string()),
    };
    match (a, b) {
        (NameValue::Array(a), NameValue::Array(b)) => {
            let new_val = op(a, b);
            NameValue::Array(new_val)
        }
        (NameValue::Error(e), _) => NameValue::Error(e),
        (_, NameValue::Error(e)) => NameValue::Error(e),
        _ => NameValue::Error("some error during operation".to_string()),
    }
}

fn op_name_plus(a: NameValue, b: NameValue) -> NameValue {
    apply_name_op(a, b, |mut a, mut b| {
        a.append(&mut b);
        a
    })
}

fn op_name_colon(a: NameValue, b: NameValue) -> NameValue {
    apply_name_op(a, b, |a, b| {
        b.into_iter()
            .flat_map(move |b_| a.clone().into_iter().map(move |a_| format!("{a_}:{b_}")))
            .collect::<Vec<String>>()
    })
}
fn op_name_power(a: NameValue, b: NameValue) -> NameValue {
    match (a, b) {
        (NameValue::Array(old_names), NameValue::Scalar(sc)) => {
            let new_names = old_names
                .into_iter()
                .map(|on| format!("{on}^{sc}"))
                .collect();
            NameValue::Array(new_names)
        }
        _ => NameValue::Error("power can only be applied to matrix and skalar".to_string()),
    }
}

ops_factory!(
    NameOps,
    NameValue,
    Operator::make_bin(
        "+",
        BinOp {
            apply: op_name_plus,
            prio: 0,
            is_commutative: false,
        },
    ),
    Operator::make_bin(
        ":",
        BinOp {
            apply: op_name_colon,
            prio: 1,
            is_commutative: false
        }
    ),
    Operator::make_bin(
        "^",
        BinOp {
            apply: op_name_power,
            prio: 2,
            is_commutative: false
        }
    )
);
pub type ExprNames = FlatEx<NameValue, NameOps>;
#[derive(Clone, Debug)]
pub struct ColCountOps;

impl MakeOperators<usize> for ColCountOps {
    fn make<'a>() -> Vec<Operator<'a, usize>> {
        vec![
            Operator::make_bin(
                "^",
                BinOp {
                    apply: |a, _| a,
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                ":",
                BinOp {
                    apply: |a, b| a * b,
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "+",
                BinOp {
                    apply: |a, b| a + b,
                    prio: 0,
                    is_commutative: false,
                },
            ),
        ]
    }
}

pub type ExprColCount = FlatEx<usize, ColCountOps>;

#[cfg(test)]
fn get_cat_fortest(n_rows: Option<usize>) -> (Vec<String>, String, Value) {
    let feature_name = "animal".to_string();
    let cats_all = ["zebra", "alpaca", "boa", "dragon", "zebra", "dragon"];
    let cats = if let Some(n_rows) = n_rows {
        &["zebra", "alpaca", "boa", "dragon", "zebra", "dragon"][0..n_rows]
    } else {
        &cats_all
    };
    let mut cats = cats.iter().map(|c| c.to_string()).collect::<Vec<_>>();
    let val = Value::Cats(cats.clone());
    cats.sort();
    cats.dedup();
    cats.pop();
    (
        cats.iter()
            .map(|c| format!("{feature_name}_{c}"))
            .collect::<Vec<_>>(),
        feature_name,
        val,
    )
}

#[cfg(test)]
fn names_equal(names: &[String], expected: &[&str]) {
    assert_eq!(names.len(), expected.len());
    for (n1, n2) in names.iter().zip(expected.iter()) {
        assert_eq!(n1, n2);
    }
}
#[cfg(test)]
fn array_almost_equal(a1: Array2d, a2: Array2d) {
    assert_eq!(a1.n_cols, a2.n_cols);
    assert_eq!(a1.n_rows, a2.n_rows);
    for (elt1, elt2) in a1.data.iter().zip(a2.data.iter()) {
        assert!((elt1 - elt2).abs() < 1e-12);
    }
}

#[test]
fn test_cat_to_dummy() {
    let (reference_names, feature_name, cats) = get_cat_fortest(None);
    let cat_name = NameValue::cats_from_value(feature_name, cats.clone()).unwrap();
    if let NameValue::Array(names) = cat_to_dummy_name(cat_name).unwrap() {
        println!("{names:?}");
        println!("{reference_names:?}");
        assert_eq!(names, reference_names);
    } else {
        panic!("could not get names");
    }
    if let Value::Array(arr) = ops_common::cat_to_dummy(cats.clone()).unwrap() {
        println!("{arr:?}");
        assert_eq!(arr.n_cols, 3);
        assert_eq!(arr.get(0, 0), 0.0);
        assert_eq!(arr.get(1, 0), 1.0);
        assert_eq!(arr.get(2, 1), 1.0);
        assert_eq!(arr.get(3, 2), 1.0);
        assert_eq!(arr.get(4, 2), 0.0);
        assert_eq!(arr.get(5, 2), 1.0);
    } else {
        assert!(false);
    }
}

#[rustfmt::skip]
#[test]
fn test_ops() {
    let mut a = Array2d::zeros(2, 3);
    a.set(0, 1, 0.5);
    a.set(0, 2, 1.0);
    a.set(1, 0, 2.0);
    println!("{a:?}");
    let mut b = Array2d::zeros(2, 1);
    b.set(0, 0, 0.5);
    b.set(1, 0, 1.0);
    println!("{b:?}");
    let na = NameValue::Array(vec!["a1".to_string(),"a2".to_string(),"a3".to_string()]);
    let nb = NameValue::Array(vec!["b".to_string()]);
    let a = Value::Array(a);
    let b = Value::Array(b);

    // colon
    let expected = Array2d::from_iter([
        0.0, 0.25, 0.5, 
        2.0, 0.0, 0.0
    ].iter(), 2, 3).unwrap();
    if let NameValue::Array(n) = op_name_colon(na.clone(), nb.clone()) {
        println!("{n:?}");
        names_equal(&n, &["a1:b", "a2:b", "a3:b"]);
    } else {
        panic!("couldn't get names for colon op")
    }
    if let Value::Array(colon) = op_multiply(a.clone(), b.clone()) {
        array_almost_equal(colon, expected); 
    } else {
        assert!(false);
    }

    // Plus
    let expected = Array2d::from_iter([
        0.0, 0.5, 1.0, 0.5, 
        2.0, 0.0, 0.0, 1.0
    ].iter(), 2, 4).unwrap();
    if let NameValue::Array(n) = op_name_plus(na.clone(), nb.clone()) {
        println!("{n:?}");
        names_equal(&n, &["a1", "a2", "a3", "b"]);
    } else {
        panic!("couldn't get names for plus op")
    }
    if let Value::Array(plus) = op_concat(a.clone(), b.clone()){
        array_almost_equal(plus, expected); 
    } else {
        assert!(false);
    }

    // Power2
    let expected = Array2d::from_iter([
        0.0, 0.25, 1.0,  
        4.0, 0.0, 0.0
    ].iter(), 2, 3).unwrap();
    if let NameValue::Array(n) = op_name_power(na.clone(), NameValue::Scalar("2".to_string())) {
        println!("{n:?}");
        names_equal(&n, &["a1^2", "a2^2", "a3^2"]);
    } else {
        panic!("couldn't get names for power op")
    }
    let power_value = ops_common::op_power(a.clone(), Value::Scalar(2.0));
    if let Value::Array(power2) = power_value {
        array_almost_equal(power2, expected); 
    } else {
        assert!(false);
    }

    // Plus cat
    let (reference_names, feature_name, c) = get_cat_fortest(Some(2));
    let mut reference_names = reference_names.iter().map(|n|n.as_str()).collect::<Vec<_>>();
    let expected = Array2d::from_iter([
        0.0, 0.5, 1.0, 0.0, 
        2.0, 0.0, 0.0, 1.0
    ].iter(), 2, 4).unwrap();
    let plus_names = op_name_plus(na, NameValue::cats_from_value(feature_name, c.clone()).unwrap());
    
    if let NameValue::Array(n) = plus_names {
        let mut all_names = vec!["a1", "a2", "a3"];
        all_names.append(&mut reference_names);
        println!("{n:?}");
        names_equal(&n, &all_names);
    } else {
        panic!("couldn't get names for power op")
    }
    let plus = op_concat(a.clone(), c.clone());
    println!("{plus:?}");
    if let Value::Array(plus) = plus {
        array_almost_equal(plus, expected); 
    } else {
        assert!(false);
    }
}

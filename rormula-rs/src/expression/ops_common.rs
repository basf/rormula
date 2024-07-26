use crate::{
    array::{Array2d, MemOrder},
    result::RoResult,
    roerr,
};
use std::mem;

use super::Value;

pub fn unique_cats(cats: &[String]) -> RoResult<(Vec<&String>, &String)> {
    let mut unique = cats.iter().collect::<Vec<_>>();
    unique.sort();
    unique.dedup();
    let removed_cat = unique.pop().ok_or_else(|| roerr!("cats are empty?",))?;
    Ok((unique, removed_cat))
}

pub fn cat_to_dummy<M: MemOrder>(c: Value<M>) -> RoResult<Value<M>> {
    if let Value::Cats(cats) = c {
        let (unique, removed_cat) = unique_cats(&cats)?;
        let (n_rows, n_cols) = (cats.len(), unique.len());
        let mut dummy_encoding = Array2d::zeros(n_rows, n_cols);
        for (row, cat) in cats.iter().enumerate() {
            if cat != removed_cat {
                let col = unique.iter().position(|cat_name| *cat_name == cat).unwrap();
                dummy_encoding.set(row, col, 1.0);
            }
        }
        Ok(Value::Array(dummy_encoding))
    } else {
        Ok(c)
    }
}

pub fn op_componentwise_array<M: MemOrder>(
    a: Array2d<M>,
    b: Array2d<M>,
    op: &impl Fn(f64, f64) -> f64,
) -> RoResult<Array2d<M>> {
    a.componentwise(b, op)
}

pub fn op_scalar<M: MemOrder + Default>(
    a: Value<M>,
    b: Value<M>,
    op: &impl Fn(f64, f64) -> f64,
) -> Value<M> {
    let arr_vs_sc = |arr: &mut Array2d<M>, sc| {
        arr.elt_mutate(&|elt| op(elt, sc));
        Value::Array(mem::take(arr))
    };
    let sc_vs_arr = |sc, arr: &mut Array2d<M>| {
        arr.elt_mutate(&|elt| op(sc, elt));
        Value::Array(mem::take(arr))
    };
    let sc_vs_sc = |sc1, sc2| Value::Scalar(op(sc1, sc2));

    match (a, b) {
        (Value::Array(mut arr), Value::Scalar(sc)) => arr_vs_sc(&mut arr, sc),
        (Value::Scalar(sc), Value::Array(mut arr)) => sc_vs_arr(sc, &mut arr),
        (Value::Scalar(sc1), Value::Scalar(sc2)) => sc_vs_sc(sc1, sc2),
        _ => Value::Error(
            "scalar op can only be applied to matrix and scalar or scalar and scalar".to_string(),
        ),
    }
}

pub fn op_power<M: MemOrder + Default>(a: Value<M>, b: Value<M>) -> Value<M> {
    op_scalar(a, b, &|x, y| x.powf(y))
}

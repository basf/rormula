use crate::{array::Array2d, result::RoResult, roerr};
use std::mem;

use super::Value;

pub fn unique_cats(cats: &[String]) -> RoResult<(Vec<&String>, &String)> {
    let mut unique = cats.iter().collect::<Vec<_>>();
    unique.sort();
    unique.dedup();
    let removed_cat = unique.pop().ok_or_else(|| roerr!("cats are empty?",))?;
    Ok((unique, removed_cat))
}

pub fn cat_to_dummy(c: Value) -> RoResult<Value> {
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

pub fn op_componentwise_array(
    mut a: Array2d,
    b: Array2d,
    op: &impl Fn(f64, f64) -> f64,
) -> RoResult<Array2d> {
    if a.n_rows == b.n_rows {
        #[cfg(feature = "print_timings")]
        let now = std::time::Instant::now();

        let n_initial_cols_a = a.n_cols;
        for b_col in 0..b.n_cols {
            let mul_col_with_bcol = |row_idx: usize, x: f64| op(x, b.get(row_idx, b_col));
            if b_col == b.n_cols - 1 {
                // last col of b -> re-use memory of a
                for a_col in 0..n_initial_cols_a {
                    a.column_mutate(a_col, &mul_col_with_bcol);
                }
            } else {
                // not last col of b -> append to a
                for a_col in 0..n_initial_cols_a {
                    let mut new_col = a.column_copy(a_col);
                    new_col.column_mutate(0, &mul_col_with_bcol);
                    a = a.concatenate_cols(new_col)?;
                }
            }
        }
        let n_elts = a.data.len();
        a.data.rotate_right(n_elts - n_initial_cols_a * a.n_rows);

        #[cfg(feature = "print_timings")]
        eprintln!("colon op {}", now.elapsed().as_nanos());

        Ok(a)
    } else {
        Err(roerr!(
            "number of rows don't match, {}, {}",
            a.n_rows,
            b.n_rows
        ))
    }
}

pub fn op_scalar(a: Value, b: Value, op: &impl Fn(f64, f64) -> f64) -> Value {
    let arr_vs_sc = |arr: &mut Array2d, sc| {
        for elt in &mut arr.data {
            *elt = op(*elt, sc);
        }
        Value::Array(mem::take(arr))
    };
    let sc_vs_arr = |sc, arr: &mut Array2d| {
        for elt in &mut arr.data {
            *elt = op(sc, *elt);
        }
        Value::Array(mem::take(arr))
    };
    match (a, b) {
        (Value::Array(mut arr), Value::Scalar(sc)) => arr_vs_sc(&mut arr, sc),
        (Value::Scalar(sc), Value::Array(mut arr)) => sc_vs_arr(sc, &mut arr),
        _ => Value::Error("power can only be applied to matrix and skalar".to_string()),
    }
}

pub fn op_power(a: Value, b: Value) -> Value {
    op_scalar(a, b, &|x, y| x.powf(y))
}

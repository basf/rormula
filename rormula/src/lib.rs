use numpy::{
    ndarray::{s, Array2, ArrayView1},
    IntoPyArray, PyArray2, PyReadonlyArray2,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
pub use rormula_rs::exmex::prelude::*;
pub use rormula_rs::exmex::ExError;
use rormula_rs::expression::{ExprColCount, ExprNames, ExprWilkinson, NameValue, Value};
use rormula_rs::result::RoErr;
use rormula_rs::{array::Array2d, expression::ExprArithmetic};

fn ex_to_pyerr(e: ExError) -> PyErr {
    PyTypeError::new_err(e.msg().to_string())
}
fn ro_to_pyerr(e: RoErr) -> PyErr {
    PyValueError::new_err(e.msg().to_string())
}

#[pyfunction]
fn eval_arithmetic<'py>(
    py: Python<'py>,
    ror: &RormulaArithmetic,
    numerical_data: PyReadonlyArray2<f64>,
    numerical_cols: Vec<&'py str>,
) -> PyResult<&'py PyArray2<f64>> {
    let numerical_data = numerical_data.as_array();
    let vars = ror
        .expr
        .var_names()
        .iter()
        .map(|vn| {
            if let Some(num_idx) = numerical_cols.iter().position(|num_name| vn == num_name) {
                let s: ArrayView1<'_, f64> = numerical_data.slice(s![.., num_idx]);
                let n_rows = s.dim();
                Ok(Value::Array(
                    Array2d::from_iter(s.into_iter(), n_rows, 1).map_err(ro_to_pyerr)?,
                ))
            } else {
                Err(PyValueError::new_err(format!(
                    "did not find Variable {vn} in the data"
                )))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let vars: Vec<Value> = vars.into_iter().collect();

    if vars.len() != ror.expr.var_names().len() {
        Err(PyValueError::new_err(
            "there is a column missing for a variable in the formula",
        ))
    } else {
        let result_data = ror.expr.eval_vec(vars).map_err(ex_to_pyerr)?;

        match result_data {
            Value::Array(a) => {
                let mut pya = Array2::<f64>::ones([a.n_rows, a.n_cols]);
                for col in 0..a.n_cols {
                    for row in 0..a.n_rows {
                        pya[(row, col)] = a.get(row, col);
                    }
                }
                let res = pya.into_pyarray(py);

                Ok(res)
            }
            Value::Cats(_) => Err(PyValueError::new_err("result cannot be cat".to_string())),
            Value::Scalar(s) => Err(PyValueError::new_err(format!(
                "result cannot be skalar but got {s}"
            ))),
            Value::Error(e) => Err(PyValueError::new_err(format!("computation failed, {e:?}"))),
        }
    }
}
#[pyfunction]
fn eval_wilkinson<'py>(
    py: Python<'py>,
    ror: &RormulaWilkinson,
    numerical_data: PyReadonlyArray2<f64>,
    numerical_cols: Vec<&'py str>,
    cat_data: PyReadonlyArray2<PyObject>,
    cat_cols: Vec<&'py str>,
    skip_names: bool,
) -> PyResult<(Option<Vec<String>>, &'py PyArray2<f64>)> {
    let numerical_data = numerical_data.as_array();
    let cat_data = cat_data.as_array();
    let vars = ror
        .expr
        .var_names()
        .iter()
        .map(|vn| {
            if let Some(num_idx) = numerical_cols.iter().position(|num_name| vn == num_name) {
                let s: ArrayView1<'_, f64> = numerical_data.slice(s![.., num_idx]);
                let n_rows = s.dim();
                let names = if skip_names {
                    None
                } else {
                    Some(NameValue::Array(vec![numerical_cols[num_idx].to_string()]))
                };
                Ok((
                    names,
                    Value::Array(
                        Array2d::from_iter(s.into_iter(), n_rows, 1).map_err(ro_to_pyerr)?,
                    ),
                ))
            } else if let Some(cat_idx) = cat_cols.iter().position(|cat_name| vn == cat_name) {
                let col: ArrayView1<'_, Py<PyAny>> = cat_data.slice(s![.., cat_idx]);
                let col = col
                    .iter()
                    .map(|s: &pyo3::Py<pyo3::PyAny>| Ok(s.extract::<&str>(py)?.to_string()))
                    .collect::<PyResult<Vec<_>>>()?;
                let x = Value::Cats(col);
                let feature_name = if skip_names {
                    None
                } else {
                    Some(
                        NameValue::cats_from_value(cat_cols[cat_idx].to_string(), x.clone())
                            .unwrap(),
                    )
                };
                Ok((feature_name, x))
            } else {
                Err(PyValueError::new_err(format!(
                    "did not find Variable {vn} in the data"
                )))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let (vars_name, mut vars): (Vec<Option<NameValue>>, Vec<Value>) = vars.into_iter().unzip();
    let vars_name: Vec<NameValue> = vars_name.into_iter().flatten().collect();

    if vars.len() != ror.expr.var_names().len() {
        Err(PyValueError::new_err(
            "there is a column missing for a variable in the formula",
        ))
    } else {
        let count_vars = vec![1; vars.len()];
        let n_cols = ror.expr_count.eval(&count_vars).map_err(ex_to_pyerr)?;
        let var_indices_ordered = ror.expr.var_indices_ordered();
        // increase capacity of first array
        for var_idx in var_indices_ordered {
            if let Value::Array(arr) = &mut vars[var_idx] {
                arr.capacity = Some(n_cols * arr.n_rows - arr.data.len());
                break;
            }
        }
        let result_data = ror.expr.eval_vec(vars).map_err(ex_to_pyerr)?;
        let result_names = if !vars_name.is_empty() {
            Some(ror.expr_names.eval_vec(vars_name).map_err(ex_to_pyerr)?)
        } else {
            None
        };

        match result_data {
            Value::Array(a) => {
                let names = if let Some(NameValue::Array(mut names)) = result_names {
                    names.insert(0, "Intercept".to_string());
                    Some(names)
                } else {
                    None
                };
                let mut pya = Array2::<f64>::ones([a.n_rows, a.n_cols + 1]);
                for col in 0..a.n_cols {
                    for row in 0..a.n_rows {
                        pya[(row, col + 1)] = a.get(row, col);
                    }
                }
                let res = pya.into_pyarray(py);

                Ok((names, res))
            }
            Value::Cats(_) => Err(PyValueError::new_err("result cannot be cat".to_string())),
            Value::Scalar(s) => Err(PyValueError::new_err(format!(
                "result cannot be skalar but got {s}"
            ))),
            Value::Error(e) => Err(PyValueError::new_err(format!("computation failed, {e:?}"))),
        }
    }
}

#[pyfunction]
fn parse_arithmetic(s: &str) -> PyResult<RormulaArithmetic> {
    Ok(RormulaArithmetic {
        expr: ExprArithmetic::parse(s).map_err(ex_to_pyerr)?,
    })
}
#[derive(Debug)]
#[pyclass]
struct RormulaArithmetic {
    expr: ExprArithmetic,
}

#[derive(Debug)]
#[pyclass]
struct RormulaWilkinson {
    expr: ExprWilkinson,
    expr_names: ExprNames,
    expr_count: ExprColCount,
}
#[pyfunction]
fn parse_wilkinson(s: &str) -> PyResult<RormulaWilkinson> {
    Ok(RormulaWilkinson {
        expr: ExprWilkinson::parse(s).map_err(ex_to_pyerr)?,
        expr_names: ExprNames::parse(s).map_err(ex_to_pyerr)?,
        expr_count: ExprColCount::parse(s).map_err(ex_to_pyerr)?,
    })
}

#[pymodule]
fn _rormula(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_wilkinson, m)?)?;
    m.add_function(wrap_pyfunction!(eval_wilkinson, m)?)?;
    m.add_function(wrap_pyfunction!(parse_arithmetic, m)?)?;
    m.add_function(wrap_pyfunction!(eval_arithmetic, m)?)?;
    m.add_class::<RormulaWilkinson>()?;
    m.add_class::<RormulaArithmetic>()?;
    Ok(())
}

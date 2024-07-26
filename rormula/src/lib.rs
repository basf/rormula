use numpy::{
    ndarray::{concatenate, s, Array2, ArrayView1, Axis, Dim},
    IntoPyArray, PyArray2, PyReadonlyArray2,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyList,
};
pub use rormula_rs::exmex::prelude::*;
pub use rormula_rs::exmex::ExError;
use rormula_rs::{
    array::Array2d,
    expression::{has_row_change_op, ExprArithmetic},
};
use rormula_rs::{array::DefaultOrder, result::RoErr};
use rormula_rs::{
    expression::{ExprColCount, ExprNames, ExprWilkinson, NameValue, Value},
    timing,
};

fn ex_to_pyerr(e: ExError) -> PyErr {
    PyTypeError::new_err(e.msg().to_string())
}
fn ro_to_pyerr(e: RoErr) -> PyErr {
    PyValueError::new_err(e.msg().to_string())
}

fn find_col(cols: &Bound<'_, PyList>, needle: &str) -> Option<usize> {
    cols.iter().position(|num_name| {
        let num_name = num_name.extract::<&str>();
        if let Ok(num_name) = num_name {
            num_name == needle
        } else {
            false
        }
    })
}

#[pyfunction]
fn eval_arithmetic<'py>(
    py: Python<'py>,
    ror: &Arithmetic,
    numerical_data: PyReadonlyArray2<f64>,
    numerical_cols: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let numerical_data = numerical_data.as_array();
    let vars = ror
        .expr
        .var_names()
        .iter()
        .map(|vn: &String| {
            if let Some(num_idx) = find_col(numerical_cols, vn) {
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
    let vars: Vec<Value<DefaultOrder>> = vars.into_iter().collect();

    if vars.len() != ror.expr.var_names().len() {
        Err(PyValueError::new_err(
            "there is a column missing for a variable in the formula",
        ))
    } else {
        let result_data = ror.expr.eval_vec(vars).map_err(ex_to_pyerr)?;

        match result_data {
            Value::Array(a) => {
                let mut pya = Array2::<f64>::ones([a.n_rows(), a.n_cols()]);
                for col in 0..a.n_cols() {
                    for row in 0..a.n_rows() {
                        pya[(row, col)] = a.get(row, col);
                    }
                }
                let res = pya.into_pyarray_bound(py);

                Ok(res)
            }
            Value::RowInds(row_inds) => {
                let mut pya = Array2::<f64>::ones([row_inds.len(), 1]);
                for row in 0..row_inds.len() {
                    pya[(row, 0)] = row_inds[row] as f64;
                }
                let res = pya.into_pyarray_bound(py);
                Ok(res)
            }
            Value::Scalar(s) => Ok(Array2::<f64>::from_elem((1, 1), s).into_pyarray_bound(py)),
            Value::Cats(_) => Err(PyValueError::new_err("result cannot be cat".to_string())),
            Value::Error(e) => Err(PyValueError::new_err(format!("computation failed, {e:?}"))),
        }
    }
}

type WilkonsonReturnType<'py> = (Option<Vec<String>>, Bound<'py, PyArray2<f64>>);

#[pyfunction]
fn eval_wilkinson<'py>(
    py: Python<'py>,
    ror: &Wilkinson,
    numerical_data: PyReadonlyArray2<f64>,
    numerical_cols: &Bound<'py, PyList>,
    cat_data: PyReadonlyArray2<PyObject>,
    cat_cols: &Bound<'py, PyList>,
    skip_names: bool,
) -> PyResult<WilkonsonReturnType<'py>> {
    let numerical_data = numerical_data.as_array();
    let cat_data = cat_data.as_array();

    let vars = timing!(
        ror.expr
            .var_names()
            .iter()
            .map(|vn| {
                if let Some(num_idx) = find_col(numerical_cols, vn) {
                    let s: ArrayView1<'_, f64> = numerical_data.slice(s![.., num_idx]);
                    let n_rows = s.dim();
                    let names = if skip_names {
                        None
                    } else {
                        Some(NameValue::Array(vec![numerical_cols
                            .get_item(num_idx)?
                            .extract::<String>()?]))
                    };
                    timing!(
                        Ok((
                            names,
                            Value::Array(
                                Array2d::from_vec(s.to_vec(), n_rows, 1).map_err(ro_to_pyerr)?,
                            ),
                        )),
                        "arr from pyarray"
                    )
                } else if let Some(cat_idx) = find_col(cat_cols, vn) {
                    let col: ArrayView1<'_, Py<PyAny>> = cat_data.slice(s![.., cat_idx]);
                    let col = timing!(
                        col.iter()
                            .map(|s: &pyo3::Py<pyo3::PyAny>| Ok(s.extract::<&str>(py)?.to_string()))
                            .collect::<PyResult<Vec<_>>>()?,
                        "categorical conversion"
                    );
                    let x = Value::Cats(col);
                    let feature_name = if skip_names {
                        None
                    } else {
                        Some(
                            NameValue::cats_from_value(
                                cat_cols.get_item(cat_idx)?.extract::<String>()?,
                                x.clone(),
                            )
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
            .collect::<PyResult<Vec<_>>>()?,
        "vars"
    );
    let (vars_name, mut vars): (Vec<Option<NameValue>>, Vec<Value<DefaultOrder>>) =
        vars.into_iter().unzip();
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
                arr.set_capacity(n_cols * arr.n_rows() - arr.len());
                break;
            }
        }
        let result_data = ror.expr.eval_vec(vars).map_err(ex_to_pyerr)?;
        let result_names = if !vars_name.is_empty() {
            Some(ror.expr_names.eval_vec(vars_name).map_err(ex_to_pyerr)?)
        } else {
            None
        };

        timing!(
            match result_data {
                Value::Array(a) => {
                    let names = if let Some(NameValue::Array(mut names)) = result_names {
                        names.insert(0, "Intercept".to_string());
                        Some(names)
                    } else {
                        None
                    };
                    let intercept = timing!(Array2::ones(Dim([a.n_rows(), 1])), "intercept alloc");

                    let pya = timing!(a.to_ndarray().map_err(ro_to_pyerr)?, "to ndarray");
                    let pya = timing!(concatenate![Axis(1), intercept, pya], "intercept");
                    let res = timing!(pya.into_pyarray_bound(py), "into bound");

                    Ok((names, res))
                }
                Value::Cats(_) => Err(PyValueError::new_err("result cannot be cat".to_string())),
                Value::RowInds(_) => Err(PyValueError::new_err(
                    "result cannot be row indices".to_string(),
                )),
                Value::Scalar(s) => Err(PyValueError::new_err(format!(
                    "result cannot be skalar but got {s}"
                ))),
                Value::Error(e) => Err(PyValueError::new_err(format!("computation failed, {e:?}"))),
            },
            "convert res"
        )
    }
}

#[pyfunction]
fn parse_arithmetic(s: &str) -> PyResult<Arithmetic> {
    Ok(Arithmetic {
        expr: ExprArithmetic::parse(s).map_err(ex_to_pyerr)?,
    })
}
#[derive(Debug)]
#[pyclass]
struct Arithmetic {
    expr: ExprArithmetic,
}
#[pymethods]
impl Arithmetic {
    pub fn has_row_change_op(&self) -> PyResult<bool> {
        Ok(has_row_change_op(&self.expr))
    }
    pub fn unparse(&self) -> PyResult<String> {
        Ok(self.expr.unparse().to_string())
    }
}

#[derive(Debug)]
#[pyclass]
struct Wilkinson {
    expr: ExprWilkinson,
    expr_names: ExprNames,
    expr_count: ExprColCount,
}
#[pyfunction]
fn parse_wilkinson(s: &str) -> PyResult<Wilkinson> {
    Ok(timing!(
        Wilkinson {
            expr: ExprWilkinson::parse(s).map_err(ex_to_pyerr)?,
            expr_names: ExprNames::parse(s).map_err(ex_to_pyerr)?,
            expr_count: ExprColCount::parse(s).map_err(ex_to_pyerr)?,
        },
        "parse"
    ))
}

#[pymodule]
fn rormula(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_wilkinson, m)?)?;
    m.add_function(wrap_pyfunction!(eval_wilkinson, m)?)?;
    m.add_function(wrap_pyfunction!(parse_arithmetic, m)?)?;
    m.add_function(wrap_pyfunction!(eval_arithmetic, m)?)?;
    m.add_class::<Wilkinson>()?;
    m.add_class::<Arithmetic>()?;
    Ok(())
}

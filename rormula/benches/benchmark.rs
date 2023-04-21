use criterion::{black_box, criterion_group, criterion_main, Criterion};
use numpy::{
    ndarray::{concatenate, Array1, Array2, Axis},
    pyo3::Python,
    PyArray2, PyReadonlyArray2,
};
use pyo3::PyResult;
use rormula_rs::array::Array2d;
use std::mem;

/// Workaround from https://github.com/PyO3/pyo3/issues/1554
pub fn initialize_python() -> PyResult<()> {
    // Due to https://github.com/ContinuumIO/anaconda-issues/issues/11439,
    // we first need to set PYTHONHOME. To do so, we will look for whatever
    // directory on PATH currently has python.exe.
    let python_exe = which::which("python").unwrap();
    let python_home = python_exe.parent().unwrap();

    // The Python C API uses null-terminated UTF-16 strings, so we need to
    // encode the path into that format here.
    // We could use the Windows FFI modules provided in the standard library,
    // but we want this to work cross-platform, so we do things more manually.
    let mut python_home = python_home
        .to_str()
        .unwrap()
        .encode_utf16()
        .collect::<Vec<u16>>();
    python_home.push(0);
    unsafe {
        pyo3::ffi::Py_SetPythonHome(python_home.as_ptr());
    }

    // Once we've set the configuration we need, we can go on and manually
    // initialize PyO3.
    pyo3::prepare_freethreaded_python();

    Ok(())
}

fn from_pyarray_nd(pyarray: &PyReadonlyArray2<f64>) -> Array2<f64> {
    pyarray.to_owned_array()
}
fn from_pyarray(pyarray: &PyReadonlyArray2<f64>) -> Array2d {
    let view = pyarray.as_array();
    let n_rows = view.shape()[0];
    let n_cols = view.shape()[1];
    Array2d::from_iter(view.into_iter(), n_rows, n_cols).unwrap()
}

fn add_col_nd(arr1: Array2<f64>, arr2: Array2<f64>) -> Array2<f64> {
    concatenate(Axis(1), &[arr1.view(), arr2.view()]).unwrap()
}
fn add_col(arr1: Array2d, arr2: Array2d) -> Array2d {
    arr1.concatenate_cols(arr2).unwrap()
}

fn compute_nd(arr1: &mut Array1<f64>, arr2: &Array1<f64>) -> Array1<f64> {
    arr1.clone() * arr2
}
fn compute(arr1: &mut Array2d, arr2: &Array2d) -> Array2d {
    let col_idx = 0;

    let mutate = |idx, val| val * arr2.get(idx, col_idx);
    arr1.column_mutate(col_idx, &mutate);

    mem::take(arr1)
}

fn criterion_benchmark(c: &mut Criterion) {
    initialize_python().unwrap();
    Python::with_gil(|py| {
        let pyarray = PyArray2::<f64>::zeros(py, (500000, 3), false);
        let readonly = pyarray.readonly();
        c.bench_function("create nd", |b| {
            b.iter(|| from_pyarray_nd(black_box(&readonly)))
        });
        c.bench_function("create array", |b| {
            b.iter(|| from_pyarray(black_box(&readonly)))
        });
        let arr_nd = from_pyarray_nd(&readonly);
        let arr = from_pyarray(&readonly);

        c.bench_function("conc nd", |b| {
            b.iter(|| add_col_nd(black_box(arr_nd.clone()), arr_nd.clone()))
        });
        c.bench_function("conc array", |b| {
            b.iter(|| add_col(black_box(arr.clone()), black_box(arr.clone())))
        });
        let col_0 = arr_nd.column(0).to_owned();
        let mut col_0_mut = col_0.clone();
        c.bench_function("mutate nd", |b| {
            b.iter(|| col_0_mut = compute_nd(black_box(&mut col_0_mut), black_box(&col_0)))
        });
        let mut arr_mut = arr.clone();
        c.bench_function("mutate", |b| {
            b.iter(|| arr_mut = compute(black_box(&mut arr_mut), black_box(&arr)))
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

use std::fmt::Debug;

use crate::{result::RoResult, roerr};

/// Col major ordering which is non-standard!
/// column major means the next element is the next row in memory, i.e., you iterate along the column
#[derive(Default)]
pub struct Array2d {
    pub data: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub capacity: Option<usize>,
}
impl Array2d {
    pub fn from_iter<'a>(
        mut row_major_it: impl Iterator<Item = &'a f64>,
        n_rows: usize,
        n_cols: usize,
    ) -> RoResult<Self> {
        let mut result = Self::zeros(n_rows, n_cols);

        for row in 0..n_rows {
            for col in 0..n_cols {
                result.set(
                    row,
                    col,
                    *row_major_it
                        .next()
                        .ok_or_else(|| roerr!("dimension of input data does not fit",))?,
                );
            }
        }
        if row_major_it.next().is_none() {
            Ok(result)
        } else {
            Err(roerr!("input iterator not fully consumed",))
        }
    }
    pub fn column_copy(&self, col_idx: usize) -> Self {
        let data = self.data[(col_idx * self.n_rows)..((col_idx + 1) * self.n_rows)].to_vec();
        Self {
            data,
            n_rows: self.n_rows,
            n_cols: 1,
            capacity: None,
        }
    }
    pub fn ones(n_rows: usize, n_cols: usize) -> Self {
        let data = vec![1.0; n_rows * n_cols];
        Self {
            data,
            n_rows,
            n_cols,
            capacity: None,
        }
    }
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        let data = vec![0.0; n_rows * n_cols];
        Self {
            data,
            n_rows,
            n_cols,
            capacity: None,
        }
    }
    #[inline]
    pub fn set(&mut self, row_idx: usize, col_idx: usize, value: f64) {
        self.data[row_idx + self.n_rows * col_idx] = value;
    }
    #[inline]
    pub fn get(&self, row_idx: usize, col_idx: usize) -> f64 {
        self.data[row_idx + self.n_rows * col_idx]
    }
    pub fn concatenate_cols(mut self, mut other: Self) -> RoResult<Self> {
        if self.n_rows == other.n_rows {
            let n_other_cols = other.n_cols;
            self.data.append(&mut other.data);
            Ok(Self {
                data: self.data,
                n_rows: self.n_rows,
                n_cols: self.n_cols + n_other_cols,
                capacity: None,
            })
        } else {
            Err(roerr!(
                "not matching number of rows, {} vs {}",
                self.n_rows,
                other.n_rows
            ))
        }
    }
    pub fn column_mutate(&mut self, col_idx: usize, mutate: &impl Fn(usize, f64) -> f64) {
        for row in 0..self.n_rows {
            self.set(row, col_idx, mutate(row, self.get(row, col_idx)));
        }
    }
    pub fn elt_mutate(&mut self, mutate: &impl Fn(f64) -> f64) {
        for elt in &mut self.data {
            *elt = mutate(*elt);
        }
    }
}
impl Debug for Array2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = "".to_string();
        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                let v = self.get(row, col);
                s = format!("{s} {v:0.3}")
            }
            s = format!("{s}\n");
        }
        f.write_str(s.as_str())
    }
}
impl Clone for Array2d {
    fn clone(&self) -> Self {
        let data = if let Some(capa) = self.capacity {
            let mut data = self.data.clone();
            data.reserve(capa);
            data
        } else {
            self.data.clone()
        };
        Self {
            data,
            n_cols: self.n_cols,
            n_rows: self.n_rows,
            capacity: self.capacity,
        }
    }
}
impl PartialEq for Array2d {
    fn eq(&self, other: &Self) -> bool {
        if self.n_cols != other.n_cols || self.n_rows != other.n_rows {
            false
        } else {
            for (s, o) in self.data.iter().zip(other.data.iter()) {
                if (s - o).abs() > 1e-12 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_capa() {
    fn test(a: Array2d) {
        println!("{}", a.data.capacity());
    }
    let mut a = Array2d::from_iter([0.0, 2.0, 3.0, 4.0].iter(), 1, 4).unwrap();
    a.data.reserve(1000000);
    test(a);
}

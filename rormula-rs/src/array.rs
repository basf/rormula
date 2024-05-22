use std::fmt::Debug;

use crate::{result::RoResult, roerr, timing};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemOrder {
    #[default]
    ColMajor,
    RowMajor,
}

/// Col major ordering which is non-standard!
/// column major means the next element is the next row in memory, i.e., you iterate along the column
#[derive(Default)]
pub struct Array2d {
    data: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    capacity: Option<usize>,
    order: MemOrder,
}
impl Array2d {
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
    pub fn capacity(&self) -> Option<usize> {
        self.capacity
    }
    pub fn order(&self) -> MemOrder {
        self.order
    }
    pub fn set_capacity(&mut self, capa: usize) {
        self.capacity = Some(capa);
    }
    pub fn new(data: Vec<f64>, n_rows: usize, n_cols: usize, order: MemOrder) -> RoResult<Self> {
        if data.len() != n_rows * n_cols {
            Err(roerr!("dimension of input data does not fit"))
        } else {
            Ok(Self {
                data,
                n_rows,
                n_cols,
                capacity: None,
                order,
            })
        }
    }
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
        match self.order {
            MemOrder::ColMajor => {
                let data =
                    self.data[(col_idx * self.n_rows)..((col_idx + 1) * self.n_rows)].to_vec();
                Self {
                    data,
                    n_rows: self.n_rows,
                    n_cols: 1,
                    capacity: None,
                    order: MemOrder::ColMajor,
                }
            }
            MemOrder::RowMajor => {
                let mut data = Vec::with_capacity(self.n_rows);
                for row in 0..self.n_rows {
                    data.push(self.get(row, col_idx));
                }
                Self {
                    data,
                    n_rows: self.n_rows,
                    n_cols: 1,
                    capacity: None,
                    order: MemOrder::RowMajor,
                }
            }
        }
    }
    pub fn ones(n_rows: usize, n_cols: usize) -> Self {
        let data = vec![1.0; n_rows * n_cols];
        Self {
            data,
            n_rows,
            n_cols,
            capacity: None,
            order: MemOrder::default(),
        }
    }
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        let data = vec![0.0; n_rows * n_cols];
        Self {
            data,
            n_rows,
            n_cols,
            capacity: None,
            order: MemOrder::default(),
        }
    }
    #[inline]
    pub fn set(&mut self, row_idx: usize, col_idx: usize, value: f64) {
        match self.order {
            MemOrder::RowMajor => self.data[row_idx * self.n_cols + col_idx] = value,
            MemOrder::ColMajor => self.data[row_idx + self.n_rows * col_idx] = value,
        }
    }
    #[inline]
    pub fn get(&self, row_idx: usize, col_idx: usize) -> f64 {
        match self.order {
            MemOrder::RowMajor => self.data[row_idx * self.n_cols + col_idx],
            MemOrder::ColMajor => self.data[row_idx + self.n_rows * col_idx],
        }
    }
    pub fn concatenate_cols(mut self, mut other: Self) -> RoResult<Self> {
        timing!(
            {
                if other.order != self.order {
                    return Err(roerr!("order of arrays does not match",));
                }
                if self.n_rows == other.n_rows {
                    let n_cols = self.n_cols + other.n_cols;
                    match self.order {
                        MemOrder::ColMajor => {
                            self.data.append(&mut other.data);
                            Ok(Self {
                                data: self.data,
                                n_rows: self.n_rows,
                                n_cols,
                                capacity: self.capacity,
                                order: MemOrder::ColMajor,
                            })
                        }
                        MemOrder::RowMajor => {
                            timing!(self.data.resize(n_cols * self.n_rows, 0.0), "resize");
                            timing!(
                                for row in (1..self.n_rows).rev() {
                                    let src = row * self.n_cols;
                                    let dest = row * n_cols;
                                    self.data.copy_within(src..(src + self.n_cols), dest);
                                },
                                "within"
                            );
                            let n_old_cols = self.n_cols;
                            self.n_cols = n_cols;
                            timing!(
                                for row in 0..self.n_rows {
                                    for col in 0..other.n_cols {
                                        self.set(row, n_old_cols + col, other.get(row, col));
                                    }
                                },
                                "other"
                            );
                            Ok(self)
                        }
                    }
                } else {
                    Err(roerr!(
                        "not matching number of rows, {} vs {}",
                        self.n_rows,
                        other.n_rows
                    ))
                }
            },
            "conc"
        )
    }
    pub fn column_mutate(&mut self, col_idx: usize, mutate: &impl Fn(usize, f64) -> f64) {
        timing!(
            {
                for row in 0..self.n_rows {
                    self.set(row, col_idx, mutate(row, self.get(row, col_idx)));
                }
            },
            "colmut"
        );
    }
    pub fn elt_mutate(&mut self, mutate: &impl Fn(f64) -> f64) {
        for elt in &mut self.data {
            *elt = mutate(*elt);
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.data.iter().copied()
    }

    pub fn componentwise(mut self, b: Self, op: &impl Fn(f64, f64) -> f64) -> RoResult<Self> {
        if self.n_rows == b.n_rows {
            let n_initial_cols_a = self.n_cols;
            for b_col in 0..b.n_cols {
                let op_selfcol_with_bcol = |row_idx: usize, x: f64| op(x, b.get(row_idx, b_col));
                if b_col == b.n_cols - 1 {
                    // last col of b -> re-use memory of a
                    for a_col in 0..n_initial_cols_a {
                        self.column_mutate(a_col, &op_selfcol_with_bcol);
                    }
                } else {
                    // not last col of b -> append to a
                    for a_col in 0..n_initial_cols_a {
                        let mut new_col = self.column_copy(a_col);
                        new_col.column_mutate(0, &op_selfcol_with_bcol);
                        self = self.concatenate_cols(new_col)?;
                    }
                }
            }
            match self.order {
                MemOrder::ColMajor => {
                    let n_elts = self.data.len();
                    self.data
                        .rotate_right(n_elts - n_initial_cols_a * self.n_rows);
                    Ok(self)
                }
                MemOrder::RowMajor => Ok(self),
            }
        } else {
            Err(roerr!(
                "number of rows don't match, {}, {}",
                self.n_rows,
                b.n_rows
            ))
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
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
            order: self.order,
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

#[rustfmt::skip]
#[test]
fn test_default_colmutate() {
    let mut a = Array2d::from_iter(
        vec![
            1.0, 0.0, 
            1.0, 2.0, 
            1.0, 3.0, 
            1.0, 4.0].iter(),
         4,
         2
    ).unwrap();
    assert_eq!(a.get(0, 1), 0.0);
    assert_eq!(a.get(1, 1), 2.0);
    assert_eq!(a.get(2, 1), 3.0);
    assert_eq!(a.get(3, 1), 4.0);
    a.column_mutate(1, &|row_idx, val| row_idx as f64 + val + 1.0);
    assert_eq!(a.get(0, 1), 1.0);
    assert_eq!(a.get(1, 1), 4.0);
    assert_eq!(a.get(2, 1), 6.0);
    assert_eq!(a.get(3, 1), 8.0);

    let a = a.clone().concatenate_cols(a.clone()).unwrap();
    assert_eq!(a.get(0, 1), 1.0);
    assert_eq!(a.get(1, 1), 4.0);
    assert_eq!(a.get(2, 1), 6.0);
    assert_eq!(a.get(3, 1), 8.0);
    assert_eq!(a.get(0, 3), 1.0);
    assert_eq!(a.get(1, 3), 4.0);
    assert_eq!(a.get(2, 3), 6.0);
    assert_eq!(a.get(3, 3), 8.0);

    let b = a.column_copy(1);
    println!("a \n{:?}", a);
    println!("b \n{:?}", b);
    let a = a.componentwise(b, &|x, y| x + y).unwrap();
    println!("a \n{:?}", a);
    assert_eq!(a.get(0, 1), 2.0);
    assert_eq!(a.get(1, 1), 8.0);
    assert_eq!(a.get(2, 1), 12.0);
    assert_eq!(a.get(3, 1), 16.0);
}

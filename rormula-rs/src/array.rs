use std::fmt::Debug;

use numpy::ndarray::{Array2, Dim, Shape, ShapeBuilder};

use crate::{
    result::{to_ro, RoResult},
    roerr, timing,
};

pub trait MemOrder: Default + Debug + Clone + PartialEq {
    fn get(data: &[f64], row_idx: usize, col_idx: usize, n_rows: usize, n_cols: usize) -> f64;
    fn set(
        data: &mut [f64],
        row_idx: usize,
        col_idx: usize,
        value: f64,
        n_rows: usize,
        n_cols: usize,
    );
    fn column_copy(data: &[f64], col_idx: usize, n_rows: usize, n_cols: usize) -> Vec<f64>;
    fn concat_cols(
        self_data: Vec<f64>,
        self_n_rows: usize,
        self_n_cols: usize,
        other_data: Vec<f64>,
        other_n_rows: usize,
        other_n_cols: usize,
    ) -> RoResult<(Vec<f64>, usize, usize)>;
    fn pproc_compontentwise(data: Vec<f64>, n_initial_cols: usize, n_rows: usize) -> Vec<f64>;
    fn to_ndarray(data: Vec<f64>, n_rows: usize, n_cols: usize) -> RoResult<Array2<f64>>;
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct ColMajor;
impl MemOrder for ColMajor {
    fn get(data: &[f64], row_idx: usize, col_idx: usize, n_rows: usize, _: usize) -> f64 {
        data[row_idx + n_rows * col_idx]
    }
    fn set(data: &mut [f64], row_idx: usize, col_idx: usize, value: f64, n_rows: usize, _: usize) {
        data[row_idx + n_rows * col_idx] = value;
    }
    fn column_copy(data: &[f64], col_idx: usize, n_rows: usize, _: usize) -> Vec<f64> {
        data[(col_idx * n_rows)..((col_idx + 1) * n_rows)].to_vec()
    }
    fn concat_cols(
        self_data: Vec<f64>,
        self_n_rows: usize,
        self_n_cols: usize,
        other_data: Vec<f64>,
        other_n_rows: usize,
        other_n_cols: usize,
    ) -> RoResult<(Vec<f64>, usize, usize)> {
        if self_n_rows == other_n_rows {
            let n_cols = self_n_cols + other_n_cols;
            let mut data = self_data;
            data.extend(other_data);
            Ok((data, self_n_rows, n_cols))
        } else {
            Err(roerr!(
                "not matching number of rows, {} vs {}",
                self_n_rows,
                other_n_rows
            ))
        }
    }
    fn pproc_compontentwise(mut data: Vec<f64>, n_initial_cols: usize, n_rows: usize) -> Vec<f64> {
        let n_elts = data.len();
        data.rotate_right(n_elts - n_initial_cols * n_rows);
        data
    }
    fn to_ndarray(data: Vec<f64>, n_rows: usize, n_cols: usize) -> RoResult<Array2<f64>> {
        let sh = Shape::from(Dim([n_rows, n_cols])).f();
        Array2::from_shape_vec(sh, data).map_err(to_ro)
    }
}
#[derive(Default, Debug, Clone, PartialEq)]
pub struct RowMajor;
impl MemOrder for RowMajor {
    fn get(data: &[f64], row_idx: usize, col_idx: usize, _: usize, n_cols: usize) -> f64 {
        data[row_idx * n_cols + col_idx]
    }
    fn set(data: &mut [f64], row_idx: usize, col_idx: usize, value: f64, _: usize, n_cols: usize) {
        data[row_idx * n_cols + col_idx] = value;
    }
    fn column_copy(data: &[f64], col_idx: usize, n_rows: usize, n_cols: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            result.push(Self::get(data, row, col_idx, n_rows, n_cols));
        }
        result
    }
    fn concat_cols(
        self_data: Vec<f64>,
        self_n_rows: usize,
        self_n_cols: usize,
        other_data: Vec<f64>,
        other_n_rows: usize,
        other_n_cols: usize,
    ) -> RoResult<(Vec<f64>, usize, usize)> {
        if self_n_rows == other_n_rows {
            let n_cols = self_n_cols + other_n_cols;
            let mut data = self_data;
            data.resize(n_cols * self_n_rows, 0.0);
            timing!(
                for row in (1..self_n_rows).rev() {
                    let src = row * self_n_cols;
                    let dest = row * n_cols;
                    data.copy_within(src..(src + self_n_cols), dest);
                },
                "copy_within"
            );
            let n_old_cols = self_n_cols;
            timing!(
                for row in 0..self_n_rows {
                    for col in 0..other_n_cols {
                        Self::set(
                            &mut data,
                            row,
                            n_old_cols + col,
                            Self::get(&other_data, row, col, other_n_rows, other_n_cols),
                            self_n_rows,
                            n_cols,
                        )
                    }
                },
                "get/set"
            );
            Ok((data, self_n_rows, n_cols))
        } else {
            Err(roerr!(
                "not matching number of rows, {} vs {}",
                self_n_rows,
                other_n_rows
            ))
        }
    }
    fn pproc_compontentwise(data: Vec<f64>, _: usize, _: usize) -> Vec<f64> {
        data
    }

    fn to_ndarray(data: Vec<f64>, n_rows: usize, n_cols: usize) -> RoResult<Array2<f64>> {
        Array2::from_shape_vec(Dim([n_rows, n_cols]), data).map_err(to_ro)
    }
}

pub type DefaultOrder = ColMajor;

/// Col major ordering which is non-standard!
/// column major means the next element is the next row in memory, i.e., you iterate along the column
#[derive(Default)]
pub struct Array2d<M> {
    data: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    capacity: Option<usize>,
    phantom: std::marker::PhantomData<M>,
}
impl<M> Array2d<M>
where
    M: MemOrder,
{
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
    pub fn capacity(&self) -> Option<usize> {
        self.capacity
    }
    pub fn set_capacity(&mut self, capa: usize) {
        self.capacity = Some(capa);
    }
    pub fn new(data: Vec<f64>, n_rows: usize, n_cols: usize) -> RoResult<Self> {
        if data.len() != n_rows * n_cols {
            Err(roerr!("dimension of input data does not fit"))
        } else {
            Ok(Self {
                data,
                n_rows,
                n_cols,
                capacity: None,
                phantom: std::marker::PhantomData,
            })
        }
    }
    pub fn from_vec(data: Vec<f64>, n_rows: usize, n_cols: usize) -> RoResult<Self> {
        if data.len() != n_rows * n_cols {
            Err(roerr!("dimension of input data does not fit"))
        } else {
            Ok(Self {
                data,
                n_rows,
                n_cols,
                capacity: None,
                phantom: std::marker::PhantomData,
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
        let data = M::column_copy(&self.data, col_idx, self.n_rows, self.n_cols);
        Self {
            data,
            n_rows: self.n_rows,
            n_cols: 1,
            capacity: None,
            phantom: std::marker::PhantomData,
        }
    }
    pub fn ones(n_rows: usize, n_cols: usize) -> Self {
        let data = vec![1.0; n_rows * n_cols];
        Self {
            data,
            n_rows,
            n_cols,
            capacity: None,
            phantom: std::marker::PhantomData,
        }
    }
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        let data = vec![0.0; n_rows * n_cols];
        Self {
            data,
            n_rows,
            n_cols,
            capacity: None,
            phantom: std::marker::PhantomData,
        }
    }
    #[inline]
    pub fn set(&mut self, row_idx: usize, col_idx: usize, value: f64) {
        M::set(
            &mut self.data,
            row_idx,
            col_idx,
            value,
            self.n_rows,
            self.n_cols,
        );
    }
    #[inline]
    pub fn get(&self, row_idx: usize, col_idx: usize) -> f64 {
        M::get(&self.data, row_idx, col_idx, self.n_rows, self.n_cols)
    }
    pub fn concatenate_cols(self, other: Self) -> RoResult<Self> {
        let (data, n_rows, n_cols) = timing!({
            M::concat_cols(
                self.data,
                self.n_rows,
                self.n_cols,
                other.data,
                other.n_rows,
                other.n_cols,
            )
        })?;
        Ok(Self {
            data,
            n_rows,
            n_cols,
            capacity: self.capacity,
            phantom: std::marker::PhantomData,
        })
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
        timing!(
            if self.n_rows == b.n_rows {
                let n_initial_cols_a = self.n_cols;
                for b_col in 0..b.n_cols {
                    let op_selfcol_with_bcol =
                        |row_idx: usize, x: f64| op(x, b.get(row_idx, b_col));
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
                Ok(Self {
                    data: M::pproc_compontentwise(self.data, n_initial_cols_a, self.n_rows),
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                    capacity: self.capacity,
                    phantom: std::marker::PhantomData,
                })
            } else {
                Err(roerr!(
                    "number of rows don't match, {}, {}",
                    self.n_rows,
                    b.n_rows
                ))
            },
            "componentwise"
        )
    }
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn to_ndarray(self) -> RoResult<Array2<f64>> {
        M::to_ndarray(self.data, self.n_rows, self.n_cols)
    }
}
impl<M> Debug for Array2d<M>
where
    M: MemOrder,
{
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
impl<M> Clone for Array2d<M> {
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
            phantom: std::marker::PhantomData,
        }
    }
}
impl<M> PartialEq for Array2d<M> {
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
    fn test(a: Array2d<RowMajor>) {
        println!("{}", a.data.capacity());
    }
    let mut a = Array2d::from_iter([0.0, 2.0, 3.0, 4.0].iter(), 1, 4).unwrap();
    a.data.reserve(1000000);
    test(a);
}

#[test]
fn test_colmutate() {
    fn test<M>()
    where
        M: MemOrder,
    {
        let mut a =
            Array2d::<M>::from_iter(vec![1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0].iter(), 4, 2)
                .unwrap();
        println!("{:?}", a);
        println!("{:?}", a.data);
        assert_eq!(a.get(0, 1), 0.0);
        assert_eq!(a.get(1, 1), 2.0);
        assert_eq!(a.get(2, 1), 3.0);
        assert_eq!(a.get(3, 1), 4.0);
        a.column_mutate(1, &|row_idx, val| row_idx as f64 + val + 1.0);
        assert_eq!(a.get(0, 1), 1.0);
        assert_eq!(a.get(1, 1), 4.0);
        assert_eq!(a.get(2, 1), 6.0);
        assert_eq!(a.get(3, 1), 8.0);

        println!("{:?}", a);
        let a = a.clone().concatenate_cols(a.clone()).unwrap();
        println!("{:?}", a);
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
    test::<RowMajor>();
    test::<ColMajor>();
}

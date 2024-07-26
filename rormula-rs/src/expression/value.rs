use std::str::FromStr;

use crate::{
    array::{Array2d, MemOrder},
    result::RoErr,
    roerr,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Value<M>
where
    M: MemOrder,
{
    Array(Array2d<M>),
    RowInds(Vec<usize>),
    /// String is the name of the categorical
    Cats(Vec<String>),
    Scalar(f64),
    /// String is the error message
    Error(String),
}
impl<M: MemOrder> Default for Value<M> {
    fn default() -> Self {
        Self::Error("default".to_string())
    }
}
impl<M: MemOrder> FromStr for Value<M> {
    type Err = RoErr;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Value::Scalar(
            s.parse::<f64>()
                .map_err(|_| roerr!("could not parse {}", s))?,
        ))
    }
}
#[derive(Clone, Debug)]
pub enum NameValue {
    Cats((String, Vec<String>)),
    Array(Vec<String>),
    Scalar(String),
    Error(String),
}
impl NameValue {
    pub fn cats_from_value<M: MemOrder>(feature_name: String, cats: Value<M>) -> Option<Self> {
        if let Value::Cats(c) = cats {
            Some(Self::Cats((feature_name, c)))
        } else {
            None
        }
    }
}
impl Default for NameValue {
    fn default() -> Self {
        Self::Error("default".to_string())
    }
}
impl FromStr for NameValue {
    type Err = RoErr;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(NameValue::Scalar(
            // we parse to make sure it is actually a number
            s.parse::<f64>()
                .map_err(|_| roerr!("could not parse {}", s))?
                .to_string(),
        ))
    }
}

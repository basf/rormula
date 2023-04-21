use std::str::FromStr;

use crate::{array::Array2d, result::RoErr, roerr};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Vec<String> are the names of the columns, i.e., the resulting names of the new features
    Array(Array2d),
    /// String is the name of the categorical
    Cats(Vec<String>),
    Scalar(f64),
    /// String is the error message
    Error(String),
}
impl Default for Value {
    fn default() -> Self {
        Self::Error("default".to_string())
    }
}
impl FromStr for Value {
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
    pub fn cats_from_value(feature_name: String, cats: Value) -> Option<Self> {
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

use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};

/// This will be thrown at you if the somehting within Exmex went wrong. Ok, obviously it is not an
/// exception, so thrown needs to be understood figuratively.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct RoErr {
    msg: String,
}
impl RoErr {
    pub fn new(msg: &str) -> RoErr {
        RoErr {
            msg: msg.to_string(),
        }
    }
    pub fn msg(&self) -> &str {
        &self.msg
    }
}
impl Display for RoErr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for RoErr {}

/// Rormula's result type with [`RoError`](RoError) as error type.
pub type RoResult<U> = Result<U, RoErr>;

pub fn to_ro(err: impl Error) -> RoErr {
    RoErr::new(err.to_string().as_str())
}

/// Creates an [`RoError`](RoError) with a formatted message.
/// ```rust
/// # use std::error::Error;
/// use rormula_rs::{roerr, {result::RoErr}};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// assert_eq!(roerr!("some error {}", 1), RoErr::new(format!("some error {}", 1).as_str()));
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! roerr {
    ($s:literal) => {
        $crate::result::RoErr::new(format!($s).as_str())
    };
    ($s:literal, $( $exps:expr),*) => {
        $crate::result::RoErr::new(format!($s, $($exps,)*).as_str())
    }
}

pub mod array;
pub mod expression;
pub mod result;
pub use exmex;

#[macro_export]
macro_rules! timing {
    ($block:expr) => {{
        $block
    }};
    ($block:expr, $name:expr) => {{
        #[cfg(feature = "print_timings")]
        let now = std::time::Instant::now();
        let res = $block;
        #[cfg(feature = "print_timings")]
        eprintln!("{} {}", $name, now.elapsed().as_micros());
        res
    }};
}

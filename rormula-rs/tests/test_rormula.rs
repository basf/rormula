use exmex::Express;
use rormula_rs::{
    array::Array2d,
    expression::{ExprArithmetic, Value},
    expression::{ExprNames, ExprWilkinson, NameValue},
};

#[test]
fn test_wilkinson() {
    let n1 = NameValue::Array(vec!["n".to_string()]);
    let v1 = Value::Array(Array2d::from_iter([0.1, 0.2, 0.3].iter(), 3, 1).unwrap());
    let n2 = NameValue::Array(vec!["o".to_string()]);
    let v2 = Value::Array(Array2d::from_iter([0.4, 0.5, 0.6].iter(), 3, 1).unwrap());
    let s = "n+o+n";
    let expr = ExprWilkinson::parse(s).unwrap();
    match expr.eval_vec(vec![v1, v2]).unwrap() {
        Value::Array(a) => {
            a.data
                .iter()
                .zip([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3].iter())
                .for_each(|(x, y)| assert!((x - y).abs() < 1e-12));
        }
        Value::Error(e) => {
            println!("{e}");
            assert!(false)
        }
        _ => assert!(false),
    }
    let name_expr = ExprNames::parse(s).unwrap();
    match name_expr.eval(&[n1, n2]).unwrap() {
        NameValue::Array(names) => {
            assert_eq!(names, vec!["n", "o", "n"]);
        }
        _ => panic!("need names as result"),
    }
}

#[test]
fn test_more_formulas() {
    let cols = ["alpha", "beta", "gamma", "eta"];
    let formula_str = "(alpha + beta):(gamma + eta)";
    let expr = ExprWilkinson::parse(formula_str).unwrap();
    let vars = (0..cols.len())
        .map(|_| Value::Array(Array2d::zeros(5, 1)))
        .collect::<Vec<_>>();
    expr.eval_vec(vars).unwrap();
}

#[test]
fn test_arithmetic() {
    let cols = ["alpha", "beta", "gamma", "eta"];
    let s = "(3.0 * alpha + 1^beta) * (gamma - eta + eta) / 2.0";
    let expr = ExprArithmetic::parse(s).unwrap();
    let vars = (0..cols.len())
        .map(|_| Value::Array(Array2d::ones(5, 1)))
        .collect::<Vec<_>>();
    let res = expr.eval_vec(vars).unwrap();
    let s_ref = "x * -2.0";
    let expr_ref = ExprArithmetic::parse(s_ref).unwrap();
    let rev_val = expr_ref.eval(&[Value::Array(Array2d::ones(5, 1))]).unwrap();
    assert_eq!(res, rev_val);
}
#[test]
fn test_restrict() {
    let cols = ["first_var", "second.var"];
    // - has higher precedence than ==

    let s = "(first_var|{second.var}==1.0) - (first_var|{second.var}==1.0)";

    let mut vars = (0..cols.len())
        .map(|_| Value::Array(Array2d::ones(5, 1)))
        .collect::<Vec<_>>();
    vars[0] = Value::Array(Array2d::zeros(5, 1));
    let exp = ExprArithmetic::parse(s).unwrap();
    println!("{:?}", exp.var_indices_ordered());
    let res = exp.eval_vec(vars).unwrap();
    println!("{:?}", res);
    if let Value::Array(a) = res {
        assert_eq!(a.data, vec![0.0; 5]);
    } else {
        assert!(false);
    }
    let s = "first_var|{second.var}==1.0 - first_var|{second.var}==1.0";

    let vars = (0..cols.len())
        .map(|_| Value::Array(Array2d::ones(5, 1)))
        .collect::<Vec<_>>();
    let exp = ExprArithmetic::parse(s).unwrap();
    let res = exp.eval_vec(vars).unwrap();
    if let Value::Error(_) = res {
        assert!(true);
    } else {
        assert!(false);
    }
}

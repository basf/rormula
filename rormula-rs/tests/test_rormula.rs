use exmex::Express;
use rormula_rs::{
    array::{Array2d, ColMajor, MemOrder, RowMajor},
    expression::{ExprArithmetic, ExprNames, ExprWilkinson, NameValue, Value},
};

#[test]
fn test_wilkinson() {
    let v1 = Array2d::from_iter([0.1, 0.2, 0.3].iter(), 3, 1).unwrap();
    let v2 = Array2d::from_iter([0.4, 0.5, 0.6].iter(), 3, 1).unwrap();
    let n1 = NameValue::Array(vec!["n".to_string()]);
    let v1 = Value::Array(v1);
    let n2 = NameValue::Array(vec!["o".to_string()]);
    let v2 = Value::Array(v2);
    let s = "n+o+n";
    let expr = ExprWilkinson::parse(s).unwrap();
    let ref_arr: Array2d<ColMajor> =
        Array2d::from_iter([0.1, 0.4, 0.1, 0.2, 0.5, 0.2, 0.3, 0.6, 0.3].iter(), 3, 3).unwrap();
    let res = expr.eval_vec(vec![v1, v2]).unwrap();
    match res {
        Value::Array(a) => {
            a.iter()
                .zip(ref_arr.iter())
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
    fn test<O: MemOrder>() {
        let cols = ["alpha", "beta", "gamma", "eta"];
        let s = "(3.0 * alpha + 1^beta) * (gamma - eta + eta) / 2.0";
        let expr = ExprArithmetic::parse(s).unwrap();
        let vars = (0..cols.len())
            .map(|_| Value::Array(Array2d::<O>::ones(5, 1)))
            .collect::<Vec<_>>();
        let res = expr.eval_vec(vars).unwrap();
        let s_ref = "x * -2.0";
        let expr_ref = ExprArithmetic::parse(s_ref).unwrap();
        let rev_val = expr_ref
            .eval(&[Value::Array(Array2d::<O>::ones(5, 1))])
            .unwrap();
        assert_eq!(res, rev_val);
        let s = "4*3";
        let expr = ExprArithmetic::<O>::parse(s).unwrap();
        let res = expr.eval_vec(vec![]).unwrap();
        let sc_ref = Value::Scalar(12.0);
        assert_eq!(res, sc_ref);
        let s = "5/3 * alpha / beta * (0.2 / 200.0 / (29.22+gamma+epsilon+phi) / 7500)";
        let _ = ExprArithmetic::<O>::parse(s).unwrap();
    }
    test::<ColMajor>();
    test::<RowMajor>();
}
#[test]
fn test_restrict() {
    let cols = ["first_var", "second.var"];
    // - has higher precedence than ==

    let s = "(first_var|{second.var}==1.0) - (first_var|{second.var}==1.0)";

    let mut vars = (0..cols.len())
        .map(|_| Value::Array(Array2d::<ColMajor>::ones(5, 1)))
        .collect::<Vec<_>>();
    vars[0] = Value::Array(Array2d::<ColMajor>::zeros(5, 1));
    let exp = ExprArithmetic::parse(s).unwrap();
    println!("{:?}", exp.var_indices_ordered());
    let res = exp.eval_vec(vars).unwrap();
    println!("{:?}", res);
    if let Value::Array(a) = res {
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![0.0; 5]);
    } else {
        assert!(false);
    }
    let s = "first_var|{second.var}==1.0 - first_var|{second.var}==1.0";

    let vars = (0..cols.len())
        .map(|_| Value::Array(Array2d::<ColMajor>::ones(5, 1)))
        .collect::<Vec<_>>();
    let exp = ExprArithmetic::parse(s).unwrap();
    let res = exp.eval_vec(vars).unwrap();
    if let Value::Error(_) = res {
        assert!(true);
    } else {
        assert!(false);
    }
}

#[test]
fn test_binop() {
    let s = "4/3 * a / b * (1.3 / 112.12 / ((21.0+x+y+z) / 2000))";
    let exp = ExprArithmetic::<ColMajor>::parse(s).unwrap();
    let a = exp.operator_reprs();
    assert_eq!(a.len(), 3);
    assert_eq!(a.as_ref(), &["*", "+", "/"]);
    let b = exp.binary_reprs();
    assert_eq!(b.len(), 3);
    assert_eq!(b.as_ref(), &["*", "+", "/"]);
}

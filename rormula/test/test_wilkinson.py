from functools import partial
from time import perf_counter
from typing import cast
import formulaic
from rormula import RormulaWilkinson, SeparatedData
import rormula as ror
import numpy as np
import pandas as pd


FORMULA_STR_NUMERICAL = "a+b+a:b+c+d+c:d+e+f+e:f"
COLS_NUMERICAL = ["a", "b", "c", "d", "e", "f"]


def get_numerical_data(n_rows):
    return np.random.random((n_rows, len(COLS_NUMERICAL)))


def test_num_cat():
    n_rows = 100
    formula_str = f"{FORMULA_STR_NUMERICAL}+animal"
    animal_row = np.array(["dog", "cat", "horse", "okapi"] * (n_rows // 4))
    cols = COLS_NUMERICAL + ["animal"]
    data = pd.DataFrame(data=get_numerical_data(n_rows), columns=COLS_NUMERICAL)
    data[cols[-1]] = animal_row
    timing_and_test(data, formula_str)


def test_numerical():
    data_numerical = pd.DataFrame(data=get_numerical_data(100), columns=COLS_NUMERICAL)
    timing_and_test(data_numerical, FORMULA_STR_NUMERICAL)


def test_small_numerical():
    matrix = np.arange(6, dtype=np.float64).reshape((2, 3))
    data = pd.DataFrame(data=matrix, columns=["a", "b", "c"])
    rormula = RormulaWilkinson("a+b+c")
    _, res = rormula.eval(data)
    ref = pd.concat(
        [pd.DataFrame(data=np.ones((2, 1)), columns=["Intercept"]), data], axis=1
    )
    np.allclose(res, ref)
    rormula = RormulaWilkinson("a+b+c+a:b+c^2")
    res = rormula.eval_asdf(data)
    ref = pd.concat(
        [
            ref,
            pd.Series(data=[0, 12], name="a:b"),
            pd.Series(data=[9, 36], name="c^2"),
        ],
        axis=1,
    )
    np.allclose(res, ref)


def test_missing_name_in_str():
    formular_str = FORMULA_STR_NUMERICAL
    cols = COLS_NUMERICAL + ["acdc"]
    n_rows = 6
    data = np.concatenate([get_numerical_data(n_rows), np.ones((n_rows, 1))], axis=1)
    data = pd.DataFrame(data=data, columns=cols)
    rormula = RormulaWilkinson(formular_str)
    _, M_r = rormula.eval(data)
    assert np.allclose(data.to_numpy()[:, 1], M_r[:, 2])


def test_missing_name_in_col():
    formular_str = FORMULA_STR_NUMERICAL
    cols = COLS_NUMERICAL[:-1]
    data = get_numerical_data(6)[:, :-1]
    data = pd.DataFrame(data=data, columns=cols)
    rormula = RormulaWilkinson(formular_str)
    try:
        rormula.eval(data)
        assert False
    except ValueError:
        pass
    except Exception:
        assert False


def timing(f, name):
    t = perf_counter()
    res = None
    for _ in range(100):
        res = f()
    t1 = perf_counter()
    print(f"{name} took {t1 - t:0.4f}s")
    return res


def timing_and_test(data, formula_str):
    rormula = RormulaWilkinson(formula_str)
    # keeping data numerical and categorical data separated is faster
    separated_data = ror.separate_numerical_categorical(data)
    M_r = timing(partial(rormula.eval, data=separated_data), "Rormula")

    assert M_r is not None
    names, M_r = M_r
    if len(names) == 0:
        return
    M_r = pd.DataFrame(data=M_r, columns=names)
    formula = formulaic.Formula(formula_str.replace("^", "**"))
    M_f = timing(partial(formula.get_model_matrix, data=data), "Formulaic")

    assert M_f is not None
    assert np.allclose(cast(pd.Series, M_f["e:f"]), M_r["e:f"])
    assert np.allclose(cast(pd.Series, M_f["f"]), M_r["f"])
    assert np.allclose(cast(pd.Series, M_f["c:d"]), M_r["c:d"])
    if "animal_dog" in M_r.columns:
        assert np.allclose(cast(pd.Series, M_f["animal[T.dog]"]), M_r["animal_dog"])
    assert np.allclose(cast(pd.Series, M_f["Intercept"]), M_r["Intercept"])


def test_more_formulas():
    def test(formula_str, extract_reference, extract_result):

        cols = ["alpha", "beta", "gamma", "eta", "theta", "omega"]
        data = np.random.random((100, len(cols)))
        data = pd.DataFrame(data=data, columns=cols)
        rormula = ror.RormulaWilkinson(formula_str)
        M_r = rormula.eval_asdf(data)
        a = extract_reference(data)
        b = extract_result(M_r)
        np.allclose(a, b)

    formula_str = "(beta):gamma"
    extract_reference = lambda data: data["beta"] * data["gamma"]
    extract_result = lambda M_r: M_r["beta:gamma"]
    test(formula_str, extract_reference, extract_result)

    formula_str = "(alpha + beta):gamma"
    extract_reference = lambda data: pd.concat(
        [data["alpha"] * data["gamma"], data["beta"] * data["gamma"]], axis=1
    )
    extract_result = lambda M_r: M_r[["alpha:gamma", "beta:gamma"]]
    test(formula_str, extract_reference, extract_result)

    formula_str = "(alpha + beta):(gamma + eta) + theta + omega"
    extract_reference = lambda data: pd.concat(
        [
            data["alpha"] * data["gamma"],
            data["beta"] * data["gamma"],
            data["alpha"] * data["eta"],
            data["beta"] * data["eta"],
            data["theta"],
            data["omega"],
        ],
        axis=1,
    )
    extract_result = lambda M_r: M_r[
        ["alpha:gamma", "beta:gamma", "alpha:eta", "beta:eta", "theta", "omega"]
    ]
    test(formula_str, extract_reference, extract_result)


def test_separated():
    data = pd.DataFrame(
        data=np.random.random((100, 3)),
        columns=["alpha", "beta", "gamma"],
    )
    separated_data = SeparatedData(
        numerical_cols=data.columns.to_list(),
        numerical_data=data.to_numpy(),
        categorical_cols=[],
        categorical_data=np.zeros((100, 0), dtype="O"),
    )
    rormula = RormulaWilkinson("alpha + beta + alpha:gamma")
    names, mm = rormula.eval(separated_data)
    assert names == ["Intercept", "alpha", "beta", "alpha:gamma"]
    assert mm.shape == (100, 4)


if __name__ == "__main__":
    test_num_cat()

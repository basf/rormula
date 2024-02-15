from time import perf_counter
import numpy as np
import pandas as pd

from rormula import Arithmetic


def timing(f, name):
    t = perf_counter()
    res = None
    for _ in range(100):
        res = f()
    t1 = perf_counter()
    print(f"{name} took {t1 - t:0.4f}s")
    return res


def test_arithmetic():
    df = pd.DataFrame(
        data=np.random.random((100, 3)), columns=["alpha", "beta", "gamma"]
    )
    s = "beta*alpha - 1 + 2^beta + alpha / gamma"
    rormula = Arithmetic(s, "s")
    df_ror = rormula.eval_asdf(df.copy())
    pd_s = f's={s.replace("^", "**")}'
    assert df_ror.shape == (100, 4)
    assert np.allclose(df_ror, df.eval(pd_s))

    def eval_asdf():
        df_ror = df.copy()
        rormula.eval_asdf(df_ror)

    timing(eval_asdf, "eval_asdf")
    timing(lambda: df.eval(pd_s), "eval_pd")
    timing(lambda: rormula.eval(df), "eval_asdf")
    timing(lambda: df.eval(pd_s.split("=")[1]), "eval_pd")

    data = np.ones((100, 3))
    data[5, :] = 2.5
    data[7, :] = 2.5
    df = pd.DataFrame(data=data, columns=["alpha", "beta", "gamma"])
    s = "beta|alpha==2.5"
    rormula = Arithmetic(s, "reduced")
    res = rormula.eval_asdf(df)
    assert res.shape == (2, 1)
    assert np.allclose(res, 2.5)
    s = "beta|alpha>=2.5"
    rormula = Arithmetic(s, s)
    res = rormula.eval_asdf(df)
    assert res.shape == (2, 1)
    assert np.allclose(res, 2.5)
    s = "beta|alpha<2.5"
    rormula = Arithmetic(s, s)
    res = rormula.eval_asdf(df)
    assert res.shape == (98, 1)

    s = "((first_var|{second.var}==5.0) - (first_var|{second.var}==2.5)) / 4.0"
    data[7, :] = 5.0
    df = pd.DataFrame(data=data[:10, :2], columns=["first_var", "second.var"])
    rormula = Arithmetic(s, "reduced")
    res = rormula.eval_asdf(df)
    assert np.allclose(res.to_numpy().item(), (5.0 - 2.5) / 4.0)

def test_scalar_scalar():
    name = "test_scalar"
    data = np.random.random((100, 6)) * 1000
    df = pd.DataFrame(
        data=data, columns=["alpha", "beta", "gamma", "delta", "epsilon", "phi"]
    )
    s = "5/3 * alpha / beta * (0.2 / 200.0 / (29.22+gamma+epsilon+phi) / 1000)"
    rormula = Arithmetic(s, name)
    res = rormula.eval_asdf(df)
    ref = df.eval(s)
    np.allclose(res[name].to_numpy(), ref.values)
    s = "5/3"
    rormula = Arithmetic(s, name)
    res = rormula.eval_asdf(df)
    ref = df.eval(s)    
    np.allclose(res[name].to_numpy(), ref)


if __name__ == "__main__":
    test_scalar_scalar()

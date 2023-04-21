# Rormula

[![CI](https://github.com/basf/rormula/actions/workflows/ci.yml/badge.svg)](https://github.com/basf/rormula/actions)
[![PyPI](https://img.shields.io/pypi/v/rormula.svg?color=%2334D058)](https://pypi.org/project/rormula)

Rormula uses the Wilkinson notation to create model matrices often used in design of experiments. 
Additionally it can also be used in a similar way like
`df.eval`  where `df` is a `pd.Dataframe`. Rormula significantly faster for small matrices, 
implemented in Rust, and still a not well tested prototype. Rormula comes with Python bindings, 
i.e., it is usable like a normal Python module.

## Getting Started

```
pip install rormula
```
Currently, the supported operations are `+`, `:`, and `^`. We can add new operators easily but we have to do
this explicitly. There
are several options how to provide inputs and how to receive results.

```python
import numpy as np
import pandas as pd
from rormula import RormulaWilkinson, SeparatedData
data_np = np.random.random((10, 2))
data = pd.DataFrame(data=data_np, columns=["a", "b"])
ror = RormulaWilkinson("a+b+a:b")

# option 1 returns the model matrix as pandas dataframe
mm_df = ror.eval_asdf(data)
assert isinstance(mm_df, pd.DataFrame)
print(mm_df)

# option 2 is faster
mm_names, mm = ror.eval(data)
assert isinstance(mm, np.ndarray)
assert isinstance(mm_names, list)
```

Keeping categorical and numerical data separated is the fastest option even if there is no categorical data. 
The categorical data is expected to have the object-`dtype` `O`. 
Admittedly, the current interface is rather tedious.

```python
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
ror = RormulaWilkinson("alpha + beta + alpha:gamma")
names, mm = ror.eval(separated_data)
assert names == ["Intercept", "alpha", "beta", "alpha:gamma"]
assert mm.shape == (100, 4)
```

## Contribute

To run the tests, you need to have [Rust](https://www.rust-lang.org/tools/install) installed. Further:
1. Install Formulaic via
   ```
   pip install formulaic pytest maturin
   ```
2. Install Maturin
   ```
   pip install maturin
   ```
3. Create a development build of Rormula
   ```
   maturin develop --release
   ```
4. Run 
   ```
   python test/test.py
   ```

## Rough Time Measurements
We compare Rormula to the well-established and way more mature package [Formulaic](https://github.com/matthewwardrop/formulaic).
The [tests](test/test_wilkinson.py) create a formula and sample 100 random data points. The output on my machine is 
```
Rormula took 0.0040s
Formulaic took 0.7854s
```
We have separated categorical and numerical data beforehand. If we let rormula do the separation and pass a Pandas dataframe, we obtain
```
Rormula took 0.0487s
Formulaic took 0.7699s
```
Rormula returns a list of column names and the data as Numpy array. If we want a Pandas as result dataframe we obtain
```
Rormula took 0.0744s
Formulaic took 0.7639s
```
The time is measured for 100 applications of the formula. We used a small data set with 100 rows. For more rows, e.g., 10k+, formulaic becomes competetive and better.

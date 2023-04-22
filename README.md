# Rormula

[![Test](https://github.com/basf/rormula/actions/workflows/test.yml/badge.svg)](https://github.com/basf/rormula/actions)
[![PyPI](https://img.shields.io/pypi/v/rormula.svg?color=%2334D058)](https://pypi.org/project/rormula)

Rormula is a Python package that parses the Wilkinson notation to create model matrices useful in design of experiments. 
Additionally, it can be used for column arithmetics similar to
`df.eval` where `df` is a Pandas dataframe. Rormula is significantly faster for small matrices than `df.eval` or [Formulaic](https://github.com/matthewwardrop/formulaic)
and still a not well tested prototype.



## Getting Started with Wilkinson Notation 

```
pip install rormula
```
Currently, the supported operations are `+`, `:`, and `^`. We can add new operators easily but we have to do
this explicitly. There
are different options how to receive results and provide inputs.
The result can either be a Pandas dataframe or a list of names and a Numpy array.

```python
import numpy as np
import pandas as pd
from rormula import Wilkinson, SeparatedData
data_np = np.random.random((10, 2))
data = pd.DataFrame(data=data_np, columns=["a", "b"])
ror = Wilkinson("a+b+a:b")

# option 1 returns the model matrix as pandas dataframe
mm_df = ror.eval_asdf(data)
assert isinstance(mm_df, pd.DataFrame)
print(mm_df)

# option 2 is faster
mm_names, mm = ror.eval(data)
assert isinstance(mm, np.ndarray)
assert isinstance(mm_names, list)
```

Regarding inputs, the fastest option is to use the interface with separated categorical and numerical data, even if there is no categorical data. 
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
ror = Wilkinson("alpha + beta + alpha:gamma")
names, mm = ror.eval(separated_data)
assert names == ["Intercept", "alpha", "beta", "alpha:gamma"]
assert mm.shape == (100, 4)
```

## Contribute

To run the tests, you need to have [Rust](https://www.rust-lang.org/tools/install) installed. 

### Python Tests

1. Go to the directory of the Python package
   ```
   cd rormula
   ```
2. Install dev dependencies via
   ```
   pip install -r requirements.txt
   ```
3. Create a development build of Rormula
   ```
   maturin develop --release
   ```
4. Run 
   ```
   python test/test.py
   ```

### Rust Tests
Run
```
cargo test
```
from the project's root.

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
Rormula returns a list of column names and the data as Numpy array. If we want a Pandas dataframe as result we obtain
```
Rormula took 0.0744s
Formulaic took 0.7639s
```
The time is measured for 100 applications of the formula. We used a small data set with 100 rows. For more rows, e.g., 10k+, formulaic becomes competetive and better.

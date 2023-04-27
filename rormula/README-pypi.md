# Rormula

Rormula is a Python package that parses the Wilkinson notation to create model matrices useful in design of experiments. 
Additionally, it can be used for column arithmetics similar to
`df.eval` where `df` is a Pandas dataframe. Rormula is significantly faster for small matrices than `df.eval` or [Formulaic](https://github.com/matthewwardrop/formulaic), does not allow arbitrary code execution but only pre-defined operators,
and still a not well tested prototype.

More information can be found under https://github.com/basf/rormula.

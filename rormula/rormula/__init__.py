from typing import List, NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
from .rormula import parse_wilkinson, eval_wilkinson, parse_arithmetic, eval_arithmetic


class SeparatedData(NamedTuple):
    numerical_cols: List[str]
    numerical_data: np.ndarray
    categorical_cols: List[str]
    categorical_data: np.ndarray


def separate_num_cat(
    data: pd.DataFrame,
) -> SeparatedData:
    """Separates numerical and categorical data for faster evaluation."""
    numerical = data.select_dtypes(include="number")
    categorical = data.select_dtypes(exclude="number")
    cat_cols = categorical.columns.to_list()
    if categorical.shape[1] == 0:
        categorical = categorical.to_numpy().astype("O")
    else:
        categorical = categorical.to_numpy()
    return SeparatedData(
        numerical.columns.to_list(),
        numerical.to_numpy(),
        cat_cols,
        categorical,
    )


class Wilkinson:
    def __init__(self, formula: str):
        self.ror = parse_wilkinson(formula)

    def eval(
        self, data: Union[pd.DataFrame, SeparatedData], skip_names: bool = False
    ) -> Tuple[List[str], np.ndarray]:
        if isinstance(data, SeparatedData):
            numerical_cols, numerical_data, categorical_cols, categorical_data = data
        else:
            (
                numerical_cols,
                numerical_data,
                categorical_cols,
                categorical_data,
            ) = separate_num_cat(data)

        names, resulting_data = eval_wilkinson(
            self.ror,
            numerical_data,
            numerical_cols,
            categorical_data,
            categorical_cols,
            skip_names=skip_names,
        )
        if names is None:
            names = []
        return names, resulting_data

    def eval_asdf(
        self, data: Union[pd.DataFrame, SeparatedData], skip_names: bool = False
    ):
        names, resulting_data = self.eval(data, skip_names=skip_names)
        return pd.DataFrame(data=resulting_data, columns=names)


class Arithmetic:
    def __init__(self, formula: str, name: str):
        self.ror = parse_arithmetic(formula)
        self.name = name

    def eval(self, data: pd.DataFrame) -> np.ndarray:
        numerical_cols = data.columns.to_list()
        numerical_data = data.to_numpy()

        resulting_data = eval_arithmetic(
            self.ror,
            numerical_data,
            numerical_cols,
        )

        return resulting_data

    def eval_asdf(self, data: pd.DataFrame):
        resulting_data = self.eval(data)
        if resulting_data.shape[0] == data.shape[0]:
            data[self.name] = resulting_data
        else:
            data = pd.DataFrame(data=resulting_data, columns=[self.name])
        return data

    def unparse(self) -> str:
        return self.ror.unparse()

    def has_row_change_op(self) -> bool:
        return self.ror.has_row_change_op()

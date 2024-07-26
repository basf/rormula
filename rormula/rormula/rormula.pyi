from typing import List, NamedTuple, Optional, Sequence, Tuple
import numpy as np

class Wilkinson:
    pass

class SeparatedData(NamedTuple):
    numerical_cols: List[str]
    numerical_data: np.ndarray
    categorical_cols: List[str]
    categorical_data: np.ndarray

def parse_wilkinson(s: str) -> Wilkinson: ...
def eval_wilkinson(
    ror: Wilkinson,
    numerical_data: np.ndarray,
    numerical_cols: Sequence[str],
    cat_data: np.ndarray,
    cat_cols: Sequence[str],
    skip_names: bool = False,
) -> Tuple[Optional[List[str]], np.ndarray]: ...

class Arithmetic:
    def has_row_change_op(self) -> bool: ...
    def unparse(self) -> str: ...

def parse_arithmetic(s: str) -> Arithmetic: ...
def eval_arithmetic(
    ror: Arithmetic,
    numerical_data: np.ndarray,
    numerical_cols: Sequence[str],
) -> np.ndarray: ...

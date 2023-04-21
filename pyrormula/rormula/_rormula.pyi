from typing import List, NamedTuple, Optional, Sequence, Tuple
import numpy as np

class RormulaWilkinson:
    pass

class SeparatedData(NamedTuple):
    numerical_cols: List[str]
    numerical_data: np.ndarray
    categorical_cols: List[str]
    categorical_data: np.ndarray


def parse_wilkinson(s: str) -> RormulaWilkinson: ...
def eval_wilkinson(
    ror: RormulaWilkinson,
    numerical_data: np.ndarray,
    numerical_cols: Sequence[str],
    cat_data: np.ndarray,
    cat_cols: Sequence[str],
    skip_names: bool = False,
) -> Tuple[Optional[List[str]], np.ndarray]: ...

class RormulaArithmetic:
    pass

def parse_arithmetic(s: str) -> RormulaArithmetic: ...
def eval_arithmetic(
    ror: RormulaArithmetic,
    numerical_data: np.ndarray,
    numerical_cols: Sequence[str],
) -> np.ndarray: ...
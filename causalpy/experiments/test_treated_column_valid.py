import pandas as pd
import pytest

def _check_treated_column_validity(df, treated_col_name):
    treated_col = df[treated_col_name]
    if not pd.api.types.is_bool_dtype(treated_col):
        raise ValueError(f"The '{treated_col_name}' column must be of boolean dtype (True/False).")

def test_treated_column_with_integers():
    df = pd.DataFrame({"treated": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="treated.*must be of boolean dtype"):
        _check_treated_column_validity(df, "treated")

def test_treated_column_with_booleans():
    df = pd.DataFrame({"treated": [True, False, True, False]})
    try:
        _check_treated_column_validity(df, "treated")
    except ValueError:
        pytest.fail("Unexpected ValueError raised")

import pandas as pd
import pytest


@pytest.fixture
def data():
    """Pytest fixture to load the transaction dataset."""
    return pd.read_csv("data/transactions_2022.csv")


def test_no_missing_values(data):
    """Tests that there are no null values in the dataset."""
    assert not data.isnull().values.any(), "Dataset contains missing values"


def test_core_columns_exist(data):
    """Tests that all expected columns are present."""
    expected_cols = {'Time', 'Amount', 'Class'} | {f'V{i}' for i in range(1, 29)}
   
    # This test is robust; it will pass even if extra columns like 'location' exist
    assert expected_cols.issubset(set(data.columns)), "Dataset is missing core columns"


def test_target_column_classes(data):
    """Tests that the target 'Class' column is binary (0 or 1)."""
    unique_classes = set(data["Class"].unique())
    assert unique_classes == {0, 1}, "Target column should only contain 0 and 1"

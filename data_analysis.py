import pandas as pd
from file_processing import load_from_json

def load_dataset(file_path):
    """Create pandas dataframe."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as missing_error:
        print(f"File not found error: {missing_error.args}")
    else:
        return df


def check_data_sanity(df):
    """Check dataframe for any unnamed columns."""
    try:
        assert (len(df.iloc[0]) == 133)
        assert (len(df.iloc[:, 0]) == df.shape[0])
    except AssertionError as assert_err:
        df.drop(df.columns[df.columns.str.contains(
            "unnamed", case=False)], axis=1, inplace=True)


def convert_features_to_float(df):
    """Convert features to float64"""
    target = df.iloc[:, -1].astype("str")
    df = df.iloc[:, :-1].astype("float64")
    df["disease"] = target
    return df


def create_column_total_symptoms(df):
    """Sum symptoms by row and plot"""
    df["total_count"] = df.iloc[:, :-1].sum(axis=1)
    return df


def convert_target_column(disease):
    """Convert str to int from json"""
    dic = load_from_json("disease.json")
    for key, value in dic.items():
        if value == disease:
            return int(key)


def create_target_to_int_column(df):
    """Apply conversion to target column"""
    df["disease_int"] = df["disease"].apply(convert_target_column)
    return df
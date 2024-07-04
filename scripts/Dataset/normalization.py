import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from typing import Callable


def normalization(
    df: pd.DataFrame,
    df_original: Optional[pd.DataFrame] = None,
    type_normalization: Optional["str"] = None,
) -> pd.DataFrame:
    def standardize() -> pd.DataFrame:
        scaler = StandardScaler()
        if df_original is None:
            normalize_array = scaler.fit_transform(df_copy)
        else:
            scaler.fit(df_original_copy)
            normalize_array = scaler.transform(df_copy)
        return pd.DataFrame(normalize_array, columns=df.columns)

    def minmax() -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if df_original is None:
            normalize_array = scaler.fit_transform(df_copy)
        else:
            scaler.fit(df_original_copy)
            normalize_array = scaler.transform(df_copy)
        return pd.DataFrame(normalize_array, columns=df.columns)

    df_copy = df.copy()
    if df_original is not None:
        df_original_copy = df_original.copy()
    with open("./templates/network_templates.json", "r") as file:
        norm_template = json.load(file)["Normalization"]
    if type_normalization == None:
        type_normalization = norm_template["type_normalization"]
    for column in df.columns:
        if column in norm_template["log_norm"]:
            df_copy[column] = np.log10(df[column])
            if df_original is not None:
                df_original_copy[column] = np.log10(df_original[column])

    if type_normalization == "minmax":
        return minmax()
    else:
        return standardize()

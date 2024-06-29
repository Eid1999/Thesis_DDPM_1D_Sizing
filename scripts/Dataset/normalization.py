import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from typing import Callable


def normalization(
    df: pd.DataFrame,
    original: Optional[pd.DataFrame] = None,
    type_normalization: Optional["str"] = None,
) -> pd.DataFrame:
    def standardize() -> pd.DataFrame:
        scaler = StandardScaler()
        if original is None:
            normalize_array = scaler.fit_transform(df)
        else:
            scaler.fit(original)
            normalize_array = scaler.transform(df)
        return pd.DataFrame(normalize_array, columns=df.columns)

    def minmax() -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if original is None:
            normalize_array = scaler.fit_transform(df)
        else:
            scaler.fit(original)
            normalize_array = scaler.transform(df)
        return pd.DataFrame(normalize_array, columns=df.columns)

    df_copy = df.copy()
    if original is not None:
        original_copy = original.copy()
    if type_normalization == None:
        with open("./templates/network_templates.json", "r") as file:
            norm_template = json.load(file)["Normalization"]
        for column in df.columns:
            if column in norm_template["log_norm"]:
                df_copy[column] = np.log10(df[column])
                if original is not None:
                    original_copy[column] = np.log10(original[column])
        df = df_copy
        if original is not None:
            original = original_copy
        type_normalization = norm_template["type_normalization"]
    if type_normalization == "minmax":
        return minmax()
    else:
        return standardize()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from typing import Callable


def normalization(
    df: pd.DataFrame,
    df_original: Optional[pd.DataFrame] = None,
    type_normalization: Optional["str"] = None,
    data_type: str = "vcota",
    poly_bool: bool = False,
) -> pd.DataFrame:
    def standardize() -> pd.DataFrame:
        scaler = StandardScaler()
        if df_original_copy is None:
            normalize_array = scaler.fit_transform(df_copy)
        else:
            scaler.fit(df_original_copy)
            normalize_array = scaler.transform(df_copy)
        return pd.DataFrame(normalize_array, columns=df_copy.columns)

    def minmax() -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if df_original_copy is None:
            normalize_array = scaler.fit_transform(df_copy)
        else:
            scaler.fit(df_original_copy)
            normalize_array = scaler.transform(df_copy)
        return pd.DataFrame(normalize_array, columns=df_copy.columns)

    df_copy = df.copy()
    if df_original is not None:
        df_original_copy = df_original.copy()
    else:
        df_original_copy = None
    if poly_bool:
        df_copy, df_original_copy = Polyfit(df_copy, df_original_copy)
    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        norm_template = json.load(file)["Normalization"]
    if type_normalization == None:
        type_normalization = norm_template["type_normalization"]

    for column in df_copy.columns:
        if any(log_col in column for log_col in norm_template["log_norm"]):
            df_copy[column] = np.log10(df_copy[column])
            if df_original_copy is not None:
                df_original_copy[column] = np.log10(df_original_copy[column])

    if type_normalization == "minmax":
        return minmax()
    else:
        return standardize()


def Polyfit(df_copy, df_original_copy):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    # Fit and transform the df_copy and df_original_copy if they are present
    if df_original_copy is None:
        copy_poly = poly.fit_transform(df_copy)
    else:
        original_poly = poly.fit_transform(df_original_copy)
        copy_poly = poly.fit_transform(df_copy)
        df_original_copy = pd.DataFrame(
            original_poly, columns=poly.get_feature_names_out(df_copy.columns)
        )
    df_copy = pd.DataFrame(
        copy_poly, columns=poly.get_feature_names_out(df_copy.columns)
    )
    return df_copy, df_original_copy

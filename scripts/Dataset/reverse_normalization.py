from libraries import *


def reverse_normalization(
    df: pd.DataFrame,
    df_original: pd.DataFrame,
    data_type: str = "vcota",
) -> pd.DataFrame:

    def reverse_standardize() -> pd.DataFrame:
        scaler = StandardScaler()
        scaler.fit(df_original_copy)
        return pd.DataFrame(scaler.inverse_transform(df.values), columns=df.columns)

    def reverse_minmax() -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df_original_copy.values)
        return pd.DataFrame(
            scaler.inverse_transform(df.values), columns=df_original_copy.columns
        )

    df_original_copy = df_original.copy()
    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        norm_template = json.load(file)["Normalization"]
    for column in df.columns:
        if column in norm_template["log_norm"]:
            df_original_copy[column] = np.log10(df_original_copy[column])
    norm = norm_template["type_normalization"]
    if norm == "minmax":
        de_norm = reverse_minmax()
    else:
        de_norm = reverse_standardize()
    for column in df.columns:
        if column in norm_template["log_norm"]:
            de_norm[column] = 10 ** (de_norm[column])
        if "nf" in column:
            de_norm[column] = de_norm[column].round(0).astype(int)
    return de_norm

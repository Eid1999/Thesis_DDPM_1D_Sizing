from libraries import *


def reverse_normalization(
    df_scaled: pd.DataFrame,
    df_original: pd.DataFrame,
) -> pd.DataFrame:

    def reverse_standardize() -> pd.DataFrame:
        scaler = StandardScaler()
        scaler.fit(df_original)  # Fit the scaler on original data (important step!)
        return pd.DataFrame(
            scaler.inverse_transform(df_scaled.values), columns=df_scaled.columns
        )

    def reverse_minmax() -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(
            df_original.values
        )  # Fit the scaler on original data (important step!)
        return pd.DataFrame(
            scaler.inverse_transform(df_scaled.values), columns=df_original.columns
        )

    with open("./templates/network_templates.json", "r") as file:
        norm_template = json.load(file)["Normalization"]
    for column in df_scaled.columns:
        if column in norm_template["log_norm"]:
            df_original[column] = np.log10(df_original[column])
    norm = norm_template["type_normalization"]
    if norm == "minmax":
        de_norm = reverse_minmax()
    else:
        de_norm = reverse_standardize()
    for column in df_scaled.columns:
        if column in norm_template["log_norm"]:
            de_norm[column] = 10 ** (de_norm[column])
    return de_norm

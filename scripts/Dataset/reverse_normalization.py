from requirements import *


def reverse_normalization(df_scaled, df_original):
    def reverse_standardize():
        scaler = StandardScaler()
        scaler.fit(df_original)  # Fit the scaler on original data (important step!)
        return scaler.inverse_transform(df_scaled)

    def reverse_minmax():
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df_original)  # Fit the scaler on original data (important step!)
        return scaler.inverse_transform(df_scaled)

    if norm == "minmax":
        return reverse_minmax()
    else:
        return reverse_standardize()

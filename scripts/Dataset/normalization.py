from requirements import *


def normalization(df, original=None, type_normalization=norm):
    def standardize():
        scaler = StandardScaler()
        if original is None:
            return scaler.fit_transform(df)
        else:
            scaler.fit(original)
            return scaler.transform(df)

    def minmax():
        scaler = MinMaxScaler(feature_range=(-1, 1))
        if original is None:
            return scaler.fit_transform(df)
        else:
            scaler.fit(original)
            return scaler.transform(df)

    if type_normalization == "minmax":
        return minmax()
    else:
        return standardize()

from requirements import *
from sklearn.preprocessing import StandardScaler

# norm = "minmax"

norm = "standardize"


def load_FCA():
    GBW = 0
    GDC = 1
    PM = 2
    POWER = 3
    CLOAD = 4
    Values = ["1.0", "1.5", "2.5", "3.5", "4.0", "4.5", "5.0"]
    final_matrix_y = np.empty((0, 5))
    final_matrix_X = np.empty((0, 19))
    for c_value in Values:
        data = loadmat(
            "/media/ssd/Anot/MEEC/Tese/Code/data/FCA/different Cload POFs_v2/Cload"
            + c_value
            + "00000e-13/matlab.mat"
        )
        X = np.zeros((100, 5))
        y = np.array(data["archive3d"])[:, :, -1]
        X[:, GBW] = np.array(data["archive_bw"])[-1, :]
        X[:, GDC] = np.array(data["archive_gain"])[-1, :]
        X[:, PM] = np.array(data["archive_pm"])[-1, :]
        X[:, POWER] = np.array(data["archive_power"])[-1, :]

        X[:, CLOAD] = data["Cload"][0, 0]
        final_matrix_X = np.concatenate((final_matrix_X, y), axis=0)
        final_matrix_y = np.concatenate((final_matrix_y, X), axis=0)
    df_y = pd.DataFrame(final_matrix_y, columns=["GBW", "GDC", "PM", "POWER", "CLOAD"])
    df_X = pd.DataFrame(
        final_matrix_X,
        columns=[
            "LM1",
            "LM2",
            "LM3",
            "LM4",
            "LM5",
            "LM6",
            "LM7",
            "LM8",
            "WM1",
            "WM2",
            "WM3",
            "WM4",
            "WM5",
            "WM6",
            "WM7",
            "WM8",
            "Vcm1",
            "Vcm2",
            "Rb",
        ],
    )
    return df_y, df_X


def dataset_info(df):

    print(df.shape)
    print(df.columns)
    print(df.describe().T)


def normalize_values(values, min_val=None, max_val=None, new_min=-1, new_max=1):

    if min_val is None:
        min_val = np.min(values, axis=0)
        max_val = np.max(values, axis=0)

    # Normalize based on the range and new min-max
    range = max_val - min_val
    normalized_values = (values - min_val) / range * (new_max - new_min) + new_min
    return normalized_values, min_val, max_val


def reverse_normalize_values(
    values, min_val=None, max_val=None, original_min=-1, original_max=1
):

    if min_val is None or max_val is None:
        raise ValueError(
            "Both min_val and max_val of original data are required for reverse normalization."
        )

    # Reverse normalization based on original range and new min-max
    range = original_max - original_min
    reversed_values = (values - original_min) / (original_max - original_min) * (
        max_val - min_val
    ) + min_val
    return reversed_values


def augment_data(X, y, target, repetition_factor=10, scale=0.2):

    y_rep = np.repeat(y, repetition_factor, axis=0)
    X_rep = np.repeat(X, repetition_factor, axis=0)
    m, n_x = X_rep.shape
    # -1 means that specifications with a smaller value are also meet by the design, e.g GDC
    #  1 means that specifications with a larger value are also meet by the design, e.g IDD
    target_scale = scale * np.mean(X, axis=0) * target
    y_rep = np.concatenate((y, y_rep), axis=0)
    X_rep = np.concatenate((X, X_rep + np.random.rand(m, n_x) * target_scale), axis=0)

    return (X_rep, y_rep)


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

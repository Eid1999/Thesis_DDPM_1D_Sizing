import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *

from Dataset import normalization, reverse_normalization


def Test_error(
    y_test: torch.Tensor,
    DDPM,
    best_weight: int,
    X_test: np.ndarray,
    df_X: pd.DataFrame,
    data_type: str = "vcota",
) -> None:
    print("\n\n\nTest Error")
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(),
        y_test.shape[0],
        y_test,
        weight=best_weight,
    )
    df_Sampled = pd.DataFrame(
        X_Sampled.cpu().numpy(),
        columns=df_X.columns,
    )
    df_X_test = pd.DataFrame(X_test, columns=df_X.columns)
    df_Sampled = reverse_normalization(
        df_Sampled,
        df_X.copy(),
        data_type=data_type,
    )
    df_X_test = reverse_normalization(
        df_X_test,
        df_X.copy(),
        data_type=data_type,
    )
    error = np.mean(
        np.abs(
            np.divide(
                (df_X_test - df_Sampled),
                df_Sampled,
                out=np.zeros_like(df_Sampled),
                where=(df_X_test != 0),
            )
        ),
        axis=0,
    )
    print(f"\n{error}")
    plt.subplot(1, 2, 1)
    sns.heatmap(
        normalization(df_Sampled, df_original=df_X.copy(), type_normalization="minmax"),
        cmap="Spectral",
        xticklabels=True,
        vmin=1,
        vmax=-1,
    )
    # plt.colorbar()
    plt.title("VCOTA Sample Dataset")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        normalization(df_X_test, df_original=df_X.copy(), type_normalization="minmax"),
        cmap="Spectral",
        xticklabels=True,
        vmin=1,
        vmax=-1,
    )
    plt.title("VCOTA test Dataset")
    plt.show()

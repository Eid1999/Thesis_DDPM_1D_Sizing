from requirements import *

from Dataset import normalization, reverse_normalization


def Test_error(
    y_test,
    DDPM,
    best_weight,
    X_test,
    df_X,
):
    print("\n\n\nTest Error")
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test.shape[0], y_test, weight=best_weight
    )

    df_Sampled = reverse_normalization(X_Sampled.cpu().numpy(), df_X)
    df_X_test = reverse_normalization(X_test, df_X)
    df_Sampled = pd.DataFrame(df_Sampled, columns=df_X.columns)
    df_X_test = pd.DataFrame(df_X_test, columns=df_X.columns)
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
        pd.DataFrame(
            normalization(df_Sampled, original=df_X, type_normalization="minmax"),
            columns=df_X.columns,
        ),
        cmap="Spectral",
        xticklabels=True,
    )
    # plt.colorbar()
    plt.title("VCOTA Sample Dataset")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        pd.DataFrame(
            normalization(df_X_test, original=df_X, type_normalization="minmax"),
            columns=df_X.columns,
        ),
        cmap="Spectral",
        xticklabels=True,
    )
    plt.title("VCOTA test Dataset")
    plt.show()

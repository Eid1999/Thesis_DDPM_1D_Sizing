from requirements import *

from Dataset import normalization, reverse_normalization


def histogram(
    DDPM,
    best_weight,
    y,
    df_X,
    X_train,
):
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(),
        y.shape[0],
        y,
        weight=best_weight,
    )

    df_Sampled_hist = reverse_normalization(
        X_Sampled.cpu().numpy(),
        df_X,
    )
    X_train_hist = reverse_normalization(
        X_train.cpu(),
        df_X,
    )
    df_Sampled_hist = pd.DataFrame(
        df_Sampled_hist,
        columns=df_X.columns,
    )
    X_train_hist = pd.DataFrame(
        X_train_hist,
        columns=df_X.columns,
    )

    fig, axs = plt.subplots(2, X_Sampled.shape[1])
    for i in range(df_Sampled_hist.shape[1]):
        axs[0, i].set_title(f"Sample {df_Sampled_hist.columns[i]}")
        axs[0, i].hist(
            df_Sampled_hist.values[:, i],
            bins=50,
            range=[
                df_Sampled_hist.values[:, i].min(),
                df_Sampled_hist.values[:, i].max(),
            ],
        )
        # axs[0, i].set_ylim(0, len(df_Sampled_hist))
        axs[1, i].set_title(f"Input {df_Sampled_hist.columns[i]}")
        axs[1, i].hist(
            X_train_hist.values[:, i],
            bins=50,
            range=[
                df_Sampled_hist.values[:, i].min(),
                df_Sampled_hist.values[:, i].max(),
            ],
        )
        y_max1 = axs[1, i].get_ylim()[1]
        y_max2 = axs[0, i].get_ylim()[1]
        y_max = max(y_max1, y_max2)
        axs[1, i].set_ylim(0, y_max)
        axs[0, i].set_ylim(0, y_max)
        # axs[1, i].set_ylim(0, len(df_Sampled_hist))
    plt.show()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *

from Dataset import normalization, reverse_normalization


def histogram(
    DDPM,
    best_weight: float,
    y: torch.Tensor,
    df_X: pd.DataFrame,
    X_train: torch.Tensor,
) -> None:
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(),
        y.shape[0],
        y,
        weight=best_weight,
    )
    X_Sampled = pd.DataFrame(
        X_Sampled.cpu().numpy(),
        columns=df_X.columns,
    )
    df_X_train = pd.DataFrame(
        X_train.cpu().numpy(),
        columns=df_X.columns,
    )
    df_Sampled_hist = reverse_normalization(
        X_Sampled,
        df_X.copy(),
    )
    X_train_hist = reverse_normalization(
        df_X_train,
        df_X.copy(),
    )

    fig, axs = plt.subplots(2, X_Sampled.shape[1])
    for i in range(df_Sampled_hist.shape[1]):
        axs[0, i].set_title(f"Sample {df_Sampled_hist.columns[i]}")
        axs[0, i].hist(
            df_Sampled_hist.values[:, i],
            bins=100,
            log_scaled=True,
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
            log_scaled=True,
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

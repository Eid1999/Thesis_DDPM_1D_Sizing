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
    for i, col in enumerate(df_Sampled_hist.columns):
        axs[0, i].set_title(f"Sample {df_Sampled_hist.columns[i]}")
        sns.histplot(
            df_Sampled_hist[col].to_frame(),
            ax=axs[0, i],
            log_scale=True,
            bins=100,
            legend=False,
            kde=True,  # If you want to add a KDE line
            color="blue",  # Change to your desired color
            edgecolor="black",  # Change edge color if desired
        )
        # axs[0, i].set_ylim(0, len(df_Sampled_hist))
        # pdb.set_trace()
        sns.histplot(
            X_train_hist[col].to_frame(),
            ax=axs[1, i],
            log_scale=True,
            bins=100,
            legend=False,
            kde=True,  # If you want to add a KDE line
            color="blue",  # Change to your desired color
            edgecolor="black",  # Change edge color if desired
        )
        # pdb.set_trace()

        y_max = max(axs[1, i].get_ylim()[1], axs[0, i].get_ylim()[1])
        axs[1, i].set_ylim(0, y_max)
        axs[0, i].set_ylim(0, y_max)
        x_max = max(axs[1, i].get_xlim()[1], axs[0, i].get_xlim()[1])
        x_min = min(axs[1, i].get_xlim()[0], axs[0, i].get_xlim()[0])
        axs[1, i].set_xlim(x_min, x_max)
        axs[0, i].set_xlim(x_min, x_max)
    plt.show()

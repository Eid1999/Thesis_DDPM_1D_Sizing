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

    fig, axs = plt.subplots(1, X_Sampled.shape[1])
    for i, col in enumerate(df_Sampled_hist.columns):
        axs[i].set_title(f"Sample {df_Sampled_hist.columns[i]}")
        plot_data = pd.DataFrame(
            {"Sampled Data": df_Sampled_hist[col], "Real data": X_train_hist[col]}
        )
        sns.histplot(
            plot_data,
            ax=axs[i],
            log_scale=True,
            bins=100,
            kde=True,  # If you want to add a KDE line
            legend=True if i == len(axs) - 1 else False,
        )
        # axs[0, i].set_ylim(0, len(df_Sampled_hist))
        # pdb.set_trace()
    sns.move_legend(
        axs[len(axs) - 1],
        loc="center left",
        bbox_to_anchor=[1, 0.5],
        fancybox=True,
        shadow=True,
    )

    plt.show()

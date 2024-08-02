import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *

from Dataset import normalization, reverse_normalization
import matplotlib.ticker as ticker


def histogram(
    DDPM,
    best_weight: float,
    y: torch.Tensor,
    df_X: pd.DataFrame,
    X_train: torch.Tensor,
    data_type: str = "vcota",
) -> None:
    X_Sampled = DDPM.sampling(
        DDPM.model,
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
        data_type=data_type,
    )
    X_train_hist = reverse_normalization(
        df_X_train,
        df_X.copy(),
        data_type=data_type,
    )

    fig, axs = plt.subplots(math.ceil(X_Sampled.shape[1] / 2), 2)
    idx = 0
    for i in range(axs.shape[1]):
        for j in range(axs.shape[0]):
            if X_Sampled.shape[1] - 1 < idx:
                fig.delaxes(axs[j, i])
                break
            axs[j, i].set_title(f"Sample {df_Sampled_hist.columns[idx]}")
            plot_data = pd.DataFrame(
                {
                    "Sampled Data": df_Sampled_hist.iloc[:, idx],
                    "Real data": X_train_hist.iloc[:, idx],
                }
            )
            sns.histplot(
                plot_data,
                ax=axs[j, i],
                log_scale=True,
                bins=100,
                kde=True,  # If you want to add a KDE line
                legend=(
                    True
                    if j == axs.shape[0] // 2 - 1 and i == axs.shape[1] - 1
                    else False
                ),
            )

            idx += 1

            # if j == 0:
            #     for ind, label in enumerate(axs[j, i].get_xticklabels()):
            #         label.set_visible(False)

    sns.move_legend(
        axs[axs.shape[0] // 2 - 1, axs.shape[1] - 1],
        loc="center left",
        bbox_to_anchor=[1, 0.5],
        fancybox=True,
        shadow=True,
        fontsize="10",
    )

    plt.show()

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
    display: bool = False,
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
    if display:
        fig, axs = plt.subplots(math.ceil(X_Sampled.shape[1] / 2), 2)
        idx = 0
        for i in range(axs.shape[1]):
            for j in range(axs.shape[0]):
                if X_Sampled.shape[1] - 1 < idx:
                    fig.delaxes(axs[j, i])
                    break
                axs[j, i].set_title(f"Sample {df_Sampled.columns[idx]}")
                plot_data = pd.DataFrame(
                    {
                        "Sampled Data": df_Sampled.iloc[:, idx],
                        "Real data": df_X_test.iloc[:, idx],
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

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *

from Dataset import normalization, reverse_normalization


def see_noise_data(
    DDPM,
    x: torch.Tensor,
    df_X: pd.DataFrame,
) -> None:
    fig, axs = plt.subplots(1, 5)
    original_matrix = x.cpu().squeeze().numpy()
    original_matrix = pd.DataFrame(
        original_matrix,
        columns=df_X.columns,
    )
    norm_original_matrix = reverse_normalization(
        original_matrix,
        df_X.copy(),
        data_type="vcota",
    )
    error = np.abs(
        np.divide(
            (norm_original_matrix - norm_original_matrix),
            norm_original_matrix,
            out=np.zeros_like(norm_original_matrix),
            where=(original_matrix != 0),
        )
    )
    strip_columns = df_X.columns.str.replace("_", "")
    original_matrix.columns = strip_columns
    plt.suptitle("Forward Process", fontsize=16)
    axs[0].set_title(f"Noise:0%, Sizing Error:0%", fontsize=14)
    sns.histplot(
        original_matrix,
        legend=False,
        # cmap="crest",
        ax=axs[0],
    )
    for i in range(1, len(axs)):
        noise_step = np.min(
            [
                int(DDPM.noise_steps * i / (len(axs) - 1)),
                DDPM.noise_steps - 1,
            ]
        )

        noise_vect, _ = DDPM.forward_process(
            x,
            torch.full(
                (x.shape[0],),
                noise_step,
                device=DDPM.device,
            ),
        )
        matrix_with_noise_array = noise_vect.cpu().squeeze().numpy()
        matrix_with_noise_array = pd.DataFrame(
            matrix_with_noise_array, columns=df_X.columns
        )
        norm_matrix_with_noise_array = reverse_normalization(
            matrix_with_noise_array,
            df_X.copy(),
            data_type=data_type,
        )

        error = np.abs(
            np.divide(
                (norm_original_matrix - norm_matrix_with_noise_array),
                norm_matrix_with_noise_array,
                out=np.zeros_like(matrix_with_noise_array),
                where=(matrix_with_noise_array != 0),
            )
        )
        axs[i].set_title(
            f"Noise:{int(noise_step*100/(DDPM.noise_steps - 1))}%,Sizing Error:{error.mean().mean()*100:.1f}%",
            fontsize=14,
        )
        matrix_with_noise_array.columns = strip_columns
        sns.histplot(
            matrix_with_noise_array,
            legend=True if i == len(axs) - 1 else False,
            ax=axs[i],
        )
        # axs[i].set_ylim(top=16000)
    sns.move_legend(
        axs[len(axs) - 1],
        loc="center left",
        bbox_to_anchor=[1, 0.5],
        fancybox=True,
        shadow=True,
    )

    plt.show()

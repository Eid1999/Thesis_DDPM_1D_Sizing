from requirements import *

from Dataset import normalization, reverse_normalization


def see_noise_data(DDPM, x, df_X):
    fig, axs = plt.subplots(1, 5)
    original_matrix = x.cpu().squeeze().numpy()

    original_matrix = reverse_normalization(original_matrix, df_X)
    original_matrix = pd.DataFrame(original_matrix, columns=df_X.columns)
    error = np.abs(
        np.divide(
            (original_matrix - original_matrix),
            original_matrix,
            out=np.zeros_like(original_matrix),
            where=(original_matrix != 0),
        )
    )
    plt.suptitle("Error added by Noise Step", fontsize=14)
    axs[0].set_title(f"Noise:0%, Mean Error:0", fontsize=14)
    sns.heatmap(
        pd.DataFrame(
            # normalization(original_matrix, original=df_X, type_normalization="minmax"),
            error,
            columns=df_X.columns,
        ),
        cbar=False,
        vmin=0,
        vmax=0.2,
        # cmap="crest",
        xticklabels=True,
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
        matrix_with_noise_array = reverse_normalization(matrix_with_noise_array, df_X)

        matrix_with_noise_array = pd.DataFrame(
            matrix_with_noise_array, columns=df_X.columns
        )
        error = np.abs(
            np.divide(
                (original_matrix - matrix_with_noise_array),
                original_matrix,
                out=np.zeros_like(matrix_with_noise_array),
                where=(matrix_with_noise_array != 0),
            )
        )
        axs[i].set_title(
            f"Noise:{int(noise_step*100/(DDPM.noise_steps - 1))}%, Mean Error:{error.mean().mean():.3f}",
            fontsize=14,
        )
        sns.heatmap(
            pd.DataFrame(
                error,
                columns=df_X.columns,
            ),
            vmin=0,
            vmax=0.2,
            ax=axs[i],
            # cmap="crest",
            cbar=False if i != len(axs) - 1 else True,
            xticklabels=True,
        )
    cbar = axs[-1].collections[0].colorbar
    cbar.set_ticks([0, 0.05, 0.1, 0.15, 0.2])
    cbar.set_ticklabels(["0%", "5%", "10%", "15%", "20%"])

    plt.show()

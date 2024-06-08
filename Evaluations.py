from matplotlib.font_manager import font_scalings
from requirements import *
from dataset import (
    normalization,
    reverse_normalization,
)
from Simulator import Simulator
import seaborn as sns


def plot_dataset(df_y):
    fig, ax = plt.subplots(1, 2)
    sns.scatterplot(
        data=df_y,
        y="gbw",
        x="gdc",
        ax=ax[0],
    )

    sns.scatterplot(
        data=df_y,
        y="idd",
        x="pm",
        ax=ax[1],
    )
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_xlabel("Gain[dB]", fontsize=14)
    ax[0].set_ylabel("Gain Bandwidth Product[Hz]", fontsize=14)
    ax[1].set_xlabel("Phase [deg]", fontsize=14)
    ax[1].set_ylabel("Bias Current[\\mu A]", fontsize=14)
    plt.suptitle("VCOTA Performace Dataset", fontsize=14)
    plt.show()


def Train_error(
    y_train,
    DDPM,
    best_weight,
    X_train,
    df_X,
):
    print("\n\n\nTrain Error")
    y_test = np.tile(y_train[-1].cpu().numpy(), (1000, 1))
    y_test = torch.tensor(y_test, dtype=torch.float32, device=DDPM.device)

    # start_time = time.time()
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test.shape[0], y_test, weight=best_weight
    )
    end_time = time.time()
    # y_test=pd.DataFrame(y_test.cpu().numpy(),columns=df_y.columns)
    # y_test.to_csv('y_test.csv')

    X_train = np.array(np.tile(X_train[-1].cpu().numpy(), (1000, 1)))

    df_Sampled = reverse_normalization(X_Sampled.cpu().numpy(), df_X)
    X_test = reverse_normalization(X_train, df_X)
    df_Sampled = pd.DataFrame(df_Sampled, columns=df_X.columns)
    X_train = pd.DataFrame(X_train, columns=df_X.columns)
    error = np.mean(
        np.abs(
            np.divide(
                (X_test - df_Sampled),
                df_Sampled,
                out=np.zeros_like(X_train),
                where=(X_train != 0),
            )
        ),
        axis=0,
    )
    print(f"\n{error}")


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
            normalization(df_Sampled, original=df_X, type_normilization="minmax"),
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
            normalization(df_X_test, original=df_X, type_normilization="minmax"),
            columns=df_X.columns,
        ),
        cmap="Spectral",
        xticklabels=True,
    )
    plt.title("VCOTA test Dataset")
    plt.show()


def target_Predictions(
    y_test,
    DDPM,
    best_weight,
    df_X,
    df_y,
    display=False,
    n_samples=100,
):
    if display:
        print("\n\n\nTarget Predictions")
    y_test = np.array(
        np.concatenate(
            [
                np.tile([50, 300e-6, 60e6, 65], (n_samples, 1)),
                np.tile([40, 700e-6, 150e6, 55], (n_samples, 1)),
                np.tile([50, 150e-6, 30e6, 65], (n_samples, 1)),
            ]
        )
    )
    y_test = pd.DataFrame(y_test, columns=df_y.columns)
    y_test.to_csv("y_test.csv")

    y_test_norm = normalization(y_test, original=df_y)
    y_test_norm = torch.tensor(y_test_norm, dtype=torch.float32, device=DDPM.device)
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test_norm.shape[0], y_test_norm, weight=best_weight
    )
    with open("best_simulator.json", "r") as file:
        hyper_parameters = json.load(file)
    simulator = Simulator(
        df_X.shape[-1],
        df_y.shape[-1],
        hidden_layers=hyper_parameters["hidden_layers"],
    ).to("cuda")
    simulator.load_state_dict(torch.load("Simulator.pth"))

    y_Sampled = simulator(X_Sampled)

    y_Sampled = reverse_normalization(y_Sampled.detach().cpu().numpy(), df_y)
    y_Sampled = pd.DataFrame(y_Sampled, columns=df_y.columns)

    X_Sampled = reverse_normalization(X_Sampled.cpu().numpy(), df_X)
    X_Sampled = pd.DataFrame(X_Sampled, columns=df_X.columns)
    error_1 = np.mean(
        np.abs(
            np.divide(
                (y_test[:n_samples] - y_Sampled[:n_samples]),
                y_test[:n_samples],
                out=np.zeros_like(y_test[:n_samples]),
                where=(y_test[:n_samples] != 0),
            )
        ),
        axis=0,
    )
    error_2 = np.mean(
        np.abs(
            np.divide(
                (
                    y_test[n_samples : n_samples * 2]
                    - y_Sampled[n_samples : n_samples * 2]
                ),
                y_test[n_samples : n_samples * 2],
                out=np.zeros_like(y_test[n_samples : n_samples * 2]),
                where=(y_test[n_samples : n_samples * 2] != 0),
            )
        ),
        axis=0,
    )
    error_3 = np.mean(
        np.abs(
            np.divide(
                (y_test[n_samples * 2 :] - y_Sampled[n_samples * 2 :]),
                y_test[n_samples * 2 :],
                out=np.zeros_like(y_test[n_samples * 2 :]),
                where=(y_test[n_samples * 2 :] != 0),
            )
        ),
        axis=0,
    )
    if display:

        print(f"\nTarget1:\n{X_Sampled[:n_samples].describe().loc[['mean', 'std']].T}")
        print(f"Error:\n{error_1}")
        print(
            f"\nTarget2:\n{X_Sampled[n_samples:n_samples*2].describe().loc[['mean', 'std']].T}"
        )
        print(f"Error:\n{error_2}")
        print(
            f"\nTarget3:\n{X_Sampled[n_samples*2:].describe().loc[['mean', 'std']].T}"
        )
        print(f"Error:\n{error_3}")


def test_performaces(
    y_test,
    DDPM,
    best_weight,
    df_X,
    df_y,
    display=False,
):
    if display:
        print("\n\n\nPerformance Error")
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test.shape[0], y_test, weight=best_weight
    )
    with open("best_simulator.json", "r") as file:
        hyper_parameters = json.load(file)
    simulator = Simulator(
        df_X.shape[-1],
        df_y.shape[-1],
        hidden_layers=hyper_parameters["hidden_layers"],
    ).to("cuda")
    simulator.load_state_dict(torch.load("Simulator.pth"))

    y_Sampled = simulator(X_Sampled)

    y_Sampled = reverse_normalization(y_Sampled.detach().cpu().numpy(), df_y)
    y_test = reverse_normalization(y_test.cpu().numpy(), df_y)
    y_Sampled = pd.DataFrame(y_Sampled, columns=df_y.columns)
    y_test = pd.DataFrame(y_test, columns=df_y.columns)
    error = np.mean(
        np.abs(
            np.divide(
                (y_test - y_Sampled),
                y_test,
                out=np.zeros_like(y_test),
                where=(y_test != 0),
            )
        ),
        axis=0,
    )
    if display:
        print(f"\n{error}")
    return error


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
    axs[0].set_title(f"Noise Step Percentage:{0}%", fontsize=14)
    sns.heatmap(
        pd.DataFrame(
            # normalization(original_matrix, original=df_X, type_normilization="minmax"),
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
        axs[i].set_title(
            f"Noise Step Percentage:{int(noise_step*100/(DDPM.noise_steps - 1))}%",
            fontsize=14,
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

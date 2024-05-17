from requirements import *
from dataset import (
    normalization,
    reverse_normalization,
)
from Simulator import Simulator


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
                (X_train - df_Sampled),
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
    X_test = reverse_normalization(X_test, df_X)
    df_Sampled = pd.DataFrame(df_Sampled, columns=df_X.columns)
    X_test = pd.DataFrame(X_test, columns=df_X.columns)
    error = np.mean(
        np.abs(
            np.divide(
                (X_test - df_Sampled),
                X_test,
                out=np.zeros_like(X_test),
                where=(X_test != 0),
            )
        ),
        axis=0,
    )
    print(f"\n{error}")
    plt.subplot(1, 2, 1)
    plt.imshow(df_Sampled, cmap="viridis", aspect="auto")
    # plt.colorbar()
    plt.title("VCOTA Sample Dataset")

    plt.subplot(1, 2, 2)
    plt.imshow(X_test, cmap="viridis", aspect="auto")
    plt.title("VCOTA test Dataset")
    plt.show()


def target_Predictions(
    df_y,
    DDPM,
    df_X,
    best_weight,
    n_samples=100,
):
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

    y_test = normalization(y_test, original=df_y)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=DDPM.device)
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test.shape[0], y_test, weight=best_weight
    )

    df_Sampled = reverse_normalization(X_Sampled.cpu().numpy(), df_X)
    df_Sampled = pd.DataFrame(df_Sampled, columns=df_X.columns)
    df_Sampled.to_csv("Sampled.csv")
    print(f"\nTarget1:\n{df_Sampled[:n_samples].describe().loc[['mean', 'std']].T}")
    print(
        f"\nTarget2:\n{df_Sampled[n_samples:n_samples*2].describe().loc[['mean', 'std']].T}"
    )
    print(f"\nTarget3:\n{df_Sampled[n_samples*2:].describe().loc[['mean', 'std']].T}")


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


def histogram(DDPM, best_weight, y_test, df_X, X_train):
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test.shape[0], y_test, weight=best_weight
    )
    end_time = time.time()
    df_Sampled = pd.DataFrame(X_Sampled.cpu().numpy(), columns=df_X.columns)

    df_Sampled_hist = reverse_normalization(df_Sampled, df_X)
    X_train_hist = reverse_normalization(X_train.cpu(), df_X)
    df_Sampled_hist = pd.DataFrame(
        df_Sampled_hist,
        columns=df_X.columns,
    )
    X_train_hist = pd.DataFrame(
        X_train_hist,
        columns=df_X.columns,
    )

    fig, axs = plt.subplots(2, df_Sampled.shape[1])
    for i in range(df_Sampled_hist.shape[1]):
        axs[0, i].set_title(f"Sample {df_Sampled_hist.columns[i]}")
        axs[0, i].hist(df_Sampled_hist.values[:, i], bins=50)
        axs[1, i].set_title(f"Input {df_Sampled_hist.columns[i]}")
        axs[1, i].hist(X_train_hist.values[:, i], bins=50)
    plt.show()

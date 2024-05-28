from requirements import *
from dataset import (
    reverse_normalize_values,
    normalize_values,
    augment_data,
    normalization,
    reverse_normalization,
)
from Evaluations import (
    Test_error,
    Train_error,
    target_Predictions,
    test_performaces,
    histogram,
)
import seaborn as sns
from optimization import HYPERPARAMETERS_OPT, GUIDANCE_WEIGTH_OPT
from DDPM import Diffusion

type = "MLP_skip"


def main():
    network = MLP if type == "MLP" else MLP_skip
    hyper_parameters = {
        "hidden_layers": [60],
        "learning_rate": 0.00001,
        "loss_type": "l2",
    }
    best_weight = 10

    ############################### VCOTA DATASET #############################

    dataframe = pd.read_csv("data/vcota.csv")
    df_X = dataframe[
        ["w8", "w6", "w4", "w10", "w1", "w0", "l8", "l6", "l4", "l10", "l1", "l0"]
    ]
    df_y = dataframe[["gdc", "idd", "gbw", "pm"]]
    fid, ax = plt.subplots(1, 2)
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
    X = normalization(df_X)
    y = normalization(df_y)

    plt.show()
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=0.5,
    )

    # y_train,X_train = augment_data(y_train,X_train,np.array([-1,1, -1, 0]), repetition_factor=10)

    DDPM = Diffusion(vect_size=X.shape[1])
    X = torch.tensor(X, dtype=torch.float32, device=DDPM.device)
    y = torch.tensor(y, dtype=torch.float32, device=DDPM.device)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=DDPM.device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=DDPM.device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=DDPM.device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=DDPM.device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=DDPM.device)

    # HYPERPARAMETERS_OPT(
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     df_X,
    #     df_y,
    #     network,
    #     type,
    #     epoch=50,
    #     n_trials=10,
    # )
    with open(f"best_hyperparameters{type}.json", "r") as file:
        hyper_parameters = json.load(file)

    DDPM.reverse_process(
        X,
        y,
        network,
        df_X,
        df_y,
        X_val=X_val,
        y_val=y_val,
        epochs=500,
        early_stop=False,
        **hyper_parameters,
    )
    torch.save(DDPM.model.state_dict(), f"{type}.pth")

    DDPM.model = network(
        input_size=X.shape[1],
        output_size=X.shape[1],
        y_dim=y.shape[-1],
        hidden_layers=hyper_parameters["hidden_layers"],
    )
    DDPM.model.load_state_dict(torch.load(f"{type}.pth"))

    GUIDANCE_WEIGTH_OPT(DDPM, y_val, df_X, df_y, type)
    with open(f"best_weight{type}.json", "r") as file:
        best_weight = json.load(file)["weight"]

    # start_time = time.time()

    # df_Sampled.to_csv('Uncod_Sampled.csv')
    histogram(DDPM, best_weight, y, df_X, X_train)
    Train_error(
        y_train,
        DDPM,
        best_weight,
        X_train,
        df_X,
    )

    Test_error(
        y_test,
        DDPM,
        best_weight,
        X_test,
        df_X,
    )

    test_performaces(
        y_test,
        DDPM,
        best_weight,
        df_X,
        df_y,
        display=True,
    )
    target_Predictions(
        y_test,
        DDPM,
        best_weight,
        df_X,
        df_y,
        display=True,
    )

    # Calculate mean squared error


if __name__ == "__main__":
    main()

    # Obter media e desvio padrao dos targets
    # UNET MLP

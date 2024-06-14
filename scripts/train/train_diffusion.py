import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from requirements import *
from Dataset import (
    normalization,
    reverse_normalization,
)


from Evaluations import (
    Test_error,
    Train_error,
    inference_error,
    test_performaces,
    histogram,
    see_noise_data,
    plot_dataset,
)
import seaborn as sns
from Optimizations import HYPERPARAMETERS_DDPM, GUIDANCE_WEIGTH_OPT
from DDPM import Diffusion


guidance = True
from Networks import MLP, MLP_skip


def main():
    network = MLP if type == "MLP" else MLP_skip
    hyper_parameters = {
        "hidden_layers": [60],
        "learning_rate": 0.00001,
        "loss_type": "l2",
    }
    best_weight = 10

    ############################### VCOTA DATASET #############################

    dataframe = pd.read_csv("./data/vcota.csv")
    df_X = dataframe[
        ["w8", "w6", "w4", "w10", "w1", "w0", "l8", "l6", "l4", "l10", "l1", "l0"]
    ]
    df_y = dataframe[["gdc", "idd", "gbw", "pm"]]

    # plot_dataset(df_y)

    X = normalization(df_X)
    y = normalization(df_y)

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
    DDPM.X_norm_min = torch.tensor(
        X.min(axis=0), device=DDPM.device, dtype=torch.float32
    )
    DDPM.X_norm_max = torch.tensor(
        X.max(axis=0), device=DDPM.device, dtype=torch.float32
    )

    X = torch.tensor(X, dtype=torch.float32, device=DDPM.device)
    y = torch.tensor(y, dtype=torch.float32, device=DDPM.device)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=DDPM.device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=DDPM.device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=DDPM.device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=DDPM.device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=DDPM.device)
    see_noise_data(DDPM, X_train, df_X)
    # HYPERPARAMETERS_DDPM(
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     df_X,
    #     df_y,
    #     network,
    #     type,
    #     epoch=2,
    #     n_trials=2,
    #     X_min=DDPM.X_norm_min,
    #     X_max=DDPM.X_norm_max,
    # )
    with open(f"./templates/best_hyperparameters{type}.json", "r") as file:
        hyper_parameters = json.load(file)

    DDPM.reverse_process(
        X,
        y,
        network,
        df_X,
        df_y,
        X_val=X_val,
        y_val=y_val,
        epochs=501,
        early_stop=False,
        **hyper_parameters,
    )
    torch.save(DDPM.model.state_dict(), f"./weights/{type}.pth")

    DDPM.model = network(
        input_size=X.shape[1],
        output_size=X.shape[1],
        y_dim=y.shape[-1],
        hidden_layers=hyper_parameters["hidden_layers"],
    )
    DDPM.model.load_state_dict(torch.load(f"./weights/{type}.pth"))
    GUIDANCE_WEIGTH_OPT(DDPM, y_val, df_X, df_y, type)
    with open(f"./templates/best_hyperparameters{type}.json", "r") as file:
        hyper_parameters = json.load(file)
    ##### EVALUATIONS #################################

    histogram(
        DDPM,
        hyper_parameters["guidance_weight"],
        y,
        df_X,
        X,
    )
    Train_error(
        y_train,
        DDPM,
        hyper_parameters["guidance_weight"],
        X_train,
        df_X,
    )

    Test_error(
        y_test,
        DDPM,
        hyper_parameters["guidance_weight"],
        X_test,
        df_X,
    )

    test_performaces(
        y_test,
        DDPM,
        hyper_parameters["guidance_weight"],
        df_X,
        df_y,
        display=True,
    )
    inference_error(
        y_test,
        DDPM,
        hyper_parameters["guidance_weight"],
        df_X,
        df_y,
        display=True,
    )

    # Calculate mean squared error


if __name__ == "__main__":
    main()

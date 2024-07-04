import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libraries import *
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
from Optimizations import HYPERPARAMETERS_DDPM, GUIDANCE_WEIGHT_OPT
from Diffusion import DiffusionDPM


guidance = True
from Networks import MLP, MLP_skip, EoT

nn_type = "MLP"  ## define NN type


def main():
    nn_map = {
        "MLP_skip": MLP_skip,
        "MLP": MLP,
        "EoT": EoT,
    }
    network = nn_map[nn_type]

    ############################### VCOTA DATASET #############################

    torch.cuda.empty_cache()
    dataframe = pd.read_csv("./data/vcota.csv")
    df_X = dataframe[
        [
            "w8",
            "w6",
            "w4",
            "w10",
            "w1",
            "w0",
            "l8",
            "l6",
            "l4",
            "l10",
            "l1",
            "l0",
        ]
    ]
    df_y = dataframe[
        [
            "gdc",
            "idd",
            "gbw",
            "pm",
        ]
    ]

    # plot_dataset(df_y)

    X = normalization(df_X.copy()).values
    y = normalization(df_y.copy()).values

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

    norm_min = normalization(
        pd.DataFrame(
            np.array(
                [
                    1e-6,
                    1e-6,
                    1e-6,
                    1e-6,
                    1e-6,
                    1e-6,
                    0.34e-6,
                    0.34e-6,
                    0.34e-6,
                    0.34e-6,
                    0.34e-6,
                    0.34e-6,
                ]
            )[None],
            columns=df_X.columns,
        ),
        df_original=df_X,
    ).values
    norm_max = normalization(
        pd.DataFrame(
            np.array(
                [
                    100e-6,
                    100e-6,
                    100e-6,
                    100e-6,
                    100e-6,
                    100e-6,
                    0.94e-6,
                    0.94e-6,
                    0.94e-6,
                    0.94e-6,
                    0.94e-6,
                    0.94e-6,
                ]
            )[None],
            columns=df_X.columns,
        ),
        df_original=df_X,
    ).values
    with open(f"./templates/network_templates.json", "r") as file:
        data = json.load(file)
    hyper_parameters = data[nn_type]
    DDPM = DiffusionDPM(
        vect_size=X.shape[1],
        X_norm_max=norm_max,
        X_norm_min=norm_min,
        noise_steps=hyper_parameters["noise_steps"],
    )

    X = torch.tensor(X, dtype=torch.float32, device=DDPM.device)
    y = torch.tensor(y, dtype=torch.float32, device=DDPM.device)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=DDPM.device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=DDPM.device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=DDPM.device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=DDPM.device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=DDPM.device)
    # see_noise_data(DDPM, X_train, df_X)
    # hyper_parameters = HYPERPARAMETERS_DDPM(
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     df_X,
    #     df_y,
    #     network,
    #     nn_type,
    #     hyper_parameters["noise_steps"],
    #     epoch=1000,
    #     n_trials=20,
    #     X_min=norm_min,
    #     X_max=norm_max,
    #     frequency_print=200,
    #     delete_previous_study=True,
    #     guidance_weight=hyper_parameters["guidance_weight"],
    # )

    # DDPM.reverse_process(
    #     X_train,
    #     y_train,
    #     network,
    #     df_X,
    #     df_y,
    #     nn_type,
    #     X_val=X_val,
    #     y_val=y_val,
    #     epochs=10000,
    #     early_stop=False,
    #     **hyper_parameters,
    #     frequency_print=100,
    # )

    DDPM.model = network(
        input_size=X.shape[1],
        output_size=X.shape[1],
        y_dim=y.shape[-1],
        **hyper_parameters["nn_template"],
    ).cuda()
    path_DDPM = max(
        glob.glob(f"./weights/{nn_type}/noise{DDPM.noise_steps}/*.pth"),
        key=os.path.getctime,
    )
    DDPM.model.load_state_dict(torch.load(path_DDPM))
    # hyper_parameters = GUIDANCE_WEIGHT_OPT(
    #     DDPM,
    #     y_val,
    #     df_X,
    #     df_y,
    #     nn_type,
    #     n_trials=20,
    # )

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


if __name__ == "__main__":
    main()

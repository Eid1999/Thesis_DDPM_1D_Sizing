import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libraries import *
from Dataset import (
    normalization,
    reverse_normalization,
)

os.environ[" SEGMENT_ANYTHING_FAST_USE_FLASH_4"] = "0"


from Evaluations import (
    Test_error,
    Train_error,
    inference_error,
    test_performaces,
    histogram,
    see_noise_data,
    plot_dataset,
    plot_targets,
)
import seaborn as sns
from Optimizations import HYPERPARAMETERS_DDPM, GUIDANCE_WEIGHT_OPT
from Diffusion import DiffusionDPM


guidance = True
from Networks import MLP, MLP_skip, EoT

nn_type = "MLP"  ## define NN type


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    nn_map = {
        "MLP_skip": MLP_skip,
        "MLP": MLP,
        "EoT": EoT,
    }
    network = nn_map[nn_type]
    torch.cuda.empty_cache()
    ############################### DATASET #############################
    if data_type == "vcota":

        dataframe = pd.read_csv("./data/vcota.csv", sep="\s+")  # type: ignore # ignore
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
                        0.3e-6,
                        0.34e-6,
                    ]
                )[None],
                columns=df_X.columns,
            ),
            data_type=data_type,
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
            data_type=data_type,
            df_original=df_X,
        ).values
    if data_type == "folded_vcota":
        dataframe = pd.read_csv("./data/folded_vcota.csv")
        df_X = dataframe[
            [
                "_wpmbiasp",
                "_wp5",
                "_wp4",
                "_wp1",
                "_wp0",
                "_wnmbiasp",
                "_wnmbiasn",
                "_wn8",
                "_wn6",
                "_wn4",
                "_wn0",
                "_nfpmbiasp",
                "_nfp5",
                "_nfp4",
                "_nfp1",
                "_nfp0",
                "_nfnmbiasp",
                "_nfnmbiasn",
                "_nfn8",
                "_nfn6",
                "_nfn4",
                "_nfn0",
                "_lpmbiasp",
                "_lp5",
                "_lp4",
                "_lp1",
                "_lp0",
                "_lnmbiasp",
                "_lnmbiasn",
                "_ln8",
                "_ln6",
                "_ln4",
                "_ln0",
            ]
        ]
        df_y = dataframe[
            [
                "gdc",
                "idd",
                "gbw",
                "pm",
                "cload",
            ]
        ]

    # plot_dataset(df_y)
    # plot_targets(df_y, data_type)
    X = normalization(
        df_X.copy(),
        data_type=data_type,
    ).values
    y = normalization(
        df_y.copy(),
        data_type=data_type,
        poly_bool=True,
    ).values
    if data_type == "folded_vcota":
        norm_min = X.min()
        norm_max = X.max()

    if False:
        value = np.unique(y[:, -1])[0]
        idx = np.any(np.isin(y, value), axis=1)
        mask = np.ones(X.shape[0], dtype=bool)
        idxa = np.where(idx)[0]
        mask[idxa] = False
        y_test = y[idxa]
        X_test = X[idxa]
        y = y[mask]
        X = X[mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val,
        y_val,
        test_size=0.5,
        random_state=0,
    )
    plot_targets(
        df_y,
        data_type,
        y_test=y_test,
    )
    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        data = json.load(file)
    hyper_parameters = data[nn_type]
    DDPM = DiffusionDPM(
        vect_size=X.shape[1],
        X_norm_max=norm_max,
        X_norm_min=norm_min,
        noise_steps=hyper_parameters["noise_steps"],
    )
    print(f"Training {nn_type} with the {data_type}")

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
    #     frequency_print=100,
    #     delete_previous_study=True,
    #     #     guidance_weight=hyper_parameters["guidance_weight"],
    #     data_type=data_type,
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
    #     frequency_print=50,
    #     data_type=data_type,
    #     # guidance_weight=hyper_parameters["guidance_weight"],
    # )

    DDPM.model = network(
        input_size=X.shape[1],
        output_size=X.shape[1],
        y_dim=y.shape[-1],
        **hyper_parameters["nn_template"],
    ).to(DDPM.device)
    pattern = re.compile(r"EPOCH\d+-PError: (\d+\.\d+)\.pth")
    smallest_error = float("inf")
    for filename in os.listdir(
        f"./weights/{data_type}/{nn_type}/noise{DDPM.noise_steps}"
    ):
        match = pattern.match(filename)
        if match:
            perror = float(match.group(1))
            if perror < smallest_error:
                smallest_error = perror
                path_DDPM = filename
    path_DDPM = f"./weights/{data_type}/{nn_type}/noise{DDPM.noise_steps}/{path_DDPM}"
    DDPM.model.load_state_dict(torch.load(path_DDPM, weights_only=True))
    # hyper_parameters = GUIDANCE_WEIGHT_OPT(
    #     DDPM,
    #     y_val,
    #     df_X,
    #     df_y,
    #     nn_type,
    #     n_trials=50,
    #     save_graph=False,
    #     data_type=data_type,
    # )

    ##### EVALUATIONS #################################

    # histogram(
    #     DDPM,
    #     hyper_parameters["guidance_weight"],
    #     y,
    #     df_X,
    #     X,
    #     data_type=data_type,
    # )
    # Train_error(
    #     y_train,
    #     DDPM,
    #     hyper_parameters["guidance_weight"],
    #     X_train,
    #     df_X,
    #     data_type=data_type,
    # )

    # Test_error(
    #     y_test,
    #     DDPM,
    #     hyper_parameters["guidance_weight"],
    #     X_test,
    #     df_X,
    #     data_type=data_type,
    #     display=False,
    # )

    # test_performaces(
    #     y_test,
    #     DDPM,
    #     hyper_parameters["guidance_weight"],
    #     df_X,
    #     df_y,
    #     display=True,
    #     save=False,
    #     nn_type=nn_type,
    #     data_type=data_type,  # ignore
    # )
    # inference_error(
    #     nn_type,
    #     DDPM,
    #     hyper_parameters["guidance_weight"],
    #     df_X,
    #     df_y,
    #     display=True,
    #     save=True,
    #     data_type=data_type,
    # )
    # print(path_DDPM)


if __name__ == "__main__":
    main()

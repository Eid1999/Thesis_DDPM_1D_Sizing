import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Networks import Simulator
from Dataset import normalization, reverse_normalization


def inference_error(
    nn_type: str,
    DDPM,
    best_weight: float,
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    display: bool = False,
    n_samples: int = 100,
    save: bool = False,
    data_type: str = "vcota",
) -> None:
    columns = df_y.columns if data_type == "vcota" else df_y.columns[:-1]
    mean = df_y[columns].mean()
    std_dev = df_y[columns].std()
    # start = idx + 2
    if data_type == "folded_vcota":
        cload_value = [df_y["cload"].mode()[0]]

        point1 = mean
        point2 = mean + std_dev
        point3 = mean + 2 * std_dev
        point4 = mean + 3 * std_dev
        y_target = pd.DataFrame(
            np.concatenate(
                [
                    np.tile(np.concatenate([point1, cload_value]), (n_samples, 1)),
                    np.tile(np.concatenate([point2, cload_value]), (n_samples, 1)),
                    np.tile(np.concatenate([point3, cload_value]), (n_samples, 1)),
                    np.tile(np.concatenate([point4, cload_value]), (n_samples, 1)),
                    # np.tile(np.concatenate([point1, cload_value]),(n_samples, 1))
                ]
            ),
            columns=df_y.columns,
        )
    if data_type == "vcota":
        y_target = np.array(
            np.concatenate(
                [
                    np.tile([50, 300e-6, 60e6, 65], (n_samples, 1)),
                    np.tile([40, 700e-6, 150e6, 55], (n_samples, 1)),
                    np.tile([50, 150e-6, 30e6, 65], (n_samples, 1)),
                    np.tile([53, 350e-6, 65e6, 55], (n_samples, 1)),
                ]
            )
        )
        y_target = pd.DataFrame(y_target, columns=df_y.columns)
    if display:
        print("\n\n\nTarget Predictions")

    y_target_norm = normalization(
        y_target,
        df_original=df_y,
        data_type=data_type,
        poly_bool=True,
    ).values
    y_target_norm = torch.tensor(
        y_target_norm,
        dtype=torch.float32,
        device=DDPM.device,
    )
    X_Sampled = DDPM.sampling(
        DDPM.model,
        y_target_norm.shape[0],
        y_target_norm,
        weight=best_weight,
    )
    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        hyper_parameters = json.load(file)["Simulator"]
    simulator = Simulator(
        df_X.shape[-1] + 1 if data_type == "folded_vcota" else df_X.shape[-1],
        df_y.shape[-1] - 1 if data_type == "folded_vcota" else df_y.shape[-1],
        **hyper_parameters["nn_template"],
    ).to("cuda")
    simulator.load_state_dict(
        torch.load(f"./weights/{data_type}/Simulator.pth", weights_only=True)
    )

    y_target_norm = y_target_norm[:, : df_y.shape[-1]]
    y_Sampled = simulator(
        (
            torch.cat((X_Sampled, y_target_norm[:, -1][:, None]), dim=-1)
            if data_type == "folded_vcota"
            else X_Sampled
        ),
    )
    if data_type == "folded_vcota":
        y_Sampled = torch.cat(
            (
                y_Sampled,
                y_target_norm[:, -1][:, None],
            ),
            dim=-1,
        )
    y_Sampled = pd.DataFrame(
        y_Sampled.detach().cpu().numpy(),
        columns=df_y.columns,
    )
    X_Sampled = pd.DataFrame(
        X_Sampled.cpu().numpy(),
        columns=df_X.columns,
    )
    y_Sampled = reverse_normalization(
        y_Sampled,
        df_y.copy(),
        data_type=data_type,
    )
    X_Sampled = reverse_normalization(
        X_Sampled,
        df_X.copy(),
        data_type=data_type,
    )
    for i in range(len(y_target) // n_samples):
        error = np.min(
            np.abs(
                np.divide(
                    (
                        y_target[n_samples * i : (1 + i) * n_samples]
                        - y_Sampled[n_samples * i : (1 + i) * n_samples]
                    ),
                    y_target[n_samples * i : (1 + i) * n_samples],
                    out=np.zeros_like(y_target[n_samples * i : (1 + i) * n_samples]),
                    where=(y_target[n_samples * i : (1 + i) * n_samples] != 0),
                )
            ),
            axis=0,
        )
        if display:

            print(
                f"\nTarget{i+1}:\n{X_Sampled[n_samples * i : (1 + i) * n_samples].describe().loc[['mean', 'std']].T}"
            )
            print(f"Error:\n{error}")
    if save:
        path = f"points_to_simulate/{data_type}/target/"
        if data_type == "folded_vcota":
            X_Sampled = pd.concat(
                (X_Sampled, y_target["cload"]),
                axis=1,
            )
        os.makedirs(path, exist_ok=True)
        X_Sampled.to_csv(f"{path}sizing_target{nn_type}.csv")
        y_target.to_csv(f"{path}real_target{nn_type}.csv")
        y_Sampled.to_csv(f"{path}nn_target{nn_type}.csv")

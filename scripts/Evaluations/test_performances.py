import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Dataset import normalization, reverse_normalization
from Networks import Simulator


def test_performaces(
    y_test: torch.Tensor,
    DDPM,
    best_weight: float,
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    display: bool = False,
    save: bool = False,
    nn_type: str = "MLP",
    data_type: str = "vcota",
):
    if display:
        print("\n\n\nTest Performance Error")
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(),
        y_test.shape[0],
        y_test,
        weight=best_weight,
    )
    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        hyper_parameters = json.load(file)["Simulator"]
    simulator = Simulator(
        df_X.shape[-1] + 1 if data_type == "folded_vcota" else df_X.shape[-1],
        df_y.shape[-1] - 1 if data_type == "folded_vcota" else df_y.shape[-1],
        **hyper_parameters["nn_template"],
    ).to("cuda")
    simulator.load_state_dict(torch.load(f"./weights/{data_type}/Simulator.pth"))

    y_Sampled = simulator(
        (
            torch.cat(
                (X_Sampled, y_test[:, -1][:, None]),
                dim=-1,
            )
            if data_type == "folded_vcota"
            else X_Sampled
        ),
    )
    if data_type == "folded_vcota":
        y_Sampled = torch.cat(
            (
                y_Sampled,
                y_test[:, -1][:, None],
            ),
            dim=-1,
        )

    df_y_Sampled = pd.DataFrame(
        y_Sampled.detach().cpu().numpy(),
        columns=df_y.columns,
    )
    df_y_Sampled = reverse_normalization(
        df_y_Sampled,
        df_y.copy(),
        data_type=data_type,
    )
    df_y_test = pd.DataFrame(
        y_test.detach().cpu().numpy(),
        columns=df_y.columns,
    )
    df_y_test = reverse_normalization(
        df_y_test,
        df_y.copy(),
        data_type=data_type,
    )
    df_y_test_aux = df_y_test.copy()
    df_y_Sampled_aux = df_y_Sampled.copy()
    if data_type == "folded_vcota":
        df_y_test_aux.drop(columns=["cload"], inplace=True)
        df_y_Sampled_aux.drop(columns=["cload"], inplace=True)
    error = np.mean(
        np.abs(
            np.divide(
                (df_y_test_aux - df_y_Sampled_aux),
                df_y_test_aux,
                out=np.zeros_like(df_y_test_aux),
                where=(df_y_test_aux != 0),
            )
        ),
        axis=0,
    )
    if display:
        print(f"\n{error} \n Mean: {error.mean()}")
    if save:
        path = f"points_to_simulate/{data_type}/test/"
        df_X_Sampled = reverse_normalization(
            pd.DataFrame(X_Sampled.detach().cpu().numpy(), columns=df_X.columns),
            df_X.copy(),
            data_type=data_type,
        )
        if data_type == "folded_vcota":
            df_X_Sampled = pd.concat(
                (df_X_Sampled, df_y_test["cload"]),
                axis=1,
            )
        os.makedirs(path, exist_ok=True)
        df_X_Sampled.to_csv(f"{path}sizing_test{nn_type}.csv")
        df_y_test.to_csv(f"{path}real_test{nn_type}.csv")
        df_y_Sampled.to_csv(f"{path}nn_test{nn_type}.csv")
    return error

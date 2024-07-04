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
):
    if display:
        print("\n\n\nPerformance Error")
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(),
        y_test.shape[0],
        y_test,
        weight=best_weight,
    )
    with open("./templates/network_templates.json", "r") as file:
        hyper_parameters = json.load(file)["Simulator"]
    simulator = Simulator(
        df_X.shape[-1],
        df_y.shape[-1],
        **hyper_parameters["nn_template"],
    ).to("cuda")
    simulator.load_state_dict(torch.load("./weights/Simulator.pth"))

    y_Sampled = simulator(X_Sampled)
    df_y_Sampled = pd.DataFrame(
        y_Sampled.detach().cpu().numpy(),
        columns=df_y.columns,
    )
    df_y_Sampled = reverse_normalization(
        df_y_Sampled,
        df_y.copy(),
    )
    df_y_test = pd.DataFrame(
        y_test.detach().cpu().numpy(),
        columns=df_y.columns,
    )
    df_y_test = reverse_normalization(
        df_y_test,
        df_y.copy(),
    )
    error = np.mean(
        np.abs(
            np.divide(
                (df_y_test - df_y_Sampled),
                df_y_test,
                out=np.zeros_like(df_y_test),
                where=(df_y_test != 0),
            )
        ),
        axis=0,
    )
    if display:
        print(f"\n{error}")
    return error

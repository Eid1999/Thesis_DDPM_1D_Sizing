from requirements import *

from Dataset import normalization, reverse_normalization
from Networks import Simulator


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
        DDPM.model.cuda(),
        y_test.shape[0],
        y_test,
        weight=best_weight,
    )
    with open("./templates/best_simulator.json", "r") as file:
        hyper_parameters = json.load(file)
    simulator = Simulator(
        df_X.shape[-1],
        df_y.shape[-1],
        hidden_layers=hyper_parameters["hidden_layers"],
    ).to("cuda")
    simulator.load_state_dict(torch.load("./weights/Simulator.pth"))

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

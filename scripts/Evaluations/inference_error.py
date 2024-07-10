import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Networks import Simulator
from Dataset import normalization, reverse_normalization


def inference_error(
    y_test,
    DDPM,
    best_weight,
    df_X,
    df_y,
    display=False,
    n_samples=100,
) -> None:
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

    y_test_norm = normalization(y_test, df_original=df_y).values
    y_test_norm = torch.tensor(y_test_norm, dtype=torch.float32, device=DDPM.device)
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test_norm.shape[0], y_test_norm, weight=best_weight
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
    y_Sampled = pd.DataFrame(
        y_Sampled.detach().cpu().numpy(),
        columns=df_y.columns,
    )
    X_Sampled = pd.DataFrame(
        X_Sampled.cpu().numpy(),
        columns=df_X.columns,
    )
    y_Sampled = reverse_normalization(y_Sampled, df_y.copy())
    X_Sampled = reverse_normalization(X_Sampled, df_X.copy())
    error_1 = np.min(
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
    error_2 = np.min(
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
    error_3 = np.min(
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

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Diffusion import DiffusionDPM
from Evaluations import test_performaces
from optuna.visualization import (
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_optimization_history,
    plot_timeline,
    plot_slice,
    plot_edf,
)


def GUIDANCE_WEIGHT_OPT(
    DDPM: DiffusionDPM,
    y_val: torch.Tensor,
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    nn_type: str,
    n_trials: int = 2,
) -> dict:
    def objective(trial):
        params = {
            "weight": trial.suggest_float("weight", 0.1, 20, log=True),
        }
        error = test_performaces(
            y_val,
            DDPM,
            params["weight"],
            df_X,
            df_y,
            display=False,
        )

        return np.mean(error)

    sampler = (optuna.samplers.TPESampler(seed=0),)
    study = optuna.create_study(
        direction="minimize"
    )  # We want to minimize the objective function
    study.optimize(objective, n_trials=n_trials)
    with open(f"./templates/network_templates.json", "r") as file:
        data = json.load(file)
    best_trial = study.best_trial
    data[nn_type]["guidance_weight"] = best_trial.params["weight"]

    with open(f"./templates/network_templates.json", "w") as file:
        json.dump(data, file, indent=4)
    x_values = []
    y_values = []
    hyperparameter = data[nn_type]
    for trial in study.trials:
        x_values.append(trial.params["weight"])
        values = trial.values[0] if trial.values != None else 0  # x-axis value
        y_values.append(values * 100)
    data = {
        "Weights": x_values,
        "Mean Performance Error": y_values,
    }

    data = pd.DataFrame(data)
    sns.lineplot(
        data=data,
        x="Weights",
        y="Mean Performance Error",
        marker="o",
    )

    plt.xlabel("Weights", fontsize=14)
    plt.ylabel("Mean Performance Error[%]", fontsize=14)
    plt.title("Epochs=500,Noise Step=100, Scaler=0.05", fontsize=14)
    plt.xscale("log")
    plt.show()

    return hyperparameter

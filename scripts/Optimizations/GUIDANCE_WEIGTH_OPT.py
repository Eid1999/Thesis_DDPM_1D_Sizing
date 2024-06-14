from requirements import *
from DDPM import Diffusion
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
import seaborn as sns


def GUIDANCE_WEIGTH_OPT(DDPM, y_val, df_X, df_y, type, n_trials=50):
    def objective(trial):
        params = {
            "weight": trial.suggest_float("weight", 2, 100, log=True),
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
    with open(f"best_hyperparameters{type}.json", "r") as file:
        hyper_parameters = json.load(file)
    best_trial = study.best_trial
    hyper_parameters["guidance_weight"] = best_trial.params["weight"]
    with open(f"best_hyperparameters{type}.json", "w") as file:
        json.dump(hyper_parameters, file, indent=4)
    x_values = []
    y_values = []
    for trial in study.trials:
        x_values.append(trial.params["weight"])
        values = trial.values if trial.values != None else 0  # x-axis value
        y_values.append(values * 100)
    data = {
        "Weigths": x_values,
        "Mean Performance Error": y_values,
    }

    data = pd.DataFrame(data)
    sns.lineplot(
        data=data,
        x="Weigths",
        y="Mean Performance Error",
        marker="o",
    )

    plt.xlabel("Weigths", fontsize=14)
    plt.ylabel("Mean Performance Error[%]", fontsize=14)
    plt.title("Epochs=500,Noise Step=100, Scaler=0.05", fontsize=14)
    # plt.xscale("log")
    plt.show()

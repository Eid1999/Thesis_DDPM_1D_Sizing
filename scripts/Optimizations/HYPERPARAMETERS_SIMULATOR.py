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
import seaborn as sns
from utils.utils_Simulator import epoch_loop


def HYPERPARAMETERS_OPT(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_trials: int = 20,
    num_epochs: int = 100,
    delete_previous_study: bool = True,
) -> None:
    def objective(trial: Trial) -> float:
        num_layers = trial.suggest_int("num_layers", 2, 10)
        params = {
            "nn_template": {
                "hidden_layers": [
                    trial.suggest_int(
                        f"hidden_size{i}",
                        20,
                        5000,
                        log=True,
                    )
                    for i in range(num_layers)
                ],
            },
            "batch_size": trial.suggest_int(
                "batch_size",
                32,
                500,
                log=True,
            ),
            "lr": trial.suggest_float(
                "lr",
                1e-6,
                1e-3,
                log=True,
            ),
        }
        _, val_loss = epoch_loop(
            X_train,
            y_train,
            X_val,
            y_val,
            **params,
            n_epoch=num_epochs,
            trial=trial,
        )

        return val_loss

    with open("./templates/network_templates.json", "r") as file:
        hyper_parameters = json.load(file)
    if delete_previous_study:
        try:
            os.remove(f"optuna_studies/Simulator.db")
        except FileNotFoundError:
            pass
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=50),
        study_name=f"study",
        storage=f"sqlite:///optuna_studies/Simulator.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=num_trials)
    best_trial = study.best_trial
    hyper_parameters["Simulator"].update(
        {
            "nn_template": {
                "hidden_layers": [
                    best_trial.params[f"hidden_size{i}"]
                    for i in range(best_trial.params["num_layers"])
                ],
            },
            "batch_size": best_trial.params["batch_size"],
            "lr": best_trial.params["lr"],
        }
    )
    plot_intermediate_values(study).update_layout(
        xaxis_title="Epoch",
        yaxis_title="Validation Loss",
    ).write_html(f"./html_graphs/epoch_graphSimulator.html")
    plot_timeline(study).write_html("./html_graphs/plot_timelineSimulator.html")
    # plot_parallel_coordinate(study).write_html("parallel_coordinate.html")
    plot_param_importances(study).write_html(
        "./html_graphs/param_importancesSimulator.html"
    )
    plot_optimization_history(study).update_layout(
        xaxis_title="Trials",
        yaxis_title="Validation Loss",
    ).write_html(f"./html_graphs/optimization_historySimulator.html")
    with open("./templates/network_templates.json", "w") as file:
        json.dump(hyper_parameters, file, indent=4)

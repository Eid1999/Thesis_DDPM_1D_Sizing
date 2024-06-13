from requirements import *
from scripts.DDPM import Diffusion
from scripts.Evaluations import test_performaces
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
from scripts.utils.Simulator import epoch_loop


def HYPERPARAMETERS_OPT(X_train, y_train, X_val, y_val):
    def objective(trial):
        num_layers = trial.suggest_int("num_layers", 2, 20)
        params = {
            "hidden_layers": [
                trial.suggest_int(
                    f"hidden_size{i}",
                    20,
                    5000,
                    log=True,
                )
                for i in range(num_layers)
            ],
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
            n_epoch=100,
            trial=trial,
        )

        return val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(objective, n_trials=20)
    best_trial = study.best_trial
    best_params = {
        "hidden_layers": [
            best_trial.params[f"hidden_size{i}"]
            for i in range(best_trial.params["num_layers"])
        ],
        "batch_size": best_trial.params["batch_size"],
        "lr": best_trial.params["lr"],
    }
    plot_intermediate_values(study).update_layout(
        xaxis_title="Epoch",
        yaxis_title="Validation Loss",
    ).write_html(f"epoch_graphSimulator.html")
    plot_timeline(study).write_html("plot_timelineSimulator.html")
    # plot_parallel_coordinate(study).write_html("parallel_coordinate.html")
    plot_param_importances(study).write_html("param_importancesSimulator.html")
    plot_optimization_history(study).update_layout(
        xaxis_title="Trials",
        yaxis_title="Validation Loss",
    ).write_html(f"optimization_historySimulator.html")
    with open("best_simulator.json", "w") as file:
        json.dump(best_params, file, indent=4)

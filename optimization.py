from requirements import *
from DDPM import Diffusion
from Evaluations import test_performaces, target_Predictions
from optuna.visualization import (
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_optimization_history,
    plot_timeline,
)


def HYPERPARAMETERS_OPT(
    X_train,
    y_train,
    X_val,
    y_val,
    df_X,
    df_y,
    network,
    type,
    epoch=100,
    n_trials=20,
):
    def objective(trial):
        num_layers = trial.suggest_int("num_layers", 5, 30)
        params = {
            "hidden_layers": [
                trial.suggest_int(f"hidden_size{i}", 20, 5000, log=True)
                for i in range(num_layers)
            ],
            "batch_size": trial.suggest_int("batch_size", 32, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "loss_type": trial.suggest_categorical("loss_type", ["l1", "l2"]),
        }
        DDPM = Diffusion(vect_size=X_train.shape[1])
        loss, error = DDPM.reverse_process(
            X_train,
            y_train,
            network,
            df_X,
            df_y,
            epochs=epoch,
            X_val=X_val,
            y_val=y_val,
            **params,
            trial=trial,
            early_stop=False,
        )

        return np.mean(error)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        # pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(objective, n_trials=n_trials)  # type: ignore
    best_trial = study.best_trial
    best_params = {
        "hidden_layers": [
            best_trial.params[f"hidden_size{i}"]
            for i in range(best_trial.params["num_layers"])
        ],
        "batch_size": best_trial.params["batch_size"],
        "learning_rate": best_trial.params["learning_rate"],
        "loss_type": best_trial.params["loss_type"],
    }

    plot_intermediate_values(study).update_layout(
        xaxis_title="Epoch",
        yaxis_title="Performace Error",
    ).write_html(f"epoch_graph{type}.html")
    plot_timeline(study).write_html(f"plot_timeline{type}.html")
    # plot_parallel_coordinate(study).write_html("parallel_coordinate.html")
    plot_param_importances(study).write_html(f"param_importances{type}.html")
    plot_optimization_history(study).update_layout(
        xaxis_title="Trials",
        yaxis_title="Performace Error",
    ).write_html(f"optimization_history{type}.html")
    with open(f"best_hyperparameters{type}.json", "w") as file:
        json.dump(best_params, file, indent=4)


def GUIDANCE_WEIGTH_OPT(DDPM, y_val, df_X, df_y, type):
    def objective(trial):
        params = {
            "weight": trial.suggest_float("weight", 0.01, 10, log=True),
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
    study.optimize(objective, n_trials=20)
    with open(f"best_weight{type}.json", "w") as file:
        json.dump(study.best_params, file, indent=4)

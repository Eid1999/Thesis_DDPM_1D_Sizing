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


def HYPERPARAMETERS_DDPM(
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
    X_min=None,
    X_max=None,
):
    def objective(trial):
        num_layers = trial.suggest_int("num_layers", 7, 20)
        params = {
            "hidden_layers": [
                trial.suggest_int(f"hidden_size{i+1}", 800, 5000, log=True)
                for i in range(num_layers)
            ],
            "batch_size": trial.suggest_int("batch_size", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True),
            "loss_type": trial.suggest_categorical("loss_type", ["l1", "l2"]),
            "guidance_weight": trial.suggest_float(
                "guidance_weight",
                0.1,
                40,
                log=True,
            ),
        }
        DDPM = Diffusion(vect_size=X_train.shape[1])

        if X_max != None and X_min != None:
            DDPM.X_norm_max = X_max
            DDPM.X_norm_min = X_min

        loss, error = DDPM.reverse_process(
            X_train,
            y_train,
            network,
            df_X,
            df_y,
            epochs=epoch + 1,
            X_val=X_val,
            y_val=y_val,
            **params,
            trial=trial,
            early_stop=False,
        )
        del DDPM
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
            best_trial.params[f"hidden_size{i+1}"]
            for i in range(best_trial.params["num_layers"])
        ],
        "batch_size": best_trial.params["batch_size"],
        "learning_rate": best_trial.params["learning_rate"],
        "loss_type": best_trial.params["loss_type"],
        "guidance_weight": best_trial.params["guidance_weight"],
    }

    plot_intermediate_values(study).update_layout(
        xaxis_title="Epoch",
        yaxis_title="Mean Performace Error",
    ).write_html(f"./html_graphs/epoch_graph{type}.html")
    plot_timeline(study).write_html(f"./html_graphs/plot_timeline{type}.html")
    # plot_parallel_coordinate(study).write_html("parallel_coordinate.html")
    plot_param_importances(study).write_html(
        f"./html_graphs/param_importances{type}.html"
    )
    plot_optimization_history(study).update_layout(
        xaxis_title="Trials",
        yaxis_title="Mean Performace Error",
    ).write_html(f"./html_graphs/optimization_history{type}.html")
    with open(f"./templates/best_hyperparameters{type}.json", "w") as file:
        json.dump(best_params, file, indent=4)

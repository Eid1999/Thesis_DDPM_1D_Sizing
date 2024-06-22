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
import math


def HYPERPARAMETERS_DDPM(
    X_train,
    y_train,
    X_val,
    y_val,
    df_X,
    df_y,
    network,
    type,
    noise_steps,
    epoch=100,
    n_trials=20,
    X_min=None,
    X_max=None,
    guidance_weight=0.3,
    frequency_print=200,
    delete_previous_study=True,
):
    def objective(trial):
        real_layers = trial.suggest_int("num_layers", 5, 30)

        num_layers = (
            real_layers
            if type == "MLP"
            else real_layers // 2 + math.ceil(real_layers / 2) - real_layers // 2
        )
        params = {
            "hidden_layers": [
                trial.suggest_int(f"hidden_size{i+1}", 100, 2000, log=True)
                for i in range(num_layers)
            ],
            "num_heads": [
                trial.suggest_int(f"num_heads{i+1}", 1, 32, log=True)
                for i in range(num_layers)
            ],
            "batch_size": trial.suggest_int(
                "batch_size",
                70,
                300,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-6,
                1e-4,
                log=True,
            ),
            "loss_type": trial.suggest_categorical(
                "loss_type",
                ["l1", "mse"],
            ),
            "guidance_weight": trial.suggest_float(
                "guidance_weight",
                0.1,
                40,
                log=True,
            ),
        }
        noise_steps = trial.suggest_int("noise_steps", 1, 1000, log=True)
        if "guidance_weight" not in params:
            params["guidance_weight"] = guidance_weight
        if type != "MLP":
            for i in range(num_layers - math.ceil(real_layers / 2) - real_layers // 2):
                j = num_layers + i
                params["hidden_layers"][f"hidden_size{j+1}"] = params["hidden_layers"][
                    f"hidden_size{i}"
                ]
                params["num_heads"][f"num_heads{j+1}"] = params["num_heads"][
                    f"num_heads{i}"
                ]
        DDPM = Diffusion(
            vect_size=X_train.shape[1],
            noise_steps=noise_steps,
            X_norm_max=X_max,
            X_norm_min=X_min,
        )

        loss, error = DDPM.reverse_process(
            X_train,
            y_train,
            network,
            df_X,
            df_y,
            type,
            epochs=epoch + 1,
            X_val=X_val,
            y_val=y_val,
            **params,
            trial=trial,
            early_stop=False,
            frequency_print=frequency_print,
        )
        del DDPM
        return np.mean(error)

    if delete_previous_study:
        files = glob.glob(f"./optuna_studies/{type}*")
        if len(files) != 0:
            for f in files:
                os.remove(f)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        study_name=f"{type}study",
        storage=f"sqlite:///optuna_studies/{type}.db",
        load_if_exists=True,
    )
    # Unique identifier of the study.
    # pruner=optuna.pruners.SuccessiveHalvingPruner(),
    study.optimize(objective, n_trials=n_trials)  # type: ignore
    best_trial = study.best_trial
    with open(f"./templates/best_hyperparameters{type}.json", "r") as file:
        data = json.load(file)
    best_params = {
        "hidden_layers": [
            best_trial.params[f"hidden_size{i+1}"]
            for i in range(best_trial.params["num_layers"])
        ],
        "num_heads": [
            best_trial.params[f"num_heads{i+1}"]
            for i in range(best_trial.params["num_layers"])
        ],
        "batch_size": best_trial.params["batch_size"],
        "learning_rate": best_trial.params["learning_rate"],
        "loss_type": best_trial.params["loss_type"],
        "guidance_weight": best_trial.params["guidance_weight"],
        "noise_steps": best_trial.params["noise_steps"],
    }
    data.update(best_params)
    with open(f"./templates/best_hyperparameters{type}.json", "w") as file:
        json.dump(data, file, indent=4)

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
    return data

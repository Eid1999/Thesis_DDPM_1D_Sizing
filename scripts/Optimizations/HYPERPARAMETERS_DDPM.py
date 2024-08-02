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


def HYPERPARAMETERS_DDPM(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    network: Type[nn.Module],
    nn_type: str,
    noise_steps: int,
    epoch: int = 100,
    n_trials: int = 20,
    X_max: np.ndarray = np.array([1] * 12),
    X_min: np.ndarray = np.array([-1] * 12),
    guidance_weight: float = 0.3,
    frequency_print: int = 200,
    delete_previous_study: bool = True,
    data_type: str = "vcota",
):
    def objective(trial):
        real_layers = trial.suggest_int("num_layers", 3, 30)

        num_layers = real_layers if nn_type in ("MLP") else math.ceil(real_layers / 2)

        params = {
            "nn_template": {
                "hidden_layers": [
                    trial.suggest_int(f"hidden_size{i+1}", 200, 2000, log=True)
                    for i in range(num_layers)
                ],
            },
            "batch_size": trial.suggest_int(
                "batch_size",
                70,
                300,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-5,
                1e-2,
                log=True,
            ),
            "loss_type": trial.suggest_categorical(
                "loss_type",
                ["l1", "mse"],
            ),
            # "guidance_weight": trial.suggest_float(
            #     "guidance_weight",
            #     0.1,
            #     40,
            #     log=True,
            # ),
        }
        if nn_type == "EoT":
            params["nn_template"]["num_heads"] = [
                trial.suggest_int(f"num_heads{i+1}", 1, 32, log=True)
                for i in range(real_layers + 1)
            ]
        if "guidance_weight" not in params:
            params["guidance_weight"] = guidance_weight
        if nn_type in ("MLP_skip", "EoT"):
            for i in reversed(range(real_layers // 2)):
                params["nn_template"]["hidden_layers"].append(
                    params["nn_template"]["hidden_layers"][i]
                )
        DDPM = DiffusionDPM(
            vect_size=X_train.shape[1],
            noise_steps=noise_steps,
            X_norm_max=X_max,
            X_norm_min=X_min,
        )

        error = DDPM.reverse_process(
            X_train,
            y_train,
            network,
            df_X,
            df_y,
            nn_type,
            epochs=epoch + 1,
            X_val=X_val,
            y_val=y_val,
            **params,
            trial=trial,
            early_stop=False,
            frequency_print=frequency_print,
            data_type=data_type,
        )
        del DDPM
        if error is None:
            exit()
        return error

    os.makedirs(f"./optuna_studies/{data_type}/{nn_type}", exist_ok=True)
    if delete_previous_study:
        files = glob.glob(
            f"./optuna_studies/{data_type}/{nn_type}/Noise_Steps{noise_steps}*"
        )
        if len(files) != 0:
            for f in files:
                os.remove(f)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        study_name=f"{nn_type}study",
        storage=f"sqlite:///optuna_studies/{data_type}/{nn_type}/Noise_Steps{noise_steps}.db",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=3),
    )
    study.optimize(objective, n_trials=n_trials)  # nn_type: ignore
    best_trial = study.best_trial
    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        data = json.load(file)
    real_layers = best_trial.params["num_layers"]

    num_layers = real_layers if nn_type in ("MLP") else math.ceil(real_layers / 2)
    data[nn_type].update(
        {
            "nn_template": {
                "hidden_layers": [
                    best_trial.params[f"hidden_size{i+1}"] for i in range(num_layers)
                ],
            },
            "batch_size": best_trial.params["batch_size"],
            "learning_rate": best_trial.params["learning_rate"],
            "loss_type": best_trial.params["loss_type"],
            # "guidance_weight": best_trial.params["guidance_weight"],
        }
    )
    if nn_type in ("MLP_skip", "EoT"):
        for i in reversed(range(real_layers // 2)):
            data[nn_type]["nn_template"]["hidden_layers"].append(
                best_trial.params[f"hidden_size{i+1}"]
            )

    if nn_type == "EoT":
        data[nn_type]["nn_template"]["num_heads"] = [
            best_trial.params[f"num_heads{i+1}"] for i in range(real_layers + 1)
        ]
    with open(f"./templates/network_templates_{data_type}.json", "w") as file:
        json.dump(data, file, indent=4)

    plot_intermediate_values(study).update_layout(
        xaxis_title="Epoch",
        yaxis_title="Mean Performace Error",
    ).write_html(f"./html_graphs/{data_type}/epoch_graph{nn_type}.html")
    plot_timeline(study).write_html(
        f"./html_graphs/{data_type}/plot_timeline{nn_type}.html"
    )
    # plot_parallel_coordinate(study).write_html("parallel_coordinate.html")
    plot_param_importances(study).write_html(
        f"./html_graphs/{data_type}/param_importances{nn_type}.html"
    )
    plot_optimization_history(study).update_layout(
        xaxis_title="Trials",
        yaxis_title="Mean Performace Error",
    ).update_yaxes(
        type="log",
        showgrid=True,
        gridcolor="lightgrey",
        zeroline=False,
        showline=True,
        linecolor="black",
        ticks="outside",
        tickformat=".2f",
    ).write_html(
        f"./html_graphs/{data_type}/optimization_history{nn_type}.html"
    )
    return data[nn_type]

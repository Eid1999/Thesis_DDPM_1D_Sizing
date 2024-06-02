import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import time
import itertools
import json

import torch.optim.lr_scheduler as lr_scheduler
import optuna
from optuna.visualization import (
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_optimization_history,
    plot_timeline,
)
from dataset import (
    normalization,
    reverse_normalization,
)

torch.manual_seed(0)


seed_value = 0
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
np.random.seed(seed_value)


class Simulator(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers=[10],
    ):

        # Define hidden layer sizes (you can adjust these)
        super(Simulator, self).__init__()
        # Create a list of linear layers for hidden layers
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=input_size if i == 0 else hidden_layers[i],
                    out_features=(
                        hidden_layers[i + 1]
                        if i + 1 < len(hidden_layers)
                        else output_size
                    ),
                )
                for i in range(len(hidden_layers))
            ]
        )
        self.activation_layer = nn.ReLU()

        # Output layer

    def forward(self, x):
        # Pass through hidden layers with activation function (e.g., ReLU)
        for i, layer in enumerate(self.hidden_layers):
            x = (
                self.activation_layer(layer(x))
                if i < len(self.hidden_layers)
                else layer(x)
            )

        # Output layer without activation (depends on loss function)
        return x


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


def epoch_loop(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_layers=[10],
    batch_size=32,
    lr=1e-5,
    n_epoch=100,
    trial=None,
):
    last_val_loss = float("inf")
    model = Simulator(
        X_train.shape[-1],
        y_train.shape[-1],
        hidden_layers=hidden_layers,
    ).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    patience = 0
    dataloader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    loss_fn = nn.MSELoss(reduction="mean")
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    val_loss = Eval_loop(X_val, y_val, model, loss_fn)
    pbar = tqdm(range(n_epoch))
    for epoch in pbar:
        pbar.set_description(f"Validarion Loss: {val_loss}")
        Train_loop(dataloader, model, optimizer, loss_fn)
        scheduler.step()
        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        val_loss = Eval_loop(X_val, y_val, model, loss_fn)
        if val_loss < last_val_loss:
            last_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience == 5:
                break
    return model, val_loss


def Eval_loop(X, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
    return loss


def Train_loop(data_loader_train, model, optimizer, loss_fn):
    model.train()
    for X, y in data_loader_train:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def Test_error(X_test, y_test, model, original_y):
    y = model(X_test)
    y = reverse_normalization(y.detach().cpu().numpy(), original_y)
    y_test = reverse_normalization(y_test.cpu().numpy(), original_y)
    error = np.mean(
        np.abs(
            np.divide(
                (y - y_test),
                y,  # type: ignore
                out=np.zeros_like(y_test),
                where=(y_test != 0),
            )  # type: ignore
        ),  # type: ignore
        axis=0,
    )  # type: ignore
    print(f"\n{error}")


def main():
    device = "cuda"
    dataframe = pd.read_csv("data/vcota.csv")
    df_X = dataframe[
        ["w8", "w6", "w4", "w10", "w1", "w0", "l8", "l6", "l4", "l10", "l1", "l0"]
    ]
    df_y = dataframe[["gdc", "idd", "gbw", "pm"]]
    X = normalization(df_X.values)
    y = normalization(df_y.values)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

    X_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    HYPERPARAMETERS_OPT(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    with open("best_simulator.json", "r") as file:
        hyper_parameters = json.load(file)
    model, _ = epoch_loop(
        X_train,
        y_train,
        X_val,
        y_val,
        **hyper_parameters,
        n_epoch=100,
    )
    Test_error(X_test, y_test, model, df_y.values)
    torch.save(model.state_dict(), "Simulator.pth")


if __name__ == "__main__":
    main()

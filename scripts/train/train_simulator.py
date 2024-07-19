import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


from Dataset import (
    normalization,
    reverse_normalization,
)
from Networks import Simulator
from Optimizations import HYPERPARAMETERS_SIMULATOR
from utils.utils_Simulator import epoch_loop
from Evaluations.Simulator import Test_error


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda"
    if data_type == "vcota":
        dataframe = pd.DataFrame(pd.read_csv("./data/vcota.csv"))
        df_X = dataframe[
            [
                "w8",
                "w6",
                "w4",
                "w10",
                "w1",
                "w0",
                "l8",
                "l6",
                "l4",
                "l10",
                "l1",
                "l0",
            ]
        ]
        df_y = dataframe[
            [
                "gdc",
                "idd",
                "gbw",
                "pm",
            ]
        ]
    elif data_type == "folded_vcota":
        dataframe = pd.read_csv("./data/folded_vcota.csv")
        df_X = dataframe[
            [
                "_wpmbiasp",
                "_wp5",
                "_wp4",
                "_wp1",
                "_wp0",
                "_wnmbiasp",
                "_wnmbiasn",
                "_wn8",
                "_wn6",
                "_wn4",
                "_wn0",
                "_lp5",
                "_lp4",
                "_lp1",
                "_lp0",
                "_lnmbiasp",
                "_lnmbiasn",
                "_ln8",
                "_ln6",
                "_ln4",
                "_ln0",
                "cload",
            ]
        ]
        df_y = dataframe[
            [
                "gdc",
                "idd",
                "gbw",
                "pm",
            ]
        ]

    X = normalization(df_X.copy()).values
    y = normalization(df_y.copy()).values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)

    X_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    # HYPERPARAMETERS_SIMULATOR(
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     num_trials=10,
    #     num_epochs=500,
    #     data_type=data_type,
    #     delete_previous_study=True,
    # )

    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        hyper_parameters = json.load(file)
    model, _ = epoch_loop(
        X_train,
        y_train,
        X_val,
        y_val,
        **hyper_parameters["Simulator"],
        n_epoch=500,
    )
    Test_error(
        X_test,
        y_test,
        model,
        df_y,
        data_type=data_type,
    )
    os.makedirs(f"./weights/{data_type}", exist_ok=True)
    torch.save(model.state_dict(), f"./weights/{data_type}/Simulator.pth")


if __name__ == "__main__":
    main()

from requirements import *


from Dataset import (
    normalization,
    reverse_normalization,
)
from Networks import Simulator
from Optimizations import HYPERPARAMETERS_SIMULATOR
from utils.Simulator import epoch_loop
from Evaluations.Simulator import Test_error


def main():
    device = "cuda"
    dataframe = pd.read_csv("./data/vcota.csv")
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

    HYPERPARAMETERS_SIMULATOR(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    with open("./templates/best_simulator.json", "r") as file:
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
    torch.save(model.state_dict(), "./weights/Simulator.pth")


if __name__ == "__main__":
    main()

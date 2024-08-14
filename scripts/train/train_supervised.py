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
        dataframe = pd.DataFrame(pd.read_csv("./data/vcota.csv", sep="\s+"))  # type: ignore # ignore
        df_y = dataframe[
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
        df_X = dataframe[
            [
                "gdc",
                "idd",
                "gbw",
                "pm",
            ]
        ]
    elif data_type == "folded_vcota":
        dataframe = pd.read_csv("./data/folded_vcota.csv")
        df_y = dataframe[
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
                "_nfpmbiasp",
                "_nfp5",
                "_nfp4",
                "_nfp1",
                "_nfp0",
                "_nfnmbiasp",
                "_nfnmbiasn",
                "_nfn8",
                "_nfn6",
                "_nfn4",
                "_nfn0",
                "_lpmbiasp",
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
            ]
        ]
        df_X = dataframe[
            [
                "gdc",
                "idd",
                "gbw",
                "pm",
                "cload",
            ]
        ]

    X = normalization(
        df_X.copy(),
        data_type=data_type,
        poly_bool=True,
    ).values
    y = normalization(
        df_y.copy(),
        data_type=data_type,
        # poly_bool=True,
    ).values
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test,
        y_test,
        test_size=0.5,
        random_state=0,
    )

    X_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    print(data_type)
    # HYPERPARAMETERS_SIMULATOR(
    #     X_train,
    #     y_train,
    #     X_val,
    #     y_val,
    #     num_trials=10,
    #     num_epochs=500,
    #     data_type=data_type,
    #     delete_previous_study=True,
    #     type="Supervised",
    # )

    with open(f"./templates/network_templates_{data_type}.json", "r") as file:
        hyper_parameters = json.load(file)
    model, _ = epoch_loop(
        X_train,
        y_train,
        X_val,
        y_val,
        **hyper_parameters["Supervised"],
        n_epoch=2000,
    )

    os.makedirs(f"./weights/{data_type}", exist_ok=True)
    torch.save(model.state_dict(), f"./weights/{data_type}/Supervised.pth")

    model = Simulator(
        X_train.shape[-1],
        y_train.shape[-1],
        **hyper_parameters["Supervised"]["nn_template"],
    ).to("cuda")
    model.load_state_dict(
        torch.load(f"./weights/{data_type}/Supervised.pth", weights_only=True)
    )
    Test_error(
        X_test,
        y_test,
        model,
        df_y,
        data_type=data_type,
        df_X=df_X,
        save=True,
    )
    targets = pd.DataFrame(
        np.array(
            [
                [49, 0.00013418900, 146217581, 63.39, 9.5e-12],
                [51.08, 0.0002073, 239794398, 66.413, 9.5e-12],
                [53.159, 0.00028044, 333371215, 69.43251, 9.5e-12],
                [55.2382, 0.00035357, 426948032, 72.451, 9.5e-12],
            ]
        ),
        columns=df_X.columns,
    )
    targets_norm = torch.tensor(
        normalization(
            targets,
            data_type=data_type,
            poly_bool=True,
            df_original=df_X.copy(),
        ).values,
        device="cuda",
        dtype=torch.float32,
    )
    sizing_targets = model(targets_norm)
    sizing_targets = pd.DataFrame(
        sizing_targets.detach().cpu().numpy(),
        columns=df_y.columns,
    )
    sizing_targets = reverse_normalization(
        sizing_targets,
        df_y,
        data_type=data_type,
    )
    if data_type == "folded_vcota":
        sizing_targets = pd.concat(
            (sizing_targets, targets["cload"]),
            axis=1,
        )
    path = f"points_to_simulate/{data_type}/target/"
    sizing_targets.to_csv(f"{path}sizing_targetSupervised.csv")


if __name__ == "__main__":
    main()

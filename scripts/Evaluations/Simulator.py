import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from Dataset import normalization, reverse_normalization


def Test_error(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model: nn.Module,
    df_y_original: pd.DataFrame,
    data_type: str = "vcota",
    save: bool = False,
    df_X: Union[pd.DataFrame, None] = None,
):
    model.eval()
    y = model(X_test)
    df_y = pd.DataFrame(
        y.detach().cpu().numpy(),
        columns=df_y_original.columns,
    )
    df_y_test = pd.DataFrame(
        y_test.detach().cpu().numpy(), columns=df_y_original.columns
    )
    df_y = reverse_normalization(
        df_y,
        df_y_original.copy(),
        data_type=data_type,
    )
    df_y_test = reverse_normalization(
        df_y_test,
        df_y_original.copy(),
        data_type=data_type,
    )
    error = np.mean(
        np.abs(
            np.divide(
                (df_y - df_y_test),
                df_y,  # type: ignore
                out=np.zeros_like(df_y_test),
                where=(df_y_test != 0),
            )  # type: ignore
        ),  # type: ignore
        axis=0,
    )  # type: ignore
    print(f"\n{error} \n Mean Error:{error.mean()}")
    if save and df_X is not None:
        path = f"points_to_simulate/{data_type}/test/"
        df_X_test = reverse_normalization(
            pd.DataFrame(
                X_test[:, : df_X.shape[1]].detach().cpu().numpy(),
                columns=df_X.columns,
            ),
            df_X.copy(),
            data_type=data_type,
        )
        if data_type == "folded_vcota":
            df_y_test = pd.concat(
                (df_y_test, df_X_test["cload"]),
                axis=1,
            )
        os.makedirs(path, exist_ok=True)
        df_y_test.to_csv(f"{path}sizing_testSupervised.csv")
        df_X_test.to_csv(f"{path}real_testSupervised.csv")

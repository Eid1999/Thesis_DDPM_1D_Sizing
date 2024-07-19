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

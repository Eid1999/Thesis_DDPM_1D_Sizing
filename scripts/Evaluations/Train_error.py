import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *

from Dataset import normalization, reverse_normalization


def Train_error(
    y_train: torch.Tensor,
    DDPM,
    best_weight: int,
    X_train: torch.Tensor,
    df_X: pd.DataFrame,
):
    print("\n\n\nTrain Error")
    tile_y_train = np.tile(y_train[-1].cpu().numpy(), (1000, 1))
    y_train = torch.tensor(tile_y_train, dtype=torch.float32, device=DDPM.device)

    # start_time = time.time()
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_train.shape[0], y_train, weight=best_weight
    )
    # y_train=pd.DataFrame(y_train.cpu().numpy(),columns=df_y.columns)
    # y_train.to_csv('y_train.csv')

    tile_x_train = np.array(np.tile(X_train[-1].cpu().numpy(), (1000, 1)))
    df_X_train = pd.DataFrame(
        tile_x_train,
        columns=df_X.columns,
    )
    X_Sampled = pd.DataFrame(
        X_Sampled.cpu().numpy(),
        columns=df_X.columns,
    )
    df_Sampled = reverse_normalization(X_Sampled, df_X.copy())
    df_X_train = reverse_normalization(df_X_train, df_X.copy())
    error = np.mean(
        np.abs(
            np.divide(
                (df_X_train - df_Sampled),
                df_Sampled,
                out=np.zeros_like(df_X_train),
                where=(df_X_train != 0),
            )
        ),
        axis=0,
    )
    print(f"\n{error}")

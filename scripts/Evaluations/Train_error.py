from requirements import *

from scripts.Dataset import normalization, reverse_normalization


def Train_error(
    y_train,
    DDPM,
    best_weight,
    X_train,
    df_X,
):
    print("\n\n\nTrain Error")
    y_test = np.tile(y_train[-1].cpu().numpy(), (1000, 1))
    y_test = torch.tensor(y_test, dtype=torch.float32, device=DDPM.device)

    # start_time = time.time()
    X_Sampled = DDPM.sampling(
        DDPM.model.cuda(), y_test.shape[0], y_test, weight=best_weight
    )
    end_time = time.time()
    # y_test=pd.DataFrame(y_test.cpu().numpy(),columns=df_y.columns)
    # y_test.to_csv('y_test.csv')

    X_train = np.array(np.tile(X_train[-1].cpu().numpy(), (1000, 1)))

    df_Sampled = reverse_normalization(X_Sampled.cpu().numpy(), df_X)
    X_test = reverse_normalization(X_train, df_X)
    df_Sampled = pd.DataFrame(df_Sampled, columns=df_X.columns)
    X_train = pd.DataFrame(X_train, columns=df_X.columns)
    error = np.mean(
        np.abs(
            np.divide(
                (X_test - df_Sampled),
                df_Sampled,
                out=np.zeros_like(X_train),
                where=(X_train != 0),
            )
        ),
        axis=0,
    )
    print(f"\n{error}")

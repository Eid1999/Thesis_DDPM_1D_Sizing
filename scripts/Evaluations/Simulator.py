from requirements import *
from Dataset import normalization, reverse_normalization


def Test_error(X_test, y_test, model, original_y):
    model.eval()
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

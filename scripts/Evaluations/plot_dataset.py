import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


def plot_dataset(df_y):
    labels = df_y.columns
    fig, ax = plt.subplots(math.ceil(len(df_y.columns) / 2), 2)

    idx = 0
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            sns.histplot(
                data=df_y,
                x=labels[idx],
                ax=ax[i, j],
                log_scale=True if labels[idx] not in ("cload", "gdc", "idd") else False,
                bins=100,
                kde=True,
            )
            if labels[idx] == "cload":
                fig.delaxes(ax[i, j + 1])
                break

            idx += 1

    plt.suptitle("Folded VCOTA Performace Dataset", fontsize=14)
    plt.show()

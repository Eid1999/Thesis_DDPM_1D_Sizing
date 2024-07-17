import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


def plot_dataset(df_y):
    fig, ax = plt.subplots(2, 2)
    labels = ["gbw", "idd", "gdc", "pm"]
    idx = 0
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            sns.histplot(
                data=df_y,
                x=labels[idx],
                ax=ax[i, j],
                log_scale=True,
                bins=100,
                kde=True,
            )
            idx += 1

    plt.suptitle("VCOTA Performace Dataset", fontsize=14)
    plt.show()

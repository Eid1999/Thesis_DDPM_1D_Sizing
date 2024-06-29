import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *


def plot_dataset(df_y):
    fig, ax = plt.subplots(1, 2)
    sns.scatterplot(
        data=df_y,
        y="gbw",
        x="gdc",
        ax=ax[0],
    )

    sns.scatterplot(
        data=df_y,
        y="idd",
        x="pm",
        ax=ax[1],
    )
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].set_xlabel("Gain[dB]", fontsize=14)
    ax[0].set_ylabel("Gain Bandwidth Product[Hz]", fontsize=14)
    ax[1].set_xlabel("Phase [deg]", fontsize=14)
    ax[1].set_ylabel("Bias Current[\\mu A]", fontsize=14)
    plt.suptitle("VCOTA Performace Dataset", fontsize=14)
    plt.show()

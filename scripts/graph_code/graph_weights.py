import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mtick
from matplotlib.ticker import LogLocator


df_MLP_skip = pd.read_csv("scripts/graph_code/weights_data_MLP_skip.csv")
df_MLP = pd.read_csv("scripts/graph_code/weights_data_MLP.csv")
df_MLP_skip["Neural Networks"] = "MLP_skip"
df_MLP["Neural Networks"] = "MLP"
fig, ax = plt.subplots()
data = pd.concat([df_MLP_skip, df_MLP], ignore_index=True)
sns.lineplot(
    data=data,
    x="Weights",
    y="Mean Performance Error",
    hue="Neural Networks",
    marker="o",
    ax=ax,
)
plt.xlabel("Classifier-Free Guidance Weights", fontsize=14)
plt.ylabel("Mean Performance Error[%]", fontsize=14)
plt.title("Epochs=1000, Time step=10", fontsize=14)
plt.xscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())
plt.yscale("log")
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=5, subs=(0.5, 2)))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mtick

data = {}
Noise_Steps = [10, 50, 100, 1000, 2000]
MLP_values = [0.0295, 0.022, 0.039, 0.567, 0.807]
MLP_skip_values = [0.019, 0.023, 0.103, 0.603, 0.904]
EoT_values = []
data["Neural Networks"] = (
    ["MLP"] * len(Noise_Steps)
    + ["MLP_skip"] * len(Noise_Steps)
    # + ["EoT"] * len(Noise_Steps)
)
data["Noise Steps"] = Noise_Steps * 2
data["Mean Performance Error"] = MLP_values + MLP_skip_values + EoT_values

fig, ax = plt.subplots()
data = pd.DataFrame(data)
sns.lineplot(
    data=data,
    x="Noise Steps",
    y="Mean Performance Error",
    hue="Neural Networks",
    marker="o",
    ax=ax,
)
plt.xlabel("Noise Steps", fontsize=14)
plt.ylabel("Mean Performance Error[%]", fontsize=14)
# plt.title("Epochs=1000", fontsize=14)
plt.xscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
# plt.yscale("log")
plt.show()

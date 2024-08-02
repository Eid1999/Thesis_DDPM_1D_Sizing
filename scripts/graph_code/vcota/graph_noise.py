import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mtick
from matplotlib.ticker import LogLocator

data = {}
Noise_Steps = [
    1,
    5,
    10,
    50,
    100,
    1000,
    2000,
]
MLP_values = [
    1.47,
    0.12,
    0.03,
    0.025,
    0.024,
    0.567,
    0.607,
]
MLP_skip_values = [
    1.2,
    0.15,
    0.025,
    0.023,
    0.023,
    0.403,
    0.504,
]
EoT_values = [
    0.60,
    0.08,
    0.025,
    0.023,
    0.0178,
    0.10,
    0.20,
]
data["Neural Networks"] = (
    ["MLP"] * len(Noise_Steps)
    + ["ResMLP"] * len(Noise_Steps)
    + ["EoT"] * len(Noise_Steps)
)
data["Noise Steps"] = Noise_Steps * 3
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
plt.xlabel("Time Steps", fontsize=14)
plt.ylabel("Mean Performance Error[%]", fontsize=14)
plt.title("Epochs=1000", fontsize=14)
plt.xscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.yscale("log")
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=5, subs=(0.5, 2)))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
sns.move_legend(
    ax,
    loc="center left",
    bbox_to_anchor=[1, 0.5],
    fancybox=True,
    shadow=True,
    fontsize="10",
)
plt.show()

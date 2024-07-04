import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {}
Noise_Steps = [10, 50, 100, 1000, 2000]
MLP_values = [2.95, 2.2, 3.9, 56.7, 80.7]
MLP_skip_values = []
EoT_values = []
data["labels"] = (
    ["MLP"] * len(Noise_Steps)
    + ["MLP_skip"] * len(Noise_Steps)
    + ["EoT"] * len(Noise_Steps)
)
data["Noise Steps"] = Noise_Steps * 3
data["Mean Performance Error"] = MLP_values + MLP_skip_values + EoT_values


data = pd.DataFrame(data)
sns.lineplot(
    data=data,
    x="Noise Steps",
    y="Mean Performance Error",
    marker="o",
)
plt.xlabel("Noise Steps", fontsize=14)
plt.ylabel("Mean Performance Error[%]", fontsize=14)
plt.title("MLP: Epochs=1000", fontsize=14)
plt.xscale("log")
plt.yscale("log")
plt.show()

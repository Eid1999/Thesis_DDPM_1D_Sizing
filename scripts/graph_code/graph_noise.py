import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {}
Noise_Steps = [1000, 2000]
MLP_values = [0.2321, 0.2279]
MLP_skip_values = []
data["labels"] = ["MLP"] * len(data["Noise Steps"])

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
plt.show()

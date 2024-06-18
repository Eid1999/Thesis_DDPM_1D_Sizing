import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {}
Noise_Steps = [50]
MLP_values = [17.5]
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
plt.title("MLP: Epochs=500,Noise", fontsize=14)
plt.xscale("log")
plt.show()

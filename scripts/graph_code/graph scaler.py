import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Scaler value": [1, 0.5, 0.2, 0.1, 0.05, 0.03, 0.01],
    "Mean Performance Error": [
        18.460,
        7.208,
        6.46,
        5.6186,
        5.46,
        6.07,
        8.4837,
    ],
}
data = pd.DataFrame(data)
sns.lineplot(
    data=data,
    x="Scaler value",
    y="Mean Performance Error",
    marker="o",
)
plt.xlabel("Scaler Values", fontsize=14)
plt.ylabel("Mean Performance Error[%]", fontsize=14)
plt.title("Epochs=500,Noise Step=100", fontsize=14)
plt.xscale("log")
plt.show()

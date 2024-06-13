import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Noise Steps": [5, 10, 20, 50, 100, 500, 1000],
    "Mean Performance Error": [
        5.82,
        4.742,
        4.900,
        4.986,
        5.4645,
        7.52,
        11.58,
    ],
}
data = pd.DataFrame(data)
sns.lineplot(
    data=data,
    x="Noise Steps",
    y="Mean Performance Error",
    marker="o",
)
plt.xlabel("Noise Steps", fontsize=14)
plt.ylabel("Mean Performance Error[%]", fontsize=14)
plt.title("Epochs=500,Noise Scaler=0.05", fontsize=14)
plt.xscale("log")
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Scaler value": [1, 0.5, 0.2, 0.1, 0.05],
    "Mean Performance Error": [12.460, 7.208, 5.705068, 5.6186, 6.49983286857605],
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

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from tabulate import tabulate

data_type = "vcota"


def main():
    sim = {}
    real = {}
    size = {}
    # NN = {}
    NN_type = ["MLP", "MLP_skip", "EoT"]
    delta_threshold = 0.08 if "folded_vcota" else 0.05
    vov_threshold = 0.05

    for nn in NN_type:
        sim[nn] = pd.read_csv(
            f"./points_to_simulate/{data_type}/target/sizing_target{nn}_Performances.csv",
            sep="\s+",  # type: ignore
        )
        size[nn] = pd.read_csv(
            f"./points_to_simulate/{data_type}/target/sizing_target{nn}.csv",
        )
        real[nn] = pd.read_csv(
            f"./points_to_simulate/{data_type}/target/real_target{nn}.csv"
        )[["gdc", "idd", "gbw", "pm"]]

        delta_indices = (
            sim[nn][sim[nn].filter(like="delta").lt(delta_threshold)]
            .stack()
            .index.get_level_values(0)
        )

        # Find indices where values in 'vov' columns are below the threshold
        vov_indices = (
            sim[nn][sim[nn].filter(like="vov").lt(vov_threshold)]
            .stack()
            .index.get_level_values(0)
        )

        # Combine indices
        all_indices = pd.Index(delta_indices).union(vov_indices)

        # Replace values in all columns at those indices with NaN
        sim[nn].loc[all_indices] = np.nan
    for i in range(4):
        # print("a")
        for nn in NN_type:
            print(f"{nn}\n\n\n")

            real_error = (
                (np.abs((sim[nn] - real[nn]) / sim[nn])[100 * i : 100 * (i + 1)])
                # .replace(np.nan, np.inf)
                .mean(axis=1)
            )

            sim[nn]["Target"] = i
            sim[nn]["error"] = real_error
            size[nn]["fom"] = sim[nn]["fom"][100 * i : 100 * (i + 1)]
            print(
                f"Best Target {i}: \n {sim[nn].iloc[100 * i+np.argmin(real_error)][real[nn].columns]}"
            )
            print(f"Fom:{sim[nn].iloc[100 * i+np.argmin(real_error)]['fom']}\n\n")
            print(np.min(real_error))
            # print(
            #     f"Best FOM Target {i}: \n {sim[nn].iloc[100 * i+np.argmax(sim[nn]['fom'][100 * i : 100 * (i + 1)])][real[nn].columns]}"
            # )
            # print(
            #     f"Fom {i}: \n {sim[nn].iloc[100 * i+np.argmax(sim[nn]['fom'][100 * i : 100 * (i + 1)])]['fom']}\n\n\n"
            # )
            for col in real[nn].columns:
                print(col)
                print(
                    sim[nn].iloc[
                        (sim[nn][col] - real[nn].loc[100 * i, col])
                        .abs()
                        .values.argsort()[0]
                    ][col]
                )
    print("a")


#  sim['MLP'].iloc[(sim['MLP']['gbw']-real['MLP'].iloc[300,2]).abs().values.argsort()[0]]['gbw']
if __name__ == "__main__":
    main()

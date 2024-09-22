import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from tabulate import tabulate

data_type = "vcota"


def main():
    sim = {}
    real = {}
    # NN = {}
    NN_type = ["MLP", "MLP_skip", "EoT", "Supervised"]
    delta_threshold = 0.08 if "folded_vcota" else 0.05
    vov_threshold = 0.05
    types = ["test"]
    type_folder = "test"
    for type in types:
        sim[type] = {}
        # NN[type] = {}
        real[type] = {}
        for nn in NN_type:
            sim[type][nn] = pd.read_csv(
                f"./points_to_simulate/{data_type}/{type}/sizing_{type}{nn}_Performances.csv",
                sep="\s+",  # type: ignore
            )

            real[type][nn] = pd.read_csv(
                f"./points_to_simulate/{data_type}/{type}/real_{type}{nn}.csv"
            )[["gdc", "idd", "gbw", "pm"]]
            # NN[type][nn] = pd.read_csv(
            #     f"./points_to_simulate/{data_type}/{type}/nn_{type}{nn}.csv"
            # )[["gdc", "idd", "gbw", "pm"]]

            # Find indices where values in 'delta' columns are below the threshold
            delta_indices = (
                sim[type][nn][sim[type][nn].filter(like="delta").lt(delta_threshold)]
                .stack()
                .index.get_level_values(0)
            )

            # Find indices where values in 'vov' columns are below the threshold
            vov_indices = (
                sim[type][nn][sim[type][nn].filter(like="vov").lt(vov_threshold)]
                .stack()
                .index.get_level_values(0)
            )

            # Combine indices
            all_indices = pd.Index(delta_indices).union(vov_indices)

            # Replace values in all columns at those indices with NaN
            sim[type][nn].loc[all_indices] = np.nan

    print(f"Test\n")

    for nn in NN_type:
        print(f"\t{nn}: \n")
        real_error = (
            (np.abs((real["test"][nn] - sim["test"][nn]) / real["test"][nn])).median()
        ).dropna()
        # nn_error = (
        # np.abs((sim["test"][nn] - NN["test"][nn]) / sim["test"][nn]).median()
        # ).dropna()
        print(f"\t\tPredicted Error: \n{real_error}\n{real_error.mean()}\n")
        # print(f"\t\tAuxiliary Network Error: \n{nn_error}\n{nn_error.mean()}\n")
        print(
            f"Out of saturation: {sim['test'][nn].isna().sum()[1]/len(sim['test'][nn])}"
        )


if __name__ == "__main__":
    main()

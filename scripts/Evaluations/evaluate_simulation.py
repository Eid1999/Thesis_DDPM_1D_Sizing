import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libraries import *
from tabulate import tabulate

data_type = "folded_vcota"


def main():
    sim = {}
    real = {}
    NN = {}
    NN_type = ["MLP_skip", "EoT"]
    types = ["test"] if data_type == "folded_vcota" else ["test", "target"]
    for type in types:
        sim[type] = {}
        NN[type] = {}
        real[type] = {}
        for nn in NN_type:
            sim[type][nn] = pd.read_csv(
                f"./points_to_simulate/{data_type}/{type}/sizing_{type}{nn}_Performances.csv",
                sep="\s+",  # type: ignore
            )[["gdc", "idd", "gbw", "pm"]]

            real[type][nn] = pd.read_csv(
                f"./points_to_simulate/{data_type}/{type}/real_{type}{nn}.csv"
            )[["gdc", "idd", "gbw", "pm"]]
            NN[type][nn] = pd.read_csv(
                f"./points_to_simulate/{data_type}/{type}/nn_{type}{nn}.csv"
            )[["gdc", "idd", "gbw", "pm"]]

    print(f"Test\n")
    for nn in NN_type:
        print(f"\t{nn}: \n")
        real_error = (
            np.abs((sim["test"][nn] - real["test"][nn]) / sim["test"][nn])
            .dropna()
            .median()
        )
        nn_error = np.abs((sim["test"][nn] - NN["test"][nn]) / sim["test"][nn]).median()
        print(f"\t\tPredicted Error: \n{real_error}\n{real_error.mean()}\n")
        print(f"\t\tAuxiliary Network Error: \n{nn_error}\n{nn_error.mean()}\n")
    if data_type == "vcota":
        print(f"Target:\n")
        for nn in NN_type:
            print(f"\t{nn}: \n")
            for i in range(4):
                real_error = np.abs(
                    (sim["target"][nn] - real["target"][nn]) / sim["target"][nn]
                )[100 * i : 100 * (i + 1)].median()
                nn_error = np.abs(
                    (sim["target"][nn] - NN["target"][nn]) / sim["target"][nn]
                )[100 * i : 100 * (i + 1)].median()
                print(
                    f"\t\tTarget{i} \nPredicted Error: \n{real_error}\n{real_error.mean()}\n"
                )
                print(f"\t\tAuxiliary Network Error: \n{nn_error}\n{nn_error.mean()}")


if __name__ == "__main__":
    main()

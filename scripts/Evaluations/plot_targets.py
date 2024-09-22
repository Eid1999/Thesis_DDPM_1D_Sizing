from libraries import *
from Dataset import (
    normalization,
    reverse_normalization,
)


def plot_targets(df_y, data_type, y_test=None):
    _, axis = plt.subplots(1, 2)
    start = 0
    df_y_copy = df_y.copy()
    default_size = 20
    if y_test is not None:
        y_test = pd.DataFrame(y_test[:, : df_y.shape[1]], columns=df_y.columns)
        y_test = reverse_normalization(y_test, df_y, data_type=data_type)
        y_test["type"] = "Test Dataset"
        y_test["size"] = default_size
        plt.suptitle("Locations of the Test points")
    df_y_copy["type"] = "Whole Datset"
    df_y_copy["size"] = default_size

    # for idx, ax in enumerate(axis):
    #     col1, col2 = df_y.columns[idx * 2 : (idx + 1) * 2]
    #     sns.scatterplot(
    #         df_y_copy,
    #         x=col1,
    #         y=col2,
    #         ax=ax,
    #     )

    columns = df_y.columns if data_type == "vcota" else df_y.columns[:-1]
    if y_test is None:

        if data_type == "folded_vcota":
            cload_value = df_y["cload"].mode()[0]
            df_y_copy = df_y_copy[df_y["cload"] == cload_value]
            mean = df_y_copy[columns].mean()
            std_dev = df_y_copy[columns].std()
            point1 = mean
            point2 = mean + std_dev
            point3 = mean + 2 * std_dev
            point4 = mean + 3 * std_dev
            targets = pd.DataFrame(
                np.array(
                    [
                        point1,
                        point2,
                        point3,
                        point4,
                    ]
                ),
                columns=columns,
            )
            plt.suptitle(f"Locations of the Target points(Cload : {cload_value})")

        else:
            targets = np.array(
                [
                    [50, 300e-6, 60e6, 65],
                    [40, 700e-6, 150e6, 55],
                    [50, 150e-6, 30e6, 65],
                    [53, 350e-6, 65e6, 55],
                ]
            )
            targets = pd.DataFrame(targets, columns=df_y.columns)
            plt.suptitle(f"Locations of the Target points")

        for i in range(4):
            targets.loc[i, "type"] = f"Target {i}"
        targets["size"] = 80
    dark_palette = sns.color_palette(
        ["#4682B4", "#641E16", "#4A235A", "#873600", "#7D6608", "#1F618D"]
    )

    # Set the dark background style
    # sns.set(style="darkgrid")
    comb_df = pd.concat([df_y_copy, targets if y_test is None else y_test])
    for idx, ax in enumerate(axis):
        col1, col2 = columns[idx * 2 : (idx + 1) * 2]
        sns.scatterplot(
            comb_df,
            x=col1,
            y=col2,
            ax=ax,
            hue="type",
            # palette=dark_palette,
            size="size",
            legend=False if idx != 1 else True,
        )
        # if col2 in ["idd", "pm"]:
        #     ax.set_yscale("log")
        ax.set_xlabel(f"{units[col1]}")
        ax.set_ylabel(f"{units[col2]}")
        # if col1 == "gbw":
        #     ax.set_xscale("log")
    h, l = ax.get_legend_handles_labels()

    # slice the appropriate section of l and h to include in the legend
    end = 6 if y_test is None else 3
    ax.legend(
        h[1:end],
        l[1:end],
        loc="center left",
        bbox_to_anchor=[1, 0.5],
        # borderaxespad=0.0,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )
    # sns.move_legend(
    #     axis[1],
    #     loc="center left",
    #     bbox_to_anchor=[1, 0.5],
    #     fancybox=True,
    #     shadow=True,
    #     fontsize="10",
    # )
    plt.show()
    print("a")

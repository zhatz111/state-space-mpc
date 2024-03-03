"""_summary_
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5
sns.set_style("ticks")

data_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/AR24-005_MPC_DoE/AR24-005_MasterDataTable_2.xlsx",
)

df_test = pd.read_excel(data_path, sheet_name="Difference Plot Data")
df_test["Temp"] = df_test["Temp"].astype(int)

# Uncomment to filter out center point conditions
# batch_list = ["AR24-005-008","AR24-005-022","AR24-005-023","AR24-005-024"]
# df_test = df_test[~df_test["Batch"].isin(batch_list)]

# Filter out days that no cedex titer was available for
day_list = [5, 7]
df_test = df_test[~df_test["Day"].isin(day_list)]

g = sns.FacetGrid(
    df_test,
    col="iVCC",
    height=4,
    sharex=False,
    sharey=True,
    despine=False,
)


def boxplot(data, **kwargs):
    sns.lineplot(
        x="Day",
        y="Measured",
        hue="Controller",
        marker="o",
        hue_order=["Julia", "Python", "No MPC"],
        palette=["C0", "C1", "k"],
        markersize=8,
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
        errorbar="ci",
        data=data,
        **kwargs,
    )
    plt.plot(df_test["Day"], df_test["Setpoint"], "r--")
    plt.grid(axis="x", linestyle="--", color="gray")
    plt.grid(axis="y", linestyle="--", color="gray")


g.map_dataframe(boxplot)
g.add_legend(title="MPC Controller")
plt.show()

g = sns.FacetGrid(
    df_test,
    col="iVCC",
    height=4,
    sharex=False,
    sharey=True,
    despine=False,
    ylim=(-30, 30),
)
sns.set_style("white")


def boxplot_2(data, **kwargs):
    sns.lineplot(
        x="Day",
        y="% Difference from Setpoint",
        hue="Controller",
        marker="o",
        hue_order=["Julia", "Python", "No MPC"],
        palette=["C0", "C1", "k"],
        markersize=10,
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
        errorbar="ci",
        data=data,
        **kwargs,
    )
    plt.axhline(y=0, color="r", linestyle="-.")
    plt.grid(axis="x", linestyle="--", color="gray")
    plt.grid(axis="y", linestyle="--", color="gray")


g.map_dataframe(boxplot_2)
g.add_legend(title="MPC Controller")
plt.show()

g = sns.FacetGrid(
    df_test,
    col="Temp/pH",
    height=4,
    sharex=False,
    sharey=True,
    despine=False,
    ylim=(-30, 30),
)
sns.set_style("white")


def boxplot_3(data, **kwargs):
    sns.lineplot(
        x="Day",
        y="% Difference from Setpoint",
        hue="Controller",
        marker="o",
        hue_order=["Julia", "Python", "No MPC"],
        palette=["C0", "C1", "k"],
        markersize=10,
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
        errorbar="ci",
        data=data,
        **kwargs,
    )
    plt.axhline(y=0, color="r", linestyle="-.")
    plt.grid(axis="x", linestyle="--", color="gray")
    plt.grid(axis="y", linestyle="--", color="gray")


g.map_dataframe(boxplot_3)
g.add_legend()
plt.show()

sns.pointplot(
    x="Day",
    y="% Difference from Setpoint",
    hue="Controller",
    hue_order=["Julia", "Python", "No MPC"],
    palette=["C0", "C1", "k"],
    capsize=0.2,
    linewidth=2,
    data=df_test,
    errorbar="ci",
)
plt.axhline(y=0, color="r", linestyle="--")
plt.grid(axis="x", linestyle="--", color="gray")
plt.grid(axis="y", linestyle="--", color="gray")
plt.ylim(-30, 30)
plt.show()

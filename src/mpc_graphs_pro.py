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

# Retrieve measurements
data_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/AR24-005_MPC_DoE/AR24-005_MasterDataTable_2.xlsx",
)
df_data = pd.read_excel(data_path, skiprows=[0])
df_data["Temp"] = df_data["Temp"].astype(int)
df_data["Bioreactor"] = [int(x[-3:]) for x in df_data["Batch"].values]
df_data["Temp/pH"] = [f"{x[0]}, {int(x[1])}" for x in df_data.loc[:,["pH","Temp"]].values]
df_data["Measured"] = df_data["IGG"]
df_data_selected = df_data.loc[:,["Bioreactor","Day","Batch","Controller","iVCC","pH","Temp","IGG","Temp/pH","Measured"]]

# Retrieve setpoint from the master sheet directly
top_dir = Path().absolute()
sp_path = Path(top_dir, f"data/simulation/AR24-005/ar24-005-mpc.csv")
df_sp = pd.read_csv(sp_path).rename(columns={"IGG--STATE_SP":"Setpoint"})
df_sp_selected = df_sp.loc[:,["Bioreactor","Day","Setpoint"]]

# Left join the two dfs
df_joined = pd.merge(df_data_selected,df_sp_selected,how="inner",left_on=["Bioreactor","Day"],right_on=["Bioreactor","Day"])
df_joined["% Difference from Setpoint"] = (df_joined["IGG"] - df_joined["Setpoint"])/df_joined["Setpoint"]*100

# Uncomment to filter out center point conditions
# batch_list = ["AR24-005-008","AR24-005-022","AR24-005-023","AR24-005-024"]
# df_test = df_test[~df_test["Batch"].isin(batch_list)]

# Filter out days that no cedex titer was available for
# day_list = [5, 7]
# df_data = df_data[~df_data["Day"].isin(day_list)]

# Setpoint tracking (all)
sns.pointplot(
    x="Day",
    y="Measured",
    hue="Controller",
    hue_order=["Julia", "Python", "No MPC"],
    palette=["C0", "C1", "k"],
    capsize=0.2,
    linewidth=2,
    data=df_joined,
    errorbar="ci",
)
plt.plot(df_joined["Day"], df_joined["Setpoint"], "r--")
plt.grid(axis="x", linestyle="--", color="gray")
plt.grid(axis="y", linestyle="--", color="gray")
plt.show()

# Setpoint tracking (iVCC)
g = sns.FacetGrid(
    df_joined,
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
    plt.plot(df_joined["Day"], df_joined["Setpoint"], "r--")
    plt.grid(axis="x", linestyle="--", color="gray")
    plt.grid(axis="y", linestyle="--", color="gray")


g.map_dataframe(boxplot)
g.add_legend(title="MPC Controller")
plt.show()

# Setpoint deviation (all)
sns.pointplot(
    x="Day",
    y="% Difference from Setpoint",
    hue="Controller",
    hue_order=["Julia", "Python", "No MPC"],
    palette=["C0", "C1", "k"],
    capsize=0.2,
    linewidth=2,
    data=df_joined,
    errorbar="ci",
)
plt.axhline(y=0, color="r", linestyle="--")
plt.grid(axis="x", linestyle="--", color="gray")
plt.grid(axis="y", linestyle="--", color="gray")
plt.ylim(-30, 30)
plt.show()

# Setpoint deviation (iVCC)
g = sns.FacetGrid(
    df_joined,
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

# Setpoint deviation (Temp/pH)
g = sns.FacetGrid(
    df_joined,
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
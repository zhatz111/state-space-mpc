"""_summary_
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5
sns.set_style("ticks")

# Retrieve measurements
data_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/AR24-005_MPC_DoE/AR24-005_MasterDataTable_2.xlsx",
)
fig_path = Path(data_path.parent,"mpc-performance-figs").expanduser()
fig_path.mkdir(parents=True, exist_ok=True)
df_data = pd.read_excel(
    data_path, skiprows=[0]
    ).rename(
        columns={"Cumulative_Feed":"Total Feed","Cumulative_Glucose":"Total Glucose","pCO2_at_temp":"pCO2"}
        ).sort_values(by=["Batch","Day"])
df_data["Temp"] = df_data["Temp"].astype(int)
df_data["Bioreactor"] = [int(x[-3:]) for x in df_data["Batch"].values]
df_data["Temp/pH"] = [f"{x[0]}, {int(x[1])}" for x in df_data.loc[:,["pH","Temp"]].values]
df_data["Measured"] = df_data["IGG"]
total_feed_diff = np.append(np.diff(df_data["Total Feed"]),0)
daily_feed = np.zeros((len(total_feed_diff),))
daily_feed[total_feed_diff > 0] = total_feed_diff[total_feed_diff > 0]
df_data["Daily Feed"] = daily_feed
total_glc_diff = np.append(np.diff(df_data["Total Glucose"]),0)
daily_glc = np.zeros((len(total_glc_diff),))
daily_glc[total_glc_diff > 0] = total_glc_diff[total_glc_diff > 0]
df_data["Daily Glucose"] = daily_glc
df_data_selected = df_data.loc[:,["Bioreactor","Day","Batch","Controller","iVCC","pH","Temp","IGG","VCC","Viability","Lactate","Glucose","pCO2","Temp/pH","Measured","Total Feed","Total Glucose","Daily Feed","Daily Glucose"]]

DISPLAY_VARIABLE = "IGG"

# Retrieve setpoint from the master sheet directly
top_dir = Path().absolute()
sp_path = Path(top_dir, f"data/simulation/AR24-005/ar24-005-mpc.csv")
df_sp = pd.read_csv(sp_path).rename(columns={"IGG--STATE_SP":"Setpoint"})
df_sp.loc[df_sp["Day"] == 0,"Setpoint"] = float("NaN")
df_sp_selected = df_sp.loc[:,["Bioreactor","Day","Setpoint"]]

# Left join the two dfs
df_joined = pd.merge(df_data_selected,df_sp_selected,how="inner",left_on=["Bioreactor","Day"],right_on=["Bioreactor","Day"])
df_joined["% Difference from Setpoint"] = (df_joined["IGG"] - df_joined["Setpoint"])/df_joined["Setpoint"]*100

# Setpoint tracking (Controller)
g = sns.FacetGrid(
    df_joined,
    col="Controller",
    height=4,
    sharex=False,
    sharey=True,
    despine=False,
)

def data_grp_by_ctrl(data, **kwargs):
    sns.lineplot(
        x="Day",
        y=DISPLAY_VARIABLE,
        hue="iVCC",
        marker="o",
        hue_order=[12, 15, 18],
        palette=["r", "k", "g"],
        markersize=8,
        err_style="bars",
        err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
        errorbar="ci",
        data=data,
        **kwargs,
    )
    if DISPLAY_VARIABLE == "IGG":
        plt.plot(df_joined["Day"], df_joined["Setpoint"], "b--")
    
    plt.grid(axis="x", linestyle="--", color="gray")
    plt.grid(axis="y", linestyle="--", color="gray")


g.map_dataframe(data_grp_by_ctrl)
g.add_legend(title="iVCC")
# plt.show()
plt.savefig(fname=Path(fig_path,f"{DISPLAY_VARIABLE}_grp_by_ctrl.png"))

# # Setpoint tracking (iVCC)
# g = sns.FacetGrid(
#     df_joined,
#     col="iVCC",
#     height=4,
#     sharex=False,
#     sharey=True,
#     despine=False,
# )

# def data_grp_by_ivcc(data, **kwargs):
#     sns.lineplot(
#         x="Day",
#         y=DISPLAY_VARIABLE,
#         hue="Controller",
#         marker="o",
#         hue_order=["Julia", "Python", "No MPC"],
#         palette=["C0", "C1", "k"],
#         markersize=8,
#         err_style="bars",
#         err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
#         errorbar="ci",
#         data=data,
#         **kwargs,
#     )
#     if DISPLAY_VARIABLE == "IGG":
#         plt.plot(df_joined["Day"], df_joined["Setpoint"], "b--")
#     plt.grid(axis="x", linestyle="--", color="gray")
#     plt.grid(axis="y", linestyle="--", color="gray")


# g.map_dataframe(data_grp_by_ivcc)
# g.add_legend(title="MPC Controller")
# # plt.show()
# plt.savefig(fname=Path(fig_path,f"{DISPLAY_VARIABLE}_grp_by_ivcc.png"))

if DISPLAY_VARIABLE == "IGG":

    # Setpoint deviation (Controller)
    g = sns.FacetGrid(
        df_joined,
        col="Controller",
        height=4,
        sharex=False,
        sharey=True,
        despine=False,
        ylim=(-30, 30),
    )
    sns.set_style("white")

    def dev_grp_by_ctrl(data, **kwargs):
        sns.lineplot(
            x="Day",
            y="% Difference from Setpoint",
            hue="iVCC",
            marker="o",
            hue_order=[12,15,18],
            palette=["r", "k", "g"],
            markersize=10,
            err_style="bars",
            err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
            errorbar="ci",
            data=data,
            **kwargs,
        )
        plt.axhline(y=0, color="b", linestyle="--")
        plt.grid(axis="x", linestyle="--", color="gray")
        plt.grid(axis="y", linestyle="--", color="gray")

    g.map_dataframe(dev_grp_by_ctrl)
    g.add_legend(title="iVCC")
    # plt.show()
    plt.savefig(fname=Path(fig_path,"dev_grp_by_ctrl.png"))

    # # Setpoint deviation (iVCC)
    # g = sns.FacetGrid(
    #     df_joined,
    #     col="iVCC",
    #     height=4,
    #     sharex=False,
    #     sharey=True,
    #     despine=False,
    #     ylim=(-30, 30),
    # )
    # sns.set_style("white")


    # def dev_grp_by_ivcc(data, **kwargs):
    #     sns.lineplot(
    #         x="Day",
    #         y="% Difference from Setpoint",
    #         hue="Controller",
    #         marker="o",
    #         hue_order=["Julia", "Python", "No MPC"],
    #         palette=["C0", "C1", "k"],
    #         markersize=10,
    #         err_style="bars",
    #         err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
    #         errorbar="ci",
    #         data=data,
    #         **kwargs,
    #     )
    #     plt.axhline(y=0, color="b", linestyle="-.")
    #     plt.grid(axis="x", linestyle="--", color="gray")
    #     plt.grid(axis="y", linestyle="--", color="gray")


    # g.map_dataframe(dev_grp_by_ivcc)
    # g.add_legend(title="iVCC")
    # # plt.show()
    # plt.savefig(fname=Path(fig_path,"dev_grp_by_ivcc.png"))
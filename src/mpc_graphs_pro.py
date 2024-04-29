"""_summary_
"""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5
sns.set_style("ticks")

# Options
MPC_GRP = "linear"
CONFIG = {
    'all':{'dest':'mpc-performance-figs-all','controller':'Linear MPC|Nonlinear MPC|No MPC','col':'Controller','col_order':["Linear MPC","Nonlinear MPC","No MPC"],'hue':'iVCC','hue_order':[12, 15, 18]},
    'linear':{'dest':'mpc-performance-figs-linear','controller':'Linear MPC','col':'iVCC','col_order':[12, 15, 18],'hue':'Temp/pH','hue_order':['7.1, 32', '7.2, 34', '7.3, 35']},
    'nonlinear':{'dest':'mpc-performance-figs-nonlinear','controller':'Nonlinear MPC','col':'iVCC','col_order':[12, 15, 18],'hue':'Temp/pH','hue_order':['7.1, 32', '7.2, 34', '7.3, 35']},
    'no':{'dest':'mpc-performance-figs-no','controller':'No MPC','col':'iVCC','col_order':[12, 15, 18],'hue':'Temp/pH','hue_order':['7.1, 32', '7.2, 34', '7.3, 35']},
}
DISP_VARS = ["Cedex Titer","Total Feed","Daily Feed","Total Glucose","Daily Glucose","Viability","VCC"]#,"HPLC Titer","Lactate","Glucose","pCO2"]
PALETTE = sns.color_palette("rocket",3)

# Retrieve measurements
data_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/AR24-005_MPC_DoE/AR24-005_MasterDataTable_2.xlsx",
)
fig_path = Path(data_path.parent,CONFIG[MPC_GRP]['dest']).expanduser()
fig_path.mkdir(parents=True, exist_ok=True)
df_data = (
    pd.read_excel(data_path, skiprows=[0])
    .rename(
        columns={
            "Cumulative_Feed": "Total Feed",
            "Cumulative_Glucose": "Total Glucose",
            "pCO2_at_temp": "pCO2",
            "IGG": "Cedex Titer",
        }
    )
    .sort_values(by=["Batch", "Day"])
    .replace({"Controller": {"Python": "Linear MPC", "Julia": "Nonlinear MPC"}})
)
df_data["Temp"] = df_data["Temp"].astype(int)
df_data["Bioreactor"] = [int(x[-3:]) for x in df_data["Batch"].values]
df_data["Temp/pH"] = [
    f"{x[0]}, {int(x[1])}" for x in df_data.loc[:, ["pH", "Temp"]].values
]
total_feed_diff = np.append(np.diff(df_data["Total Feed"]), 0)
daily_feed = np.zeros((len(total_feed_diff),))
daily_feed[total_feed_diff > 0] = total_feed_diff[total_feed_diff > 0]
df_data["Daily Feed"] = daily_feed
total_glc_diff = np.append(np.diff(df_data["Total Glucose"]), 0)
daily_glc = np.zeros((len(total_glc_diff),))
daily_glc[total_glc_diff > 0] = total_glc_diff[total_glc_diff > 0]
df_data["Daily Glucose"] = daily_glc
df_data_selected = df_data.loc[df_data['Controller'].str.contains(CONFIG[MPC_GRP]['controller']),["Bioreactor","Day","Batch","Controller","iVCC","pH","Temp","Cedex Titer","HPLC Titer","VCC","Viability","Lactate","Glucose","pCO2","Temp/pH","Total Feed","Total Glucose","Daily Feed","Daily Glucose"]]

# Retrieve setpoint from the master sheet directly
top_dir = Path().absolute()
sp_path = Path(top_dir, "data/simulation/AR24-005/ar24-005-mpc.csv")
df_sp = pd.read_csv(sp_path).rename(columns={"IGG--STATE_SP": "Setpoint"})
df_sp.loc[df_sp["Day"] == 0, "Setpoint"] = float("NaN")
df_sp_selected = df_sp.loc[:, ["Bioreactor", "Day", "Setpoint"]]

# Left join the two dfs
df_joined = pd.merge(
    df_data_selected,
    df_sp_selected,
    how="inner",
    left_on=["Bioreactor", "Day"],
    right_on=["Bioreactor", "Day"],
)
df_joined["Cedex Titer Tracking Error (%)"] = (
    (df_joined["Cedex Titer"] - df_joined["Setpoint"]) / df_joined["Setpoint"] * 100
)
df_joined["Cedex Titer Absolute Tracking Error (%)"] = (
    np.abs(df_joined["Cedex Titer"] - df_joined["Setpoint"])
    / df_joined["Setpoint"]
    * 100
)
df_joined["HPLC Titer Tracking Error (%)"] = (
    (df_joined["HPLC Titer"] - df_joined["Setpoint"]) / df_joined["Setpoint"] * 100
)
df_joined["HPLC Titer Absolute Tracking Error (%)"] = (
    np.abs(df_joined["HPLC Titer"] - df_joined["Setpoint"])
    / df_joined["Setpoint"]
    * 100
)

i = 1
for disp_var in DISP_VARS:

    print(f"Generating figures for {disp_var}")

    # Setpoint tracking (Controller)
    g = sns.FacetGrid(
        df_joined,
        col=CONFIG[MPC_GRP]['col'],
        height=4,
        sharex=False,
        sharey=True,
        despine=False,
        xlim=(-0.25, np.max(df_joined["Day"]) + 0.25),
        col_order=CONFIG[MPC_GRP]['col_order']
    )

    def plot_measured(data, **kwargs):
        sns.lineplot(
            x="Day",
            y=disp_var,
            hue=CONFIG[MPC_GRP]['hue'],
            marker="o",
            hue_order=CONFIG[MPC_GRP]['hue_order'],
            palette=PALETTE,
            markersize=8,
            err_style="bars",
            err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
            errorbar="ci",
            data=data,
            **kwargs,
        )
        if "Titer" in disp_var:
            plt.plot(df_joined["Day"], df_joined["Setpoint"], "b--")

        plt.grid(axis="x", linestyle="--", color="gray")
        plt.grid(axis="y", linestyle="--", color="gray")

    g.map_dataframe(plot_measured)
    g.add_legend(title=CONFIG[MPC_GRP]['hue'])
    # plt.show()
    plt.savefig(fname=Path(fig_path, f"{i}-{disp_var}-1a-measured.png"))

    # Setpoint tracking (Controller, grand average)
    g = sns.FacetGrid(
        df_joined,
        col=CONFIG[MPC_GRP]['col'],
        height=4,
        sharex=False,
        sharey=True,
        despine=False,
        xlim=(-0.25, np.max(df_joined["Day"]) + 0.25),
        col_order=CONFIG[MPC_GRP]['col_order']
    )    

    def plot_measured_grand_avg(data, **kwargs):
        sns.lineplot(
            x="Day",
            y=disp_var,
            marker="o",
            markersize=8,
            err_style="bars",
            err_kws={"capsize": 2, "elinewidth": 2, "capthick": 2},
            errorbar="ci",
            data=data,
            **kwargs,
        )
        if "Titer" in disp_var:
            plt.plot(df_joined["Day"], df_joined["Setpoint"], "b--")

        plt.grid(axis="x", linestyle="--", color="gray")
        plt.grid(axis="y", linestyle="--", color="gray")

    g.map_dataframe(plot_measured_grand_avg)
    # plt.show()
    plt.savefig(fname=Path(fig_path, f"{i}-{disp_var}-1b-measured_grand_avg.png"))

    if "Titer" in disp_var:
        # Tracking error (Controller)
        g = sns.FacetGrid(
            df_joined,
            col=CONFIG[MPC_GRP]['col'],
            height=4,
            sharex=False,
            sharey=True,
            despine=False,
            ylim=(-30, 30),
            xlim=(-0.25, np.max(df_joined["Day"]) + 0.25),
            col_order=CONFIG[MPC_GRP]['col_order']
        )
        sns.set_style("white")

        def plot_error(data, **kwargs):
            sns.lineplot(
                x="Day",
                y=f"{disp_var} Tracking Error (%)",
                hue=CONFIG[MPC_GRP]['hue'],
                marker="o",
                hue_order=CONFIG[MPC_GRP]['hue_order'],
                palette=PALETTE,
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

        g.map_dataframe(plot_error)
        g.add_legend(title=CONFIG[MPC_GRP]['hue'])
        # plt.show()
        plt.savefig(fname=Path(fig_path, f"{i}-{disp_var}-2a-error.png"))

        # Tracking error (Controller, grand average)
        g = sns.FacetGrid(
            df_joined,
            col=CONFIG[MPC_GRP]['col'],
            height=4,
            sharex=False,
            sharey=True,
            despine=False,
            ylim=(-30, 30),
            xlim=(-0.25, np.max(df_joined["Day"]) + 0.25),
            col_order=CONFIG[MPC_GRP]['col_order']
        )
        sns.set_style("white")

        def plot_error_grand_avg(data, **kwargs):
            sns.lineplot(
                x="Day",
                y=f"{disp_var} Tracking Error (%)",
                marker="o",
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

        g.map_dataframe(plot_error_grand_avg)
        # plt.show()
        plt.savefig(fname=Path(fig_path,f"{i}-{disp_var}-2b-error_grand_avg.png")) 

        # Tracking error (Controller, no grouping)
        g = sns.FacetGrid(
            df_joined,
            col=CONFIG[MPC_GRP]['col'],
            height=4,
            sharex=False,
            sharey=True,
            despine=False,
            ylim=(-30, 30),
            xlim=(-0.25, np.max(df_joined["Day"]) + 0.25),
            col_order=CONFIG[MPC_GRP]['col_order']
        )
        sns.set_style("white")

        def plot_error_no_grp(data, **kwargs):
            sns.lineplot(
                x="Day",
                y=f"{disp_var} Tracking Error (%)",
                hue="Bioreactor",
                marker=None,
                palette=["k"],
                data=data,
                estimator=None,
                **kwargs,
            )
            plt.axhline(y=0, color="b", linestyle="--")
            # plt.grid(axis="x", linestyle="--", color="gray")
            # plt.grid(axis="y", linestyle="--", color="gray")

        g.map_dataframe(plot_error_no_grp)
        plt.savefig(fname=Path(fig_path,f"{i}-{disp_var}-2c-error_no_grp.png"))        

        # Absolute tracking error (Controller)
        g = sns.FacetGrid(
            df_joined,
            col=CONFIG[MPC_GRP]['col'],
            height=4,
            sharex=False,
            sharey=True,
            despine=False,
            ylim=(0, 30),
            xlim=(-0.25, np.max(df_joined["Day"]) + 0.25),
            col_order=CONFIG[MPC_GRP]['col_order']
        )
        sns.set_style("white")

        def plot_error_abs(data, **kwargs):
            sns.lineplot(
                x="Day",
                y=f"{disp_var} Absolute Tracking Error (%)",
                hue=CONFIG[MPC_GRP]['hue'],
                marker="o",
                hue_order=CONFIG[MPC_GRP]['hue_order'],
                palette=PALETTE,
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

        g.map_dataframe(plot_error_abs)
        g.add_legend(title=CONFIG[MPC_GRP]['hue'])
        plt.savefig(fname=Path(fig_path,f"{i}-{disp_var}-3a-abs_error.png"))   

        # Absolute tracking error (Controller, grand average)
        g = sns.FacetGrid(
            df_joined,
            col=CONFIG[MPC_GRP]['col'],
            height=4,
            sharex=False,
            sharey=True,
            despine=False,
            ylim=(0, 30),
            xlim=(-0.25, np.max(df_joined["Day"]) + 0.25),
            col_order=CONFIG[MPC_GRP]['col_order']
        )
        sns.set_style("white")

        def error_abs_grand_avg(data, **kwargs):
            sns.lineplot(
                x="Day",
                y=f"{disp_var} Absolute Tracking Error (%)",
                marker="o",
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

        g.map_dataframe(error_abs_grand_avg)
        plt.savefig(fname=Path(fig_path,f"{i}-{disp_var}-3b-abs_error_grand_avg.png"))   

    i = i + 1        

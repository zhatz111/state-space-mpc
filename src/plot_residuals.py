"""_plot residuals based on mpc .csv output file_
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

# Retrieve measurements
EXP_NUM = "AR24-005"
RAW_DATA_PATH = "AR24-005_MPC_DoE"
MASTER_DATA_TABLE = "AR24-005-daily_feed_plus_eor_data"
VESSELS = [3,5,6,9,13,15,18,20]
DISP_VAR = "IGG"
YLIM = [-1000,1000]
EST_HORIZON = 3

# Specify batch sheet path and load the read-only "master" sheet
top_dir = Path().absolute()
fig_path_lv1 = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/", RAW_DATA_PATH
)
data_path = Path(top_dir, f"data/simulation/{EXP_NUM}")
df = pd.read_csv(
    Path(data_path, f"{MASTER_DATA_TABLE}.csv")
)
df["Code_Run_Date"] = pd.to_datetime(df["Code_Run_Date"],format="mixed")

# Create figure output folder
fig_path_lv2 = Path(fig_path_lv1.expanduser(), MASTER_DATA_TABLE)
fig_path_lv2.mkdir(parents=True, exist_ok=True)

# Key column names
col_data = f"{DISP_VAR}--STATE_DATA"
col_est = f"{DISP_VAR}--STATE_EST"
col_pred = f"{DISP_VAR}--STATE_PRED"
col_mod = f"{DISP_VAR}--STATE_MOD"

for curr_vessel in VESSELS:

    df_br = df.loc[
        df["Bioreactor"] == curr_vessel,:
    ]
    unique_code_run_dates = df_br["Code_Run_Date"].unique()

    last_date = max(unique_code_run_dates)
    df_br_last = df_br.loc[df_br["Code_Run_Date"] == last_date,:]

    fig, ax = plt.subplots(3, 5, figsize=(19.2, 10.8))
    sub_ax = ax.flatten()
    identifier = f"{MASTER_DATA_TABLE}: BR{curr_vessel:02d}, {DISP_VAR}"
    save_path=fig_path_lv2/f"{DISP_VAR}-BR{curr_vessel:02d}.png"
    
    for count, curr_date in enumerate(unique_code_run_dates):
        df_br_day = df_br.loc[df_br["Code_Run_Date"] == curr_date,:]

        # Left join
        plot_data = pd.merge(
            df_br_day.loc[:,["Day",col_est,col_pred,col_mod]],
            df_br_last.loc[:, ["Day",col_data]],
            how="left",
            on="Day"
        )

        # Current day has the last modifier
        last_valid_mod_ind = plot_data[col_mod].last_valid_index()
        curr_time = plot_data.loc[last_valid_mod_ind,"Day"]

        # Retrieve the latest modifier
        modifiers_data = plot_data[col_mod].values
        latest_modifier = modifiers_data[last_valid_mod_ind]

        # Use the correct modifiers for days outside the est horizon (2024-02-26)
        modifiers = modifiers_data.copy()
        modifiers[:] = latest_modifier
        if sum(~np.isnan(modifiers_data)) > EST_HORIZON:
            for m in range(sum(~np.isnan(modifiers_data)) - EST_HORIZON):
                modifiers[m] = modifiers_data[m + EST_HORIZON - 1]
        
        # Combine past estimates and future predictions
        est_pred = plot_data[col_pred]
        est_pred[plot_data["Day"] <= curr_time] = np.multiply(
            plot_data[col_est].loc[plot_data["Day"] <= curr_time],
            modifiers[plot_data["Day"] <= curr_time]
            )
        
        # RMSE
        data_all = plot_data[col_data].values
        est_pred_all = est_pred.values
        rmse_all = np.round(np.sqrt(np.nanmean(np.square(est_pred_all - data_all))),1)
        data_pred = plot_data.loc[plot_data["Day"] >= curr_time,col_data].values
        pred_pred = est_pred[plot_data["Day"] >= curr_time].values
        rmse_pred = np.round(np.sqrt(np.nanmean(np.square(pred_pred - data_pred))),1)
        data_est = plot_data.loc[plot_data["Day"] <= curr_time,col_data].values
        est_est = est_pred[plot_data["Day"] <= curr_time].values
        rmse_est = np.round(np.sqrt(np.nanmean(np.square(est_est - data_est))),1)
        

        # Plot
        sub_ax[count].plot(
            plot_data["Day"].loc[plot_data["Day"] >= curr_time],
            pred_pred - data_pred,
            "ro",
            label=f"Prediction RMSE = {rmse_pred}",
        )
        sub_ax[count].plot(
            plot_data["Day"].loc[plot_data["Day"] <= curr_time],
            est_est - data_est,
            "bo",
            label=f"Estimation RMSE = {rmse_est}",
        )
        sub_ax[count].plot(
            plot_data["Day"],
            np.zeros(len(plot_data["Day"])),
            "k--",
        )        
        sub_ax[count].set_ylim(YLIM[0],YLIM[1])
        sub_ax[count].legend(prop={"size": 9})
        sub_ax[count].set_title(f"Day {curr_time}: RMSE = {rmse_all}", fontweight="bold")

    font = {
                "family": "sans-serif",
                "color": "gray",
                "weight": "normal",
                "size": 20,
            }
    fig.text(
        0.977,
        0.5,
        identifier,
        fontdict=font,
        va="center",
        ha="center",
        rotation="vertical",
    )
    plt.tight_layout(pad=1, h_pad=1, w_pad=1)
    plt.subplots_adjust(right=0.95, left=0.04, top=0.96, bottom=0.04)
    fig.savefig(fname=save_path)
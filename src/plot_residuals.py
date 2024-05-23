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
VESSELS = [3, 5, 6, 9, 13, 15, 18, 20]
DISP_VARS = ["IGG", "VIABILITY", "VCC"]
YLIM = [-100, 100]
EST_HORIZON = 3

# Specify batch sheet path and load the read-only "master" sheet
top_dir = Path().absolute()
fig_path_lv1 = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/", RAW_DATA_PATH
)
data_path = Path(top_dir, f"data/simulation/{EXP_NUM}")
df = pd.read_csv(Path(data_path, f"{MASTER_DATA_TABLE}.csv"))
df["Code_Run_Date"] = pd.to_datetime(df["Code_Run_Date"], format="mixed")

# Create figure output folder
fig_path_lv2 = Path(fig_path_lv1.expanduser(), MASTER_DATA_TABLE)
fig_path_lv2.mkdir(parents=True, exist_ok=True)

for disp_var in DISP_VARS:
    # Key column names
    col_data = f"{disp_var}--STATE_DATA"
    col_est = f"{disp_var}--STATE_EST"
    col_pred = f"{disp_var}--STATE_PRED"
    col_mod = f"{disp_var}--STATE_MOD"
    col_sp = f"{disp_var}--STATE_SP"

    # Day containers
    unique_code_run_dates = df["Code_Run_Date"].unique()
    fig_days = []
    sub_ax_days = []
    curr_times = np.zeros(len(unique_code_run_dates))
    for count_date, curr_date in enumerate(unique_code_run_dates):
        fig_day, ax_day = plt.subplots(3, 3, figsize=(19.2, 10.8))
        sub_ax_day = ax_day.flatten()
        fig_days.append(fig_day)
        sub_ax_days.append(sub_ax_day)

    for count_br, curr_vessel in enumerate(VESSELS):
        df_br = df.loc[df["Bioreactor"] == curr_vessel, :]

        last_date = max(unique_code_run_dates)
        df_br_last = df_br.loc[df_br["Code_Run_Date"] == last_date, :]

        fig_br, ax_br = plt.subplots(3, 4, figsize=(19.2, 10.8))
        sub_ax = ax_br.flatten()

        identifier_br = f"{MASTER_DATA_TABLE}: BR{curr_vessel:02d}, {disp_var}"
        save_path_br = fig_path_lv2 / f"{disp_var}-BR{curr_vessel:02d}.png"

        for count_date, curr_date in enumerate(unique_code_run_dates):
            df_br_day = df_br.loc[df_br["Code_Run_Date"] == curr_date, :]

            plot_data = pd.merge(
                df_br_day.loc[:, ["Day", col_est, col_pred, col_mod]],
                df_br_last.loc[:, ["Day", col_data]],
                how="left",
                on="Day",
            )

            # # Left join
            # try:
            #     plot_data = pd.merge(
            #         df_br_day.loc[:,["Day",col_est,col_pred,col_mod]],
            #         df_br_last.loc[:, ["Day",col_data,col_sp]],
            #         how="left",
            #         on="Day"
            #     )
            # except KeyError:
            #     plot_data = pd.merge(
            #         df_br_day.loc[:,["Day",col_est,col_pred,col_mod]],
            #         df_br_last.loc[:, ["Day",col_data]],
            #         how="left",
            #         on="Day"
            #     )

            # Current day has the last modifier
            last_valid_mod_ind = plot_data[col_mod].last_valid_index()
            curr_time = plot_data.loc[last_valid_mod_ind, "Day"]

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
                modifiers[plot_data["Day"] <= curr_time],
            )

            # RMSE/MAPE
            data_all = plot_data[col_data].values
            est_pred_all = est_pred.values
            rmse_all = np.round(
                np.sqrt(np.nanmean(np.square(est_pred_all - data_all))), 1
            )
            mape_all = np.round(
                np.nanmean(np.abs((est_pred_all - data_all) / data_all * 100)), 0
            )
            data_pred = plot_data.loc[plot_data["Day"] >= curr_time, col_data].values
            pred_pred = est_pred[plot_data["Day"] >= curr_time].values
            rmse_pred = np.round(
                np.sqrt(np.nanmean(np.square(pred_pred - data_pred))), 1
            )
            mape_pred = np.round(
                np.nanmean(np.abs((pred_pred - data_pred) / data_pred * 100)), 0
            )
            data_est = plot_data.loc[plot_data["Day"] <= curr_time, col_data].values
            est_est = est_pred[plot_data["Day"] <= curr_time].values
            rmse_est = np.round(np.sqrt(np.nanmean(np.square(est_est - data_est))), 1)
            mape_est = np.round(
                np.nanmean(np.abs((est_est - data_est) / data_est * 100)), 0
            )

            # Labels
            label_pred = f"Pred. RMSE = {rmse_pred:.1f}; MAPE = {mape_pred:.0f}"
            label_est = f"Est. RMSE = {rmse_est:.1f}; MAPE = {mape_est:.0f}"
            label_all = f"{disp_var}, BR{curr_vessel:02d}, Day {curr_time:.0f}: RMSE = {rmse_all:.1f}; MAPE = {mape_all:.0f}"

            # # Tracking error
            # try:
            #     sp_all = plot_data[col_sp].values
            #     rmse_ctrl = np.round(np.sqrt(np.nanmean(np.square(data_all - sp_all))),1)
            #     mape_ctrl = np.round(np.nanmean(np.abs((data_all - sp_all)/sp_all*100)),0)
            #     label_ctrl = f"Ctrl. RMSE = {rmse_ctrl:.1f}; MAPE = {mape_ctrl:.0f}"
            # except KeyError:
            #     pass

            # Plot (by reactor)
            sub_ax[count_date].stem(
                plot_data["Day"].loc[plot_data["Day"] >= curr_time],
                (pred_pred - data_pred) / data_pred * 100,
                linefmt="r-",
                markerfmt="ro",
                basefmt="k--",
                label=label_pred,
            )
            sub_ax[count_date].stem(
                plot_data["Day"].loc[plot_data["Day"] <= curr_time],
                (est_est - data_est) / data_est * 100,
                linefmt="b-",
                markerfmt="bo",
                basefmt="k--",
                label=label_est,
            )
            # try:
            #     sub_ax[count_date].stem(
            #         plot_data["Day"],
            #         (data_all - sp_all)/sp_all*100,
            #         linefmt='k-',markerfmt='ko',basefmt='k--',
            #         label=label_ctrl,
            #     )
            # except KeyError:
            #     pass
            sub_ax[count_date].set_ylim(YLIM[0], YLIM[1])
            sub_ax[count_date].legend(prop={"size": 9})
            sub_ax[count_date].set_title(label_all, fontweight="bold")

            # Plot (by day)
            curr_times[count_date] = curr_time
            sub_ax_days[count_date][count_br].stem(
                plot_data["Day"].loc[plot_data["Day"] >= curr_time],
                (pred_pred - data_pred) / data_pred * 100,
                linefmt="r-",
                markerfmt="ro",
                basefmt="k--",
                label=label_pred,
            )
            sub_ax_days[count_date][count_br].stem(
                plot_data["Day"].loc[plot_data["Day"] <= curr_time],
                (est_est - data_est) / data_est * 100,
                linefmt="b-",
                markerfmt="bo",
                basefmt="k--",
                label=label_est,
            )
            # try:
            #     sub_ax_days[count_date][count_br].stem(
            #         plot_data["Day"],
            #         (data_all - sp_all)/sp_all*100,
            #         linefmt='k-',markerfmt='ko',basefmt='k--',
            #         label=label_ctrl,
            #     )
            # except KeyError:
            #     pass
            sub_ax_days[count_date][count_br].set_ylim(YLIM[0], YLIM[1])
            sub_ax_days[count_date][count_br].legend(prop={"size": 9})
            sub_ax_days[count_date][count_br].set_title(label_all, fontweight="bold")

        font = {
            "family": "sans-serif",
            "color": "gray",
            "weight": "normal",
            "size": 20,
        }

        # Decorate (by reactor)
        fig_br.text(
            0.977,
            0.5,
            identifier_br,
            fontdict=font,
            va="center",
            ha="center",
            rotation="vertical",
        )
        plt.tight_layout(pad=1, h_pad=1, w_pad=1)
        plt.subplots_adjust(right=0.95, left=0.04, top=0.96, bottom=0.04)
        if count_date + 1 < len(sub_ax):
            for sub_ax_single in sub_ax[count_date + 1 :]:
                sub_ax_single.set_axis_off()
        fig_br.savefig(fname=save_path_br)
        print(f"{disp_var}: BR{curr_vessel:02d}")

    # Decorate (by day)
    for count_date, curr_date in enumerate(unique_code_run_dates):
        curr_time = curr_times[count_date]
        identifier_day = f"{MASTER_DATA_TABLE}: Day {curr_time:.0f}, {disp_var}"
        save_path_day = fig_path_lv2 / f"{disp_var}-D{curr_time:.0f}.png"
        fig_day = fig_days[count_date]
        fig_day.text(
            0.977,
            0.5,
            identifier_day,
            fontdict=font,
            va="center",
            ha="center",
            rotation="vertical",
        )
        plt.tight_layout(pad=1, h_pad=1, w_pad=1)
        plt.subplots_adjust(right=0.95, left=0.04, top=0.96, bottom=0.04)
        if count_br + 1 < len(sub_ax_days[count_date]):
            for sub_ax_single in sub_ax_days[count_date][count_br + 1 :]:
                sub_ax_single.set_axis_off()
        fig_day.savefig(fname=save_path_day)
        print(f"{disp_var}: Day {curr_time:.0f}")

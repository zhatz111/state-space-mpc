"""Main code for visualizing daily MPC trends
    Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2022-11-04
    Modified: 2024-04-29
"""

# Imports from Standard Library
import math
import warnings
from typing import Union, Optional
from pathlib import Path

# Imports from 3rd party library
# import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports from Classes in Repository
from src.mpc.mpc_optimizer import Bioreactor, Controller

# RMSE
from sklearn.metrics import mean_squared_error

# suppress warnings
warnings.filterwarnings("ignore")


class MPCVisualizer:
    """_summary_"""

    def __init__(
        self,
        bioreactor: Union[list[Bioreactor], Bioreactor],
        controller: Union[list[Controller], Controller],
    ):
        if isinstance(bioreactor, list) and isinstance(controller, list):
            if len(bioreactor) != len(controller):
                raise ValueError(
                    "List of Bioreactor instances must be the same length as list of Controller Instances."
                )
            for b, c in zip(bioreactor, controller):
                if not isinstance(b, Bioreactor):
                    raise ValueError(f"{b} is not an instance of the Bioreactor Class.")
                if not isinstance(c, Controller):
                    raise ValueError(f"{c} is not an instance of the Controller Class.")
            self.bioreactor = bioreactor
            self.controller = controller
        elif isinstance(bioreactor, Bioreactor) == isinstance(controller, Controller):
            self.bioreactor = bioreactor
            self.controller = controller
        elif isinstance(bioreactor, Bioreactor) != isinstance(controller, Controller):
            raise ValueError(
                "Class inputs must either be both lists or both instances of the Bioreactor and Controller Classes"
            )
        else:
            raise ValueError("Provided inputs to class are not correct.")

    def mpc_daily_plot(
        self,
        save_paths: Union[list, tuple, None] = None,
        metadata: Optional[dict] = None,
        unit_dict: Optional[dict] = None,
        PV: str = "",
        identifier: str = "",
        display=False,
    ):
        # I want this function to plot the trajectory of the process variable on any given day with
        # the STATE_DATA, STATE_EST, and STATE_PRED, graphed together for comparison, I want everything
        # after the curr_time of the bioreactor class to be a different color (red) and everything
        # before that should be black since it cannot be changed. Dashed lines to refer to predictions
        # and solid line to refer to estimations
        if isinstance(self.bioreactor, Bioreactor) and isinstance(
            self.controller, Controller
        ):
            if not PV:
                y_var = [x.split("--")[0] for x in self.controller.pv_names]
            else:
                y_var = PV

            # if not MV:
            #     MV = self.controller.mv_names[0]
            #     input_var = MV.split("--")[0]
            #     # input_var = "DAILY_NORMALIZED_FEED"
            # else:
            #     input_var = MV

            PV_SP_SUFFIX = "--STATE_SP"
            PV_SUFFIX = "--STATE_DATA"
            PRED_SUFFIX = "--STATE_PRED"
            EST_SUFFIX = "--STATE_EST"
            MV_SUFFIX = "--INPUT_DATA"
            MV_REF_SUFFIX = "--INPUT_REF"

            plot_data = self.bioreactor.return_data(show_daily_feed=True)
            fig, ax = plt.subplots(3, 3, figsize=(19.2, 10.8))

            # Plot the Bioreactor Data

            # Create a mask for NaN Values
            last_ax_used = 0
            sub_ax = ax.flatten()
            for count, state in enumerate(self.bioreactor.process_model.states):

                # Retrieve the latest modifier
                modifiers_data = plot_data[state + '--STATE_MOD'].values
                latest_modifier = modifiers_data[~pd.isnull(modifiers_data)][-1]

                # # Use the correct modifiers for days outside the est horizon (2024-02-26)
                # modifiers = modifiers_data.copy()
                # modifiers[:] = latest_modifier
                # if sum(~pd.isnull(modifiers_data)) > self.controller.est_horizon:
                #     for m in range(sum(~pd.isnull(modifiers_data)) - self.controller.est_horizon):
                #         modifiers[m] = modifiers_data[m + self.controller.est_horizon - 1]

                # Additive modifier (2024-06-20)
                modifiers = modifiers_data.copy()
                modifiers[np.where(np.isnan(modifiers))] = latest_modifier   

                # Calculate est. error
                y_data = plot_data[state + PV_SUFFIX].loc[plot_data["Day"] <= self.bioreactor.curr_time].values
                y_est = plot_data[state + EST_SUFFIX].loc[plot_data["Day"] <= self.bioreactor.curr_time].values
                y_data_est_have_values = ~np.logical_or(np.isnan(y_data),np.isnan(y_est))
                rmse_est = np.sqrt(mean_squared_error(y_data[y_data_est_have_values],y_est[y_data_est_have_values]))

                if state in y_var:
                    measured_mask = np.isfinite(plot_data[state + PV_SUFFIX])
                    sub_ax[count].plot(
                        plot_data["Day"][measured_mask],
                        plot_data[state + PV_SUFFIX][measured_mask],
                        "ks",
                        label="Measured Output",
                        # markersize=10
                    )
                    sub_ax[count].plot(
                        plot_data["Day"].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                        plot_data[state + PRED_SUFFIX].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                        "r-o",
                        label="Predicted Output",
                    )

                    sub_ax[count].plot(
                        plot_data["Day"].loc[plot_data["Day"] <= self.bioreactor.curr_time],
                        plot_data[state + EST_SUFFIX].loc[plot_data["Day"] <= self.bioreactor.curr_time],
                        # np.add(
                        #     plot_data[state + EST_SUFFIX].loc[plot_data["Day"] <= self.bioreactor.curr_time],
                        #     modifiers[plot_data["Day"] <= self.bioreactor.curr_time]
                        #     ),
                        "b-o",
                        label=f"Estimated Output ({np.round(rmse_est,2)})",
                    )
                    try:

                        y_sp = plot_data[state + PV_SP_SUFFIX].loc[plot_data["Day"] <= self.bioreactor.curr_time].values
                        y_data_sp_have_values = ~np.logical_or(np.isnan(y_data),np.isnan(y_est))
                        rmse_ctrl = np.sqrt(mean_squared_error(y_data[y_data_sp_have_values],y_sp[y_data_sp_have_values]))

                        sub_ax[count].plot(
                            plot_data["Day"],
                            plot_data[state + PV_SP_SUFFIX],
                            "g--",
                            label=f"Setpoint ({np.round(rmse_ctrl,2)})",
                        )
                    except KeyError:
                        pass
                    if isinstance(unit_dict, dict):
                        sub_ax[count].set_title(f"{state} (PV) {unit_dict[state]}", fontweight="bold")
                    else:
                        sub_ax[count].set_title(f"{state} (PV)", fontweight="bold")
                    sub_ax[count].legend(prop={"size": 9})
                else:
                    measured_mask = np.isfinite(plot_data[state + PV_SUFFIX])
                    sub_ax[count].plot(
                        plot_data["Day"][measured_mask],
                        plot_data[state + PV_SUFFIX][measured_mask],
                        "ks",
                        label="Measured Output",
                    )
                    sub_ax[count].plot(
                        plot_data["Day"].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                        plot_data[state + PRED_SUFFIX].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                        "r-o",
                        label="Predicted Output",
                    )
                    sub_ax[count].plot(
                        plot_data["Day"].loc[plot_data["Day"] <= self.bioreactor.curr_time],
                        plot_data[state + EST_SUFFIX].loc[plot_data["Day"] <= self.bioreactor.curr_time],
                        # np.add(
                        #     plot_data[state + EST_SUFFIX].loc[plot_data["Day"] <= self.bioreactor.curr_time],
                        #     modifiers[plot_data["Day"] <= self.bioreactor.curr_time]
                        #     ),
                        "b-o",
                        label=f"Estimated Output ({np.round(rmse_est,2)})",
                    )
                    if isinstance(unit_dict, dict):
                        sub_ax[count].set_title(f"{state} {unit_dict[state]}", fontweight="bold")
                    else:
                        sub_ax[count].set_title(f"{state}", fontweight="bold")
                    sub_ax[count].legend(prop={"size": 9})
                last_ax_used = count

                sub_ax[count].set_xlim([0,np.max(plot_data["Day"])])

            # Plot the Controller Actions
            for count, inputs in enumerate(
                self.bioreactor.process_model.inputs, start=last_ax_used + 1
            ):
                
                # Determine if the input is an MV and has constraint
                if inputs + MV_SUFFIX in self.controller.mv_names:
                    mv_where = np.where(np.isin(self.controller.mv_names,inputs + MV_SUFFIX))[0]
                    mv_constr = self.controller.mv_constr[:,mv_where]
                    mv_constr[0] = 0
                else:
                    mv_constr = []

                if count != len(sub_ax):
                    try:
                        sub_ax[count].step(
                            plot_data["Day"].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                            plot_data[inputs + MV_SUFFIX].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                            "r-",
                            label="Predicted Control Input",
                            where="post"
                        )
                        sub_ax[count].step(
                            plot_data["Day"].loc[
                                plot_data["Day"] <= self.bioreactor.curr_time
                            ],
                            plot_data[inputs + MV_SUFFIX].loc[
                                plot_data["Day"] <= self.bioreactor.curr_time
                            ],
                            "k-",
                            label="Past Control Input",
                            where="post"
                        )
                        try:
                            sub_ax[count].step(
                                plot_data["Day"],
                                plot_data[inputs + MV_REF_SUFFIX],
                                "g--",
                                label="Historical Input Reference",
                                where="post"
                            )
                        except KeyError:
                            pass
                        if isinstance(unit_dict, dict):
                            sub_ax[count].set_title(f"{inputs} {unit_dict[inputs]}", fontweight="bold")
                        else:
                            sub_ax[count].set_title(f"{inputs}", fontweight="bold")
                    except KeyError:
                        sub_ax[count].step(
                            plot_data["Day"].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                            plot_data[self.bioreactor.daily_feed_name_data].loc[
                                plot_data["Day"] >= self.bioreactor.curr_time
                            ],
                            "r-",
                            label="Predicted Control Input",
                            where="post"
                        )
                        sub_ax[count].step(
                            plot_data["Day"].loc[
                                plot_data["Day"] <= self.bioreactor.curr_time
                            ],
                            plot_data[self.bioreactor.daily_feed_name_data].loc[
                                plot_data["Day"] <= self.bioreactor.curr_time
                            ],
                            "k-",
                            label="Past Control Input",
                            where="post"
                        )
                        sub_ax[count].step(
                            plot_data["Day"],
                            plot_data[self.bioreactor.daily_feed_name_ref],
                            "g--",
                            label="Historical Input Reference",
                            where="post"
                        )
                        if isinstance(unit_dict, dict):
                            label = self.bioreactor.daily_feed_name_ref.split('--')[0]
                            sub_ax[count].set_title(
                                f"{label} (MV) {unit_dict[label]}",
                                fontweight="bold"
                            )
                        else:
                            sub_ax[count].set_title(
                                f"{self.bioreactor.daily_feed_name_ref.split('--')[0]} (MV)",
                                fontweight="bold"
                            )
                    sub_ax[count].legend(prop={"size": 9})
                    if len(mv_constr) > 0:
                        sub_ax[count].set_ylim(mv_constr)

                sub_ax[count].set_xlim([0,np.max(plot_data["Day"])])

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
            if display:
                plt.show()

            # Save the figure if the arguments passed are the correct instances
            for save_path in save_paths:
                if isinstance(save_path, (str, Path)) and isinstance(metadata, dict):
                    fig.savefig(fname=save_path, metadata=metadata)
                elif isinstance(save_path, (str, Path)):
                    fig.savefig(fname=save_path)


        else:
            raise ValueError(
                "This method cannot be used with a list of Bioreactor and Controller Classes"
            )

    def plot_controllers(self):
        """_summary_"""
        # Plots the all the bioreactors and controllers passed to __init__
        if isinstance(self.bioreactor, list) and isinstance(self.controller, list):
            cols = 4
            rows = math.ceil(self.bioreactor[0].duration / cols)

            # Create figures for the target trajectory variable
            fig_target_list = []
            for _ in self.bioreactor:
                fig, axes = plt.subplots(rows, cols, figsize=(9, 7), squeeze=False)
                fig.subplots_adjust(top=0.8)
                fig_target_list.append([fig, axes])

            # Create figures for the input variables
            fig_input_list = []
            for _ in self.bioreactor:
                fig, axes = plt.subplots(rows, cols, figsize=(9, 7), squeeze=False)
                fig.subplots_adjust(top=0.8)
                fig_input_list.append([fig, axes])

            # Generate the target variable graphs
            for count, tup in enumerate(fig_target_list):
                for ax, key in zip(
                    tup[1].flatten(),
                    self.controller[count].data_after_optim_dict.keys(),
                ):
                    ax.plot(
                        self.controller[count].ts,
                        self.controller[count].pv_sps[:, 0],
                        "k--",
                        label="Target Trajectory",
                    )
                    # ax.plot(
                    #     self.controller[count].data_before_optim_dict[key]["Day"],
                    #     self.controller[count].data_before_optim_dict[key][
                    #         self.controller[0].pv_names
                    #     ],
                    #     "b-",
                    #     label="Un-optimized",
                    # )
                    ax.plot(
                        self.controller[count].data_after_optim_dict[key]["Day"],
                        self.controller[count].data_after_optim_dict[key][
                            self.controller[0].pv_names
                        ],
                        "r-",
                        label="MPC Optimized",
                    )
                    ax.title.set_text(f"Day: {key}")
                tup[0].supxlabel("Day", size="x-large", weight="bold")
                tup[0].supylabel("Level", size="x-large", weight="bold")
                tup[0].suptitle(
                    f"{self.controller[0].pv_names} for {self.bioreactor[count].vessel}",
                    size="x-large",
                    weight="bold",
                    y=0.98,
                )
                tup[0].tight_layout()

            # Generate the input variable graphs
            for count, tup in enumerate(fig_input_list):
                for ax, key in zip(
                    tup[1].flatten(),
                    self.controller[count].data_after_optim_dict.keys(),
                ):
                    ax.step(
                        self.controller[count].ts,
                        self.bioreactor[count].original_data[
                            self.controller[0].mv_names
                        ],
                        "k--",
                        label="Target Trajectory",
                        where="post"
                    )
                    ax.step(
                        self.controller[count].data_before_optim_dict[key]["Day"],
                        self.controller[count].data_before_optim_dict[key][
                            self.controller[count].mv_names
                        ],
                        "b-",
                        label="Un-optimized",
                        where="post"
                    )
                    ax.step(
                        self.controller[count].data_after_optim_dict[key]["Day"],
                        self.controller[count].data_after_optim_dict[key][
                            self.controller[count].mv_names
                        ],
                        "r-",
                        label="Optimized",
                        where="post"
                    )
                    ax.title.set_text(f"Day: {key}")
                tup[0].supxlabel("Day", size="x-large", weight="bold")
                tup[0].supylabel("Level", size="x-large", weight="bold")
                tup[0].suptitle(
                    f"Daily_Normalized_Feed for {self.bioreactor[count].vessel}",
                    size="x-large",
                    weight="bold",
                    y=0.98,
                )
                tup[0].tight_layout()
                tup[0].legend()

            # Save graphs to folder rather than displaying them

            plt.legend()
            plt.show()
        else:
            raise ValueError(
                "This method cannot be used with a single Bioreactor and Controller Class"
            )

    def plot_simulations(self):
        """_summary_"""
        # add subplot for each simulation in list
        # within each subplot i want to add line for projected target path without any manipulation
        # add line for open loop MPC, add line for closed loop MPC, add line for setpoint to be tracked
        # input labels for graph based on DoE design
        if isinstance(self.bioreactor, list) and isinstance(self.controller, list):
            cols = 4
            rows = math.ceil(len(self.bioreactor) / cols)
            fig, axes = plt.subplots(
                rows, cols, figsize=(9, 7), squeeze=False, constrained_layout=True
            )
            # fig.subplots_adjust(top=0.8)
            for count, ax in enumerate(axes.flatten()):
                if count < len(self.bioreactor):
                    before_keys = list(
                        self.controller[count].data_before_optim_dict.keys()
                    )
                    # Line for setpoint tracking tracjectory
                    ax.plot(
                        self.controller[count].ts,
                        self.controller[count].pv_sps[:, 0],
                        "k--",
                        label="Target Trajectory",
                    )

                    # Line for Open Loop MPC
                    ax.plot(
                        self.bioreactor[count].open_loop_df["Day"],
                        self.bioreactor[count].open_loop_df[
                            self.controller[count].pv_names
                        ],
                        "b-",
                        label="Open Loop",
                    )

                    # Line for Closed Loop MPC
                    ax.plot(
                        self.bioreactor[count].data["Day"],
                        self.bioreactor[count].data[self.controller[count].pv_names],
                        "r-",
                        label="Closed Loop MPC",
                    )
                    ax.title.set_text(self.bioreactor[count].vessel)
                if ax == axes.flatten()[0]:
                    ax.legend()

            fig2, axes2 = plt.subplots(
                rows, cols, figsize=(9, 7), squeeze=False, constrained_layout=True
            )
            # fig2.subplots_adjust(top=0.8)
            for count, ax in enumerate(axes2.flatten()):
                if count < len(self.bioreactor):
                    before_keys = list(
                        self.controller[count].data_before_optim_dict.keys()
                    )
                    # Line for setpoint tracking tracjectory
                    ax.step(
                        self.controller[count].ts,
                        self.bioreactor[count].original_data[
                            self.controller[0].mv_names
                        ],
                        "k--",
                        label="Target Trajectory",
                        where="post"
                    )

                    # Line for Open Loop MPC
                    ax.step(
                        self.controller[count].data_before_optim_dict[before_keys[0]][
                            "Day"
                        ],
                        self.controller[count].data_before_optim_dict[before_keys[0]][
                            self.controller[count].mv_names
                        ],
                        "b-",
                        label="Open Loop MPC",
                        where="post"
                    )

                    # Line for Closed Loop MPC
                    ax.step(
                        self.controller[count].data_after_optim_dict[before_keys[-1]][
                            "Day"
                        ],
                        self.controller[count].data_after_optim_dict[before_keys[-1]][
                            self.controller[count].mv_names
                        ],
                        "r-",
                        label="Closed Loop MPC",
                        where="post"
                    )

                    ax.title.set_text(self.bioreactor[count].vessel)
                if ax == axes2.flatten()[0]:
                    ax.legend()

                # if count == 0:
                #     pass
                #     # ax.legend(bbox_to_anchor=(3, 1.3), ncol=3, fancybox=True, shadow=True)

            fig.supxlabel("Day", size="x-large", weight="bold")
            fig.supylabel("Level", size="x-large", weight="bold")

            fig2.supxlabel("Day", size="x-large", weight="bold")
            fig2.supylabel("Level", size="x-large", weight="bold")

            plt.show()
        else:
            raise ValueError(
                "This method cannot be used with a single Bioreactor and Controller Class"
            )

    def output_table(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if isinstance(self.bioreactor, list) and isinstance(self.controller, list):
            output_dict = {}
            for b, c in zip(self.bioreactor, self.controller):
                optim_percent_diff = (
                    100
                    * (
                        c.data_after_optim_dict[list(c.data_after_optim_dict)[-1]][
                            c.pv_names
                        ].values.flatten()
                        - c.pv_sps[:, 0]
                    )
                    / c.pv_sps[:, 0]
                )
                unoptim_percent_diff = (
                    100
                    * (
                        c.data_after_optim_dict[list(c.data_after_optim_dict)[0]][
                            c.pv_names
                        ].values.flatten()
                        - c.pv_sps[:, 0]
                    )
                    / c.pv_sps[:, 0]
                )
                br_df = pd.DataFrame(
                    {
                        "Day": c.ts,
                        "Target Trajectory": c.pv_sps[:, 0],
                        "Predicted Trajectory": c.data_after_optim_dict[
                            list(c.data_after_optim_dict)[0]
                        ][c.pv_names].values.flatten(),
                        "Optimized Trajectory": c.data_after_optim_dict[
                            list(c.data_after_optim_dict)[-1]
                        ][c.pv_names].values.flatten(),
                        "Target Trajectory Feed Amount (mL)": b.original_data[
                            c.mv_names
                        ].values.flatten(),
                        "Optimized Trajectory Feed Amount (mL)": c.data_after_optim_dict[
                            list(c.data_after_optim_dict)[-1]
                        ][c.mv_names].values.flatten(),
                        "Unoptimized Difference %": unoptim_percent_diff,
                        "Optimized Difference %": optim_percent_diff,
                    }
                )
                output_dict[b.vessel] = br_df

            cols = 4
            rows = math.ceil(len(self.bioreactor) / cols)
            fig, axes = plt.subplots(
                rows, cols, figsize=(9, 7), squeeze=False, constrained_layout=True
            )
            dict_keys = list(output_dict.keys())
            bar_width = 0.4
            for count, ax in enumerate(axes.flatten()):
                if count < len(self.bioreactor):
                    ax.bar(
                        output_dict[dict_keys[count]]["Day"],
                        output_dict[dict_keys[count]]["Unoptimized Difference %"],
                        color="r",
                        width=bar_width,
                        label="Open Loop MPC",
                    )

                    ax.bar(
                        output_dict[dict_keys[count]]["Day"] + bar_width,
                        output_dict[dict_keys[count]]["Optimized Difference %"],
                        color="b",
                        width=bar_width,
                        label="Closed Loop MPC",
                    )
                    ax.title.set_text(self.bioreactor[count].vessel)

            fig.supxlabel("Day", size="x-large", weight="bold")
            fig.supylabel("% Difference", size="x-large", weight="bold")
            plt.show()

            return pd.concat(output_dict)
        else:
            raise ValueError(
                "This method cannot be used with a single Bioreactor and Controller Class"
            )

"""_summary_
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Imports from Standard Library
import math
import warnings

# Imports from 3rd party library
import pandas as pd
import matplotlib.pyplot as plt

# Imports from Classes in Repository
from mpc.mpc_optimizer import Bioreactor, Controller

# suppress warnings
warnings.filterwarnings("ignore")


class MPCVisualizer:
    """_summary_"""

    def __init__(
        self,
        bioreactor: list[Bioreactor],
        controller: list[Controller],
    ):
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

    def plot_controllers(self):
        """_summary_"""
        # Plots the all the bioreactors and controllers passed to __init__
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
                tup[1].flatten(), self.controller[count].data_after_optim_dict.keys()
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
                tup[1].flatten(), self.controller[count].data_after_optim_dict.keys()
            ):
                ax.step(
                    self.controller[count].ts,
                    self.bioreactor[count].original_data[self.controller[0].mv_names],
                    "k--",
                    label="Target Trajectory",
                )
                ax.step(
                    self.controller[count].data_before_optim_dict[key]["Day"],
                    self.controller[count].data_before_optim_dict[key][
                        self.controller[count].mv_names
                    ],
                    "b-",
                    label="Un-optimized",
                )
                ax.step(
                    self.controller[count].data_after_optim_dict[key]["Day"],
                    self.controller[count].data_after_optim_dict[key][
                        self.controller[count].mv_names
                    ],
                    "r-",
                    label="Optimized",
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

    def plot_simulations(self):
        """_summary_"""
        # add subplot for each simulation in list
        # within each subplot i want to add line for projected target path without any manipulation
        # add line for open loop MPC, add line for closed loop MPC, add line for setpoint to be tracked
        # input labels for graph based on DoE design

        cols = 4
        rows = math.ceil(len(self.bioreactor) / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(9, 7), squeeze=False, constrained_layout=True
        )
        # fig.subplots_adjust(top=0.8)
        for count, ax in enumerate(axes.flatten()):
            if count < len(self.bioreactor):
                before_keys = list(self.controller[count].data_before_optim_dict.keys())
                # Line for setpoint tracking tracjectory
                ax.plot(
                    self.controller[count].ts,
                    self.controller[count].pv_sps[:, 0],
                    "k--",
                    label="Target Trajectory",
                )

                # Line for Open Loop MPC
                ax.plot(
                    self.bioreactor[count].open_loop_df[
                        "Day"
                    ],
                    self.bioreactor[count].open_loop_df[
                        self.controller[count].pv_names
                    ],
                    "b-",
                    label="Open Loop",
                )

                # Line for Closed Loop MPC
                ax.plot(
                    self.bioreactor[count].data[
                        "Day"
                    ],
                    self.bioreactor[count].data[
                        self.controller[count].pv_names
                    ],
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
                before_keys = list(self.controller[count].data_before_optim_dict.keys())
                # Line for setpoint tracking tracjectory
                ax.step(
                    self.controller[count].ts,
                    self.bioreactor[count].original_data[self.controller[0].mv_names],
                    "k--",
                    label="Target Trajectory",
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

    def output_table(self):
        """_summary_

        Returns:
            _type_: _description_
        """
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
                    "Target Trajectory Feed Amount (mL)": b.original_data[c.mv_names].values.flatten(),
                    "Optimized Trajectory Feed Amount (mL)": c.data_after_optim_dict[list(c.data_after_optim_dict)[-1]][
                        c.mv_names
                    ].values.flatten(),
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
                    output_dict[dict_keys[count]]["Day"]+bar_width,
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

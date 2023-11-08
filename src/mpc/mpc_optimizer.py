"""MPC-related class definitions
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Standard Library Imports
import warnings

# 3rd Party Library Imports
import numpy as np
import pandas as pd
from scipy import optimize

# State-Space-Model Package Imports
from models.ssm import StateSpaceModel


def daily_to_cumulative_feed(model, u_matrix_daily):
    """
    The function `daily_to_cumulative_feed` converts a U matrix's daily feed column to cumulative feed
    for lsim.

    Args:
      model: The "model" parameter is a variable that represents a model object. It is not specified in
    the code snippet provided, so its exact definition and usage would depend on the context in which
    this function is being used.
      u_matrix_daily: The u_matrix_daily parameter is a numpy array representing the U matrix with daily
    feed values.

    Returns:
      the modified U matrix with the daily feed column converted to cumulative feed.
    """
    u_matrix_cumulative = np.copy(u_matrix_daily)
    cumulative_feed_loc = np.where(np.isin(model.inputs, "Cumulative_Normalized_Feed"))[
        0
    ]
    u_matrix_cumulative[:, cumulative_feed_loc] = np.cumsum(
        u_matrix_cumulative[:, cumulative_feed_loc]
    ).reshape([-1, 1])
    return u_matrix_cumulative


class Bioreactor:
    """
    The Bioreactor class represents a bioreactor object used for simulation and processing of real-time
    data, with methods for initializing, resetting, logging samples, updating inputs, and advancing the
    simulation.
    """

    def __init__(
        self,
        vessel: str,
        process_model: StateSpaceModel,
        data: pd.DataFrame,
        # plot_names: list,
        # plot_ts: list,
        # plot_sps: list,
    ):
        """
        The function initializes an object with attributes based on user input, checks the validity of
        the data, and converts cumulative feed to daily feed if necessary.

        Args:
          vessel (str): The `vessel` parameter is a string that represents the name of the vessel. It is
        used for processing multiple bioreactors.
          process_model (StateSpaceModel): The `process_model` parameter is a `StateSpaceModel` object
        that represents the mathematical model of the process being simulated. It contains information
        about the states, inputs, and outputs of the system, as well as the equations that describe the
        dynamics of the system.
          data (pd.DataFrame): The `data` parameter is a pandas DataFrame that contains the time-series
        data for the bioreactor process. It should have the following columns:
        """

        # Update attributes based on user input
        self.vessel = vessel  # Vessel name for processing multiple bioreactors
        self.process_model = (
            process_model  # Model (if provided) for simulating a process
        )
        self.data = data.copy(deep=True)
        # self.data = data # Data for storing simulation results or real data (if provided)

        # Data frame for open_loop simulation results
        self.open_loop_df = pd.DataFrame()

        # Check if the data set starts on Day 0
        if self.data["Day"].values[0] != 0:
            raise ValueError("Data set does not start on Day 0!")

        # Initialize other attributes
        self.curr_time = 0
        # self.state = self.data.filter(items=self.process_model.states)
        self.duration = self.data["Day"].values[-1]

        # Check if the data set ends on Day duration
        if self.data.shape[0] - 1 != self.duration:
            raise ValueError("Data set has missing or duplicate days!")

        # Check if days are consecutive (2023-10-21)
        if any(np.diff(self.data["Day"]) != 1):
            raise ValueError("Data set is not in 1-day increments!")

        # Convert cumulative feed to daily feed
        if np.isin("Cumulative_Normalized_Feed", self.data.columns):
            self.data["Cumulative_Normalized_Feed"] = np.append(
                np.diff(self.data.loc[:, "Cumulative_Normalized_Feed"]), 0
            )
            self.has_cumulative_feed = True
            warnings.warn(
                "Cumulative feed was converted to daily feed (variable name is unchanged)!"
            )
        else:
            self.has_cumulative_feed = False

        # Retain the original dataset
        self.original_data = self.data.copy(deep=True)
        self.tracking_dict = {}

        # Plot (2023-10-25)
        # self.plot_names = plot_names
        # self.plot_ts = plot_ts
        # self.plot_sps = plot_sps
        # cols = 4
        # rows = math.ceil(self.duration / cols)
        # figs = []
        # fig_axes = []
        # for _ in range(len(self.plot_names)):
        #     fig, axes = plt.subplots(rows, cols, figsize=(9, 7), squeeze=False)
        #     fig.subplots_adjust(top=0.8)
        #     figs.append(fig)
        #     fig_axes.append(axes)

        # self.figs = figs
        # self.fig_axes = fig_axes

    # def reset(self):
    #     """
    #     The `reset` function reinitializes an object by copying the original data, resetting the current
    #     time to 0, and updating the state based on the process model.
    #     """

    #     self.data = self.original_data.copy(deep=True)
    #     self.curr_time = 0
    #     self.state = self.data.filter(items=self.process_model.states).values

    def show_data(self):
        """
        The function `show_data` prints the dataset for a bioreactor, with accurate column names.
        """

        if self.has_cumulative_feed:
            data = self.data.copy(deep=True)
            data = data.rename(
                columns={"Cumulative_Normalized_Feed": "Daily_Normalized_Feed"}
            )
        else:
            data = self.data

        print(f"Dataset for Bioreactor: {self.vessel}")
        print(data)
        print("")

    def return_data(self):
        """
        The function `return_data` returns the dataset for a bioreactor, with accurate column names.
        """

        if self.has_cumulative_feed:
            data = self.data.copy(deep=True)
            data = data.rename(
                columns={"Cumulative_Normalized_Feed": "Daily_Normalized_Feed"}
            )
        else:
            data = self.data

        return data

    def log_sample(
        self, sample_day: int, sample_var_names: list[str], sample_var_vals: list[float]
    ):
        """
        The function `log_sample` logs a sample by recording the sample day, variable names, and
        variable values.

        Args:
          sample_day (int): The sample_day parameter is an integer that represents the day of the
        sample.
          sample_var_names (list[str]): A list of variable names for the sample.
          sample_var_vals (list[float]): The `sample_var_vals` parameter is a list of float values
        representing the values of the variables specified in the `sample_var_names` parameter.
        """

        self.data.loc[
            self.data["Day"] == sample_day, sample_var_names
        ] = sample_var_vals

    def update_input(self, input_days, input_var_names, input_var_vals):
        """
        The function `update_input` updates specific columns of a DataFrame based on the provided input
        days, variable names, and variable values.

        Args:
          input_days: A list of days for which the input needs to be updated.
          input_var_names: The parameter `input_var_names` is a list of column names in the `self.data`
        DataFrame that you want to update.
          input_var_vals: The parameter `input_var_vals` is a 2-dimensional array or matrix that
        contains the values to be updated in the specified columns of the input data. Each row of
        `input_var_vals` corresponds to a specific day, and each column corresponds to a specific
        variable name.
        """

        for _, i in enumerate(input_days):
            self.data.loc[
                self.data["Day"] == input_days[i], input_var_names
            ] = input_var_vals[i, :]

    def state(self):
        return self.data.loc[
            self.data["Day"] == self.curr_time, self.process_model.states
        ].values[0]
    
    def sim_from_curr_day(self):
        
        # Get all inputs with daily feed
        u_matrix_daily = self.data.loc[:, self.process_model.inputs].values

        # Convert daily feed to cumulative feed
        u_matrix_cumulative = daily_to_cumulative_feed(
            self.process_model, u_matrix_daily
        )

        # Filter future inputs
        u_matrix_cumulative = u_matrix_cumulative[self.data["Day"] >= self.curr_time, :]

        # Get time array
        ts = np.arange(u_matrix_cumulative.shape[0])

        # Solve
        x_out = self.process_model.ssm_lsim(
            initial_state=self.state(), input_matrix=u_matrix_cumulative, time=ts
        )

        # Create a DF
        x_out_df = pd.DataFrame(x_out, columns=self.process_model.states)
        x_out_df.insert(0, "Day", ts + self.curr_time)

        # Check if the simulation starts from the current state
        if max(abs(self.state() - x_out[0])) > 1e-10:  # ~np.all(self.state == x_out[0]):
            raise ValueError("Simulation did not start from the current state!")
        
        return x_out, x_out_df
        
        
    def next_day(self):
        """
        The `next_day` function advances the simulation by 24 hours, updates the state and current time,
        and returns the simulation results as a DataFrame.

        Args:
          plot: The `plot` parameter is a boolean flag that determines whether or not to generate plots
        after simulation. If `plot` is set to `True`, the code will generate plots comparing simulated PV and setpoint. Defaults to False

        Returns:
          a DataFrame object named `x_out_df`.
        """

        # Simulate from the current date
        x_out,x_out_df = self.sim_from_curr_day()

        # Update state and time
        self.curr_time = self.curr_time + 1
        self.data.loc[
            self.data["Day"] == self.curr_time, self.process_model.states
        ] = x_out[1]
        self.tracking_dict[self.curr_time] = self.data.copy()
        return x_out_df


    # def next_day(self):
    #     """
    #     The `next_day` function advances the simulation by 24 hours, updates the state and current time,
    #     and returns the simulation results as a DataFrame.

    #     Args:
    #       plot: The `plot` parameter is a boolean flag that determines whether or not to generate plots
    #     after simulation. If `plot` is set to `True`, the code will generate plots comparing simulated PV and setpoint. Defaults to False

    #     Returns:
    #       a DataFrame object named `x_out_df`.
    #     """

    #     # Get all inputs with daily feed
    #     u_matrix_daily = self.data.loc[:, self.process_model.inputs].values

    #     # Convert daily feed to cumulative feed
    #     u_matrix_cumulative = daily_to_cumulative_feed(
    #         self.process_model, u_matrix_daily
    #     )

    #     # Filter future inputs
    #     u_matrix_cumulative = u_matrix_cumulative[self.data["Day"] >= self.curr_time, :]

    #     # Get time array
    #     ts = np.arange(u_matrix_cumulative.shape[0])

    #     # Solve
    #     x_out = self.process_model.ssm_lsim(
    #         initial_state=self.state(), input_matrix=u_matrix_cumulative, time=ts
    #     )

    #     # Create a DF
    #     x_out_df = pd.DataFrame(x_out, columns=self.process_model.states)
    #     x_out_df.insert(0, "Day", ts + self.curr_time)

    #     # Check if the simulation starts from the current state
    #     if max(abs(self.state() - x_out[0])) > 1e-10:  # ~np.all(self.state == x_out[0]):
    #         raise ValueError("Simulation did not start from the current state!")

    #     # Update state and time
    #     self.curr_time = self.curr_time + 1
    #     self.data.loc[
    #         self.data["Day"] == self.curr_time, self.process_model.states
    #     ] = x_out[1]
    #     self.tracking_dict[self.curr_time] = self.data.copy()
    #     return x_out_df


class Controller:
    """
    The `Controller` class represents a controller object that is used to control a bioreactor system by
    optimizing the manipulated variables based on setpoints and weights.
    """

    def __init__(
        self,
        controller_model,
        bioreactor: Bioreactor,
        ts: np.ndarray,  # A 1D, length-T array of time
        pv_sps: np.ndarray,  # A T by P array (P process variables)
        pv_names: list[str],  # Controlled process variable names
        pv_wts: np.ndarray,  # SP tracking weights
        mv_names: list[str],  # Manipulated variables
        mv_wts: np.ndarray,  # MV cost weights
        pred_horizon: int,
        ctrl_horizon: int,
        constr: np.ndarray,  # A 2 by U array (lower and upper limits only)
    ):
        """
        The function is the initialization method for a controller object, taking in various parameters
        to set up the controller.

        Args:
          controller_model: The controller model is the model used for control, such as a PID controller
        or a model predictive controller (MPC). It defines the control algorithm and strategy used to
        manipulate the process variables.
          bioreactor (Bioreactor): The `bioreactor` parameter is an instance of the `Bioreactor` class.
        It represents the bioreactor system that the controller will be controlling.
          ts (np.ndarray): A 1D array of length T representing the time points at which the process
        variables are measured or controlled.
          pv_sps (np.ndarray): `pv_sps` is a 2D array of shape (T, P), where T is the number of time
        steps and P is the number of process variables. Each row represents the setpoint values for the
        process variables at a specific time step.
          pv_names (list[str]): The `pv_names` parameter is a list of strings that represents the names
        of the controlled process variables. These are the variables that the controller will try to
        regulate and maintain at setpoints.
          pv_wts (np.ndarray): The `pv_wts` parameter is a numpy array that represents the SP (setpoint)
        tracking weights for the controlled process variables. It is a 1D array of length P, where P is
        the number of process variables. Each element of the array represents the weight for a specific
        process variable
          mv_names (list[str]): The `mv_names` parameter is a list of strings that represents the names
        of the manipulated variables (MV) in the system. These are the variables that the controller can
        adjust to control the process variables (PV) and achieve the desired setpoints.
          mv_wts (np.ndarray): The `mv_wts` parameter is an array that represents the cost weights for
        the manipulated variables (MVs). It is a 1D array of length U, where U is the number of
        manipulated variables. Each element of the array represents the cost weight for a specific
        manipulated variable. These cost
          pred_horizon (int): The prediction horizon is the number of time steps into the future for
        which the controller will generate predictions. It determines how far ahead the controller will
        look when making decisions about the manipulated variables.
          ctrl_horizon (int): The control horizon is the number of time steps into the future that the
        controller plans for. It determines how far ahead the controller looks when making control
        decisions.
          constr (np.ndarray): The `constr` parameter is a 2 by U array that represents the lower and
        upper limits for the manipulated variables (MVs). U is the number of manipulated variables. The
        first row of the `constr` array represents the lower limits for each MV, and the second row
        represents the
        """

        # The basics
        self.controller_model = controller_model
        self.bioreactor = bioreactor
        self.curr_time = bioreactor.curr_time
        self.ts = ts
        self.pv_sps = pv_sps
        self.pv_names = pv_names
        self.pv_wts = pv_wts
        self.mv_names = mv_names
        self.mv_wts = mv_wts
        self.pred_horizon = pred_horizon
        self.ctrl_horizon = ctrl_horizon
        self.constr = constr

        # Data snapshots (2023-10-22)
        self.data_before_optim = pd.DataFrame.copy(bioreactor.data)
        self.data_after_optim = pd.DataFrame.copy(bioreactor.data)

        self.data_before_optim_dict = {}
        self.data_after_optim_dict = {}

    def optimize(self,open_loop = False):
        """
        The `optimize` function optimizes future inputs for a bioreactor system and updates the dataset
        with the optimized inputs.

        Args:
          plot: The `plot` parameter is a boolean flag that determines whether or not to generate plots
        after optimization. If `plot` is set to `True`, the code will generate plots comparing the
        un-optimized and optimized data for each process variable (PV) and manipulated variable (MV). If
        `plot`. Defaults to False
        """

        # Retrieve MVs from curr_time to EoR
        data = self.bioreactor.data
        self.curr_time = self.bioreactor.curr_time
        is_in_ctrl_horizon = np.logical_and(
            data["Day"] >= self.curr_time,
            data["Day"] < (self.curr_time + self.ctrl_horizon),
        )
        mv_matrix = data.loc[is_in_ctrl_horizon, self.mv_names].values

        # Flatten initial mv
        mv_array = mv_matrix.flatten()

        # Create constraint matrix
        constr_low_matrix = np.tile(self.constr[:, 0], (mv_matrix.shape[0], 1))
        constr_low_array = constr_low_matrix.flatten()
        constr_high_matrix = np.tile(self.constr[:, 1], (mv_matrix.shape[0], 1))
        constr_high_array = constr_high_matrix.flatten()
        bounds = np.vstack((constr_low_array, constr_high_array)).transpose()

        # Simulate before optimization
        _, x_out_before_optim = self.obj_func_wrapper(mv_array)
        data_before_optim = self.data_before_optim
        data_before_optim.loc[
            data_before_optim["Day"] >= self.curr_time, self.controller_model.states
        ] = x_out_before_optim
        data_before_optim.loc[is_in_ctrl_horizon, self.mv_names] = mv_matrix
        self.data_before_optim_dict[self.curr_time] = data_before_optim.copy()



        # Simulate after optimization
        if open_loop:
            
            # No change of inputs in open loop
            x_out_after_optim = x_out_before_optim
            mv_matrix_star = mv_array.reshape([-1, len(self.mv_names)])

        else:
          
          # Solve the optimization problem
            mv_array_star = optimize.minimize(
                fun=lambda x: self.obj_func_wrapper(x)[0],
                x0=mv_array,
                bounds=bounds,
                method="SLSQP",
                options={"disp": False, "maxiter": 100},
            )

          # Fold mv to 2D
            mv_matrix_star = mv_array_star.x.reshape([-1, len(self.mv_names)])
            _, x_out_after_optim = self.obj_func_wrapper(mv_array_star.x)
        
        # Update post-optimization (or open loop) data record
        data_after_optim = self.data_after_optim
        data_after_optim.loc[
            data_after_optim["Day"] >= self.curr_time, self.controller_model.states
        ] = x_out_after_optim
        data_after_optim.loc[is_in_ctrl_horizon, self.mv_names] = mv_matrix_star
        self.data_after_optim_dict[self.curr_time] = data_after_optim.copy()

        # Update the dataset
        data.loc[is_in_ctrl_horizon, self.mv_names] = mv_matrix_star

    def obj_func_wrapper(self, mv_array):
        """
        The `obj_func_wrapper` function is a wrapper for an objective function that takes in an array of
        manipulated variables and calculates the objective function value based on the current state of
        a bioreactor system.

        Args:
          mv_array: The `mv_array` parameter is an array that contains the manipulated variable (MV)
        values. It is used to replace the MVs in the input matrix `u_matrix_daily` with the values from
        `mv_array`. The MVs are identified by their names, which are stored in the `mv

        Returns:
          The function `obj_func_wrapper` returns a tuple containing two elements. The first element is
        the result of calling the `obj_func` function with the arguments `ts + self.curr_time`,
        `x_out[:, pv_loc]`, and `u_matrix_daily[self.bioreactor.data["Day"] >= self.curr_time, :][:,
        mv_loc]`. The second element is the `x_out`
        """

        # Rows within the control horizon (2023-10-21)
        ctrl_horizon_where = np.where(
            np.logical_and(
                self.bioreactor.data["Day"] >= self.curr_time,
                self.bioreactor.data["Day"] < (self.curr_time + self.ctrl_horizon),
            )
        )[0]

        # Fold mv_array to a 2D array
        mv_matrix = mv_array.reshape([-1, len(self.mv_names)])

        # Retrieve input from day 0 to EoR
        u_matrix_daily = self.bioreactor.data.loc[
            :, self.controller_model.inputs
        ].values

        # Replace MVs with mv_matrix
        loc_mv_in_inputs = np.where(
            np.isin(self.controller_model.inputs, self.mv_names)
        )[0]
        u_matrix_daily_ctrl_horizon = u_matrix_daily[ctrl_horizon_where, :]
        u_matrix_daily_ctrl_horizon[:, loc_mv_in_inputs] = mv_matrix
        u_matrix_daily[ctrl_horizon_where, :] = u_matrix_daily_ctrl_horizon
        u_matrix_cumulative = u_matrix_daily

        # Convert daily feed to cumulative feed
        if self.bioreactor.has_cumulative_feed:
            u_matrix_cumulative = daily_to_cumulative_feed(
                self.controller_model, u_matrix_daily
            )
            # cumulative_feed_loc = np.where(np.isin(self.controller_model.inputs,'Cumulative_Normalized_Feed'))[0]
            # u_matrix_cumulative[:,cumulative_feed_loc] = np.cumsum(u_matrix_daily[:,cumulative_feed_loc]).reshape([-1,1])

        # Time array
        ts = np.arange(
            u_matrix_daily[self.bioreactor.data["Day"] >= self.curr_time, :].shape[0]
        )

        # Sim
        x_out = self.controller_model.ssm_lsim(
            initial_state=self.bioreactor.state(),
            input_matrix=u_matrix_cumulative[
                self.bioreactor.data["Day"] >= self.curr_time, :
            ],
            time=ts,
        )

        # Obj
        pv_loc = np.where(np.isin(self.controller_model.states, self.pv_names))[0]
        mv_loc = np.where(np.isin(self.controller_model.inputs, self.mv_names))[0]
        return (
            self.obj_func(
                ts + self.curr_time,
                x_out[:, pv_loc],
                u_matrix_daily[self.bioreactor.data["Day"] >= self.curr_time, :][
                    :, mv_loc
                ],
            ),
            x_out,
        )

    def obj_func(self, ts: np.ndarray, x: np.ndarray, u: np.ndarray):
        """
        The function calculates the cost value based on the given inputs.

        Args:
          ts (np.ndarray): `ts` is a numpy array representing the time steps. It is used to index and
        trim the arrays `x`, `u`, and `pv_sps` to keep only the future entries.
          x (np.ndarray): The parameter `x` is a numpy array representing the current state of the
        system. It contains the values of the system variables at each time step.
          u (np.ndarray): The parameter `u` is a numpy array representing the control inputs. It is a
        2-dimensional array with shape `(num_samples, num_inputs)`, where `num_samples` is the number of
        samples and `num_inputs` is the number of control inputs. Each row of the array represents a

        Returns:
          the sum of the cost values for the control inputs (u3_cost) and the process variables
        (x3_cost).
        """

        # Trim to keep only future entries
        x2 = x[ts > self.curr_time, :]
        u2 = u[ts >= self.curr_time, :]
        pv_sps2 = self.pv_sps[self.ts > self.curr_time, :]

        # Trim to keep the prediction and control horizons
        x3 = x2[0 : self.pred_horizon, :]
        pv_sps3 = pv_sps2[0 : self.pred_horizon, :]
        u3 = u2[0 : self.ctrl_horizon, :]

        # Calculate the cost
        u3_diff = np.diff(u3, axis=0)
        u3_cost = np.sum(np.multiply(np.sum(np.square(u3_diff), axis=0), self.mv_wts))
        x3_diff = x3 - pv_sps3
        x3_cost = np.sum(np.multiply(np.sum(np.square(x3_diff), axis=0), self.pv_wts))
        return u3_cost + x3_cost

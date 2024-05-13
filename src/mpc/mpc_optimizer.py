"""MPC-related class definitions
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Standard Library Imports
import warnings
from typing import Optional, Union
from datetime import datetime

# 3rd Party Library Imports
import numpy as np
import pandas as pd
from scipy import optimize

# State-Space-Model Package Imports
from models.ssm import StateSpaceModel
from data.functions import daily_to_cumulative_feed


class Bioreactor:
    """
    The Bioreactor class represents a bioreactor object used for simulation and processing of real-time
    data, with methods for initializing, resetting, logging samples, updating inputs, and advancing the
    simulation.
    """

    def __init__(
        self,
        vessel: Union[str, int],
        process_model: StateSpaceModel,
        data: Optional[pd.DataFrame] = None,
        experiment_config: Optional[dict] = None,
        controller_config: Optional[dict] = None,
    ):
        """
        This Python function initializes an object with specified attributes and data, performing
        various checks and conversions on the input data.
        
        Args:
          vessel (Union[str, int]): The `vessel` parameter in the `__init__` method is used to specify
        the name or identifier of the vessel or bioreactor for processing multiple bioreactors. It is a
        required parameter and can be either a string or an integer.
          process_model (StateSpaceModel): The `process_model` parameter in the `__init__` method is
        expected to be an instance of the `StateSpaceModel` class. This parameter is used for simulating
        a process within the class.
          data (Optional[pd.DataFrame]): The `data` parameter in the `__init__` method is used to
        provide a DataFrame containing process data. If no data is provided, a DataFrame is initialized
        based on the configuration settings. The DataFrame should include columns for various process
        variables, setpoints, and references. If cumulative feed data is
          config (Optional[dict]): The `config` parameter in the `__init__` method is a dictionary that
        should contain the following keys:
            - Batch Length: length of the batch as an int
            - Column Mapping: dictionary of columns in input topic to columns in bioreactor class df
            - Process Variable Setpoints: list of setpoints trajectory for target process variable i.e. Titer
            - Manipulated Variable Reference: list of setpoints trajectory for manipulated variable i.e. Feed
            - Process Variables: list of all process variables to control at a trajectory
            - Manipulated Variables: list of all variables to manipulate controlled trajectory
        """
        if not isinstance(experiment_config, dict):
            raise ValueError(
                "Config file must specify batch length, column mapping, PV setpoints and MV reference."
            )
        # Initialize attributes
        self.curr_time = 0
        self.start_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.duration = experiment_config["Last Day"] + 1
        # Update attributes based on user input
        self.vessel = vessel  # Vessel name for processing multiple bioreactors
        self.process_model = (
            process_model  # Model (if provided) for simulating a process
        )
        # this parameter is necessary for tech automation
        # when the data vector is passed to this class the columns must be mapped
        # correctly to the dataframe initialized at instantiation
        
        # if "Constraints" in config:
        #     self.constraints = config["Constraints"]

        if data is None:
            self.column_map = experiment_config["Column Mapping"]
            # if (
            #     controller_config["Process Variable Setpoints"] is None
            #     or controller_config["Manipulated Variable Reference"] is None
            # ):
            #     raise ValueError(
            #         "Config file must contain PV setpoints/MV reference trajectory."
            #     )
            sp_cols = [f"{x}--STATE_SP" for x in controller_config['Process Variables']]
            mv_ref_cols = [f"{x}--INPUT_REF" for x in controller_config['Manipulated Variables']]
            u_cols = [f"{x}--INPUT_DATA" for x in controller_config['Input Variables']]
            x_cols = [f"{x}--STATE_DATA" for x in controller_config['State Variables']]
            cols = (
                ["Code_Run_Date", "Bioreactor", "Day", "Date"] 
                + sp_cols
                + mv_ref_cols
                # Find way to avoid hard coding these
                + self.process_model.state_data_labels
                + self.process_model.input_data_labels
                + self.process_model.state_pred_labels
                + self.process_model.state_est_labels
                + self.process_model.state_mod_labels
            )
            if u_cols != self.process_model.input_data_labels:
                raise ValueError("Input vectors must be identical between model and controller config.")
            if x_cols != self.process_model.state_data_labels:
                raise ValueError("State vectors must be identical between model and controller config.")
            zero_arr = np.zeros((self.duration, len(cols)))
            zero_arr[:] = np.nan
            data = pd.DataFrame(data=zero_arr, columns=cols)

            # Initialize reference and nominal data
            for key in controller_config['Process Variables']:
                data[f"{key}--STATE_SP"] = np.array(controller_config['Process Variables'][key]["Data"])
            for key in controller_config["Manipulated Variables"]:
                data[f"{key}--INPUT_REF"] = np.array(controller_config['Manipulated Variables'][key]["Data"])
            for key in controller_config["Input Variables"]:
                data[f"{key}--INPUT_DATA"] = np.array(controller_config['Input Variables'][key])

            # # Initialize the dataframe with input values and setpoints
            # data[sp_cols] = np.array(controller_config["Process Variable Setpoints"])
            # data[mv_ref_cols] = np.array(controller_config["Manipulated Variable Reference"])
            # data[u_cols] = np.array(controller_config["Nominal Input Recipe"])

            data["Day"] = np.arange(0, self.duration)
            data["Bioreactor"] = str(self.vessel)
            self.data = data.copy(deep=True)
        else:
            self.data = data.copy(deep=True)

        # 24 columns defined

        # for key, value in experiment_config["Controller Dictionary"].items():
        #     if self.vessel in value:
        #         controller_key = key

        self.total_feed_name = experiment_config['Total Feed Name']
        self.daily_feed_name = experiment_config["Daily Feed Name"]

        # Data frame for open_loop simulation results
        self.open_loop_df = pd.DataFrame()

        # Check if the data set starts on Day 0
        if self.data["Day"].values[0] != 0:
            raise ValueError("Data set does not start on Day 0!")

        # # self.state = self.data.filter(items=self.process_model.states)
        # self.duration = self.data["Day"].values[-1]

        # Check if the data set ends on Day duration
        if self.data.shape[0] != self.duration:
            raise ValueError("Data set has missing or duplicate days!")

        # Check if days are consecutive (2023-10-21)
        if any(np.diff(self.data["Day"]) != 1):
            raise ValueError("Data set is not in 1-day increments!")

        # Convert cumulative feed (data) to daily feed
        if np.isin(f"{self.total_feed_name}--INPUT_DATA", self.data.columns):
            self.data[f"{self.total_feed_name}--INPUT_DATA"] = np.append(
                np.diff(self.data.loc[:, f"{self.total_feed_name}--INPUT_DATA"]), 0
            )
            self.has_cumulative_feed_data = True
            warnings.warn(
                "Cumulative feed (data) was converted to daily feed (variable name is unchanged)!"
            )
        else:
            self.has_cumulative_feed_data = False
        self.daily_feed_name_data = f"{self.daily_feed_name}--INPUT_DATA"

        # Convert cumulative feed (reference) to daily feed
        if np.isin(f"{self.total_feed_name}--INPUT_REF", self.data.columns):
            self.data[f"{self.total_feed_name}--INPUT_REF"] = np.append(
                np.diff(self.data.loc[:, f"{self.total_feed_name}--INPUT_REF"]), 0
            )
            self.has_cumulative_feed_ref = True
            warnings.warn(
                "Cumulative feed (reference) was converted to daily feed (variable name is unchanged)!"
            )
        else:
            self.has_cumulative_feed_ref = False
        self.daily_feed_name_ref = f"{self.daily_feed_name}--INPUT_REF"

        # Retain the original dataset
        self.original_data = self.data.copy(deep=True)
        self.tracking_dict = {}

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

        if self.has_cumulative_feed_data:
            data = self.data.copy(deep=True)
            data = data.rename(
                columns={
                    "CUMULATIVE_NORMALIZED_FEED--INPUT_DATA": self.daily_feed_name_data
                }
            )
        else:
            data = self.data

        print(f"Dataset for Bioreactor: {self.vessel}")
        print(data)
        print("")

    def ingest_vector(self, vector: pd.Series):
        """
        The function `ingest_vector` takes a Pandas Series vector, renames its columns based on a
        provided mapping, and inserts it into a DataFrame at a specific index based on a condition.

        Args:
          vector (pd.Series): The `ingest_vector` method takes a pandas Series object as input, which is
        represented by the parameter `vector`. This method is used to ingest the input vector into the
        class dataframe after performing some operations.
        """
        if self.column_map is None:
            raise ValueError(
                "Need to instantiate column mapping to correctly ingest input vector to dataframe."
            )
        renamed_vector = vector.rename(self.column_map)
        selected_col = (
            self.process_model.state_data_labels + self.process_model.input_data_labels
        )
        if renamed_vector is not None:
            insert_index = self.data[self.data["Day"] == renamed_vector["Day"].values[0]].index[
                0
            ]
        else:
            raise ValueError(
                "Vector does not contain data, check that vector is not None or column mapping is correct"
            )
        
        # Convert bioreactor feed data from daily to total
        if np.isin(f"{self.total_feed_name}--INPUT_DATA", self.data.columns):
            feed_daily = self.data[f"{self.total_feed_name}--INPUT_DATA"]
            feed_total = np.append(0, np.cumsum(feed_daily[0:-1]))
            feed_total[insert_index] = renamed_vector[f"{self.total_feed_name}--INPUT_DATA"].values[0]
            feed_daily = np.append(np.diff(feed_total), 0)

        # Replace current day's data
        self.data.loc[insert_index,selected_col] = renamed_vector.loc[:,selected_col].values

        # Replace daily feed
        if np.isin(f"{self.total_feed_name}--INPUT_DATA", self.data.columns):
            self.data[f"{self.total_feed_name}--INPUT_DATA"] = feed_daily

        # NEED TO THINK ABOUT percent feed

    def return_data(self, show_daily_feed: bool, exec_date: bool = False):
        """
        The function `return_data` returns the dataset for a bioreactor, with accurate column names.
        """

        # loc_mv_in_inputs = np.where(
        #     np.isin(np.array(self.controller_model.input_data_labels), self.mv_names)
        # )[0]

        data = self.data.copy(deep=True)
        if exec_date:
            data["Code_Run_Date"] = datetime.today().strftime("%Y-%m-%d")
            cols = np.array(data.columns.tolist())
            new_cols = cols[
                np.concatenate(
                    (
                        np.where(np.isin(cols, "Code_Run_Date"))[0],
                        np.where(~np.isin(cols, "Code_Run_Date"))[0],
                    )
                )
            ]
            data = data[new_cols].copy(deep=True)

        if self.has_cumulative_feed_data or self.has_cumulative_feed_ref:
            # data = self.data.copy(deep=True)
            # if exec_date:
            #     data["Code_Run_Date"] = datetime.today().strftime("%Y-%m-%d")
            #     cols = data.columns.tolist()
            #     data = data[cols[-1:] + cols[:-1]].copy(deep=True)

            if show_daily_feed:
                data = data.rename(
                    columns={
                        "CUMULATIVE_NORMALIZED_FEED--INPUT_DATA": self.daily_feed_name_data,
                        "CUMULATIVE_NORMALIZED_FEED--INPUT_REF": self.daily_feed_name_ref,
                    }
                )
            else:
                if self.has_cumulative_feed_data:
                    feed_daily = data["CUMULATIVE_NORMALIZED_FEED--INPUT_DATA"]
                    feed_total = np.append(0, np.cumsum(feed_daily[0:-1]))
                    data["CUMULATIVE_NORMALIZED_FEED--INPUT_DATA"] = feed_total

                if self.has_cumulative_feed_ref:
                    feed_daily = data["CUMULATIVE_NORMALIZED_FEED--INPUT_REF"]
                    feed_total = np.append(0, np.cumsum(feed_daily[0:-1]))
                    data["CUMULATIVE_NORMALIZED_FEED--INPUT_REF"] = feed_total
        # else:
        #     data = self.data

        return data

    # def log_sample(
    #     self, sample_day: int, sample_var_names: list[str], sample_var_vals: list[float]
    # ):
    #     """
    #     The function `log_sample` logs a sample by recording the sample day, variable names, and
    #     variable values.

    #     Args:
    #       sample_day (int): The sample_day parameter is an integer that represents the day of the
    #     sample.
    #       sample_var_names (list[str]): A list of variable names for the sample.
    #       sample_var_vals (list[float]): The `sample_var_vals` parameter is a list of float values
    #     representing the values of the variables specified in the `sample_var_names` parameter.
    #     """

    #     self.data.loc[
    #         self.data["Day"] == sample_day, sample_var_names
    #     ] = sample_var_vals

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

    def state(self, day=-1):
        if day >= 0:
            curr_time = day
        else:
            curr_time = self.curr_time

        if curr_time == 0:
            state_labels = self.process_model.state_data_labels
            self.data.loc[
                self.data["Day"] == 0, self.process_model.state_est_labels
            ] = self.data.loc[self.data["Day"] == 0, state_labels].values
            self.data.loc[
                self.data["Day"] == 0, self.process_model.state_pred_labels
            ] = self.data.loc[self.data["Day"] == 0, state_labels].values
        else:
            state_labels = self.process_model.state_est_labels

        return self.data.loc[self.data["Day"] == curr_time, state_labels].values[0]

    def sim_from_day(self, day=-1, initial_state=np.array([])):
        # If not from current day
        if day >= 0:
            curr_time = day
        else:
            curr_time = self.curr_time

        # Initial state
        if initial_state.size == 0:
            x0 = self.state(curr_time)
        else:
            x0 = initial_state

        # Get all inputs with daily feed
        u_matrix_daily = self.data.loc[:, self.process_model.input_data_labels].values

        # Convert daily feed to cumulative feed
        u_matrix_cumulative = daily_to_cumulative_feed(
            self.process_model, u_matrix_daily
        )

        # Filter future inputs
        u_matrix_cumulative = u_matrix_cumulative[self.data["Day"] >= curr_time, :]

        # Get time array
        ts = np.arange(u_matrix_cumulative.shape[0])

        # Solve
        x_out, y_out = self.process_model.ssm_lsim(
            initial_state=x0,  # self.state(curr_time),
            input_matrix=u_matrix_cumulative,
            time=ts,
            # delta_p=self.data.loc[
            #     self.data["Day"] == self.curr_time, self.process_model.state_mod_labels
            # ].values,
        )

        # Create a DF for the predicted states
        x_out_df = pd.DataFrame(x_out, columns=self.process_model.state_est_labels)
        x_out_df.insert(0, "Day", ts + curr_time)

        # Create a DF for the predicted outputs
        y_out_df = pd.DataFrame(y_out, columns=self.process_model.state_pred_labels)
        y_out_df.insert(0, "Day", ts + curr_time)

        # Check if the simulation starts from the current state
        if max(abs(x0 - x_out[0])) > 1e-10:  # ~np.all(self.state == x_out[0]):
            raise ValueError("Simulation did not start from the current state!")

        return x_out, x_out_df, y_out, y_out_df
    
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
        x_out, _, _, _ = self.sim_from_day()

        # Update estimated state for next day (YL@2024-01-18)
        self.curr_time = self.curr_time + 1
        self.data.loc[
            self.data["Day"] == self.curr_time, self.process_model.state_est_labels
        ] = x_out[1]

        # # Update predicted output for the rest of the batch (YL@2024-01-18)
        # self.data.loc[
        #     self.data["Day"] >= self.curr_time, self.process_model.state_pred_labels
        # ] = y_out[1:,]

        # # Update state and time
        # self.curr_time = self.curr_time + 1
        # self.data.loc[
        #     self.data["Day"] == self.curr_time, self.process_model.state_data_labels
        # ] = x_out[1]
        # self.tracking_dict[self.curr_time] = self.data.copy()
        # return x_out_df


class Controller:
    """
    The `Controller` class represents a controller object that is used to control a
    bioreactor system by optimizing the manipulated variables based on setpoints and weights.
    """

    def __init__(
        self,
        controller_model: StateSpaceModel,
        bioreactor: Bioreactor,
        controller_config: Optional[dict] = None,
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

        # Use config file to construct other tuning parameters
        ts=np.array(controller_config["Time"])
        pv_sps = np.transpose(np.array([controller_config["Process Variables"][key]["Data"] for key in controller_config["Process Variables"]]))
        pv_wts = np.array([controller_config["Process Variables"][key]["Weight"] for key in controller_config["Process Variables"]])
        pv_names = list(controller_config["Process Variables"].keys())
        mv_wts = np.array([controller_config["Manipulated Variables"][key]["Weight"] for key in controller_config["Manipulated Variables"]])
        mv_names = list(controller_config["Manipulated Variables"].keys())
        mv_constr = np.transpose(np.array([controller_config["Manipulated Variables"][key]["Constraint"] for key in controller_config["Manipulated Variables"]]))
        est_wts = np.array([controller_config["State Variables"][key]["Weight"] for key in controller_config["State Variables"]])
        output_mods_user = np.array([controller_config["State Variables"][key]["Modifier"] for key in controller_config["State Variables"]])
        eor_names = list(controller_config["End of Run Variables"].keys())
        eor_constr = np.transpose(np.array([controller_config["End of Run Variables"][key]["Constraint"] for key in controller_config["End of Run Variables"]]))
        eor_wts = np.array([controller_config["End of Run Variables"][key]["Weight"] for key in controller_config["End of Run Variables"]])

        pred_horizon=controller_config["Prediction Horizon"]
        ctrl_horizon=controller_config["Control Horizon"]
        est_horizon=controller_config["Estimation Horizon"]
        filter_wt_on_data=controller_config["Estimation Filter Weight on Data"]

        # The basics
        self.controller_model = controller_model
        self.bioreactor = bioreactor
        self.curr_time = bioreactor.curr_time
        self.ts = ts
        self.pv_sps = pv_sps

        self.pv_wts = pv_wts
        self.est_wts = est_wts
        self.filter_wt_on_data = filter_wt_on_data

        self.mv_wts = mv_wts
        self.pred_horizon = pred_horizon
        self.ctrl_horizon = ctrl_horizon
        self.mv_constr = mv_constr

        self.output_mods_est = np.array([])  # p estimated from data
        self.output_mods_user = output_mods_user  # p specified by user
        # self.delta_p_a = []
        # self.delta_p_b = []
        self.est_horizon = est_horizon
        self.eor_names = eor_names
        self.eor_const = eor_constr
        self.eor_wts = eor_wts

        # Data snapshots (2023-10-22)
        self.data_before_optim = pd.DataFrame.copy(bioreactor.data)
        self.data_after_optim = pd.DataFrame.copy(bioreactor.data)

        PV_SUFFIX = "--STATE_DATA"
        self.pv_names = [x + PV_SUFFIX for x in pv_names]

        MV_SUFFIX = "--INPUT_DATA"
        self.mv_names = [x + MV_SUFFIX for x in mv_names]

        # self.bioreactor.data[
        #     "IGG--STATE_SP"
        # ] = self.pv_sps  # find a way to avoid hard coding target setpoint name

        self.data_before_optim_dict = {}
        self.data_after_optim_dict = {}

    def optimize(self, open_loop=False, print_pred=False):
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
            np.logical_and(
                data["Day"] >= self.curr_time,
                data["Day"] < (self.curr_time + self.ctrl_horizon),
            ),
            data["Day"] < max(data["Day"]),
        )
        is_in_pred_horizon = np.logical_and(
            data["Day"] >= self.curr_time,
            data["Day"] < (self.curr_time + self.pred_horizon),
        )

        # Do not run MPC on EoR
        pred_names = [x.replace("--STATE_DATA", "--STATE_PRED") for x in self.pv_names]
        sp_names = [x.replace("--STATE_DATA", "--STATE_SP") for x in self.pv_names]
        if not (np.any(is_in_ctrl_horizon)):
            # Update pred horizon
            data.loc[
                is_in_pred_horizon, self.controller_model.state_pred_labels
            ] = np.multiply(
                data.loc[
                    is_in_pred_horizon, self.controller_model.state_est_labels
                ].values,
                self.output_mods_est,
            )

            # Update command line output
            print(
                data.loc[
                    data["Day"] == max(data["Day"]),
                    ["Day", "Bioreactor"] + sp_names + pred_names + [str.upper(x) + "--STATE_PRED" for x in self.eor_names],
                ]
            )
            return

        control_matrix = data.loc[is_in_ctrl_horizon, self.mv_names].values

        # Flatten initial mv
        mv_array = control_matrix.flatten()

        # Create constraint matrix
        constr_low_matrix = np.tile(self.mv_constr[0, :], (control_matrix.shape[0], 1))
        constr_low_array = constr_low_matrix.flatten()
        constr_high_matrix = np.tile(self.mv_constr[1, :], (control_matrix.shape[0], 1))
        constr_high_array = constr_high_matrix.flatten()
        bounds = np.vstack((constr_low_array, constr_high_array)).transpose()

        # Simulate before optimization
        _, y_out_before_optim = self.obj_func_wrapper(mv_array)
        data_before_optim = self.data_before_optim
        data_before_optim.loc[
            data_before_optim["Day"] >= self.curr_time,
            self.controller_model.state_pred_labels,
        ] = y_out_before_optim
        data_before_optim.loc[is_in_ctrl_horizon, self.mv_names] = control_matrix
        self.data_before_optim_dict[self.curr_time] = data_before_optim.copy()

        # Simulate after optimization
        if open_loop:
            # No change of inputs in open loop
            y_out_after_optim = y_out_before_optim
            control_matrix_star = mv_array.reshape([-1, len(self.mv_names)])
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
            control_matrix_star = mv_array_star.x.reshape([-1, len(self.mv_names)])
            _, y_out_after_optim = self.obj_func_wrapper(mv_array_star.x)

        # Update post-optimization (or open loop) data record
        data_after_optim = self.data_after_optim
        data_after_optim.loc[
            data_after_optim["Day"] >= self.curr_time,
            self.controller_model.state_pred_labels,
        ] = y_out_after_optim
        data_after_optim.loc[is_in_ctrl_horizon, self.mv_names] = control_matrix_star
        self.data_after_optim_dict[self.curr_time] = data_after_optim.copy()

        # Update the dataset with new inputs
        data.loc[is_in_ctrl_horizon, self.mv_names] = control_matrix_star

        # Update the dataset with new predictions
        data.loc[
            is_in_pred_horizon, self.controller_model.state_pred_labels
        ] = data_after_optim.loc[
            is_in_pred_horizon, self.controller_model.state_pred_labels
        ]

        # Print final PV (2024-03-01)
        if print_pred:
            print(
                data.loc[
                    data["Day"] == max(data["Day"]),
                    ["Day", "Bioreactor"] + sp_names + pred_names + [str.upper(x) + "--STATE_PRED" for x in self.eor_names],
                ]
            )

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

        # Rows within the control horizon (2023-10-21/updated 2024-05-06: added the < max day condition)
        ctrl_horizon_where = np.where(
            np.logical_and(
            np.logical_and(
                self.bioreactor.data["Day"] >= self.curr_time,
                self.bioreactor.data["Day"] < (self.curr_time + self.ctrl_horizon),
            ),self.bioreactor.data["Day"] < max(self.bioreactor.data["Day"]))
        )[0]

        # Fold mv_array to a 2D array
        control_matrix = mv_array.reshape([-1, len(self.mv_names)])

        # Retrieve input from day 0 to EoR
        u_matrix_daily = self.bioreactor.data.loc[
            :, self.controller_model.input_data_labels
        ].values

        # Replace MVs with control_matrix
        loc_mv_in_inputs = np.where(
            np.isin(np.array(self.controller_model.input_data_labels), self.mv_names)
        )[0]
        u_matrix_daily_ctrl_horizon = u_matrix_daily[ctrl_horizon_where, :]
        u_matrix_daily_ctrl_horizon[:, loc_mv_in_inputs] = control_matrix
        u_matrix_daily[ctrl_horizon_where, :] = u_matrix_daily_ctrl_horizon
        u_matrix_cumulative = u_matrix_daily

        # Convert daily feed to cumulative feed
        if self.bioreactor.has_cumulative_feed_data:
            u_matrix_cumulative = daily_to_cumulative_feed(
                self.controller_model, u_matrix_daily
            )

        # Time array
        ts = np.arange(
            u_matrix_daily[self.bioreactor.data["Day"] >= self.curr_time, :].shape[0]
        )

        # Simulate the modified output
        _, y_out = self.controller_model.ssm_lsim(
            initial_state=self.bioreactor.state(),
            input_matrix=u_matrix_cumulative[
                self.bioreactor.data["Day"] >= self.curr_time, :
            ],
            time=ts,
            output_mods=self.output_mods_est,
        )

        # self.bioreactor.data.loc[self.bioreactor.data["Day"] == self.curr_time, self.bioreactor.process_model.state_mod_labels].values

        # Retrieve end of run predictions
        # below_deadband_cost = 0
        # above_deadband_cost = 0
        e = np.array([])
        if self.eor_names is not None:
            for state_count,state in enumerate(self.eor_names):
                constraint = self.eor_const[:,state_count]
                eor_wt = self.eor_wts[state_count]
                for idx, string in enumerate(self.controller_model.state_pred_labels):
                    if str.upper(state) in string:
                        if constraint[0] > y_out[-1,idx]:
                            e = np.append(e,(constraint[0] - y_out[-1,idx]) * eor_wt)
                        if constraint[1] < y_out[-1,idx]:
                            e = np.append(e,(y_out[-1,idx] - constraint[0]) * eor_wt)

        # Obj
        pv_loc = np.where(
            np.isin(np.array(self.controller_model.state_data_labels), self.pv_names)
        )[0]
        mv_loc = np.where(
            np.isin(np.array(self.controller_model.input_data_labels), self.mv_names)
        )[0]
        return (
            self.ctrl_obj_func(
                ts + self.curr_time,
                y_out[:, pv_loc],
                u_matrix_daily[self.bioreactor.data["Day"] >= self.curr_time, :][
                    :, mv_loc
                ],
                e,
            ),
            y_out,
        )

    def ctrl_obj_func(self, ts: np.ndarray, y: np.ndarray, u: np.ndarray, e: np.ndarray):
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
        y2 = y[ts > self.curr_time, :]
        u2 = u[ts >= self.curr_time, :]
        pv_sps2 = self.pv_sps[self.ts > self.curr_time, :]

        # Trim to keep the prediction and control horizons
        y3 = y2[0 : self.pred_horizon, :]
        pv_sps3 = pv_sps2[0 : self.pred_horizon, :]
        u3 = u2[0 : self.ctrl_horizon, :]

        # Calculate the cost
        u3_diff = np.diff(u3, axis=0)
        u3_cost = np.sum(np.multiply(np.sum(np.square(u3_diff), axis=0), self.mv_wts))
        y3_diff = y3 - pv_sps3
        y3_cost = np.sum(np.multiply(np.sum(np.square(y3_diff), axis=0), self.pv_wts))
        return u3_cost + y3_cost + np.sum(e**2)

    def estimate(self):
        """_summary_

        Args:
            N (int): _description_

        Returns:
            _type_: _description_
        """
        # get the last few state measurements based on the estimator horizon
        # measurements that will be input into the excel sheet. Not sure if this is correct based on
        # how the measurements will be input into the sheet.

        N = self.est_horizon

        # Point to the bioreactor data
        data = self.bioreactor.data

        # ensure curr_time is set to the time of the bioreactor
        self.curr_time = self.bioreactor.curr_time
        # print(
        #     f"{self.bioreactor.vessel}: Estimating Day {self.curr_time}'s output modifiers ..."
        # )

        # select days in the MHE horizon
        is_in_est_horizon = np.logical_and(
            data["Day"] <= self.curr_time,
            data["Day"] > (self.curr_time - N),
        )

        def est_x_obj_func(x):
            # Retrieve measurements
            measurements = data.loc[
                is_in_est_horizon, self.bioreactor.process_model.state_data_labels
            ].values

            previous_estimates = data.loc[
                is_in_est_horizon, self.bioreactor.process_model.state_est_labels
            ].values

            # Simulate
            x_out, _, _, _ = self.bioreactor.sim_from_day(day=t0, initial_state=x)

            # Estimates
            new_estimates = x_out[0 : sum(is_in_est_horizon),]

            cost = np.nansum(
                np.nanmean(
                    np.multiply(np.square(new_estimates - measurements), self.est_wts),
                    axis=0,
                )
            ) * self.filter_wt_on_data + np.nansum(
                np.nanmean(
                    np.multiply(
                        np.square(new_estimates - previous_estimates), self.est_wts
                    ),
                    axis=0,
                )
            ) * (1 - self.filter_wt_on_data)

            return cost

        # Re-estimate the initial state of the horizon
        t0 = data.loc[is_in_est_horizon, "Day"].values[0]
        x0 = self.bioreactor.state(t0)
        res1 = optimize.minimize(
            fun=est_x_obj_func,
            x0=x0,
            method="SLSQP",
        )
        x_star = res1.x
        x_out, _, _, _ = self.bioreactor.sim_from_day(day=t0, initial_state=x_star)
        new_estimates = x_out[0 : sum(is_in_est_horizon),]
        data.loc[
            is_in_est_horizon, self.bioreactor.process_model.state_est_labels
        ] = new_estimates

        # Use the previous output_mods if available
        if self.output_mods_user.size == 0:
            if self.curr_time == 0:
                self.output_mods_est = np.ones(
                    (1, len(self.controller_model.state_mod_labels))
                )
            else:
                self.output_mods_est = self.bioreactor.data.loc[
                    self.bioreactor.data["Day"] == self.curr_time - 1,
                    self.controller_model.state_mod_labels,
                ].values
        else:
            self.output_mods_est = self.output_mods_user

        # def est_mod_obj_func(p_array):
        #     # Apply the output modifiers to the stored states
        #     predictions = np.multiply(
        #         data.loc[
        #             is_in_est_horizon, self.bioreactor.process_model.state_est_labels
        #         ].values,
        #         p_array,
        #     )

        #     # Retrieve measurements
        #     measurements = data.loc[
        #         is_in_est_horizon, self.bioreactor.process_model.state_data_labels
        #     ].values

        #     cost = (
        #         np.nansum(
        #             np.nanmean(
        #                 np.multiply(
        #                     np.square(predictions - measurements), self.est_wts
        #                 ),
        #                 axis=0,
        #             )
        #         )
        #         + np.nansum(np.square(p_array - 1))
        #         + np.nansum(np.square(p_array - self.output_mods_est))
        #     )

        #     return cost

        # # flatten the optimization parameters
        # p_array0 = self.output_mods_est.flatten()

        # res2 = optimize.minimize(
        #     fun=est_mod_obj_func,
        #     x0=p_array0,
        #     method="SLSQP",
        # )

        # # Unravel final delta_p matrix back to correct shape
        # self.output_mods_est = res2.x  # [: len(self.output_mods.flatten())].reshape(
        # # self.output_mods.shape[0], self.output_mods.shape[1]
        # # )

        # Linear reg with no intercept to get mods (YL: 2024-05-09)
        p_array_star = self.output_mods_est.flatten()
        predictions = data.loc[
            is_in_est_horizon, self.bioreactor.process_model.state_est_labels
            ].values
        measurements = data.loc[
            is_in_est_horizon, self.bioreactor.process_model.state_data_labels
            ].values
        for i in range(len(self.bioreactor.process_model.state_est_labels)):
            predictions_i = predictions[:,i]
            measurements_i = measurements[:,i]
            p_has_data = np.logical_and(~np.isnan(predictions_i),~np.isnan(measurements_i))
            if any(p_has_data):
                p_star_i = np.sum(np.multiply(predictions_i[p_has_data],measurements_i[p_has_data]))/np.sum(predictions_i**2)
                p_array_star[i] = np.max((np.min((p_star_i,1.05)),0.95))
            elif np.isnan(p_array_star[i]):
                p_array_star[i] = 1
        self.output_mods_est = p_array_star


        # Store the new delta_p matrix into the dataframe
        data.loc[
            data["Day"] == self.curr_time,
            self.bioreactor.process_model.state_mod_labels,
        ] = self.output_mods_est

        # # create a matrix with values in control horizon
        # control_matrix = data.loc[is_in_est_horizon, self.mv_names].values

        # # find the values within the MHE horizon
        # ctrl_horizon_where = np.where(
        #     np.logical_and(
        #         self.bioreactor.data["Day"] <= self.curr_time,
        #         self.bioreactor.data["Day"] > (self.curr_time - N),
        #     )
        # )[0]

        # # Retrieve input from day 0 to EoR
        # u_matrix_daily = self.bioreactor.data.loc[
        #     :, self.controller_model.input_data_labels
        # ].values

        # loc_mv_in_inputs = np.where(
        #     np.isin(
        #         np.array(self.controller_model.input_data_labels), self.mv_names
        #     )
        # )[0]

        # u_matrix_daily_ctrl_horizon = u_matrix_daily[ctrl_horizon_where, :]
        # u_matrix_daily_ctrl_horizon[:, loc_mv_in_inputs] = control_matrix
        # # u_matrix_daily[ctrl_horizon_where, :] = u_matrix_daily_ctrl_horizon
        # # u_matrix_cumulative = u_matrix_daily

        # ts = np.arange(
        #     u_matrix_daily[
        #         self.bioreactor.data["Day"] >= self.curr_time - N, :
        #     ].shape[0]
        # )

        # if self.curr_time - N < 0:
        #     start_of_horizon = 0
        # else:
        #     start_of_horizon = self.curr_time - N

        # x_out, y_out = self.controller_model.ssm_lsim(
        #     initial_state=data.loc[
        #         data["Day"] == start_of_horizon,
        #         self.bioreactor.process_model.state_est_labels,
        #     ].values[0],
        #     input_matrix=u_matrix_daily[
        #         self.bioreactor.data["Day"] >= self.curr_time - N, :
        #     ],
        #     time=ts,
        #     output_mods=self.output_mods,
        # )

        # if self.curr_time - N < 0:
        #     data.loc[
        #         data["Day"] == self.curr_time,
        #         self.bioreactor.process_model.state_est_labels,
        #     ] = x_out[self.curr_time, :]
        # else:
        #     data.loc[
        #         data["Day"] == self.curr_time,
        #         self.bioreactor.process_model.state_est_labels,
        #     ] = x_out[self.bioreactor.duration - self.curr_time, :]

        # if self.curr_time - N < 0:
        #     data.loc[
        #         data["Day"] == self.curr_time,
        #         self.bioreactor.process_model.state_pred_labels,
        #     ] = y_out[self.curr_time, :]
        # else:
        #     data.loc[
        #         data["Day"] == self.curr_time,
        #         self.bioreactor.process_model.state_pred_labels,
        #     ] = y_out[self.bioreactor.duration - self.curr_time, :]

        # # Unravel delta_p back to matrix
        # delta_p = p_array[: len(self.output_mods.flatten())].reshape(
        #     self.output_mods.shape[0], self.output_mods.shape[1]
        # )

        # Cost function
        # cost = np.nansum(
        #     np.multiply(np.square(predictions - measurements), self.est_wts)
        #     + np.square(p_array - 1)
        #     + np.sum(
        #         np.square(p_array - self.output_mods
        #             # np.diff(
        #             #     data.loc[
        #             #         is_in_est_horizon,
        #             #         self.bioreactor.process_model.state_mod_labels,
        #             #     ]
        #             # )
        #         )
        #     )
        # )

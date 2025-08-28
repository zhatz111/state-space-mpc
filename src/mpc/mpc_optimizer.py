"""
MPC-related class definitions for bioreactors and controllers
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2025-08-27
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
from data.functions import daily_to_cumulative


class Bioreactor:
    """
    The Bioreactor class represents a bioreactor object used for simulation and
    processing of real-time data, with methods for initializing, resetting,
    logging samples, updating inputs, and advancing the simulation.
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
        This Python function initializes an object with specified attributes and
        data, performing various checks and conversions on the input data.

        Args:
          vessel (Union[str, int]): The `vessel` parameter in the `__init__`
          method is used to specify
        the name or identifier of the vessel or bioreactor for processing
        multiple bioreactors. It is a required parameter and can be either a
        string or an integer.
          process_model (StateSpaceModel): The `process_model` parameter in the
          `__init__` method is
        expected to be an instance of the `StateSpaceModel` class. This
        parameter is used for simulating a process within the class.
          data (Optional[pd.DataFrame]): The `data` parameter in the `__init__`
          method is used to
        provide a DataFrame containing process data. If no data is provided, a
        DataFrame is initialized based on the configuration settings. The
        DataFrame should include columns for various process variables,
        setpoints, and references. If cumulative feed data is
          config (Optional[dict]): The `config` parameter in the `__init__`
          method is a dictionary that
        should contain the following keys:
            - Batch Length: length of the batch as an int
            - Column Mapping: dictionary of columns in input topic to columns in
              bioreactor class df
            - Process Variable Setpoints: list of setpoints trajectory for
              target process variable i.e. Titer
            - Manipulated Variable Reference: list of setpoints trajectory for
              manipulated variable i.e. Feed
            - Process Variables: list of all process variables to control at a
              trajectory
            - Manipulated Variables: list of all variables to manipulate
              controlled trajectory
        """
        if not isinstance(experiment_config, dict):
            raise ValueError(
                """Config file must specify batch length, column mapping, PV
                setpoints and MV reference."""
            )
        else:
            self.experiment_config = experiment_config

        if not isinstance(controller_config, dict):
            raise ValueError("Must pass a controller configuration to this class.")
        else:
            self.controller_config = controller_config

        # Initialize attributes
        self.curr_time = 0
        self.start_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.duration = self.experiment_config["Last Day"] + 1
        self.scale = self.experiment_config["Scale"]
        self.init_vol = self.experiment_config["Initial Volumes"][self.scale]
        self.vol = self.init_vol

        # Update attributes based on user input
        self.vessel = vessel  # Vessel name for processing multiple bioreactors
        self.process_model = (
            process_model  # Model (if provided) for simulating a process
        )

        # this parameter is necessary for tech automation
        # when the data vector is passed to this class the columns must be mapped
        # correctly to the dataframe initialized at instantiation
        if data is None:
            self.column_map = self.experiment_config["Column Mapping"]
            sp_cols = [
                f"{x.upper()}{self.process_model.data_sp_suffix}"
                for x in controller_config["Process Variables"]
            ]
            mv_ref_cols = [
                f"{x.upper()}{self.process_model.input_ref_suffix}"
                for x in controller_config["Manipulated Variables"]
            ]
            u_cols = [
                f"{x.upper()}{self.process_model.input_suffix}"
                for x in controller_config["Input Variables"]
            ]
            x_cols = [
                f"{x.upper()}{self.process_model.data_suffix}"
                for x in controller_config["State Variables"]
            ]

            metadata_columns = [
                "Code_Execution_Date",
                "Bioreactor",
                "Day",
                "Date",
                "VOLUME_L",
            ]

            cols = (
                metadata_columns
                + sp_cols
                + mv_ref_cols
                + self.process_model.state_data_labels
                + self.process_model.input_data_labels
                + self.process_model.state_pred_labels
                + self.process_model.state_est_labels
                + self.process_model.state_mod_labels
            )
            if u_cols != self.process_model.input_data_labels:
                raise ValueError(
                    "Input vectors must be identical between model and controller config."
                )
            if x_cols != self.process_model.state_data_labels:
                raise ValueError(
                    "State vectors must be identical between model and controller config."
                )
            zero_arr = np.zeros((self.duration, len(cols)))
            zero_arr[:] = np.nan
            data = pd.DataFrame(data=zero_arr, columns=cols)

            # Initialize reference and nominal data
            for key in controller_config["Process Variables"]:
                data[f"{key.upper()}{self.process_model.data_sp_suffix}"] = np.array(
                    controller_config["Process Variables"][key]["Data"]
                )
            for key in controller_config["Manipulated Variables"]:
                data[f"{key.upper()}{self.process_model.input_ref_suffix}"] = np.array(
                    controller_config["Manipulated Variables"][key]["Data"]
                )
            for key in controller_config["Input Variables"]:
                data[f"{key.upper()}{self.process_model.input_suffix}"] = np.array(
                    controller_config["Input Variables"][key]
                )
            for key in controller_config["State Variables"]:
                data[f"{key.upper()}{self.process_model.data_suffix}"][0] = (
                    controller_config["State Variables"][key]["Initial"]
                )

            # Other metadata
            data["Day"] = np.arange(0, self.duration)
            data["Bioreactor"] = str(self.vessel)

            # Initialize volume
            data["VOLUME_L"] = self.init_vol

            self.data = data.copy(deep=True)
        else:
            self.data = data.copy(deep=True)

        # Feed names
        # self.total_feed_name = self.experiment_config["Total Feed Name"]
        # self.daily_feed_name = self.experiment_config["Daily Feed Name"]

        # Data frame for open_loop simulation results
        self.open_loop_df = pd.DataFrame()

        # Check if the data set starts on Day 0
        if self.data["Day"].values[0] != 0:
            raise ValueError("Data set does not start on Day 0!")

        # Check if the data set ends on Day duration
        if self.data.shape[0] != self.duration:
            raise ValueError("Data set has missing or duplicate days!")

        # Check if days are consecutive (2023-10-21)
        if any(np.diff(self.data["Day"]) != 1):
            raise ValueError("Data set is not in 1-day increments!")

        # convert monotonic cumulative manipulated variables to non-monotonic daily variables
        self.monotonic_inputs = []
        for key, value in controller_config["Manipulated Variables"].items():
            if value["Monotonic"]:
                self.data[f"{key.upper()}{self.process_model.input_suffix}"] = (
                    np.append(
                        np.diff(
                            self.data.loc[
                                :, f"{key.upper()}{self.process_model.input_suffix}"
                            ]
                        ),
                        0,
                    )
                )
                self.data[f"{key.upper()}{self.process_model.input_ref_suffix}"] = (
                    np.append(
                        np.diff(
                            self.data.loc[
                                :, f"{key.upper()}{self.process_model.input_ref_suffix}"
                            ]
                        ),
                        0,
                    )
                )
                self.monotonic_inputs.append(key.upper())
                warnings.warn(
                    f"{key.upper()} (data & reference) was converted to daily {key.upper()} (variable name is unchanged)!"
                )

        # Retain the original dataset
        self.original_data = self.data.copy(deep=True)
        self.tracking_dict = {}

    def get_result(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # Assemble result
        result = {
            "Current_Day": self.curr_time,
            "Process_Days": self.data["Day"].values,
        }

        feed_variable = [
            item
            for item in self.controller_config["Manipulated Variables"].keys()
            if "FEED" in item
        ]
        if self.experiment_config["Feed Type"] == "C":
            for var in feed_variable:
                # Retrieve daily feeds and convert to feed rates (mL/min)
                feed_daily = self.data[f"{var}{self.process_model.input_suffix}"].values
                feed_daily_ml = feed_daily * self.vol
                feed_rates_ml_min = feed_daily_ml / 24 / 60
                self.data[f"FEED_RATE_{var}{self.process_model.input_suffix}"] = (
                    feed_rates_ml_min
                )
                result[f"Feed_Rate_{var}_mL_min"] = feed_rates_ml_min

        for label in self.process_model.state_pred_labels:
            result[label] = self.data[label].values

        for label in self.process_model.input_data_labels:
            result[label] = self.data[label].values

        return result

    def show_data(self):
        """
        The function `show_data` prints the dataset for a bioreactor, with
        accurate column names.
        """

        if self.monotonic_inputs:
            data = self.data.copy(deep=True)
            for name in self.monotonic_inputs:
                data = data.rename(
                    columns={
                        f"{name.upper()}{self.process_model.input_suffix}": f"DAILY_{name.upper()}{self.process_model.input_suffix}"
                    }
                )
        else:
            data = self.data

        print(f"Dataset for Bioreactor: {self.vessel}")
        print(data)
        print("")

    def ingest_vectors(self, vector_dict: dict):
        """
        The function `ingest_vector` takes a Pandas Series vector, renames its
        columns based on a provided mapping, and inserts it into a DataFrame at
        a specific index based on a condition.

        Args:
          vector (pd.Series): The `ingest_vector` method takes a pandas Series
          object as input, which is
        represented by the parameter `vector`. This method is used to ingest the
        input vector into the class dataframe after performing some operations.
        """
        if self.column_map is None:
            raise ValueError(
                """Need to instantiate column mapping to correctly ingest input
                vectors to dataframe."""
            )
        for key, data in vector_dict.items():
            vector = pd.Series(data)

            if "nutrient_total" in vector:
                # Scale feed based on init_vol
                vector["nutrient_total"] = vector["nutrient_total"] / self.vol

            # Rename the input vector to standard column names
            renamed_vector = vector.rename(self.column_map)
            selected_col = (
                self.process_model.state_data_labels
                + self.process_model.input_data_labels
                + ["VOLUME_L"]
            )
            if renamed_vector is not None:
                # might need to change this if the day is the key
                insert_index = self.data[
                    self.data["Day"] == renamed_vector["Day"]
                ].index[0]
            else:
                raise ValueError(
                    """Vector does not contain data, check that vector is not None or
                    column mapping is correct."""
                )

            # Replace current day's data with non-NaN values in input
            self.data.loc[insert_index, renamed_vector[selected_col].dropna().index] = (
                renamed_vector[selected_col].dropna()
            )

            # Convert bioreactor data from daily to cumulative
            for name in self.monotonic_inputs:
                # Convert input data from daily to cumulative
                daily_input_data = self.data[
                    f"{name.upper()}{self.process_model.input_suffix}"
                ]
                total_input_data = np.append(0, np.cumsum(daily_input_data[0:-1]))

                # Adjust future daily values based on the offset between predicted and measured
                curr_total = renamed_vector[
                    f"{name.upper()}{self.process_model.input_suffix}"
                ]
                if not np.isnan(curr_total):
                    curr_time_offset = curr_total - total_input_data[insert_index]
                    total_input_data[insert_index:] = (
                        total_input_data[insert_index:] + curr_time_offset
                    )
                    daily_input_data = np.append(np.diff(total_input_data), 0)

                    # Check for negative daily feeds
                    if np.any(daily_input_data < 0):
                        raise ValueError("Negative daily feeds!")

                self.data[f"{name.upper()}{self.process_model.input_suffix}"] = (
                    daily_input_data
                )

            # Update volume
            if not np.isnan(renamed_vector["VOLUME_L"]):
                self.vol = renamed_vector["VOLUME_L"]

    def return_data(self, show_daily_inputs: bool = True, exec_date: bool = False):
        """
        The function `return_data` returns the dataset for a bioreactor, with
        accurate column names.
        """

        data = self.data.copy(deep=True)
        if exec_date:
            data["Code_Execution_Date"] = datetime.today().strftime("%Y-%m-%d")
            cols = np.array(data.columns.tolist())
            new_cols = cols[
                np.concatenate(
                    (
                        np.where(np.isin(cols, "Code_Execution_Date"))[0],
                        np.where(~np.isin(cols, "Code_Execution_Date"))[0],
                    )
                )
            ]
            data = data[new_cols].copy(deep=True)

        if self.monotonic_inputs:
            if show_daily_inputs:
                for name in self.monotonic_inputs:
                    data = data.rename(
                        columns={
                            f"{name.upper()}{self.process_model.input_suffix}": f"DAILY_{name.upper()}{self.process_model.input_suffix}",
                            f"{name.upper()}{self.process_model.input_ref_suffix}": f"DAILY_{name.upper()}{self.process_model.input_ref_suffix}",
                        }
                    )
            else:
                for name in self.monotonic_inputs:
                    # Convert input data back to cumulative
                    daily_input_data = data[
                        f"{name.upper()}{self.process_model.input_suffix}"
                    ]
                    total_input_data = np.append(0, np.cumsum(daily_input_data[0:-1]))
                    data[f"{name.upper()}{self.process_model.input_suffix}"] = (
                        total_input_data
                    )

                    # Convert reference data back to cumulative
                    daily_input_ref = data[
                        f"{name.upper()}{self.process_model.input_ref_suffix}"
                    ]
                    total_input_ref = np.append(0, np.cumsum(daily_input_ref[0:-1]))
                    data[f"{name.upper()}{self.process_model.input_ref_suffix}"] = (
                        total_input_ref
                    )

        return data

    def state(self, day=-1):
        """_summary_

        Args:
            day (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """
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

    def measurement(self, day=-1):
        """_summary_

        Args:
            day (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """
        if day >= 0:
            curr_time = day
        else:
            curr_time = self.curr_time

        # Measurements from the data sheet
        measurements = self.data.loc[
            self.data["Day"] == curr_time, self.process_model.state_data_labels
        ].values[0]

        # Current model predictions
        preds = self.data.loc[
            self.data["Day"] == curr_time, self.process_model.state_pred_labels
        ].values[0]

        # Replace missing data with predictions
        if np.any(np.isnan(measurements)):
            measurements[np.isnan(measurements)] = preds[np.isnan(measurements)]

        return measurements

    def sim_from_day(
        self, day=-1, initial_state=np.array([]), output_mods=np.array([])
    ):
        """_summary_

        Args:
            day (int, optional): _description_. Defaults to -1.
            initial_state (_type_, optional): _description_. Defaults to np.array([]).
            output_mods (_type_, optional): _description_. Defaults to np.array([]).

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
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

        # Convert daily variable to cumulative variable
        u_matrix_cumulative = daily_to_cumulative(
            model=self.process_model,
            input_variables=self.monotonic_inputs,
            u_matrix_daily=u_matrix_daily,
        )

        # Filter future inputs
        u_matrix_cumulative = u_matrix_cumulative[self.data["Day"] >= curr_time, :]

        # Get time array
        ts = np.arange(u_matrix_cumulative.shape[0])

        # Solve
        x_out, y_out = self.process_model.ssm_lsim(
            initial_state=x0,
            input_matrix=u_matrix_cumulative,
            time=ts,
            output_mods=output_mods,
            hidden_state=self.process_model.hidden_state,
        )

        # Create a DF for the predicted states
        x_out_df = pd.DataFrame(x_out, columns=self.process_model.state_est_labels)
        x_out_df.insert(0, "Day", ts + curr_time)

        # Create a DF for the predicted outputs
        y_out_df = pd.DataFrame(y_out, columns=self.process_model.state_pred_labels)
        y_out_df.insert(0, "Day", ts + curr_time)

        # Check if the simulation starts from the current state
        # if max(abs(x0 - x_out[0])) > 1e-10:  # ~np.all(self.state == x_out[0]):
        if not any(np.isclose(x0, x_out[0], rtol=1e-10)):
            raise ValueError("Simulation did not start from the current state!")

        return x_out, x_out_df, y_out, y_out_df

    def next_day(self):
        """
        The `next_day` function advances the simulation by 24 hours, updates the
        state and current time, and returns the simulation results as a
        DataFrame.

        Args:
          plot: The `plot` parameter is a boolean flag that determines whether
          or not to generate plots
        after simulation. If `plot` is set to `True`, the code will generate
        plots comparing simulated PV and setpoint. Defaults to False

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


class Controller:
    """
    The `Controller` class represents a controller object that is used to
    control a bioreactor system by optimizing the manipulated variables based on
    setpoints and weights.
    """

    def __init__(
        self,
        controller_model: StateSpaceModel,
        bioreactor: Bioreactor,
        controller_config: dict,
    ):
        """
        The function is the initialization method for a controller object,
        taking in various parameters to set up the controller.

        Args:
          controller_model: The controller model is the model used for control,
          such as a PID controller
        or a model predictive controller (MPC). It defines the control algorithm
        and strategy used to manipulate the process variables.
          bioreactor (Bioreactor): The `bioreactor` parameter is an instance of
          the `Bioreactor` class.
        It represents the bioreactor system that the controller will be
        controlling.
          ts (np.ndarray): A 1D array of length T representing the time points
          at which the process
        variables are measured or controlled.
          pv_sps (np.ndarray): `pv_sps` is a 2D array of shape (T, P), where T
          is the number of time
        steps and P is the number of process variables. Each row represents the
        setpoint values for the process variables at a specific time step.
          pv_names (list[str]): The `pv_names` parameter is a list of strings
          that represents the names
        of the controlled process variables. These are the variables that the
        controller will try to regulate and maintain at setpoints.
          pv_wts (np.ndarray): The `pv_wts` parameter is a numpy array that
          represents the SP (setpoint)
        tracking weights for the controlled process variables. It is a 1D array
        of length P, where P is the number of process variables. Each element of
        the array represents the weight for a specific process variable
          mv_names (list[str]): The `mv_names` parameter is a list of strings
          that represents the names
        of the manipulated variables (MV) in the system. These are the variables
        that the controller can adjust to control the process variables (PV) and
        achieve the desired setpoints.
          mv_wts (np.ndarray): The `mv_wts` parameter is an array that
          represents the cost weights for
        the manipulated variables (MVs). It is a 1D array of length U, where U
        is the number of manipulated variables. Each element of the array
        represents the cost weight for a specific manipulated variable. These
        cost
          pred_horizon (int): The prediction horizon is the number of time steps
          into the future for
        which the controller will generate predictions. It determines how far
        ahead the controller will look when making decisions about the
        manipulated variables.
          ctrl_horizon (int): The control horizon is the number of time steps
          into the future that the
        controller plans for. It determines how far ahead the controller looks
        when making control decisions.
          constr (np.ndarray): The `constr` parameter is a 2 by U array that
          represents the lower and
        upper limits for the manipulated variables (MVs). U is the number of
        manipulated variables. The first row of the `constr` array represents
        the lower limits for each MV, and the second row represents the
        """

        # Use config file to construct other tuning parameters
        ts = np.array(controller_config["Time"])
        pv_sps = np.transpose(
            np.array(
                [
                    controller_config["Process Variables"][key]["Data"]
                    for key in controller_config["Process Variables"]
                ]
            )
        )
        pv_wts = np.array(
            [
                controller_config["Process Variables"][key]["Weight"]
                for key in controller_config["Process Variables"]
            ]
        )
        pv_names = list(controller_config["Process Variables"].keys())
        mv_wts = np.array(
            [
                controller_config["Manipulated Variables"][key]["Weight"]
                for key in controller_config["Manipulated Variables"]
            ]
        )
        mv_names = list(controller_config["Manipulated Variables"].keys())
        mv_constr = np.transpose(
            np.array(
                [
                    controller_config["Manipulated Variables"][key]["Constraint"]
                    for key in controller_config["Manipulated Variables"]
                ]
            )
        )
        est_wts = np.array(
            [
                controller_config["State Variables"][key]["Weight"]
                for key in controller_config["State Variables"]
            ]
        )
        offset_individual_kps = np.array(
            [
                controller_config["State Variables"][key]["Offset Proportional Gain"]
                for key in controller_config["State Variables"]
            ]
        )
        offset_individual_kis = np.array(
            [
                controller_config["State Variables"][key]["Offset Integral Gain"]
                for key in controller_config["State Variables"]
            ]
        )
        offset_single_kp = controller_config["Offset Proportional Gain"]
        offset_single_ki = controller_config["Offset Integral Gain"]
        eor_names = list(controller_config["End of Run Variables"].keys())
        eor_constr = np.transpose(
            np.array(
                [
                    controller_config["End of Run Variables"][key]["Constraint"]
                    for key in controller_config["End of Run Variables"]
                ]
            )
        )
        eor_wts = np.array(
            [
                controller_config["End of Run Variables"][key]["Weight"]
                for key in controller_config["End of Run Variables"]
            ]
        )
        pred_horizon = controller_config["Prediction Horizon"]
        ctrl_horizon = controller_config["Control Horizon"]
        est_horizon = controller_config["Estimation Horizon"]
        filter_wt_on_data = controller_config["Estimation Filter Weight on Data"]
        persist_after_ctrl_horizon = controller_config["Persist After Control Horizon"]

        # The basics
        self.controller_model = controller_model
        self.bioreactor = bioreactor
        self.curr_time = bioreactor.curr_time
        self.ts = ts
        self.pv_sps = pv_sps

        self.pv_wts = pv_wts
        self.est_wts = est_wts
        self.filter_wt_on_data = filter_wt_on_data
        if offset_single_kp >= 0:
            self.offset_kp = offset_single_kp
        else:
            self.offset_kp = offset_individual_kps
        if offset_single_ki >= 0:
            self.offset_ki = offset_single_ki
        else:
            self.offset_ki = offset_individual_kis

        self.mv_wts = mv_wts
        self.pred_horizon = pred_horizon
        self.ctrl_horizon = ctrl_horizon
        self.mv_constr = mv_constr
        self.persist_after_ctrl_horizon = persist_after_ctrl_horizon

        self.output_mods_est = np.zeros((1, len(est_wts)))
        self.est_curr_error = np.zeros((1, len(est_wts)))
        self.est_prev_error = np.zeros((1, len(est_wts)))
        self.est_horizon = est_horizon
        self.eor_names = eor_names
        self.eor_const = eor_constr
        self.eor_wts = eor_wts

        # Data snapshots (2023-10-22)
        self.data_before_optim = pd.DataFrame.copy(bioreactor.data)
        self.data_after_optim = pd.DataFrame.copy(bioreactor.data)

        pv_suffix = self.bioreactor.process_model.data_suffix
        self.pv_names = [x.upper() + pv_suffix for x in pv_names]

        mv_suffix = self.bioreactor.process_model.input_suffix
        self.mv_names = [x.upper() + mv_suffix for x in mv_names]

        mv_ref_suffix = self.bioreactor.process_model.input_ref_suffix
        self.mv_ref_names = [x.upper() + mv_ref_suffix for x in mv_names]

        self.data_before_optim_dict = {}
        self.data_after_optim_dict = {}

    def optimize(self, open_loop=False, print_pred=False, end_of_run=False):
        """
        The `optimize` function optimizes future inputs for a bioreactor system
        and updates the dataset with the optimized inputs.

        Args:
          plot: The `plot` parameter is a boolean flag that determines whether
          or not to generate plots
        after optimization. If `plot` is set to `True`, the code will generate
        plots comparing the un-optimized and optimized data for each process
        variable (PV) and manipulated variable (MV). If `plot`. Defaults to
        False
        """

        # Retrieve MVs from curr_time to EoR
        self.curr_time = self.bioreactor.curr_time
        is_in_ctrl_horizon = np.logical_and(
            np.logical_and(
                self.bioreactor.data["Day"] >= self.curr_time,
                self.bioreactor.data["Day"] < (self.curr_time + self.ctrl_horizon),
            ),
            self.bioreactor.data["Day"] < max(self.bioreactor.data["Day"]),
        )
        # is_after_ctrl_horizon = self.bioreactor.data["Day"] >= (self.curr_time + self.ctrl_horizon)
        is_in_pred_horizon = np.logical_and(
            self.bioreactor.data["Day"] >= self.curr_time,
            self.bioreactor.data["Day"] < (self.curr_time + self.pred_horizon + 1),
        )

        # Do not run MPC on EoR
        pred_names = [
            x.replace(
                self.bioreactor.process_model.data_suffix,
                self.bioreactor.process_model.yhat_suffix,
            )
            for x in self.pv_names
        ]
        sp_names = [
            x.replace(
                self.bioreactor.process_model.data_suffix,
                self.bioreactor.process_model.data_sp_suffix,
            )
            for x in self.pv_names
        ]

        control_matrix = self.bioreactor.data.loc[
            is_in_ctrl_horizon, self.mv_names
        ].values

        # Update reference input with current input data and previously optimized data
        self.bioreactor.data[self.mv_ref_names] = self.bioreactor.data[self.mv_names]

        # Flatten initial mv
        mv_array = control_matrix.flatten()

        # Create constraint matrix
        constr_low_matrix = np.tile(self.mv_constr[0, :], (control_matrix.shape[0], 1))
        constr_low_array = constr_low_matrix.flatten()
        constr_high_matrix = np.tile(self.mv_constr[1, :], (control_matrix.shape[0], 1))
        constr_high_array = constr_high_matrix.flatten()
        bounds = np.vstack((constr_low_array, constr_high_array)).transpose()

        # Simulate before optimization
        _, y_out_before_optim, _ = self.obj_func_wrapper(mv_array)
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
        elif end_of_run:
            y_out_after_optim = self.bioreactor.state()
            control_matrix_star = []
        else:
            # Solve the optimization problem
            mv_array_star = optimize.minimize(
                fun=lambda x: self.obj_func_wrapper(x)[0],
                x0=mv_array,
                bounds=bounds,
                method="SLSQP",
                options={"disp": False, "maxiter": 100},
            )

            _, y_out_after_optim, mv_after_optim = self.obj_func_wrapper(
                mv_array_star.x
            )

        # Update post-optimization (or open loop) data record
        data_after_optim = self.data_after_optim
        data_after_optim.loc[
            data_after_optim["Day"] >= self.curr_time,
            self.controller_model.state_pred_labels,
        ] = y_out_after_optim
        data_after_optim.loc[:, self.mv_names] = mv_after_optim
        self.data_after_optim_dict[self.curr_time] = data_after_optim.copy()

        # Update the dataset with new inputs
        self.bioreactor.data.loc[:, self.mv_names] = mv_after_optim

        # Update the dataset with new predictions
        self.bioreactor.data.loc[
            is_in_pred_horizon, self.controller_model.state_pred_labels
        ] = data_after_optim.loc[
            is_in_pred_horizon, self.controller_model.state_pred_labels
        ]

        # Print final PV (2024-03-01)
        if print_pred:
            print(
                self.bioreactor.data.loc[
                    self.bioreactor.data["Day"].isin(
                        (
                            np.max(self.bioreactor.data.loc[is_in_pred_horizon, "Day"]),
                            np.max(self.bioreactor.data["Day"]),
                        )
                    ),
                    ["Day", "Bioreactor"]
                    + sp_names
                    + pred_names
                    + [
                        str.upper(x) + f"{self.controller_model.yhat_suffix}"
                        for x in self.eor_names
                    ],
                ]
            )

    def obj_func_wrapper(self, mv_array):
        """
        The `obj_func_wrapper` function is a wrapper for an objective function
        that takes in an array of manipulated variables and calculates the
        objective function value based on the current state of a bioreactor
        system.

        Args:
          mv_array: The `mv_array` parameter is an array that contains the
          manipulated variable (MV)
        values. It is used to replace the MVs in the input matrix
        `u_matrix_daily` with the values from `mv_array`. The MVs are identified
        by their names, which are stored in the `mv

        Returns:
          The function `obj_func_wrapper` returns a tuple containing two
          elements. The first element is
        the result of calling the `obj_func` function with the arguments `ts +
        self.curr_time`, `x_out[:, pv_loc]`, and
        `u_matrix_daily[self.bioreactor.data["Day"] >= self.curr_time, :][:,
        mv_loc]`. The second element is the `x_out`
        """

        # Rows within the control horizon (2023-10-21/updated 2024-05-06: added
        # the < max day condition)
        ctrl_horizon_where = np.where(
            np.logical_and(
                np.logical_and(
                    self.bioreactor.data["Day"] >= self.curr_time,
                    self.bioreactor.data["Day"] < (self.curr_time + self.ctrl_horizon),
                ),
                self.bioreactor.data["Day"] < max(self.bioreactor.data["Day"]),
            )
        )[0]

        # Remaining duration
        after_ctrl_horizon_where = np.where(
            self.bioreactor.data["Day"] >= (self.curr_time + self.ctrl_horizon)
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
        if self.persist_after_ctrl_horizon:
            u_matrix_daily[after_ctrl_horizon_where, :] = u_matrix_daily_ctrl_horizon[
                -1, :
            ]
        u_matrix_cumulative = u_matrix_daily

        # Convert daily feed to cumulative feed
        # if self.bioreactor.has_cumulative_feed_data:
        u_matrix_cumulative = daily_to_cumulative(
            model=self.controller_model,
            input_variables=self.bioreactor.monotonic_inputs,
            u_matrix_daily=u_matrix_daily,
        )

        # Time array
        ts = np.arange(
            u_matrix_daily[self.bioreactor.data["Day"] >= self.curr_time, :].shape[0]
        )

        # Simulate the modified output
        _, y_out = self.controller_model.ssm_lsim(
            initial_state=self.bioreactor.measurement(),  # state(),
            input_matrix=u_matrix_cumulative[
                self.bioreactor.data["Day"] >= self.curr_time, :
            ],
            time=ts,
            output_mods=self.output_mods_est,
            hidden_state=self.controller_model.hidden_state,
        )

        # Retrieve end of run predictions
        e = np.array([])
        if self.eor_names is not None:
            for state_count, state in enumerate(self.eor_names):
                constraint = self.eor_const[:, state_count]

                # Take the sqrt to be consistent with weight multiplied by squares in obj
                eor_wt = np.sqrt(self.eor_wts[state_count])

                for idx, string in enumerate(self.controller_model.state_pred_labels):
                    if str.upper(state) in string:
                        if constraint[0] > y_out[-1, idx]:
                            e = np.append(e, (constraint[0] - y_out[-1, idx]) * eor_wt)
                        if constraint[1] < y_out[-1, idx]:
                            e = np.append(e, (y_out[-1, idx] - constraint[0]) * eor_wt)

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
                u_matrix_daily[:, mv_loc],
                e,
            ),
            y_out,
            u_matrix_daily[:, mv_loc],
        )

    def ctrl_obj_func(
        self, ts: np.ndarray, y: np.ndarray, u: np.ndarray, e: np.ndarray
    ):
        """
        The function calculates the cost value based on the given inputs.

        Args:
          ts (np.ndarray): `ts` is a numpy array representing the time steps. It
          is used to index and
        trim the arrays `x`, `u`, and `pv_sps` to keep only the future entries.
          x (np.ndarray): The parameter `x` is a numpy array representing the
          current state of the
        system. It contains the values of the system variables at each time
        step.
          u (np.ndarray): The parameter `u` is a numpy array representing the
          control inputs. It is a
        2-dimensional array with shape `(num_samples, num_inputs)`, where
        `num_samples` is the number of samples and `num_inputs` is the number of
        control inputs. Each row of the array represents a

        Returns:
          the sum of the cost values for the control inputs (u3_cost) and the
          process variables
        (x3_cost).
        """

        # Trim to keep only future entries
        y2 = y[ts > self.curr_time, :]
        pv_sps2 = self.pv_sps[self.ts > self.curr_time, :]

        # Trim to keep the prediction and control horizons
        y3 = y2[0 : self.pred_horizon, :]
        pv_sps3 = pv_sps2[0 : self.pred_horizon, :]

        # Calculate the cost
        u_diff = np.diff(np.vstack((u[0,], u)), axis=0)  # self.mv_constr[0][0]
        u2_diff = u_diff[self.ts >= self.curr_time, :]
        u2 = u[self.ts >= self.curr_time, :]
        u3_diff = u2_diff[0 : self.ctrl_horizon, :]
        u3 = u2[0 : self.ctrl_horizon, :]
        u3_diff_norm = np.divide(u3_diff, u3)
        u3_cost = np.mean(
            np.multiply(np.sum(np.square(u3_diff_norm), axis=0), self.mv_wts)
        )
        y3_diff = y3 - pv_sps3
        y3_diff_norm = np.divide(y3_diff, pv_sps3)
        y3_cost = np.mean(
            np.multiply(np.sum(np.square(y3_diff_norm), axis=0), self.pv_wts)
        )
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

        n_horizon = self.est_horizon

        # ensure curr_time is set to the time of the bioreactor
        self.curr_time = self.bioreactor.curr_time

        # select days in the MHE horizon
        is_in_est_horizon = np.logical_and(
            self.bioreactor.data["Day"] <= self.curr_time,
            self.bioreactor.data["Day"] > (self.curr_time - n_horizon),
        )

        # Retrieve the start and end measurements within est horizon
        t0 = self.bioreactor.data.loc[is_in_est_horizon, "Day"].values[0]
        t1 = self.bioreactor.data.loc[is_in_est_horizon, "Day"].values[-1]
        x0_est = self.bioreactor.state(t0)
        x0_data = self.bioreactor.measurement(t0)
        x1_data = self.bioreactor.measurement(t1)

        # First assign data to x0 estimate
        x0 = x0_data

        # Replace missing value with the latest estimate
        if np.any(np.isnan(x0_data)):
            ind_to_estimate = np.where(np.isnan(x0_data))[0]
            x0[ind_to_estimate] = x0_est[ind_to_estimate]

        # Skip re-estimation of the start state
        x_star = x0

        # Predict current state
        x_out, _, _, _ = self.bioreactor.sim_from_day(
            day=t0, initial_state=x_star, output_mods=self.output_mods_est
        )
        new_estimates_x = x_out[0 : sum(is_in_est_horizon),]

        # Update current state
        self.bioreactor.data.loc[
            self.bioreactor.data["Day"] == self.curr_time,
            self.bioreactor.process_model.state_est_labels,
        ] = new_estimates_x[-1]

        # PI control of model error
        self.est_prev_error = self.est_curr_error
        curr_estimate = new_estimates_x[-1]
        curr_measurement = x1_data
        self.est_curr_error = curr_measurement - curr_estimate
        self.est_curr_error[np.isnan(self.est_curr_error)] = 0
        self.output_mods_est += np.multiply(
            self.est_curr_error - self.est_prev_error, self.offset_kp
        ) + np.multiply(self.est_curr_error, self.offset_ki)

        # Store the new correction into the dataframe
        self.bioreactor.data.loc[
            self.bioreactor.data["Day"] == self.curr_time,
            self.bioreactor.process_model.state_mod_labels,
        ] = self.output_mods_est

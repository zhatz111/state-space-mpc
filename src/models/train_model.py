"""
Code for training and testing state-space models
Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
Created: 2022-11-04
Modified: 2025-08-08
"""

# Standard library imports
import math
import random
import warnings
from typing import Union
from datetime import datetime
from pathlib import Path
from itertools import chain

# Imports from 3rd party libraries
import numpy as np
import pandas as pd
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

warnings.filterwarnings("ignore")


# The ModelTraining class is used for training state space models.
class ModelTraining:
    """_summary_"""

    random.seed(10)

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        states: list[str],
        inputs: list[str],
        pv_wghts: list,
        instability_weight: float,
        num_days: int,
        scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
        algorithm: str = "basinhopping",
        hidden_state: bool = False,
        rho: float = 0.5,
        af_col: np.ndarray = np.array([]),
        af_row: np.ndarray = np.array([]),
        bf_row: np.ndarray = np.array([]),
    ):
        """
        The function is an initializer for a class that takes in various parameters and initializes them
        as attributes of the class.

        Args:
          train_data (pd.DataFrame): The `train_data` parameter is a pandas DataFrame that contains the
        training data for your model. It should have the necessary columns and rows to train your model.
          test_data (pd.DataFrame): The `test_data` parameter is a pandas DataFrame that contains the
        test data for your model. It is used to evaluate the performance of your model on unseen data.
          a_matrix: The `a_matrix` parameter represents the transition matrix for the hidden states in a
        Hidden Markov Model (HMM). It is a matrix that defines the probabilities of transitioning from
        one state to another. Each row of the matrix represents the probabilities of transitioning from
        the current state to all other states.
          b_matrix: The `b_matrix` parameter is a matrix that represents the emission probabilities of
        the hidden states given the observed inputs. It is a matrix of shape (num_states, num_inputs),
        where num_states is the number of hidden states and num_inputs is the number of observed inputs.
        Each element in the matrix
          states (list): The `states` parameter is a list that represents the different states or
        variables in your system. Each element in the list represents a state or variable.
          inputs (list): The `inputs` parameter is a list that contains the names of the input variables
        or features used in the model. These inputs are used to predict the state variables.
          num_days (int): The `num_days` parameter represents the number of days for which the model
        will be trained and tested. It determines the length of the time series data that will be used
        for training and testing the model.
          scaler (MinMaxScaler): The `scaler` parameter is an instance of the `MinMaxScaler` class. It
        is used to scale the input data to a specified range, typically between 0 and 1. This scaling is
        important for certain machine learning algorithms that are sensitive to the scale of the input
        features.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.states = states
        self.inputs = inputs
        self.num_days = num_days
        self.scaler = scaler
        self.pv_wghts = pv_wghts
        self.instability_weight = instability_weight
        self.state_len = len(states)
        self.input_len = len(inputs)
        self.total = self.state_len + self.input_len
        self.algorithm = algorithm
        self.hidden_state = hidden_state

        # store training error data
        self.iters = 0
        self.model_error = 0
        self.weighted_model_error = 0
        self.lowest_model_error = np.inf
        self.weighted_lowest_model_error = np.inf
        self.model_error_dict = {}
        self.lowest_model_error_dict = {}
        self.true_model_error_dict = {}
        self.best_result = np.array([])

        # normal matrices for training and testing
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.c_matrix = np.identity(self.state_len)
        self.d_matrix = np.zeros([self.state_len, self.input_len])

        # Check to make sure matrices are correct dimensions, if not create a random matrix of values
        if (self.a_matrix.shape[0] != self.state_len) or (
            self.a_matrix.shape[1] != self.state_len
        ):
            warnings.warn(
                f"Wrong size A-Matrix ({self.a_matrix.shape[0]}x{self.a_matrix.shape[1]}) should be, {self.state_len}x{self.state_len}"
            )
            self.a_matrix = np.resize(self.a_matrix, (self.state_len, self.state_len))

        if (self.b_matrix.shape[0] != self.state_len) or (
            self.b_matrix.shape[1] != self.input_len
        ):
            warnings.warn(
                f"Wrong size B-Matrix ({self.b_matrix.shape[0]}x{self.b_matrix.shape[1]}) should be, {self.state_len}x{self.input_len}"
            )
            self.b_matrix = np.resize(self.b_matrix, (self.state_len, self.input_len))

        # augmented matrices if using feed hidden state
        if self.hidden_state:
            self.rho = rho  # feed effect decay rate
            if af_col.shape[0] == self.state_len:
                self.af_col = np.array(af_col)
            else:
                self.af_col = np.ones([self.state_len, 1])  # learnable hidden state

            if af_row.shape[0] == self.state_len:
                self.af_row = np.array(af_row)
            else:
                self.af_row = np.ones([self.state_len, 1])  # learnable hidden state

            if bf_row.shape[0] == self.input_len:
                self.bf_row = bf_row
            else:
                self.bf_row = np.zeros([1, self.input_len])
                self.bf_row[:, 1] = 1.0

            self.augmented_a_matrix = np.zeros([self.state_len + 1, self.state_len + 1])
            self.augmented_a_matrix[: self.state_len, : self.state_len] = self.a_matrix
            self.augmented_a_matrix[: self.state_len, self.state_len] = (
                self.af_col.flatten()
            )
            self.augmented_a_matrix[self.state_len, : self.state_len] = (
                self.af_row.flatten()
            )
            self.augmented_a_matrix[self.state_len, self.state_len] = self.rho

            self.augmented_b_matrix = np.zeros([self.state_len + 1, self.input_len])
            self.augmented_b_matrix[: self.state_len, :] = self.b_matrix
            self.augmented_b_matrix[self.state_len, :] = (
                self.bf_row
            )  # only feed (input index 0) affects feed-effect state (hidden state)

            self.c_matrix = np.hstack(
                [np.identity(self.state_len), np.zeros([self.state_len, 1])]
            )

    def first_pass_training(self):
        """
        The function `first_pass_training` performs linear regression on the training data to generate a
        regression matrix.

        Returns:
          a regression matrix, which is a 2D numpy array containing the coefficients of the linear
        regression models for each state variable.
        """
        x_data = np.array(
            self.train_data[
                self.train_data["Day"].between(0, self.num_days - 2)
            ].filter(items=self.states + self.inputs)
        )
        y_data = np.array(
            self.train_data[
                self.train_data["Day"].between(1, self.num_days - 1)
            ].filter(items=self.states)
        )
        regression_matrix = np.zeros([self.state_len, self.total], dtype=np.float64)
        for j in range(self.state_len):
            reg = LinearRegression().fit(x_data, y_data[:, j])
            regression_matrix[j, :] = reg.coef_
        return regression_matrix

    def train_model(
        self,
        save_path: Path,
        first_train: bool = True,
        iterations: int = 10,
        basin_temp: int = 2000,
    ):
        """
        The `train_model` function trains a system model using the provided training data and saves the
        resulting A and B matrices to CSV files.

        Args:
          save_path: The `save_path` parameter is the directory path where the A_Matrix.csv and
        B_Matrix.csv files will be saved.
          first_train: The `first_train` parameter is a boolean flag that indicates whether it is the
        first time training the model. If `first_train` is `True`, it means it is the first time
        training and the matrices `a_matrix` and `b_matrix` will be initialized with ones. If `first.
        Defaults to True
          iterations: The `iterations` parameter specifies the maximum number of iterations for the
        optimization algorithm to run. It determines how many times the objective function will be
        evaluated and the optimization process will be performed. The default value is 10. Defaults to
        10
        """
        if first_train:
            self.a_matrix = np.random.random([self.state_len, self.state_len])
            self.b_matrix = np.random.random([self.state_len, self.input_len])
            self.c_matrix = np.identity(self.state_len)
            self.d_matrix = np.zeros([self.state_len, self.input_len])

        if self.hidden_state:
            a_sim = self.augmented_a_matrix.reshape(-1, 1)
            b_sim = self.augmented_b_matrix.reshape(-1, 1)
            combined_mat = np.vstack([a_sim, b_sim]).flatten()
        else:
            a_sim = self.a_matrix.reshape(-1, 1)
            b_sim = self.b_matrix.reshape(-1, 1)
            combined_mat = np.vstack([a_sim, b_sim]).flatten()

        if self.hidden_state:
            bounds_ = [(-1, 1)] * len(combined_mat)
            bounds_[len(a_sim) - 1] = (0, 1)
        else:
            bounds_ = [(-1, 1)] * len(combined_mat)

        if self.algorithm == "minimize":
            optimize.minimize(
                fun=self.objective_func,
                x0=combined_mat,
                bounds=bounds_,
                method="SLSQP",
                options={"maxiter": iterations, "disp": False, "ftol": 1e-07},
            )
        elif self.algorithm == "evolution":
            optimize.differential_evolution(
                func=self.objective_func, bounds=bounds_, maxiter=iterations, disp=False
            )
        elif self.algorithm == "basinhopping":
            minimizer_kwargs = {
                "method": "SLSQP",
                "bounds": bounds_,
                "options": {"maxiter": iterations * 10, "disp": False, "ftol": 1e-07},
            }
            optimize.basinhopping(
                func=self.objective_func,
                x0=combined_mat,
                minimizer_kwargs=minimizer_kwargs,
                niter=iterations,
                disp=True,
                T=basin_temp,
            )
        else:
            raise KeyError("Must use either Minimize or Basin Hopping algorithms")

        print("")
        print("--------------------Final Results--------------------")
        for key, value in self.lowest_model_error_dict.items():
            print(f"{key} Error: {value:.5f}")
        print("--------------------")
        print(f"Iterations: {self.iters}")
        print(f"Weighted Model Error: {self.weighted_lowest_model_error:.5f}")
        print(f"Standard Model Error: {self.lowest_model_error:.5f}")

        # opt_matrix = res.x
        opt_matrix = self.best_result

        if self.hidden_state:
            # This returns the matrix in the correct shape for use later on in the class
            self.augmented_a_matrix = opt_matrix[: ((self.state_len + 1) ** 2)].reshape(
                self.state_len + 1, self.state_len + 1
            )
            self.augmented_b_matrix = opt_matrix[((self.state_len + 1) ** 2) :].reshape(
                self.state_len + 1, self.input_len
            )
            self.a_matrix = self.augmented_a_matrix[: self.state_len, : self.state_len]
            self.b_matrix = self.augmented_b_matrix[: self.state_len, :]
            self.rho = self.augmented_a_matrix[self.state_len, self.state_len]
            self.af_col = self.augmented_a_matrix[: self.state_len, self.state_len]
            self.af_row = self.augmented_a_matrix[self.state_len, : self.state_len]
            self.bf_row = self.augmented_b_matrix[self.state_len, :]
        else:
            # This returns the matrix in the correct shape for use later on in the class
            self.a_matrix = opt_matrix[: (self.state_len**2)].reshape(
                self.state_len, self.state_len
            )
            self.b_matrix = opt_matrix[(self.state_len**2) :].reshape(
                self.state_len, self.input_len
            )

        save_time = datetime.now().strftime("%Y-%m-%d %H%M%S")

        # historic_path = Path(save_path, "historic_matrices")
        # historic_path.mkdir(parents=True, exist_ok=True)
        # pd.DataFrame(self.a_matrix).to_csv(
        #     rf"{save_path}\A_Matrix.csv", index=False, header=False
        # )
        # pd.DataFrame(self.b_matrix).to_csv(
        #     rf"{save_path}\B_Matrix.csv", index=False, header=False
        # )

        if self.hidden_state:
            pd.DataFrame(self.augmented_a_matrix).to_csv(
                rf"{save_path}\{save_time} Augmented_A_Matrix.csv",
                index=False,
                header=False,
            )
            pd.DataFrame(self.augmented_b_matrix).to_csv(
                rf"{save_path}\{save_time} Augmented_B_Matrix.csv",
                index=False,
                header=False,
            )
        else:
            pd.DataFrame(self.a_matrix).to_csv(
                rf"{save_path}\{save_time} A_Matrix.csv", index=False, header=False
            )
            pd.DataFrame(self.b_matrix).to_csv(
                rf"{save_path}\{save_time} B_Matrix.csv", index=False, header=False
            )

    def objective_func(self, x0):
        """
        The `objective_func` function calculates the error between actual and simulated data for a
        given set of A and B matrices.

        Args:
            mat: The parameter `mat` is a 1-dimensional numpy array that contains the values of the
        matrices `a_matrix` and `b_matrix`. The first `self.state_len**2` elements of `mat`
        represent the values of `a_matrix`, while the remaining elements represent the values of `b
            info: The "info" parameter is a dictionary that contains additional information about the
        optimization process. It is used to keep track of the number of function evaluations
        (Nfeval) and can be used to store any other relevant information during the optimization
        process.

        Returns:
            the value of the objective function, which is the sum of squared errors between the actual
        and simulated values of the system.
        """
        iter_counter = 0
        train_grouped = self.train_data.groupby("Batch")
        y_sim_all = np.zeros([self.num_days, self.state_len])
        y_actual_all = np.zeros([self.num_days, self.state_len])

        if self.hidden_state:
            a_matrix = x0[: ((self.state_len + 1) ** 2)].reshape(
                self.state_len + 1, self.state_len + 1
            )
            b_matrix = x0[((self.state_len + 1) ** 2) :].reshape(
                self.state_len + 1, self.input_len
            )
        else:
            a_matrix = x0[: (self.state_len**2)].reshape(self.state_len, self.state_len)
            b_matrix = x0[(self.state_len**2) :].reshape(self.state_len, self.input_len)

        for _, group in train_grouped:
            if self.hidden_state:
                objfunc_x0 = np.append(
                    np.array(group.filter(self.states).iloc[0, :]), 0
                )
            else:
                objfunc_x0 = np.array(group.filter(self.states).iloc[0, :])

            objfunc_u = np.array(group.filter(self.inputs))
            objfunc_y = np.array(group.filter(self.states))
            time = np.arange(0, len(objfunc_u), 1)

            state = signal.StateSpace(a_matrix, b_matrix, self.c_matrix, self.d_matrix)
            _, objfunc_yout, _ = signal.lsim(
                state, objfunc_u, time, objfunc_x0, interp=False
            )

            if iter_counter == 0:
                y_sim_all = np.array(objfunc_yout, dtype=np.float64)
                y_actual_all = np.array(objfunc_y, dtype=np.float64)
            else:
                y_sim_all = np.vstack([y_sim_all, objfunc_yout])
                y_actual_all = np.vstack([y_actual_all, objfunc_y])
            iter_counter += 1

        cost_list = []
        for count, state in enumerate(self.states):
            y_diff = y_actual_all[:, count] - y_sim_all[:, count]
            feature_cost = np.square(y_diff).sum()
            negative_cost = sum(y_sim_all[:, count] < 0) * 0
            cost_list.append(feature_cost + feature_cost * negative_cost)

        eigvals = np.linalg.eigvals(a_matrix)
        real_parts = np.real(eigvals)

        # Violation amount = how far into unstable region each eigenvalue is
        violations = np.clip(
            real_parts + 1e-12, a_min=0.0, a_max=None
        )  # add small buffer to ensure stability

        # Quadratic penalty for instability
        penalty = np.sum(np.square(violations)) * self.instability_weight

        wghtd_cost = sum(np.array(cost_list) * np.array(self.pv_wghts)) + penalty
        unwghtd_cost = sum(np.array(cost_list)) + penalty

        rmse_sim_scaled = np.hstack(
            (np.array(y_sim_all), np.zeros((y_sim_all.shape[0], self.input_len)))
        )
        rmse_sim_unscaled = np.array(self.scaler.inverse_transform(rmse_sim_scaled))

        rmse_real_scaled = np.hstack(
            (np.array(y_actual_all), np.zeros((y_actual_all.shape[0], self.input_len)))
        )
        rmse_real_unscaled = np.array(self.scaler.inverse_transform(rmse_real_scaled))
        rmse_diff = (
            rmse_sim_unscaled[:, : self.state_len]
            - rmse_real_unscaled[:, : self.state_len]
        )

        if self.iters % 50 == 0:
            print("")
            print(f"Iteration: {self.iters}")
            for count, state in enumerate(self.states):
                rmse = np.sqrt(
                    np.square(rmse_diff[:, count]).sum() / rmse_diff.shape[0]
                )
                self.model_error_dict[state] = rmse
                print(f"{state} Error: {rmse:.5f}")
            print("--------------------")
            print(f"Total Weighted Error: {wghtd_cost:.5f}")
            print(f"Total Standard Error: {unwghtd_cost:.5f}")
        self.iters += 1
        if (wghtd_cost < self.weighted_model_error) or self.iters == 1:
            for count, state in enumerate(self.states):
                rmse = np.sqrt(
                    np.square(rmse_diff[:, count]).sum() / rmse_diff.shape[0]
                )
                self.true_model_error_dict[state] = rmse
            self.model_error = unwghtd_cost
            self.weighted_model_error = wghtd_cost

        if self.weighted_model_error < self.weighted_lowest_model_error:
            self.weighted_lowest_model_error = self.weighted_model_error
            self.lowest_model_error = self.model_error
            self.lowest_model_error_dict = self.model_error_dict.copy()
            self.best_result = x0.copy()

        return wghtd_cost

    def get_model_data_dict(self, data_agg="both") -> tuple[dict, dict]:
        """
        The `get_model_data_dict` function takes in a data aggregation parameter and returns two
        dictionaries containing simulation data and train/test data.

        Args:
          data_agg: The parameter `data_agg` is used to specify which data to aggregate. It can take one
        of three values:. Defaults to both

        Returns:
          a tuple containing two dictionaries. The first dictionary, `simulation_data_dict`, contains
        the simulation data for each batch, where the keys are the batch names and the values are pandas
        DataFrames with columns representing the states and inputs. The second dictionary,
        `train_test_data_dict`, contains the original train or test data for each batch, where the keys
        are the batch names and the values
        """
        if data_agg == "train":
            data = self.train_data.copy()
        elif data_agg == "test":
            data = self.test_data.copy()
        else:
            data = pd.concat([self.train_data, self.test_data], ignore_index=True)

        columns = self.states + self.inputs
        batch_grouped = data.groupby("Batch")

        simulation_data_dict = {}
        train_test_data_dict = {}
        for name, group in batch_grouped:
            if self.hidden_state:
                x0_matrix = np.append(np.array(group.filter(self.states).iloc[0, :]), 0)
            else:
                x0_matrix = np.array(group.filter(self.states).iloc[0, :])

            u_matrix = np.array(group.filter(self.inputs))
            time = np.arange(0, len(u_matrix), 1)

            if self.hidden_state:
                bioreactor = signal.StateSpace(
                    self.augmented_a_matrix,
                    self.augmented_b_matrix,
                    self.c_matrix,
                    self.d_matrix,
                )
            else:
                bioreactor = signal.StateSpace(
                    self.a_matrix, self.b_matrix, self.c_matrix, self.d_matrix
                )

            _, y_out, _ = signal.lsim(
                system=bioreactor, U=u_matrix, T=time, X0=x0_matrix, interp=False
            )
            simulation_data = pd.DataFrame(
                data=np.array(
                    self.scaler.inverse_transform(np.hstack((y_out, u_matrix)))
                ),
                columns=self.scaler.get_feature_names_out(),
            )
            simulation_data_dict[name] = pd.DataFrame(
                data=simulation_data, columns=columns
            )
            group[self.states + self.inputs] = self.scaler.inverse_transform(
                group.filter(items=self.states + self.inputs)
            )
            train_test_data_dict[name] = group

        return simulation_data_dict, train_test_data_dict

    def plot_test_data(self, test_label: str, ylim=True):
        """
        The function `plot_test_data` plots simulated and experimental data for a given test label.

        Args:
          test_label (str): The `test_label` parameter is a string that represents the label of the data
        to be plotted. It is used to access the specific data from the `simulation_dict` and
        `train_test_dict` dictionaries.
          ylim: The `ylim` parameter is used to set the y-axis limits for the plots. If `ylim` is not
        specified (i.e., `None`), the y-axis limits will be automatically determined based on the data.
        If `ylim` is specified, the y-axis limits will be set to
        """
        cols = 4
        simulation_dict, train_test_dict = self.get_model_data_dict(data_agg="test")
        rows = math.floor(len(simulation_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(9, 7), squeeze=False)
        fig.subplots_adjust(top=0.8)
        dict_keys = list(simulation_dict.keys())
        max_val_sim = max(
            chain.from_iterable(df[test_label] for df in simulation_dict.values())
        )
        max_val_exp = max(
            chain.from_iterable(df[test_label] for df in train_test_dict.values())
        )
        min_val_sim = min(
            chain.from_iterable(df[test_label] for df in train_test_dict.values())
        )
        min_val_exp = min(
            chain.from_iterable(df[test_label] for df in train_test_dict.values())
        )
        max_value = max(max_val_sim, max_val_exp)
        min_value = min(min_val_sim, min_val_exp)
        for count, ax_test in enumerate(axes.reshape(-1)):
            if len(dict_keys) == 1:
                key = dict_keys[0]
            else:
                key = dict_keys[count]
            time = np.arange(0, len(simulation_dict[key][test_label]), 1)
            ax_test.plot(
                time,
                simulation_dict[key][test_label],
                "ro-",
                label="Simulated Data",
                markersize=3.5,
            )
            ax_test.plot(
                time,
                train_test_dict[key][test_label],
                "bo-",
                label="Experimental Data",
                markersize=3.5,
            )
            ax_test.set_title(key, size="medium", weight="bold")
            if ylim:
                if min_value > 200:
                    ax_test.set_ylim(
                        min_value - (min_value * 0.2),
                        max_value + (max_value * 0.2),
                    )
                else:
                    ax_test.set_ylim(0, max_value + (max_value * 0.2))
            # if ylim is not None:
            #     ax_test.set_ylim(0, ylim)
        fig.suptitle("Testing Data Set", size="x-large", weight="bold", y=0.98)
        fig.supxlabel("Day", size="x-large", weight="bold")
        fig.supylabel(f"{test_label}", size="x-large", weight="bold")
        fig.tight_layout()
        plt.legend(loc="best")
        plt.show()

    def plot_train_data(self, test_label: str, ylim=True, random_plots=False):
        """
        The function `plot_train_data` plots simulated and experimental data, as well as a parity plot
        comparing the two.

        Args:
          test_label (str): The `test_label` parameter is a string that represents the label or variable
        you want to plot in the graphs. It is used to access the corresponding data in the
        `simulation_dict` and `train_test_dict` dictionaries.
          ylim: The `ylim` parameter is used to set the y-axis limits for the plots. If `ylim` is not
        specified, the y-axis limits will be automatically determined based on the data. If `ylim` is
        specified, the y-axis limits will be set to the specified values.
        """
        cols = 4
        simulation_dict, train_test_dict = self.get_model_data_dict(data_agg="train")
        max_val_sim = max(
            chain.from_iterable(df[test_label] for df in simulation_dict.values())
        )
        max_val_exp = max(
            chain.from_iterable(df[test_label] for df in train_test_dict.values())
        )
        min_val_sim = min(
            chain.from_iterable(df[test_label] for df in train_test_dict.values())
        )
        min_val_exp = min(
            chain.from_iterable(df[test_label] for df in train_test_dict.values())
        )
        max_value = max(max_val_sim, max_val_exp)
        min_value = min(min_val_sim, min_val_exp)

        if len(simulation_dict) > 15:
            rows = math.floor(15 / cols)
        else:
            rows = math.floor(len(simulation_dict) / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(10, 8),
            squeeze=False,  # sharex=True, sharey=True
        )
        fig.subplots_adjust(top=0.8)

        fig2, axes2 = plt.subplots(
            rows,
            cols,
            figsize=(10, 8),
            squeeze=False,  # sharex=True, sharey=True
        )
        fig2.subplots_adjust(top=0.8)

        dict_keys = list(simulation_dict.keys())
        random_nums = [
            random.randint(0, len(dict_keys)) for _, _ in enumerate(dict_keys)
        ]

        for count, ax_test in enumerate(axes.reshape(-1)):
            if random_plots:
                key = dict_keys[random_nums[count]]
            else:
                key = dict_keys[count]
            time = np.arange(0, len(simulation_dict[key][test_label]), 1)
            ax_test.plot(
                time,
                simulation_dict[key][test_label],
                "ro-",
                label="Simulated Data",
                markersize=3.5,
            )
            ax_test.plot(
                time,
                train_test_dict[key][test_label],
                "bo-",
                label="Experimental Data",
                markersize=3.5,
            )
            ax_test.set_title(key, size="medium", weight="bold")
            ax_test.grid()
            if ylim:
                if min_value > 200:
                    ax_test.set_ylim(
                        min_value - (min_value * 0.2),
                        max_value + (max_value * 0.2),
                    )
                else:
                    ax_test.set_ylim(0, max_value + (max_value * 0.2))
            # if ylim is not None:
            #     ax_test.set_ylim(0, ylim)

        axes[rows - 1][cols - 1].legend()
        fig.suptitle("Training Data Set", size="x-large", weight="bold", y=0.98)
        fig.supxlabel("Day", size="x-large", weight="bold")
        fig.supylabel(f"{test_label}", size="x-large", weight="bold")
        fig.tight_layout()

        for count, ax_test in enumerate(axes2.reshape(-1)):
            if random_plots:
                key = dict_keys[random_nums[count]]
            else:
                key = dict_keys[count]
            ax_test.plot(
                train_test_dict[key][test_label],
                simulation_dict[key][test_label],
                "bo",
                label="Simulated Data",
                markersize=4.5,
            )
            ax_test.set_title(key, size="medium", weight="bold")
            ax_test.axline((0, 0), slope=1, color="black")
            if ylim:
                if min_value > 200:
                    ax_test.set_ylim(
                        min_value - (min_value * 0.2),
                        max_value + (max_value * 0.2),
                    )
                else:
                    ax_test.set_ylim(0, max_value + (max_value * 0.2))
            # if ylim is not None:
            #     ax_test.set_ylim(0, ylim)
            #     ax_test.set_xlim(0, ylim)

        plt.legend()
        fig2.supxlabel("Measurement", size="x-large", weight="bold")
        fig2.supylabel("Prediction", size="x-large", weight="bold")
        fig2.suptitle("Parity Plot", size="x-large", weight="bold", y=0.98)
        fig2.tight_layout()
        plt.show()

    def get_rmse_table(self) -> pd.DataFrame:
        """
        The function `get_rmse_table` calculates the root mean square error (RMSE) for each state in
        each batch of data and returns a DataFrame with the results.

        Returns:
          a pandas DataFrame object containing the root mean square error (RMSE) values for each batch
        and state. The DataFrame has columns "Batch" and the names of the states, and the index is
        reset.
        """
        simulation_dict, train_test_dict = self.get_model_data_dict(data_agg="both")
        rmse_dict = {}
        for batch, _ in train_test_dict.items():
            state_rmse = []
            for state in self.states:
                y_true = np.array(train_test_dict[batch][state], dtype=np.float64)
                y_pred = np.array(simulation_dict[batch][state], dtype=np.float64)
                rmse = root_mean_squared_error(y_true, y_pred)
                # rmse = np.sqrt((y_true - y_pred) ** 2).sum() / len(
                #     y_true
                # )
                state_rmse.append(rmse)
            rmse_dict[batch] = state_rmse

        columns = self.states
        df_rmse = pd.DataFrame.from_dict(
            rmse_dict, columns=columns, orient="index"
        ).reset_index()
        return df_rmse

    def get_r2_table(self) -> pd.DataFrame:
        """
        The function `get_r2_table` calculates the R-squared values for a given set of true and
        predicted values and returns them in a pandas DataFrame.

        Returns:
          a pandas DataFrame object, df_r2.
        """
        simulation_dict, train_test_dict = self.get_model_data_dict(data_agg="both")
        r2_dict = {}
        for batch, _ in train_test_dict.items():
            state_r2 = []
            for state in self.states:
                y_true = np.array(train_test_dict[batch][state], dtype=np.float64)
                y_pred = np.array(simulation_dict[batch][state], dtype=np.float64)
                r2 = r2_score(y_true, y_pred)
                state_r2.append(r2)
            r2_dict[batch] = state_r2

        columns = self.states
        df_r2 = pd.DataFrame.from_dict(
            r2_dict, columns=columns, orient="index"
        ).reset_index()
        return df_r2

    def get_corrcoef_table(self) -> pd.DataFrame:
        """
        The function `get_corrcoef_table` calculates the correlation coefficient between predicted and
        true values for each state in each batch of data and returns the results in a pandas DataFrame.

        Returns:
          a pandas DataFrame object, df_corr, which contains the correlation coefficients between the
        predicted and true values for each state in each batch. The DataFrame has columns for "Batch"
        and each state, and the index is reset to be the default integer index.
        """
        simulation_dict, train_test_dict = self.get_model_data_dict(data_agg="both")
        corr_dict = {}
        for batch, _ in train_test_dict.items():
            state_corr = []
            for state in self.states:
                y_true = np.array(train_test_dict[batch][state], dtype=np.float64)
                y_pred = np.array(simulation_dict[batch][state], dtype=np.float64)
                corr = (np.corrcoef(np.vstack([y_pred, y_true]))) ** 2
                state_corr.append(corr[0, 1])
            corr_dict[batch] = state_corr

        columns = self.states
        df_corr = pd.DataFrame.from_dict(
            corr_dict, columns=columns, orient="index"
        ).reset_index()
        return df_corr

    def train_test_model(
        self,
        save_path,
        test_label: str,
        first_train=True,
        iterations=10,
        basin_temp: int = 2000,
    ):
        """
        The function trains a model, saves it to a specified path, and then plots test data using the
        trained model.

        Args:
          save_path: The save_path parameter is the file path where the trained model will be saved.
          test_label (str): A string that represents the label or name of the test data.
          first_train: A boolean value indicating whether it is the first time training the model or
        not. Defaults to True
          iterations: The "iterations" parameter specifies the number of training iterations or epochs
        to perform during the training process. Each iteration involves feeding the training data
        through the model, calculating the loss, and updating the model's parameters based on the loss.
        Increasing the number of iterations can potentially improve the model's performance, but.
        Defaults to 10
        """
        self.train_model(
            save_path=save_path,
            first_train=first_train,
            iterations=iterations,
            basin_temp=basin_temp,
        )

        self.plot_test_data(
            test_label=test_label,
        )

        self.plot_train_data(
            test_label=test_label,
        )

    def single_batch_test(self, test_label):
        """
        The function `single_batch_test` plots simulated and actual data for a given test label using
        matplotlib.

        Args:
          test_label: The `test_label` parameter is a string that represents the label or variable you
        want to plot in the graph. It is used to access the corresponding data in the `simulation_dict`
        and `train_test_dict` dictionaries.
        """
        cols = 1
        rows = 1
        simulation_dict, train_test_dict = self.get_model_data_dict(data_agg="both")
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10), squeeze=False)
        dict_keys = list(simulation_dict.keys())
        key = dict_keys[0]
        time = np.arange(0, len(simulation_dict[key][test_label]), 1)
        axes[0][0].plot(
            time,
            simulation_dict[key][test_label],
            "ro-",
            label="Simulated",
            markersize=3.5,
        )
        axes[0][0].plot(
            time,
            train_test_dict[key][test_label],
            "bo-",
            label="Actual",
            markersize=3.5,
        )
        axes[0][0].set_title(key)
        plt.legend(loc="best")
        fig.tight_layout()
        plt.show()

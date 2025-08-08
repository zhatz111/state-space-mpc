"""Class for setting up MPC simulations
    Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2025-08-07
    Modified: 2025-08-08
"""

# Standard library imports
import math
import random
import warnings
from typing import Union
from datetime import datetime
from pathlib import Path

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
class ModelSimulations:
    """_summary_"""

    random.seed(10)

    def __init__(
        self,
        simulation_data: pd.DataFrame,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        states: list[str],
        inputs: list[str],
        num_days: int,
        scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
        hidden_state: bool = False,
        rho: float = 0.5,
        af_col: np.ndarray = np.array([]),
        af_row: np.ndarray = np.array([]),
        bf_row: np.ndarray = np.array([])
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
        self.simulation_data = simulation_data
        self.states = states
        self.inputs = inputs
        self.num_days = num_days
        self.scaler = scaler
        self.state_len = len(states)
        self.input_len = len(inputs)
        self.total = self.state_len + self.input_len
        self.hidden_state = hidden_state

        # store training error data
        self.iters = 0
        self.model_error = 0
        self.lowest_model_error = np.inf
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
            self.rho = rho                                      # feed effect decay rate
            if af_col.shape[0] == self.state_len:
                self.af_col = np.array(af_col)
            else:
                self.af_col = np.ones([self.state_len, 1])          # learnable hidden state

            if af_row.shape[0] == self.state_len:
                self.af_row = np.array(af_row)
            else:
                self.af_row = np.ones([self.state_len, 1])          # learnable hidden state
            
            if bf_row.shape[0] == self.input_len:
                self.bf_row = bf_row
            else:
                self.bf_row = np.zeros([1, self.input_len])
                self.bf_row[:, 1] = 1.0

            self.augmented_a_matrix = np.zeros([self.state_len+1, self.state_len+1])
            self.augmented_a_matrix[:self.state_len, :self.state_len] = self.a_matrix
            self.augmented_a_matrix[:self.state_len, self.state_len] = self.af_col.flatten()
            self.augmented_a_matrix[self.state_len, :self.state_len] = self.af_row.flatten()
            self.augmented_a_matrix[self.state_len, self.state_len] = self.rho

            self.augmented_b_matrix = np.zeros([self.state_len+1, self.input_len])
            self.augmented_b_matrix[:self.state_len, :] = self.b_matrix
            self.augmented_b_matrix[self.state_len, :] = self.bf_row     # only feed (input index 0) affects feed-effect state (hidden state)

            self.c_matrix = np.hstack([np.identity(self.state_len), np.zeros([self.state_len, 1])])
    
    def get_simulated_data(self, save_path: Path) -> dict:
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
        data = self.simulation_data.copy()

        columns = self.states + self.inputs
        batch_grouped = data.groupby("Batch")

        simulation_data_dict = {}
        for name, group in batch_grouped:

            if self.hidden_state:
                x0_matrix = np.append(np.array(group.filter(self.states).iloc[0, :]), 0)
            else:
                x0_matrix = np.array(group.filter(self.states).iloc[0, :])

            u_matrix = np.array(group.filter(self.inputs))
            time = np.arange(0, len(u_matrix), 1)

            if self.hidden_state:
                bioreactor = signal.StateSpace(
                    self.augmented_a_matrix, self.augmented_b_matrix, self.c_matrix, self.d_matrix
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
            simulation_data["Day"] = time
            simulation_data_dict[name] = pd.DataFrame(
                data=simulation_data, columns=columns
            )
            group[self.states + self.inputs] = self.scaler.inverse_transform(
                group.filter(items=self.states + self.inputs)
            )
        
        df_sim = pd.concat(simulation_data_dict).reset_index()
        df_sim.to_csv(save_path)

        return simulation_data_dict

    def plot_simulation_data(self, simulation_dict: dict, target_label: str, ylim=None, random_plots=False):
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
            time = np.arange(0, len(simulation_dict[key][target_label]), 1)
            ax_test.plot(
                time,
                simulation_dict[key][target_label],
                "ro-",
                label="Simulated Data",
                markersize=3.5,
            )
            ax_test.set_title(key, size="medium", weight="bold")
            ax_test.grid()
            if ylim is not None:
                ax_test.set_ylim(0, ylim)

        axes[rows - 1][cols - 1].legend()
        fig.suptitle("Training Data Set", size="x-large", weight="bold", y=0.98)
        fig.supxlabel("Day", size="x-large", weight="bold")
        fig.supylabel(f"{target_label}", size="x-large", weight="bold")
        fig.tight_layout()
        plt.show()
    
    def simulate(self, file_save_path, target_label, ylim=None, random_plots=False):

        simulation_dict = self.get_simulated_data(save_path=file_save_path)
        self.plot_simulation_data(
            simulation_dict=simulation_dict,
            target_label=target_label,
            ylim=ylim,
            random_plots=random_plots
        )

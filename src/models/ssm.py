"""_summary_
"""
# Standard library imports
from typing import Union
from pathlib import Path

# Third party library imports
import json
import numpy as np
from scipy.signal import lsim, StateSpace
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create type hint for the scaler object being passed to SSM class
ScalerType = Union[MinMaxScaler, StandardScaler]


def scaler_tojson(scaler: MinMaxScaler, save_path: Union[str, Path]):
    """
    The function `scaler_tojson` takes a scaler object and saves its attributes to a JSON file.
    
    Args:
      scaler: The `scaler` parameter is an instance of a scaler object. It could be any scaler object
    from a machine learning library, such as `StandardScaler` from scikit-learn. The scaler object is
    used to scale or normalize data.
      save_path: The `save_path` parameter is the file path where the JSON file will be saved. It should
    include the file name and extension. For example, if you want to save the JSON file as
    "scaler_attributes.json" in the current directory, you can set `save_path` as "sc
    """
    # Prepare a dictionary to hold the scaler's attributes
    scaler_attributes = {
        attr_name: getattr(scaler, attr_name).tolist()
        if hasattr(getattr(scaler, attr_name), "tolist")
        else getattr(scaler, attr_name)
        for attr_name in vars(scaler)
    }

    # Save the attributes to a JSON file
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(scaler_attributes, file, indent=4)


def json_toscaler(json_file: Union[str, Path], minmaxscaler=True):
    """
    The function `json_toscaler` takes a JSON file containing attributes of a scaler and reconstructs
    the scaler object using the loaded attributes.
    
    Args:
      json_file: The path to the JSON file that contains the attributes of the scaler object.
      minmaxscaler: The `minmaxscaler` parameter is a boolean flag that determines whether to use the
    `MinMaxScaler` class for scaling the data. If `minmaxscaler` is set to `True`, the function will use
    `MinMaxScaler` for scaling the data. If it is set to `. Defaults to True
    
    Returns:
      a reconstructed scaler object.
    """
    # Load the attributes from the JSON file
    with open(json_file, "r", encoding="utf-8") as file:
        loaded_attributes = json.load(file)

    # Initialize a new MinMaxScaler instance
    if minmaxscaler:
        reconstructed_scaler = MinMaxScaler()
    else:
        raise ValueError("This method currently only works for the MinMaxScaler")

    # Set the loaded attributes back to the scaler
    for attr_name, attr_value in loaded_attributes.items():
        setattr(
            reconstructed_scaler,
            attr_name,
            np.array(attr_value) if isinstance(attr_value, list) else attr_value,
        )

    return reconstructed_scaler


class StateSpaceModel:
    """
    The `StateSpaceModel` class represents a mathematical model of a system in state space form. It is
    used to initialize an object with the given states, inputs, scaler, a_matrix, and b_matrix. The
    `ssm_lsim` method is used to predict a trajectory based on the state space model using initial
    conditions, inputs, and time.
    """

    def __init__(
        self,
        states: list[str],
        inputs: list[str],
        scaler: ScalerType,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        name="",
    ):
        """
        The function initializes an object with given states, inputs, scaler, a_matrix, and b_matrix
        .

        Args:
            states (list[str]): The states parameter is a list of strings representing the possible states of
        a system.
            inputs (list[str]): The `inputs` parameter is a list of strings that represents the possible input
        values for the system. These input values can be used to control or influence the behavior of the
        system.
            scaler (object): The `scaler` parameter is an object that is used to scale the input data. It is
        typically used to normalize or standardize the input values before feeding them into the model. The
        specific implementation of the scaler object will depend on the library or framework being used.
            a_matrix (np.ndarray): The `a_matrix` parameter is a numpy array representing the transition
        probabilities between states in a Markov chain. Each element `a[i][j]` represents the probability of
        transitioning from state `i` to state `j`. The shape of the array should be `(num_states,
        num_states)
            b_matrix (np.ndarray): The `b_matrix` parameter is a numpy array representing the output matrix of
        a system. It defines the relationship between the inputs and the outputs of the system. Each row of
        the matrix corresponds to a state of the system, and each column corresponds to an input.
            name (str): The `name` parameter is a str (no name by default) to identify the model
        """
        self.states = states
        self.inputs = inputs
        self.scaler = scaler
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.c_matrix = np.identity(len(states))
        self.d_matrix = np.zeros([len(states), len(inputs)])
        self.name = name

        data_suffix = "--STATE_DATA"
        xhat_suffix = "--STATE_EST"
        yhat_suffix = "--STATE_PRED"
        p_suffix = "--STATE_MOD"
        input_suffix = "--INPUT_DATA"

        self.state_data_labels = [x + data_suffix for x in self.states]
        self.state_est_labels = [x + xhat_suffix for x in self.states]
        self.state_pred_labels = [x + yhat_suffix for x in self.states]
        self.state_mod_labels = [x + p_suffix for x in self.states]
        self.input_data_labels = [x + input_suffix for x in self.inputs]

    def ssm_lsim(
        self,
        initial_state: np.ndarray,
        input_matrix: np.ndarray,
        time: np.ndarray,
        output_mods=np.array([]),
    ):
        """
        Created by ZH (zach.a.hatzenbeller@gsk.com)
        Created: 2023-10-13
        Modified: 2023-10-13

        The function `ssm_lsim` predicts a trajectory based on a state space model using initial
        conditions, inputs, and time.

        Args:
          X0: The parameter X0 represents the initial condition matrix for the state variables of the
        system. It can be either a 1-dimensional or 2-dimensional array. If it is 1-dimensional, it
        should be reshaped to a row vector with shape (1, n), where n is the number of
          U: The parameter U represents the input matrix. It is used to provide input values to the
        system being modeled. The function expects U to be a 1-dimensional or 2-dimensional array. If U
        is 1-dimensional, it will be reshaped to a 2-dimensional array with one row. If
          time: The `time` parameter is a 1D array or list that represents the time points at which the
        trajectory should be predicted. It specifies the time intervals at which the system's states
        should be calculated.

        Returns:
          the predicted trajectory of the system, represented by the variable xHat.
        """
        # Warning: Remember that the time array needs to start at 0 with the initial
        # condition at time 0

        # Check if X0 is 1d or 2d and reshape accordingly
        if initial_state.ndim == 1:
            x_row = initial_state.reshape(1, -1)
        elif initial_state.ndim == 2:
            x_row = initial_state[0, :]
        else:
            raise ValueError(
                "Initial condition matrix X0 must have at least 1 dimension"
            )

        # Default vector of ones
        if output_mods.size == 0:
            output_mods = np.ones((1, len(self.states)))

        # Check if U is 1d or 2d and reshape accordingly
        if input_matrix.ndim == 1:
            u_row = input_matrix.reshape(1, -1)
        elif input_matrix.ndim == 2:
            u_row = input_matrix
        else:
            raise ValueError("Input matrix U must have at least 1 dimension")

        # Check to see if X and U have the same # of rows and then scale them both
        # by horizontally stacking them together or creating seperate zeros matrices
        # for both X and U to get them in the correct data shape for transform
        if x_row.shape[0] == u_row.shape[0]:
            xu_row = np.hstack((x_row, u_row))
            xu_scaled = np.array(self.scaler.transform(xu_row))
            x_scaled = xu_scaled[:, : x_row.shape[1]]
            u_scaled = xu_scaled[:, x_row.shape[1] :]
        else:
            x_row_reshape = x_row.reshape(1, -1)
            xu_mask = np.hstack(
                (x_row_reshape, np.zeros((x_row.shape[0], u_row.shape[1])))
            )
            xu_mask_scaled = np.array(self.scaler.transform(xu_mask))
            ux_mask = np.hstack((np.zeros((u_row.shape[0], x_row.shape[1])), u_row))
            ux_mask_scaled = np.array(self.scaler.transform(ux_mask))
            x_scaled = xu_mask_scaled[:, : x_row.shape[1]]
            u_scaled = ux_mask_scaled[:, x_row.shape[1] :]

        # Predict the next days states based on a continuous system using lsim
        bioreactor = StateSpace(
            self.a_matrix, self.b_matrix, self.c_matrix, self.d_matrix
        )
        _, y_out, _ = lsim(bioreactor, U=u_scaled, T=time, X0=x_scaled)

        # Reshape the output, if X and U were the same initial shape, to a row matrix
        if x_row.shape[0] == u_row.shape[0]:
            xuhat_scaled = np.hstack((y_out.reshape(1, -1), u_row))
        else:
            xuhat_scaled = np.hstack((y_out, u_row))

        # Inverse transform the output to get the unscaled values for next time step
        x_hat = np.array(self.scaler.inverse_transform(xuhat_scaled))[
            :, : x_row.shape[1]
        ]

        # Modify measurement based on the correction factor delta_p (diagonal of the C matrix)
        # delta_p_matrix = np.tile(output_mods,(x_hat.shape[0],1))
        y_hat = np.multiply(x_hat, output_mods)

        return x_hat, y_hat

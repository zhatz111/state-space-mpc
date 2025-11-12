"""
Class for state space process model and simulation function
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2025-08-27
"""

# Standard library imports
from typing import Union

# Third party library imports
import numpy as np
from numpy import atleast_1d, squeeze, zeros, dot, transpose
from scipy.signal import lsim, StateSpace, lti, dlti
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create type hint for the scaler object being passed to SSM class
ScalerType = Union[MinMaxScaler, StandardScaler]


def lsim_mod(system, U, T, X0=None, interp=False, output_mods_scaled=0):
    """
    This function is identical to scipy.signal.lsim except the offset to error is applied at each time step
    """
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError("lsim can only be used with continuous-time systems.")
    else:
        sys = lti(*system)._as_ss()
    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")

    A, B, C, D = map(np.asarray, (sys.A, sys.B, sys.C, sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]

    n_steps = T.size
    if X0 is None:
        X0 = zeros(n_states, sys.A.dtype)
    xout = np.empty((n_steps, n_states), sys.A.dtype)

    if T[0] == 0:
        xout[0] = X0
    elif T[0] > 0:
        # step forward to initial time, with zero input
        xout[0] = dot(X0, linalg.expm(transpose(A) * T[0]))
    else:
        raise ValueError("Initial time must be nonnegative")

    no_input = U is None or (isinstance(U, (int, float)) and U == 0.0) or not np.any(U)

    if n_steps == 1:
        yout = squeeze(dot(xout, transpose(C)))
        if not no_input:
            yout += squeeze(dot(U, transpose(D)))
        return T, squeeze(yout), squeeze(xout)

    dt = T[1] - T[0]
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Time steps are not equally spaced.")

    if no_input:
        # Zero input: just use matrix exponential
        # take transpose because state is a row vector
        expAT_dt = linalg.expm(transpose(A) * dt)
        for i in range(1, n_steps):
            xout[i] = np.clip(dot(xout[i - 1], expAT_dt) + output_mods_scaled, a_min=0, a_max=None)
        yout = squeeze(dot(xout, transpose(C)))
        return T, squeeze(yout), squeeze(xout)

    # Nonzero input
    U = atleast_1d(U)
    if U.ndim == 1:
        U = U[:, np.newaxis]

    if U.shape[0] != n_steps:
        raise ValueError("U must have the same number of rows as elements in T.")

    if U.shape[1] != n_inputs:
        raise ValueError("System does not define that many inputs.")

    if not interp:
        # Zero-order hold
        # Algorithm: to integrate from time 0 to time dt, we solve
        #   xdot = A x + B u,  x(0) = x0
        #   udot = 0,          u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt   B*dt ] [ x0 ]
        #   [ u(dt) ] = exp [  0     0    ] [ u0 ]
        M = np.vstack(
            [np.hstack([A * dt, B * dt]), np.zeros((n_inputs, n_states + n_inputs))]
        )
        # transpose everything because the state and input are row vectors
        expMT = linalg.expm(transpose(M))
        Ad = expMT[:n_states, :n_states]
        Bd = expMT[n_states:, :n_states]
        for i in range(1, n_steps):
            xout[i] = np.clip(dot(xout[i - 1], Ad) + dot(U[i - 1], Bd) + output_mods_scaled, a_min=0, a_max=None)
    else:
        # Linear interpolation between steps
        # Algorithm: to integrate from time 0 to time dt, with linear
        # interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
        #   xdot = A x + B u,        x(0) = x0
        #   udot = (u1 - u0) / dt,   u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
        #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        M = np.vstack(
            [
                np.hstack([A * dt, B * dt, np.zeros((n_states, n_inputs))]),
                np.hstack(
                    [np.zeros((n_inputs, n_states + n_inputs)), np.identity(n_inputs)]
                ),
                np.zeros((n_inputs, n_states + 2 * n_inputs)),
            ]
        )
        expMT = linalg.expm(transpose(M))
        Ad = expMT[:n_states, :n_states]
        Bd1 = expMT[n_states + n_inputs :, :n_states]
        Bd0 = expMT[n_states : n_states + n_inputs, :n_states] - Bd1
        for i in range(1, n_steps):
            xout[i] = np.clip((
                dot(xout[i - 1], Ad) + dot(U[i - 1], Bd0) + dot(U[i], Bd1)
            ) + output_mods_scaled, a_min=0, a_max=None)

    yout = squeeze(dot(xout, transpose(C))) + squeeze(dot(U, transpose(D)))
    return T, squeeze(yout), squeeze(xout)


class StateSpaceModel:
    """
    The `StateSpaceModel` class represents a mathematical model of a system in state space form. It is
    used to initialize an object with the given states, inputs, scaler, a_matrix, and b_matrix. The
    `ssm_lsim` method is used to predict a trajectory based on the state space model using initial
    conditions, inputs, and time.
    """

    def __init__(
        self,
        model_parameters: dict,
        scaler: MinMaxScaler,
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
        # dictionary that contains all info regarding the model
        self.model_parameters = model_parameters
        self.scaler = scaler

        self.states = [x.upper() for x in self.model_parameters["Model States"]]
        self.inputs = [x.upper() for x in self.model_parameters["Model Inputs"]]

        self.a_matrix = np.array(self.model_parameters["a_matrix"])
        self.b_matrix = np.array(self.model_parameters["b_matrix"])
        self.af_col_matrix = np.array(self.model_parameters["af_col"])
        self.af_row_matrix = np.array(self.model_parameters["af_row"])
        self.bf_row_matrix = np.array(self.model_parameters["bf_row"])
        self.rho = self.model_parameters["rho"]

        self.c_matrix = np.identity(len(self.states))
        self.d_matrix = np.zeros([len(self.states), len(self.inputs)])
        self.name = self.model_parameters["Asset"]
        self.hidden_state = self.model_parameters["Hidden State"]

        self.data_suffix = "--STATE_DATA"
        self.data_sp_suffix = "--STATE_SP"
        self.xhat_suffix = "--STATE_EST"
        self.yhat_suffix = "--STATE_PRED"
        self.p_suffix = "--STATE_MOD"
        self.input_suffix = "--INPUT_DATA"
        self.input_ref_suffix = "--INPUT_REF"

        self.state_data_labels = [x + self.data_suffix for x in self.states]
        self.state_est_labels = [x + self.xhat_suffix for x in self.states]
        self.state_pred_labels = [x + self.yhat_suffix for x in self.states]
        self.state_mod_labels = [x + self.p_suffix for x in self.states]
        self.input_data_labels = [x + self.input_suffix for x in self.inputs]

        # augmented matrices if using feed hidden state
        if self.hidden_state:
            self.state_len = self.a_matrix.shape[0]
            self.input_len = self.b_matrix.shape[1]

            if self.rho is None:
                raise ValueError("rho value is none in model parameters")

            if self.af_col_matrix.shape[0] != self.state_len:
                raise ValueError("af_col matrix does not match length of states")

            if self.af_row_matrix.shape[0] != self.state_len:
                raise ValueError("af_row matrix does not match length of states")

            if self.bf_row_matrix.shape[0] != self.input_len:
                raise ValueError("bf_row matrix does not match length of inputs")

            self.augmented_a_matrix = np.zeros([self.state_len + 1, self.state_len + 1])
            self.augmented_a_matrix[: self.state_len, : self.state_len] = self.a_matrix
            self.augmented_a_matrix[: self.state_len, self.state_len] = (
                self.af_col_matrix.flatten()
            )
            self.augmented_a_matrix[self.state_len, : self.state_len] = (
                self.af_row_matrix.flatten()
            )
            self.augmented_a_matrix[self.state_len, self.state_len] = self.rho
            self.a_matrix = self.augmented_a_matrix

            self.augmented_b_matrix = np.zeros([self.state_len + 1, self.input_len])
            self.augmented_b_matrix[: self.state_len, :] = self.b_matrix
            self.augmented_b_matrix[self.state_len, :] = (
                self.bf_row_matrix
            )  # only feed (input index 0) affects feed-effect state (hidden state)
            self.b_matrix = self.augmented_b_matrix

            self.c_matrix = np.hstack(
                [np.identity(self.state_len), np.zeros([self.state_len, 1])]
            )

    def ssm_lsim(
        self,
        initial_state: np.ndarray,
        input_matrix: np.ndarray,
        time: np.ndarray,
        output_mods=np.array([]),
        hidden_state: bool = False
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
        
        # Scale offset
        if output_mods.size == 0:
            # Default no offset
            output_mods_scaled = np.zeros((1, len(self.states)))
        else:
            # Use only the scale and not the min/mean for offset
            x_scales = self.scaler.scale_[: len(self.states)]
            output_mods_scaled = np.multiply(output_mods, x_scales)
        
        if hidden_state:
            x_scaled = np.append(x_scaled, 0)
            output_mods_scaled = np.append(output_mods_scaled, 0)

        # Predict the next days states based on a continuous system using lsim
        bioreactor = StateSpace(
            self.a_matrix, self.b_matrix, self.c_matrix, self.d_matrix
        )
        _, y_out, _ = lsim_mod(
            bioreactor,
            U=u_scaled,
            T=time,
            interp=False,
            X0=x_scaled,
            output_mods_scaled=output_mods_scaled,
        )

        # Reshape the output, if X and U were the same initial shape, to a row matrix
        if x_row.shape[0] == u_row.shape[0]:
            xuhat_scaled = np.hstack((y_out.reshape(1, -1), u_row))
        else:
            xuhat_scaled = np.hstack((y_out, u_row))

        # Inverse transform the output to get the unscaled values for next time step
        x_hat = np.array(self.scaler.inverse_transform(xuhat_scaled))[
            :, : x_row.shape[1]
        ]

        # Output = state in the new offset scheme
        y_hat = x_hat

        return x_hat, y_hat

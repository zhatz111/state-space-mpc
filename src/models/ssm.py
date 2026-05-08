"""
Class for state space process model and simulation function
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2026-05-08
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


def _compute_bolus_input(t_eval, bolus_events, tau):
    """Reconstruct exponentially-decayed bolus signal. See data/functions.py for full docstring."""
    result = np.zeros(len(t_eval))
    for t_b, v in bolus_events:
        mask = t_eval >= t_b
        result[mask] += v * np.exp(-tau * (t_eval[mask] - t_b))
    return result


class StateSpaceModel:
    """
    The `StateSpaceModel` class represents a mathematical model of a system in state space form.
    Supports both time-invariant and partitioned (time-varying) models. When model_parameters
    contains a 3-D a_matrix (shape [n_partitions, n_states, n_states]) and a "Partitions" key,
    ssm_lsim uses piecewise-ZOH simulation via simulate_tv_zoh.

    Also supports automatic bolus-decay upsampling in production: when "Bolus Decay Tau" is
    present in model_parameters, the bolus input column is reconstructed as an exponentially
    decaying signal at fine intra-day resolution before simulation, matching the representation
    used during model training.
    """

    def __init__(
        self,
        model_parameters: dict,
        scaler: MinMaxScaler,
    ):
        # dictionary that contains all info regarding the model
        self.model_parameters = model_parameters
        self.scaler = scaler

        self.states = [x.upper() for x in self.model_parameters["Model States"]]
        self.inputs = [x.upper() for x in self.model_parameters["Model Inputs"]]

        a_raw = np.array(self.model_parameters["a_matrix"])
        b_raw = np.array(self.model_parameters["b_matrix"])

        # Detect partitioned model: a_matrix has shape (n_partitions, n_states, n_states)
        if a_raw.ndim == 3:
            self.is_partitioned = True
            self.a_matrices = a_raw
            self.b_matrices = b_raw
            # Use partition 0 as the "default" single matrix (for shape checks etc.)
            self.a_matrix = a_raw[0]
            self.b_matrix = b_raw[0]
            partitions_cfg = self.model_parameters.get("Partitions", {})
            self.t_partitions = np.array(
                partitions_cfg.get("time_partitions", [0.0, float(self.model_parameters.get("Process Time", 15))])
            )
        else:
            self.is_partitioned = False
            self.a_matrix = a_raw
            self.b_matrix = b_raw
            self.a_matrices = None
            self.b_matrices = None
            self.t_partitions = None

        self.c_matrix = np.identity(len(self.states))
        self.d_matrix = np.zeros([len(self.states), len(self.inputs)])
        self.name = self.model_parameters["Asset"]
        self.hidden_state = self.model_parameters.get("Hidden State", False)

        # Bolus decay config for production simulation
        self.bolus_tau = self.model_parameters.get("Bolus Decay Tau", None)
        bolus_col_candidates = [
            i for i, inp in enumerate(self.inputs) if "BOLUS" in inp.upper()
        ]
        self.bolus_col_idx = bolus_col_candidates[0] if bolus_col_candidates else None
        # Fine-resolution steps per calendar day used for bolus upsampling (default 10)
        self.bolus_upsample_resolution = int(
            self.model_parameters.get("Bolus Upsample Resolution", 10)
        )

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

        # Optional hidden-state augmentation
        af_col_raw = self.model_parameters.get("af_col", [])
        af_row_raw = self.model_parameters.get("af_row", [])
        bf_row_raw = self.model_parameters.get("bf_row", [])
        self.rho = self.model_parameters.get("rho", None)

        self.af_col_matrix = np.array(af_col_raw) if af_col_raw else np.array([])
        self.af_row_matrix = np.array(af_row_raw) if af_row_raw else np.array([])
        self.bf_row_matrix = np.array(bf_row_raw) if bf_row_raw else np.array([])

        if self.hidden_state:
            state_len = self.a_matrix.shape[0]
            input_len = self.b_matrix.shape[1]

            if self.rho is None:
                raise ValueError("rho value is none in model parameters")
            if self.af_col_matrix.shape[0] != state_len:
                raise ValueError("af_col matrix does not match length of states")
            if self.af_row_matrix.shape[0] != state_len:
                raise ValueError("af_row matrix does not match length of states")
            if self.bf_row_matrix.shape[0] != input_len:
                raise ValueError("bf_row matrix does not match length of inputs")

            aug_a = np.zeros([state_len + 1, state_len + 1])
            aug_a[:state_len, :state_len] = self.a_matrix
            aug_a[:state_len, state_len] = self.af_col_matrix.flatten()
            aug_a[state_len, :state_len] = self.af_row_matrix.flatten()
            aug_a[state_len, state_len] = self.rho
            self.a_matrix = aug_a

            aug_b = np.zeros([state_len + 1, input_len])
            aug_b[:state_len, :] = self.b_matrix
            aug_b[state_len, :] = self.bf_row_matrix
            self.b_matrix = aug_b

            self.c_matrix = np.hstack(
                [np.identity(state_len), np.zeros([state_len, 1])]
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsample_with_bolus_decay(self, input_matrix_phys, time):
        """
        Upsample daily inputs to fine intra-day resolution and apply exponential
        decay to the bolus column.  Returns (u_fine, t_fine, daily_indices) where
        daily_indices are the row positions in t_fine that correspond to the
        original daily time points.

        Only called when self.bolus_tau is set and a bolus column exists.
        """
        res = self.bolus_upsample_resolution
        n_days = len(time)
        # Build fine time grid: res sub-steps per interval, with exact daily alignment
        t_fine_segments = []
        for i in range(n_days - 1):
            t_fine_segments.append(
                np.linspace(time[i], time[i + 1], res + 1)[:-1]  # exclude right endpoint
            )
        t_fine_segments.append(np.array([time[-1]]))
        t_fine = np.concatenate(t_fine_segments)

        # Indices in t_fine that correspond to original daily time points
        daily_indices = np.arange(n_days) * res
        daily_indices[-1] = len(t_fine) - 1  # last point

        # Extract bolus events (day, magnitude) where magnitude > 0
        bolus_col = input_matrix_phys[:, self.bolus_col_idx]
        bolus_events = [
            (float(time[i]), float(bolus_col[i]))
            for i in range(n_days)
            if bolus_col[i] > 0
        ]

        # Build fine-resolution input array
        u_fine = np.zeros((len(t_fine), input_matrix_phys.shape[1]))
        for j in range(input_matrix_phys.shape[1]):
            if j == self.bolus_col_idx:
                if bolus_events:
                    u_fine[:, j] = _compute_bolus_input(t_fine, bolus_events, self.bolus_tau)
                # else: stays zero
            else:
                # ZOH: each daily value held until the next day
                for i in range(n_days - 1):
                    start = i * res
                    end = (i + 1) * res
                    u_fine[start:end, j] = input_matrix_phys[i, j]
                u_fine[-1, j] = input_matrix_phys[-1, j]

        return u_fine, t_fine, daily_indices

    # ------------------------------------------------------------------
    # Public simulation entry point
    # ------------------------------------------------------------------

    def ssm_lsim(
        self,
        initial_state: np.ndarray,
        input_matrix: np.ndarray,
        time: np.ndarray,
        output_mods=np.array([]),
        hidden_state: bool = False
    ):
        """
        Predict a trajectory based on the state space model using initial conditions,
        inputs, and time.

        Supports three modes transparently to callers:
          1. Time-invariant: single A/B, uses lsim_mod (ZOH matrix-exponential).
          2. Partitioned (time-varying): 3-D A/B, uses simulate_tv_zoh per partition.
             Time array must be in batch days so partition boundaries are resolved correctly.
          3. Bolus-decay upsampling: when Bolus Decay Tau is set in model_parameters, the
             bolus input column is upsampled to fine intra-day resolution before simulation
             and outputs are sampled back to the original daily grid.  This ensures the input
             representation matches what was used during training.

        Args:
          initial_state: shape (n_states,) or (1, n_states)
          input_matrix: shape (T, n_inputs), physical (un-scaled) units, one row per time step
          time: shape (T,) — batch days (e.g. [3, 4, ..., 14] when simulating from day 3)
          output_mods: optional state offset corrections, shape (1, n_states)
          hidden_state: whether to append hidden feed-effect state (legacy flag)
        """
        # Reshape x0 to row vector
        if initial_state.ndim == 1:
            x_row = initial_state.reshape(1, -1)
        elif initial_state.ndim == 2:
            x_row = initial_state[0:1, :]
        else:
            raise ValueError("Initial condition matrix X0 must have at least 1 dimension")

        # Reshape u to 2D
        if input_matrix.ndim == 1:
            u_row = input_matrix.reshape(1, -1)
        elif input_matrix.ndim == 2:
            u_row = input_matrix
        else:
            raise ValueError("Input matrix U must have at least 1 dimension")

        # Apply bolus-decay upsampling if configured and there is >1 time step
        use_bolus_upsample = (
            self.bolus_tau is not None
            and self.bolus_col_idx is not None
            and len(time) > 1
            and not self.is_partitioned  # TV model already handles fine resolution
        )
        if use_bolus_upsample:
            u_fine_phys, t_fine, daily_indices = self._upsample_with_bolus_decay(u_row, time)
            time_for_sim = t_fine
            u_for_scale = u_fine_phys
        else:
            time_for_sim = time
            u_for_scale = u_row

        # Scale x0 and u
        x_row_1 = x_row[0:1, :]  # always single row for x0
        xu_x_mask = np.hstack((x_row_1, np.zeros((1, u_for_scale.shape[1]))))
        xu_x_scaled = np.array(self.scaler.transform(xu_x_mask))
        x_scaled = xu_x_scaled[0, :x_row.shape[1]]

        ux_u_mask = np.hstack((np.zeros((u_for_scale.shape[0], x_row.shape[1])), u_for_scale))
        ux_u_scaled = np.array(self.scaler.transform(ux_u_mask))
        u_scaled = ux_u_scaled[:, x_row.shape[1]:]

        # Scale offset modifiers
        if output_mods.size == 0:
            output_mods_scaled = np.zeros((1, len(self.states)))
        else:
            x_scales = self.scaler.scale_[: len(self.states)]
            output_mods_scaled = np.multiply(output_mods, x_scales)

        if hidden_state or self.hidden_state:
            x_scaled = np.append(x_scaled, 0)
            output_mods_scaled = np.append(output_mods_scaled, 0)

        # -----------------------------------------------------------
        # Simulate
        # -----------------------------------------------------------
        if self.is_partitioned:
            from models.time_varying_ss import TimeVaryingStateSpace, simulate_tv_zoh
            tv_model = TimeVaryingStateSpace(
                self.t_partitions, self.a_matrices, self.b_matrices
            )
            y_out, _ = simulate_tv_zoh(
                model=tv_model,
                x0=x_scaled,
                u_array=u_scaled,
                time_array=time_for_sim,
                C=self.c_matrix,
            )
        else:
            bioreactor = StateSpace(
                self.a_matrix, self.b_matrix, self.c_matrix, self.d_matrix
            )
            _, y_out, _ = lsim_mod(
                bioreactor,
                U=u_scaled,
                T=time_for_sim,
                interp=False,
                X0=x_scaled,
                output_mods_scaled=output_mods_scaled,
            )

        # -----------------------------------------------------------
        # If bolus upsampled: sample y_out back to original daily grid
        # -----------------------------------------------------------
        if use_bolus_upsample:
            y_out = y_out[daily_indices]
            # Use original daily u for inverse transform
            u_for_invtransform = u_row
        else:
            u_for_invtransform = u_for_scale

        # Inverse transform
        y_out_2d = np.atleast_2d(y_out)
        xuhat_scaled = np.hstack((y_out_2d, u_for_invtransform))
        x_hat = np.array(self.scaler.inverse_transform(xuhat_scaled))[:, : x_row.shape[1]]

        y_hat = x_hat
        return x_hat, y_hat

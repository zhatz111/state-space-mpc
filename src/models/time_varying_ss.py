"""
Time-Varying State-Space Model with Cubic Spline Interpolation

Stores A(t) and B(t) as continuous functions of batch time using cubic splines.
Supports serialization to JSON/YAML so the model can be reconstructed later
without needing scipy — just the spline coefficients and knots.
"""

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from pathlib import Path
from data.functions import json_to_dict
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def simulate_tv_continuous(model, x0, u_array, time_array, C, D=None, method="RK45"):
    """
    Continuous-time simulation for time-varying state-space: ẋ = A(t)x + B(t)u

    Parameters
    ----------
    model : TimeVaryingStateSpace
        Your TV model with get_AB(t).
    x0 : array, shape (n_states,)
    u_array : array, shape (N, n_inputs)
        Input values at each time point.
    time_array : array, shape (N,)
        Time points corresponding to u_array (must be in same scale as model partitions).
    C : array, shape (n_outputs, n_states)
    D : array, shape (n_outputs, n_inputs), optional
    method : str, optional
        ODE solver method. Default "RK45". Use "Radau" or "BDF" if system is stiff.

    Returns
    -------
    y_out : array, shape (N, n_outputs)
    x_out : array, shape (N, n_states)
    """

    # Interpolate inputs so the ODE solver can query u at any t
    u_interp = interp1d(
        time_array,
        u_array,
        axis=0,
        kind="previous",
        bounds_error=False,
        fill_value=(u_array[0], u_array[-1]),  # type: ignore[arg-type]
    )
    # u_interp = u_array

    def dynamics(t, x):
        A, B = model.get_AB(t)
        u = u_interp(t)
        return A @ x + B @ u

    sol = solve_ivp(
        dynamics,
        t_span=(time_array[0], time_array[-1]),
        y0=x0,
        method=method,
        t_eval=time_array,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    x_out = sol.y.T  # shape (N, n_states)

    # Compute outputs y = Cx + Du
    y_out = (C @ x_out.T).T
    if D is not None:
        y_out += (D @ u_array.T).T

    return y_out, x_out


def simulate_tv_zoh(model, x0, u_array, time_array, C, D=None):
    """
    Fast piecewise ZOH simulation for time-varying state-space: ẋ = A(t)x + B(t)u

    Runs scipy.signal.lsim (matrix-exponential / zero-order hold) on each partition
    segment separately, carrying the state vector across partition boundaries.
    Produces virtually identical trajectories to simulate_tv_continuous while being
    50-100x faster — suitable for use inside tight optimisation loops.

    Parameters
    ----------
    model : TimeVaryingStateSpace
    x0 : array, shape (n_states,)
    u_array : array, shape (N, n_inputs)
    time_array : array, shape (N,)
        Must span the same range as model.t_partitions.
    C : array, shape (n_outputs, n_states)
    D : array, shape (n_outputs, n_inputs), optional

    Returns
    -------
    y_out : array, shape (N, n_outputs)
    x_out : None  — omitted for memory efficiency in training loops
    """
    d_mat = np.zeros((C.shape[0], model.n_inputs)) if D is None else D
    x_current = np.array(x0, dtype=float)
    segments_y = []

    for i in range(model.n_partitions):
        t_start = model.t_partitions[i]
        t_end = model.t_partitions[i + 1]

        # All points in [t_start, t_end); final partition includes the endpoint
        if i < model.n_partitions - 1:
            mask = (time_array >= t_start) & (time_array < t_end)
        else:
            mask = time_array >= t_start

        t_seg = time_array[mask]
        u_seg = u_array[mask]

        if len(t_seg) == 0:
            continue

        sys = signal.StateSpace(model.a_matrices[i], model.b_matrices[i], C, d_mat)
        _, y_seg, x_seg = signal.lsim(sys, u_seg, t_seg, x_current, interp=False)

        segments_y.append(y_seg)
        x_current = x_seg[-1]  # carry final state into the next partition

    y_out = (
        np.vstack(segments_y)
        if segments_y
        else np.zeros((len(time_array), C.shape[0]))
    )
    return y_out, None


class TimeVaryingStateSpace:
    """
    LPV state-space model where A(t) and B(t) are cubic spline functions of time.

    Parameters
    ----------
    t_partitions : array-like
        Time points (e.g., batch days) at which matrices were identified.
    a_matrices : ndarray, shape (n_partitions, n_states, n_states)
        Stack of A matrices at each partition.
    b_matrices : ndarray, shape (n_partitions, n_states, n_inputs)
        Stack of B matrices at each partition.
    bc_type : str, optional
        Boundary condition for cubic splines. Default "clamped" to prevent
        wild extrapolation at batch start/end.
    """

    def __init__(self, t_partitions, a_matrices, b_matrices, bc_type="clamped"):
        self.t_partitions = np.asarray(t_partitions, dtype=float)
        self.a_matrices = np.asarray(a_matrices, dtype=float)
        self.b_matrices = np.asarray(b_matrices, dtype=float)
        self.bc_type = bc_type

        # Infer dimensions from the matrices
        self.n_partitions = self.a_matrices.shape[0]
        self.n_states = self.a_matrices.shape[1]
        self.n_inputs = self.b_matrices.shape[2]

        self._validate()
        self._fit_splines()

    def _validate(self):
        assert self.a_matrices.shape == (
            self.n_partitions,
            self.n_states,
            self.n_states,
        ), (
            f"A shape mismatch: expected ({self.n_partitions}, {self.n_states}, {self.n_states}), got {self.a_matrices.shape}"
        )
        assert self.b_matrices.shape == (
            self.n_partitions,
            self.n_states,
            self.n_inputs,
        ), (
            f"B shape mismatch: expected ({self.n_partitions}, {self.n_states}, {self.n_inputs}), got {self.b_matrices.shape}"
        )
        assert len(self.t_partitions[:-1]) == self.n_partitions, (
            f"Partition count mismatch: {len(self.t_partitions[:-1])} time points vs {self.n_partitions} matrices"
        )

    def _fit_splines(self):
        """Fit cubic splines for each element of A and B."""
        self._a_splines = np.empty((self.n_states, self.n_states), dtype=object)
        self._b_splines = np.empty((self.n_states, self.n_inputs), dtype=object)

        for i in range(self.n_states):
            for j in range(self.n_states):
                self._a_splines[i, j] = CubicSpline(
                    self.t_partitions[:-1],
                    self.a_matrices[:, i, j],
                    bc_type=self.bc_type,
                )

        for i in range(self.n_states):
            for j in range(self.n_inputs):
                self._b_splines[i, j] = CubicSpline(
                    self.t_partitions[:-1],
                    self.b_matrices[:, i, j],
                    bc_type=self.bc_type,
                )

    def get_A(self, t):
        """Get interpolated A matrix at time t."""
        A = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                A[i, j] = float(self._a_splines[i, j](t))
        return A

    def get_B(self, t):
        """Get interpolated B matrix at time t."""
        B = np.zeros((self.n_states, self.n_inputs))
        for i in range(self.n_states):
            for j in range(self.n_inputs):
                B[i, j] = float(self._b_splines[i, j](t))
        return B

    # def get_AB(self, t):
    #     """Get both A(t) and B(t) in one call."""
    #     return self.get_A(t), self.get_B(t)

    def get_AB(self, t):
        # partition_bounds: e.g., [0, 25, 50, 75, 100]
        # models: list of (A_i, B_i) for each segment
        for i in range(len(self.t_partitions) - 1):
            if self.t_partitions[i] <= t < self.t_partitions[i + 1]:
                return self.a_matrices[i], self.b_matrices[i]
        # Handle t == final boundary
        return self.a_matrices[-1], self.b_matrices[-1]

    # -------------------------------------------------------------------------
    # Serialization — store spline knots + coefficients, not the scipy objects
    # -------------------------------------------------------------------------

    # def _spline_to_dict(self, spline):
    #     """Extract the raw data needed to reconstruct a cubic spline."""
    #     return {
    #         "knots": spline.x.tolist(),
    #         "coeffs": spline.c.tolist(),  # shape (4, n_segments)
    #     }

    # def _dict_to_spline(self, d):
    #     """Reconstruct a CubicSpline from stored knots and coefficients via PPoly."""
    #     from scipy.interpolate import PPoly

    #     c = np.array(d["coeffs"])
    #     x = np.array(d["knots"])
    #     return PPoly(c, x)

    # def to_dict(self):
    #     """Serialize model to a plain dictionary."""
    #     model_dict = {
    #         "metadata": {
    #             "n_states": self.n_states,
    #             "n_inputs": self.n_inputs,
    #             "n_partitions": self.n_partitions,
    #             "bc_type": self.bc_type,
    #         },
    #         "t_partitions": self.t_partitions.tolist(),
    #         "a_raw": self.a_matrices.tolist(),
    #         "b_raw": self.b_matrices.tolist(),
    #         "a_splines": [
    #             [
    #                 self._spline_to_dict(self._a_splines[i, j])
    #                 for j in range(self.n_states)
    #             ]
    #             for i in range(self.n_states)
    #         ],
    #         "b_splines": [
    #             [
    #                 self._spline_to_dict(self._b_splines[i, j])
    #                 for j in range(self.n_inputs)
    #             ]
    #             for i in range(self.n_states)
    #         ],
    #     }
    #     return model_dict

    # def save(self, filepath):
    #     """
    #     Save model to JSON or YAML based on file extension.

    #     Stores both raw matrices (for refitting) and spline coefficients
    #     (for reconstruction without refitting).
    #     """
    #     filepath = Path(filepath)
    #     model_dict = self.to_dict()

    #     if filepath.suffix in (".yaml", ".yml"):
    #         if not YAML_AVAILABLE:
    #             raise ImportError(
    #                 "PyYAML required for .yaml export: pip install pyyaml"
    #             )
    #         with open(filepath, "w") as f:
    #             yaml.dump(model_dict, f, default_flow_style=False, sort_keys=False)
    #     elif filepath.suffix == ".json":
    #         with open(filepath, "w") as f:
    #             json.dump(model_dict, f, indent=2)
    #     else:
    #         raise ValueError(
    #             f"Unsupported format: {filepath.suffix}. Use .json or .yaml"
    #         )

    #     print(f"Model saved to {filepath}")

    # @classmethod
    # def load(cls, filepath, refit=False):
    #     """
    #     Load model from JSON or YAML.

    #     Parameters
    #     ----------
    #     filepath : str or Path
    #         Path to the saved model file.
    #     refit : bool, optional
    #         If True, refit splines from raw matrices (requires scipy).
    #         If False, reconstruct splines directly from stored coefficients.
    #         Default False — this is faster and doesn't require the original
    #         fitting logic, just scipy.interpolate.PPoly.
    #     """
    #     filepath = Path(filepath)

    #     if filepath.suffix in (".yaml", ".yml"):
    #         if not YAML_AVAILABLE:
    #             raise ImportError(
    #                 "PyYAML required for .yaml import: pip install pyyaml"
    #             )
    #         with open(filepath, "r") as f:
    #             model_dict = yaml.safe_load(f)
    #     elif filepath.suffix == ".json":
    #         with open(filepath, "r") as f:
    #             model_dict = json.load(f)
    #     else:
    #         raise ValueError(f"Unsupported format: {filepath.suffix}")

    #     t_partitions = np.array(model_dict["t_partitions"])
    #     a_matrices = np.array(model_dict["a_raw"])
    #     b_matrices = np.array(model_dict["b_raw"])
    #     meta = model_dict["metadata"]

    #     if refit:
    #         # Full refit from raw data
    #         return cls(t_partitions, a_matrices, b_matrices, bc_type=meta["bc_type"])

    #     # Reconstruct from stored spline coefficients (no refitting needed)
    #     instance = cls.__new__(cls)
    #     instance.t_partitions = t_partitions
    #     instance.a_matrices = a_matrices
    #     instance.b_matrices = b_matrices
    #     instance.bc_type = meta["bc_type"]
    #     instance.n_states = meta["n_states"]
    #     instance.n_inputs = meta["n_inputs"]
    #     instance.n_partitions = meta["n_partitions"]

    #     instance._a_splines = np.empty(
    #         (instance.n_states, instance.n_states), dtype=object
    #     )
    #     instance._b_splines = np.empty(
    #         (instance.n_states, instance.n_inputs), dtype=object
    #     )

    #     for i in range(instance.n_states):
    #         for j in range(instance.n_states):
    #             instance._a_splines[i, j] = instance._dict_to_spline(
    #                 model_dict["a_splines"][i][j]
    #             )

    #     for i in range(instance.n_states):
    #         for j in range(instance.n_inputs):
    #             instance._b_splines[i, j] = instance._dict_to_spline(
    #                 model_dict["b_splines"][i][j]
    #             )

    #     return instance

    def __repr__(self):
        return (
            f"TimeVaryingStateSpace("
            f"states={self.n_states}, inputs={self.n_inputs}, "
            f"partitions={self.n_partitions}, "
            f"t_range=[{self.t_partitions[0]:.1f}, {self.t_partitions[-1]:.1f}])"
        )


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Synthetic example: 4 states, 2 inputs, 3 partitions ---
    np.random.seed(42)
    top_dir = Path().absolute()
    PARENT_FILE_PATH = top_dir / "models" / "2026-03-03 B7H4 SAM 1000L Partition Model"
    model_config = json_to_dict(Path(PARENT_FILE_PATH, "B7H4_model_parameters.json"))

    a_matrices = np.array(model_config["a_matrix"])
    b_matrices = np.array(model_config["b_matrix"])

    n_states, n_inputs, n_partitions = 6, 4, 3
    t_partitions = np.arange(0, 14, 1)  # e.g., batch days

    # a_matrices = np.random.randn(n_partitions, n_states, n_states) * 0.5
    # b_matrices = np.random.randn(n_partitions, n_states, n_inputs) * 0.3
    a1 = 2.0
    a2 = 8.0
    b1 = 3.0
    b2 = 7.0

    # 1) Create model
    model = TimeVaryingStateSpace(t_partitions, a_matrices, b_matrices)
    print(model)
    print(f"\nA at day {a1}:\n{model.get_A(a1)}")
    print(f"\nA at day {a2}:\n{model.get_A(a2)}")
    print(f"\nB at day {b1}:\n{model.get_B(b1)}")
    print(f"\nB at day {b2}:\n{model.get_B(b2)}")

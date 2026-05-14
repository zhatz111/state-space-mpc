# State Space Model — Project Context

## Overview

This is a bioreactor cell culture modeling project developed at GSK. The goal is to build
**continuous-time state-space models** of mammalian cell culture processes (e.g., B7H4, TSLP drug
targets grown in 1000L bioreactors) for use in **Model Predictive Control (MPC)**.

The system models how bioreactor states (cell density, viability, product titer, metabolites) 
evolve in response to process inputs (feed additions, temperature/pH setpoints).

---

## Mathematical Structure

### Time-Invariant Model

Continuous-time linear state-space:

```
ẋ(t) = A·x(t) + B·u(t)
y(t) = C·x(t)           (C = identity, D = 0)
```

- **States x**: IGG (titer), VCC (viable cell count), Viability, Glutamate, Ammonium
- **Inputs u**: Normalized_Bolus_Feed, Temperature_setpoint, pH_setpoint
- **A matrix**: (n_states × n_states) — system dynamics
- **B matrix**: (n_states × n_inputs) — input influence
- All values are MinMaxScaled to [0,1] before simulation; outputs are inverse-transformed back

Integrated using `scipy.signal.lsim` with zero-order hold (matrix exponential step).

### Time-Varying (Partitioned) Model

The batch is split into time partitions (e.g., days 0–7.5 and 7.5–15). Each partition has its
own A and B matrices. The model is piecewise-constant:

```
A(t), B(t) = Aᵢ, Bᵢ    for t ∈ [tᵢ, tᵢ₊₁)
```

Stored as `a_matrix` shape `(n_partitions, n_states, n_states)` in JSON config.

`TimeVaryingStateSpace` class in [src/models/time_varying_ss.py](src/models/time_varying_ss.py)
also fits cubic splines element-wise over partition knots, but the ODE simulation via
`simulate_tv_continuous` currently calls `get_AB(t)` (step function), **not** `get_A(t)`/`get_B(t)`
(spline interpolation). The spline infrastructure is built but unused in simulation.

Full-batch prediction uses `scipy.integrate.solve_ivp` with RK45.

---

## Repository Structure

```
state-space-model/
├── data/                   # Experiment YAML configs + JSON data tables
├── models/                 # Trained model folders, each with:
│   └── {date} {name}/
│       ├── *_model_parameters.json   # A/B matrices, scaler, training metadata
│       └── Matrices/                 # Timestamped backup JSONs
├── src/
│   ├── data/
│   │   ├── make_dataset.py   # ModelData: preprocessing, spline upsampling, train/test split
│   │   └── functions.py      # Utilities: JSON I/O, scaler serialization
│   ├── models/
│   │   ├── ssm.py            # StateSpaceModel: time-invariant prediction with lsim_mod
│   │   ├── train_model.py    # ModelTraining: optimization (basinhopping), objective function
│   │   ├── time_varying_ss.py # TimeVaryingStateSpace + simulate_tv_continuous
│   │   └── optimize_model.py
│   ├── mpc/                  # MPC optimizer and simulations
│   ├── generate_model.py     # Main training script (interactive folder selection)
│   └── module_tester.py      # Ad-hoc test harness
└── pyproject.toml            # uv-managed dependencies
```

---

## Data Flow

1. **Raw data**: CSV with columns mapped by YAML config → `ModelData`
2. **Preprocessing** (`ModelData.clean()`):
   - Interpolate missing values within each batch (linear interpolation between known values;
     ffill+bfill handles any leading/trailing NaN at batch edges)
   - If `Spline Points` is non-null: upsample to `n_points` per batch using signal-type-aware splines:
     - `"smooth"`: Savitzky-Golay filter + PCHIP
     - `"zoh"`: zero-order hold (setpoints)
     - `"bolus"`: exponential decay — `v · exp(-τ · (t − t_bolus))` if `Bolus Decay Tau` is non-null,
       else linear interpolation
     - `"linear"`: linear interpolation
   - If `Spline Points` is `null`: skip upsampling entirely — raw interpolated data is used as-is
   - `GroupShuffleSplit` by batch → train/test split
   - `MinMaxScaler` fit on training data, applied to both sets
3. **Output**: scaled DataFrames with columns `[Batch, Day, *states, *inputs]`

---

## Training Pipeline (`generate_model.py`)

1. Select model folder interactively (InquirerPy)
2. Load JSON config (A/B matrices, scaler, partition info)
3. `ModelData.clean()` → scaled train/test data
4. `ModelTraining.__init__()`:
   - If partitions: slices data to the current partition's time window; sets `self.a_matrix`/`self.b_matrix` to that partition's matrices
5. `train_test_model()` → `train_model()` → `basinhopping` optimizer → `objective_func()`
6. Save best A/B matrices back to JSON; archive timestamped copy in `Matrices/` folder

### Objective Function

In `objective_func` (time-invariant path):
- For each training batch: run `signal.lsim` from true initial condition → simulated output
- SSE between simulated and experimental states (in scaled space)
- Weighted by `Process Variable Weights`
- Plus `matrix_stability_cost`:
  - Eigenvalue penalty: Re(λ(A)) ≤ -0.3
  - B-matrix authority: Frobenius norm near 1.0
  - B-matrix condition number: log-penalized
  - Controllability: min singular value of `[B, AB, ..., A^(n-1)B]` ≥ 0.03

**Important**: The partitioned model's `objective_func` uses `signal.lsim`, NOT
`simulate_tv_continuous`. It trains each partition independently against that partition's
data slice, starting from the **true experimental** initial condition at the partition boundary.

---

## Bolus Feed Input Representation

### Problem

Bolus feed events are discrete additions (once per day in raw data). The original `"bolus"`
upsampling mapped each event to a **single time index** spike, concentrating the full feed
magnitude at one point. When this passes through `ẋ = Ax + Bu`, the term `B·u_bolus` becomes
very large for one time step, causing a visible discontinuous jump in the predicted state trajectory.

### Fix Implemented (2026-04-14)

The `"bolus"` signal type in `ModelData.spline_upsample` now represents each bolus event as an
**exponentially decaying input**:

```
u_bolus(t) = v · exp(−τ · (t − t_bolus))    for t ≥ t_bolus
```

Multiple bolus events superpose additively. `τ` (`Bolus Decay Tau` in the model JSON, default
`1.0` day) controls how quickly the feed effect decays — physically motivated by first-order
substrate consumption kinetics (e.g., glutamine half-life ~1 day in cell culture).

### Production / MPC Use

**Critical**: in production, never feed the model a raw `{0, v}` bolus signal. Always reconstruct
the same exponentially-decayed representation using `compute_bolus_input` from `data/functions.py`:

```python
from data.functions import compute_bolus_input

u_bolus = compute_bolus_input(
    t_eval=time_array,
    bolus_events=[(day_4, 0.05), (day_8, 0.05)],  # (t_bolus, magnitude) pairs
    tau=model_config["Bolus Decay Tau"],
)
```

For MPC future predictions, a planned bolus at time `t_f` contributes `v · exp(−τ·(t − t_f))`
to the prediction horizon — smooth, no jumps, and schedulable.

`tau` must match the value used during training (stored in the model JSON as `"Bolus Decay Tau"`).
Changing `tau` requires retraining.

---

## Known Issue: Time-Varying Model Performs Worse Than Time-Invariant

### Root Cause: Training–Evaluation Mismatch

**During training** of partition `i`:
- Data is sliced to `[start_idx : end_idx]` of each batch
- `objective_func` initializes each batch simulation from the **true experimental state** at the partition start (e.g., the real measured cell density at day 7.5)
- Each partition is trained to be locally optimal, given perfect initial conditions

**During evaluation** (`get_model_data_dict` → `simulate_tv_continuous`):
- The full batch is simulated continuously from day 0
- Partition 2 starts from whatever state **partition 1's simulation** produced at day 7.5
- If partition 1 drifted from reality (even slightly), partition 2 starts from a wrong state
- Errors from partition 1 cascade into partition 2 — error propagation

### Secondary Issues

1. **Numerical method mismatch**: Training uses `signal.lsim` (matrix-exponential, discrete
   zero-order hold); evaluation uses `solve_ivp` RK45 (continuous Runge-Kutta). Different
   numerical schemes can produce divergent results especially for the same matrices.

2. **Cubic splines built but unused**: `TimeVaryingStateSpace._fit_splines()` fits splines
   but `get_AB(t)` returns piecewise-constant matrices (step function). The commented-out
   `get_AB` at the top of `time_varying_ss.py` would use spline interpolation (smoother
   transitions). Using step functions creates abrupt discontinuities in A(t)/B(t).

3. **No cross-partition coupling in loss**: Partitions trained independently with no penalty
   for discontinuity at boundaries. Neighboring partition matrices can differ significantly.

### Fix Implemented (2026-04-14)

The partitions are now **trained jointly** to match exactly how they're evaluated:

1. **`objective_func` (train_model.py)** — when `self.partitions` is True, the flat `x0`
   vector now encodes **all** partition matrices (`[A0_flat, B0_flat, A1_flat, B1_flat, ...]`).
   Each batch is simulated with `simulate_tv_continuous` (RK45, full batch from day 0),
   so error from partition 1 propagating into partition 2 is captured in the loss.
   Stability penalties are summed across all partition matrices.

2. **`train_model` (train_model.py)** — `combined_mat` is built from all partition matrices.
   After optimization, `self.a_matrices[i]` / `self.b_matrices[i]` are all unpacked from
   `self.best_result`.

3. **`generate_model.py`** — all partition matrices and RMSE are written back to the JSON.

4. **`simulate_tv_continuous` (time_varying_ss.py)** — `fill_value` added to `u_interp`
   so the ODE solver's adaptive stepping never queries NaN outside the time range.

This is **dynamic**: works with any `num_partitions` value in the JSON config.

---

## Key Configuration Keys (JSON model parameters)

| Key | Meaning |
|-----|---------|
| `a_matrix` | For time-invariant: `(n_states, n_states)`. For partitioned: `(n_partitions, n_states, n_states)` |
| `b_matrix` | For time-invariant: `(n_states, n_inputs)`. For partitioned: `(n_partitions, n_states, n_inputs)` |
| `Partitions` | Dict with `num_partitions`, `time_partitions`, `start_idx`, `end_idx` |
| `Current Training Partition` | Index into `a_matrix`/`b_matrix` list to train |
| `Process Variable Weights` | Loss weights per state (e.g., IGG weight=30, others=3.5) |
| `Instability Weights` | Penalty weights: `Eigenvalue`, `B-matrix Authority`, `B-matrix Condition`, `Controllability` |
| `Spline Points` | How many upsampled time points per batch (e.g., 30). Set to `null` to skip upsampling entirely |
| `Bolus Decay Tau` | Exponential decay τ (days) for bolus feed input. Set to `null` to use linear interpolation instead |
| `BasinHopping Temperature` | Exploration parameter for global optimizer |

### Reverting to the old (no-processing) training method

Set both of the following in the model JSON:

```json
"Spline Points": null,
"Bolus Decay Tau": null
```

| `Spline Points` | `Bolus Decay Tau` | Behavior |
|---|---|---|
| `null` | `null` | Raw interpolated data, no upsampling — old training method |
| `14` | `null` | Upsampled to 14 pts/batch, bolus uses linear interp (no decay) |
| `14` | `1.0` | Full processing — upsampled + exponential bolus decay |
| `null` | `1.0` | Raw data (bolus_tau is ignored since there is no upsampling) |

---

## Bug Fixes (2026-05-14)

### `num_partitions: 0` partition detection

A non-empty Python dict (e.g. `{"num_partitions": 0, "time_partitions": []}`) evaluates as
truthy, so the original `if partition_data:` check incorrectly entered the partitioned code path
for models with `num_partitions: 0`. This caused `generate_model.py` to build a `(0, n, n)`
zero-length matrix array, and `ModelTraining.__init__` then crashed with
`IndexError: index 0 is out of bounds` when accessing `self.a_matrices[0]`.

**Fix**: replaced all three `if partition_data:` / `if not partition_data:` guards in
`generate_model.py` with:

```python
is_partitioned = bool(partition_data and partition_data.get("num_partitions", 0) > 0)
```

`ModelTraining` now receives `partitions_data=None` (not the raw dict) when the model is
non-partitioned, and its default changed from `{}` to `None`. For non-partitioned models, the
stored `a_matrix` (which may be 3-D from a prior partitioned training run) is automatically
squeezed to `(n_states, n_states)` by extracting `a_matrix[0]`.

### `signal.lsim` equally-spaced time requirement

`scipy.signal.lsim` requires equally spaced time steps. The upsampled path guaranteed this
via `np.linspace`. When `Spline Points: null` is set and raw measurement data is passed through,
real bioreactor sampling days are typically irregular (e.g. days 0, 1, 3, 5, 7, 10, 14).

**Fix**: in `objective_func` (non-partitioned path in `train_model.py`), after extracting
`time = np.array(group["Day"])`, the code checks `np.allclose(np.diff(time), np.diff(time)[0])`.
If the spacing is irregular, `u`, `y`, and `time` are all resampled to a `np.linspace` grid of
the same number of points before calling `lsim`. The partitioned path (`simulate_tv_zoh`) does
not have this requirement and is unaffected.

---

## Dependencies (managed via `uv`)

- numpy, scipy, pandas
- scikit-learn (MinMaxScaler, metrics)
- matplotlib
- InquirerPy (interactive CLI prompts in `generate_model.py`)
- PyYAML

---

## Active Model: B7H4 SAM 1000L Partition Model

Located at `models/2026-03-03 B7H4 SAM 1000L Partition Model/`.
- 2 partitions: days 0–7.5 (growth phase) and 7.5–15 (production phase)
- States: IGG, VCC, Viability, Glutamate, Ammonium
- Inputs: Normalized_Bolus_Feed, Temperature_setpoint, pH_setpoint
- Multiple archived training runs in `Matrices/` subfolder

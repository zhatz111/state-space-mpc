# State Space Model — Bioreactor MPC

Continuous-time linear state-space models of mammalian cell culture processes (B7H4, TSLP) grown in 1000 L bioreactors at GSK. The models are trained on historical batch data and deployed into a cloud-hosted Model Predictive Controller (MPC) that recommends daily feed, temperature, and pH setpoints.

---

## What this repo does

1. **Pre-processes** historical batch CSV data — interpolation, spline upsampling, train/test split, MinMax scaling.
2. **Trains** A and B matrices via basin-hopping optimization against scaled batch trajectories.
3. **Archives** trained models as JSON files alongside timestamped backups.
4. **Provides** `StateSpaceModel` and time-varying extensions that the cloud MPC consumes at runtime.

---

## Mathematical structure

### Time-invariant model

```
ẋ(t) = A·x(t) + B·u(t)
y(t) = C·x(t)           (C = identity, D = 0)
```

| Symbol | Meaning |
|--------|---------|
| **x** | IGG, VCC, Viability, Glutamate, Ammonium |
| **u** | Normalized_Bolus_Feed, Temperature_setpoint, pH_setpoint |
| **A** | (n_states × n_states) system dynamics |
| **B** | (n_states × n_inputs) input influence |

All values are MinMaxScaled to [0, 1] before simulation; outputs are inverse-transformed back to physical units.

### Partitioned (time-varying) model

The batch is split into time windows (e.g., days 0–7.5 and 7.5–15). Each window has independent A and B matrices:

```
A(t), B(t) = Aᵢ, Bᵢ   for t ∈ [tᵢ, tᵢ₊₁)
```

Training optimizes **all partition matrices jointly** using `simulate_tv_zoh` (piecewise ZOH, equivalent to running `signal.lsim` per partition and carrying state across boundaries). This matches the evaluation path exactly, eliminating the training/evaluation mismatch that caused the partitioned model to perform worse than the time-invariant model when partitions were trained independently.

---

## Repository structure

```
state-space-model/
├── data/                          # Per-experiment folders
│   └── {experiment}/
│       ├── {experiment}.yaml      # Experiment + controller config (read by cloud MPC)
│       └── data/                  # Daily measurement YAMLs
├── models/                        # One folder per trained model
│   └── {date} {name}/
│       ├── *_model_parameters.json   # A/B matrices, scaler, training config
│       └── Matrices/                 # Timestamped backup JSONs
├── src/
│   ├── data/
│   │   ├── make_dataset.py        # ModelData: preprocessing and upsampling
│   │   └── functions.py           # JSON/YAML I/O, scaler helpers, compute_bolus_input
│   ├── models/
│   │   ├── ssm.py                 # StateSpaceModel — used by cloud MPC at runtime
│   │   ├── train_model.py         # ModelTraining: basin-hopping optimization
│   │   ├── time_varying_ss.py     # TimeVaryingStateSpace + simulate_tv_zoh/continuous
│   │   └── optimize_model.py
│   ├── mpc/
│   │   ├── mpc_optimizer.py       # Bioreactor + Controller classes
│   │   └── simulations.py         # ModelSimulations helper
│   └── generate_model.py          # Interactive training entry point
└── pyproject.toml                 # uv-managed dependencies
```

---

## Data flow

1. **Raw CSV** → `ModelData` (interpolation, optional spline upsampling at `Spline Points` resolution, train/test split, MinMaxScaler fit on training set).
2. **Scaled data** → `ModelTraining` → basin-hopping → optimized A/B matrices.
3. **JSON** updated with new matrices, RMSE, and a timestamped backup written to `Matrices/`.

Set `"Spline Points": null` and `"Bolus Decay Tau": null` in the model JSON to skip all upsampling and bolus decay processing and train directly on raw interpolated data.

### Signal-type-aware upsampling

| Type | Columns | Method |
|------|---------|--------|
| `smooth` | States (IGG, VCC, …) | Savitzky-Golay filter → PCHIP |
| `zoh` | Setpoints (Temperature, pH) | Zero-order hold |
| `bolus` | Normalized_Bolus_Feed | Exponential decay (see below) |
| `linear` | Other | Linear interpolation |

---

## Bolus feed input representation

Raw bolus events are discrete daily additions. Representing them as single-point spikes causes a large discontinuous jump in `B·u` at that time step.

**Fix**: the `"bolus"` signal type reconstructs each event as an exponentially decaying signal:

```
u_bolus(t) = v · exp(−τ · (t − t_bolus))   for t ≥ t_bolus
```

Multiple events superpose. `τ` (`Bolus Decay Tau`, default 1.0 day) is physically motivated by first-order substrate consumption kinetics.

### Production / MPC consistency

In production, data arrives at 1 point/day. Without correction, passing daily bolus values to `lsim` (ZOH at dt = 1) integrates the full bolus magnitude over the whole day, whereas training integrated the decaying signal — a ~58 % overestimate of the bolus effect.

**Fix in `StateSpaceModel.ssm_lsim`**: when `Bolus Decay Tau` is present in the model parameters, the bolus column is automatically upsampled to `Bolus Upsample Resolution` sub-steps per day (default 10), `compute_bolus_input` is applied at that fine grid, the simulation runs at fine resolution, and outputs are sampled back to the original daily time points. This is transparent to calling code.

To deploy the same tau that was used during training, add `"Bolus Decay Tau"` to the experiment YAML's `Model Parameters` block (already present in `SAM-1000L-B7H4.yaml`). **`tau` must match between training and production; changing it requires retraining.**

---

## Training pipeline (`generate_model.py`)

```
select model folder (InquirerPy)
  → load JSON config
  → ModelData.clean() — upsample, split, scale
  → ModelTraining (train all partitions jointly)
  → basinhopping optimizer → objective_func()
  → write A/B matrices back to JSON + timestamped backup
```

### Objective function

- For each training batch: `simulate_tv_zoh` (partitioned) or `signal.lsim` (time-invariant) from true initial condition → SSE vs experimental states (scaled space).
- Weighted by `Process Variable Weights` (dict keyed by state name, e.g. `{"IGG": 100.0, "VCC": 3.0, …}`).
- Plus `matrix_stability_cost`:
  - Eigenvalue penalty: Re(λ(A)) ≤ −0.3
  - B-matrix authority: Frobenius norm near 1.0
  - B-matrix condition number: log-penalized
  - Controllability: min singular value of `[B, AB, …, A^(n-1)B]` ≥ 0.03

---

## Key JSON config fields

| Key | Meaning |
|-----|---------|
| `a_matrix` | Time-invariant: `(n_states, n_states)`. Partitioned: `(n_partitions, n_states, n_states)` |
| `b_matrix` | Time-invariant: `(n_states, n_inputs)`. Partitioned: `(n_partitions, n_states, n_inputs)` |
| `Partitions` | `num_partitions`, `time_partitions` (boundary days) |
| `Process Variable Weights` | Dict keyed by state name: `{"IGG": 100.0, "VCC": 3.0, …}` |
| `Instability Weights` | Dict: `Eigenvalue`, `B-matrix Authority`, `B-matrix Condition`, `Controllability` |
| `Bolus Decay Tau` | Exponential decay time constant τ in days (default 1.0) |
| `Spline Points` | Upsampled points per batch during training. Set to `null` to skip upsampling |
| `Bolus Decay Tau` | Exponential decay τ (days) for bolus input. Set to `null` to use linear interp |
| `BasinHopping Temperature` | Global optimizer exploration parameter |

---

## Key YAML config fields (experiment YAML)

The experiment YAML (`data/{experiment}/{experiment}.yaml`) is read by the cloud MPC at runtime. `Model Parameters` contains only the fields `StateSpaceModel` needs:

| Field | Purpose |
|-------|---------|
| `Asset` | Model identifier |
| `Model States` / `Model Inputs` | Variable lists |
| `Bolus Decay Tau` | Must match training tau — enables production bolus upsampling |
| `Bolus Upsample Resolution` | Sub-steps per day for bolus fix (default 10) |
| `Hidden State` / `af_col` / `af_row` / `bf_row` / `rho` | Hidden feed-effect state augmentation (optional) |
| `a_matrix` / `b_matrix` / `scaler` | Deployed model matrices and scaler |

Training metadata (batch lists, RMSE, stability penalties, weight configs) is **not** stored in the YAML — it lives in the model JSON.

---

## MPC integration

`StateSpaceModel` is the bridge between this repo and the cloud MPC:

- `ssm_lsim(initial_state, input_matrix, time, ...)` — time array must be in **batch days** (e.g., `[3, 4, …, 14]` when simulating from day 3). This is required for time-varying partition boundary resolution and for bolus-decay upsampling to reconstruct the correct event timing.
- Partitioned models (`a_matrix` shape `(n_partitions, n_states, n_states)`) are detected automatically and routed through `simulate_tv_zoh`.
- The `Bioreactor` and `Controller` classes in `mpc_optimizer.py` now pass batch-day time arrays to `ssm_lsim`, making both the time-varying simulation and bolus correction work correctly in the MPC loop.

---

## Upsampling and production data density

**Training**: 28–35 batches × ~50 upsampled points each = high-resolution trajectories.

**Production**: 1 measurement per day × 12–14 days per batch.

The model is trained in normalized space using the upsampled data, but the **scaler** (MinMaxScaler) was fit on the same upsampled data. At production time, daily physical measurements are scaled with the same scaler — this is correct because the scaler maps physical ranges to [0, 1] and those ranges are data-driven, not density-dependent.

The only density-sensitive component is the bolus input, which is handled by the automatic upsampling in `ssm_lsim` described above. All other inputs (temperature setpoint, pH setpoint) are ZOH-type and behave identically at daily resolution.

---

## Active model

**B7H4 SAM 1000L Partition Model** — `models/2026-03-03 B7H4 SAM 1000L Partition Model/`

- 2 partitions: days 0–7.5 (growth phase), 7.5–15 (production phase)
- States: IGG, VCC, Viability, Glutamate, Ammonium
- Inputs: Normalized_Bolus_Feed, Temperature_setpoint, pH_setpoint
- Training data: AR25-050, AR25-051, AR25-068 campaigns

---

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```shell
pip install uv
uv sync
```

To run the training script:

```shell
cd src
uv run python generate_model.py
```

---

## Changelog

### 2026-05-14
- **Null training mode**: `Spline Points: null` and `Bolus Decay Tau: null` in model JSON now skip upsampling and bolus decay respectively, reverting to raw interpolated data for training. Useful for comparing model accuracy against the pre-processing baseline. Leading/trailing NaN at batch edges are handled with ffill+bfill so the scaler does not fail on raw data.
- **Bug fix — partition detection**: `{"num_partitions": 0}` is a truthy Python dict, causing the partitioned code path to run for time-invariant models and crash with `IndexError`. Fixed by checking `partition_data.get("num_partitions", 0) > 0` in `generate_model.py`. Non-partitioned models now correctly extract a 2-D `(n_states, n_states)` matrix even if the JSON stores a 3-D array from a prior partitioned training run.
- **Bug fix — lsim equally spaced time**: `scipy.signal.lsim` requires uniform time steps. When training on raw data (irregular measurement intervals), the objective function now resamples `u`, `y`, and `time` to a `np.linspace` grid of equal length before calling `lsim`.

### 2026-05-08
- `Process Variable Weights` in model JSON and training code changed from an ordered list to a named dict (`{"IGG": 100.0, "VCC": 3.0, …}`) for readability and to prevent silent ordering bugs.
- `StateSpaceModel` (`ssm.py`) updated:
  - Hidden-state augmentation fields (`af_col`, `af_row`, `bf_row`, `rho`, `Hidden State`) are now optional — model loads without them.
  - Partitioned (time-varying) models now supported: 3-D `a_matrix` is detected automatically and `ssm_lsim` routes through `simulate_tv_zoh`.
  - Bolus-decay upsampling added to `ssm_lsim`: when `Bolus Decay Tau` is set in model parameters, the bolus column is upsampled to fine intra-day resolution before simulation and downsampled back, matching the training representation.
- `mpc_optimizer.py`: `sim_from_day` and `obj_func_wrapper` now pass batch-day time arrays (e.g. `[3,4,…,14]`) to `ssm_lsim` instead of simulation-relative arrays (`[0,1,…,11]`), which is required for correct partition boundary resolution in time-varying models.
- Experiment YAML (`SAM-1000L-B7H4.yaml`) `Model Parameters` stripped of all training metadata (batch lists, RMSE, weights, stability penalties). Now contains only the fields `StateSpaceModel` needs at runtime. Added `Bolus Decay Tau` and `Bolus Upsample Resolution`.

### 2026-04-14
- Bolus feed input representation changed from single-point spike to exponential decay: `v·exp(−τ·(t−t_bolus))`. Removes discontinuous jumps in predicted trajectories.
- Partitioned model training changed from per-partition independent optimization to joint optimization using `simulate_tv_zoh` across all partitions, eliminating the training/evaluation mismatch that caused error propagation across partition boundaries.
- `simulate_tv_continuous`: added `fill_value` to `u_interp` so ODE solver adaptive stepping never queries NaN outside the time range.

### 2026-02-27
- Added `uv` for project and dependency management (`pyproject.toml`).
- Added `simulate_tv_zoh` (fast piecewise ZOH) and `simulate_tv_continuous` (RK45) to `time_varying_ss.py`.
- `TimeVaryingStateSpace` class added with cubic spline infrastructure (built but unused in simulation — step-function `get_AB` is used instead).

### 2025-08-27
- MPC optimizer (`mpc_optimizer.py`) and `StateSpaceModel` (`ssm.py`) updated for hidden feed-effect state augmentation.
- `Bioreactor.return_data` updated to include constraint columns.

### 2024-04-19
- `functions.py` created with JSON/YAML I/O utilities and `scaler_todict`/`dict_toscaler` helpers.

### 2024-01-29
- `mpc_optimizer.py`: corrected `--INPUT_DATA` / `--INPUT_REF` column name handling for feed.

### 2024-01-18
- `ssm.py`: renamed `delta_p` to `output_mods`.
- `mpc_optimizer.py`: Day 0 state initialization, estimation horizon, open-loop mode, and `return_data` improvements.

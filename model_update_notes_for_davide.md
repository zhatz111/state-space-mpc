# Model Update Notes for Davide

## src/estimation_loop.py (main script)

- `read_config()`: this function reads the .yaml file and returns an `experiment_config` dict for the `Bioreactor` and `Controller` objects; if the optional argument `export=True`, it also copies the current .yaml file to a destination folder specified in the .yaml file
- The code reads a master data file with up-to-date measurements when running locally and reads a storing table from the input topic when running in the cloud
- The code initializes a `Bioreactor` and a `Controller` objectives outside the time loop then runs `bioreactor.ingest_vector()`, `controller.estimate()`, and `controller.optimize()` on on Day 0, Day 1, ..., until the current day
- Once the time loop completes, the code runs `bioreactor.get_result()` and print the current day's nutrient feed pump flow rate in mL/min (**we still need to make sure the units are correct across different scales**)

## src/mpc/mpc_optimizer.py (Bioreactor and Controller class definitions)

- `Bioreactor` class
	- Added `scale`, `init_vol` (initial volume), and `vol` (current volume) attributes
	- `get_result()`: this function reads `Bioreactor.data` and returns `FeedRate_mL_min` (nutrient feed pump rate), `Times_day` (days array), `TiterPred_mg_L` (predicted titer) in a dict
	- `ingest_vector()`: this function reads a PD series (single row) from the current day (or the day specified by the time loop in the main script) and updates `Bioreactor.data` 
- `Controller` class
	- Added `offset_kp`, `offset_ki`, `est_curr_error`, `est_prev_error`, and `mv_ref_names` attributes
	- Updated the command line output to include end-of-prediction-horizon prediction and end-of-run prediction
	- Added end-of-run constraints
	- Updated the way `estimate()` works

## src/models/ssm.py (state-space model)

- `lsim_mod()`: a modified version of `scipy.signal.lsim`

## src/data/functions.py (utilities)

- Minor updates

## data/dmc-test-with-mr24-030-experiment-data/dmc_test.yaml (configuration file)

- Reorganized some fields and created new fields
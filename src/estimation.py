"""Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2024-01-18
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Standard Library Imports
import warnings
from pathlib import Path
from datetime import datetime

# 3rd Party Library Imports
import joblib
import numpy as np
import pandas as pd

# State-Space-Model Package Imports
from mpc.mpc_optimizer import Bioreactor, Controller
from visualization.visualize import MPCVisualizer
from models.ssm import StateSpaceModel

warnings.filterwarnings("ignore")

todays_date = datetime.today().strftime('%Y-%m-%d')

# Specify the current time and vessel
CURR_TIME = 2
VESSEL = 1

# Load an example read-only "master" sheet
SIM_FOLDER = "mpc-simulation"
SIM_REFERENCE_DATA = f"master_sheet_example-D{CURR_TIME}"

simulation_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/", SIM_FOLDER
)
reference_data_all = pd.read_csv(Path(simulation_path, f"{SIM_REFERENCE_DATA}.csv"))

# Create output folder
batch_sheet_path = Path(simulation_path.expanduser(), SIM_REFERENCE_DATA)
batch_sheet_path.mkdir(parents=True, exist_ok=True)

# Load the model
# CONTROL_MATRIX_FOLDER_EXT = "Control_Model_Matrices"
SIM_MATRIX_FOLDER_EXT = "Sim_Model_Matrices"

# Parse the states from the reference data csv file
reference_data_this_vessel = reference_data_all.loc[reference_data_all["Bioreactor"] == VESSEL, :]
contains_state_data = reference_data_this_vessel.columns.str.contains("--STATE_DATA")
contains_input = reference_data_this_vessel.columns.str.contains("--INPUT_DATA")

# store the states and inputs as a list
STATES = [x.split("--")[0] for x in reference_data_this_vessel.columns[contains_state_data]]
INPUTS = [x.split("--")[0] for x in reference_data_this_vessel.columns[contains_input]]

# # Controller Matrices: Import matrices and scaler (worse model)
# controller_model_scaler = joblib.load(
#     Path(
#         simulation_path.expanduser(),
#         rf"{CONTROL_MATRIX_FOLDER_EXT}",
#         "model_scaler.scl",
#     ),
# )
# control_A_matrix = np.array(
#     pd.read_csv(
#         Path(
#             simulation_path.expanduser(),
#             rf"{CONTROL_MATRIX_FOLDER_EXT}",
#             "A_Matrix.csv",
#         ),
#         header=None,
#     )
# )
# control_B_matrix = np.array(
#     pd.read_csv(
#         Path(
#             simulation_path.expanduser(),
#             rf"{CONTROL_MATRIX_FOLDER_EXT}",
#             "B_Matrix.csv",
#         ),
#         header=None,
#     )
# )

# Simulation Matrices: Import matrices and scaler (better model)
sim_model_scaler = joblib.load(
    Path(simulation_path.expanduser(), rf"{SIM_MATRIX_FOLDER_EXT}", "model_scaler.scl")
)

sim_A_matrix = np.array(
    pd.read_csv(
        Path(simulation_path.expanduser(), rf"{SIM_MATRIX_FOLDER_EXT}", "A_Matrix.csv"),
        header=None,
    )
)

sim_B_matrix = np.array(
    pd.read_csv(
        Path(simulation_path.expanduser(), rf"{SIM_MATRIX_FOLDER_EXT}", "B_Matrix.csv"),
        header=None,
    )
)

# Create a state space model for control (identical to bioreactor sim model)
controller_model = StateSpaceModel(
    states=STATES,
    inputs=INPUTS,
    scaler=sim_model_scaler,
    a_matrix=sim_A_matrix,
    b_matrix=sim_B_matrix,
)

# Create a state space model for simulation
simulation_model = StateSpaceModel(
    states=STATES,
    inputs=INPUTS,
    scaler=sim_model_scaler,
    a_matrix=sim_A_matrix,
    b_matrix=sim_B_matrix,
)

# Construct a bioreactor object
bioreactor = Bioreactor(
    vessel=str(VESSEL), process_model=simulation_model, data=reference_data_this_vessel
)

# Parse the PV and MV names from the reference data csv file
contains_PV = reference_data_this_vessel.columns.str.contains("state_sp", case=False)
contains_MV = reference_data_this_vessel.columns.str.contains("input_ref", case=False)

# Construct a controller object
PRED_HORIZON = 30
CTRL_HORIZON = 3
EST_HORIZON = 2
ts = np.array(reference_data_this_vessel["Day"])

# Define the PV and MV names using the parsing from csv file
pv_names = [x.split("--")[0] for x in reference_data_this_vessel.columns[contains_PV]]
mv_names = [x.split("--")[0] for x in reference_data_this_vessel.columns[contains_MV]]

PV_WTS = np.array([1 / (1000) ** 2])
EST_WTS = np.array([2.59074E-07, 0.078638189, 0.033750463, 2.614604943, 0.006617413, 0.018507072])

MV_WTS = np.array([1 / (0.01) ** 2])
MV_BOUNDS = np.array([[0, 0.1]])  # feed

# Define the suffix after each MV name and index the matrix with these names
PV_SUFFIX = "--STATE_SP"
MV_SUFFIX = "--INPUT_REF"
pv_sps = reference_data_this_vessel[[mv + PV_SUFFIX for mv in pv_names]].values
mv_matrix = reference_data_this_vessel[[mv + MV_SUFFIX for mv in mv_names]].values

# Verify dimensions (YL@2024-01-18)
if len(EST_WTS) != len(STATES):
    raise ValueError("Wrong estimation weights dimension!")
if len(PV_WTS) != len(pv_names):
    raise ValueError("Wrong PV weights dimension!")
if len(MV_WTS) != len(mv_names):
    raise ValueError("Wrong PV weights dimension!")
if [x.upper() for x in sim_model_scaler.get_feature_names_out()] != STATES + INPUTS:
    raise ValueError("Model and CSV do not match!")

controller = Controller(
    controller_model=controller_model,
    bioreactor=bioreactor,
    ts=ts,
    pv_sps=pv_sps,
    pv_names=pv_names,
    pv_wts=PV_WTS,
    mv_names=mv_names,
    mv_wts=MV_WTS,
    pred_horizon=PRED_HORIZON,
    ctrl_horizon=CTRL_HORIZON,
    constr=MV_BOUNDS,
    output_mods_user=np.array([]),
    est_wts=EST_WTS,
    est_horizon=EST_HORIZON,
)

# Update the time cursor
bioreactor.curr_time = CURR_TIME

# Estimate CURR_DAY's state
controller.estimate() # Update bioreactor.data>STATE_MOD (curr day)
bioreactor.next_day() # Update bioreactor.data>STATE_EST (curr day + 1)
controller.optimize(open_loop=False) # Update bioreactor.data>STATE_PRED (curr day + 1:end of pred horizon)

bioreactor.return_data(show_daily_feed=True).to_csv(
    batch_sheet_path / f"{bioreactor.vessel}_daily_feed_D{CURR_TIME}-{todays_date}.csv"
)
bioreactor.return_data(show_daily_feed=False).to_csv(
    batch_sheet_path / f"{bioreactor.vessel}_total_feed_D{CURR_TIME}-{todays_date}.csv"
)

# # Plot the in-silico Simulations
# br_plots = MPCVisualizer(sim_bioreactors, controllers)
# br_plots.plot_simulations()
# # br_plots.output_table().to_clipboard()

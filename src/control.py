"""Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Standard Library Imports
import copy
import warnings
from pathlib import Path

# 3rd Party Library Imports
import joblib
import numpy as np
import pandas as pd

# State-Space-Model Package Imports
from mpc.mpc_optimizer import Bioreactor, Controller
from visualization.visualize import MPCVisualizer
from models.ssm import StateSpaceModel

warnings.filterwarnings("ignore")

# Load an example dataset
BATCH_SHEET_FOLDER = "mpc-simulation"
BATCH_SHEET_NAME = "disturbance"
SIM_SHEET_NAME = "control_data"

simulation_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/", BATCH_SHEET_FOLDER
)
data = pd.read_csv(Path(simulation_path, rf"{BATCH_SHEET_NAME}.csv"))
control_data = pd.read_csv(Path(simulation_path, rf"{SIM_SHEET_NAME}.csv"))

# Create output folder
batch_sheet_path = Path(simulation_path.expanduser(), BATCH_SHEET_NAME)
batch_sheet_path.mkdir(parents=True, exist_ok=True)

# Load the model
CONTROL_MATRIX_FOLDER_EXT = "Control_Model_Matrices"
SIM_MATRIX_FOLDER_EXT = "Sim_Model_Matrices"

STATES = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
    "Osmo",
    "pCO2_at_Temp",
]

INPUTS = [
    "Cumulative_Normalized_Feed",
    "Temperature",
    "pH_setpoint",
    "DO",
]

# Controller Matrices: Import matrices and scaler (worse model)
controller_model_scaler = joblib.load(
    Path(
        simulation_path.expanduser(),
        rf"{CONTROL_MATRIX_FOLDER_EXT}",
        "model_scaler.scl",
    ),
)
control_A_matrix = np.array(
    pd.read_csv(
        Path(
            simulation_path.expanduser(),
            rf"{CONTROL_MATRIX_FOLDER_EXT}",
            "A_Matrix.csv",
        ),
        header=None,
    )
)
control_B_matrix = np.array(
    pd.read_csv(
        Path(
            simulation_path.expanduser(),
            rf"{CONTROL_MATRIX_FOLDER_EXT}",
            "B_Matrix.csv",
        ),
        header=None,
    )
)

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

# Create a state space model for control
controller_model = StateSpaceModel(
    states=STATES,
    inputs=INPUTS,
    scaler=controller_model_scaler,
    a_matrix=control_A_matrix,
    b_matrix=control_B_matrix,
)

# Create a state space model for sim
simulation_model = StateSpaceModel(
    states=STATES,
    inputs=INPUTS,
    scaler=sim_model_scaler,
    a_matrix=sim_A_matrix,
    b_matrix=sim_B_matrix,
)

# Construct a bioreactor object
bioreactor = Bioreactor(vessel="BR1-MPC", process_model=simulation_model, data=data)

# Construct an open-loop bioreactor object
bioreactor_open_loop = copy.deepcopy(bioreactor)
bioreactor_open_loop.vessel = "BR1-Open_Loop"

# Construct a controller object
PRED_HORIZON = 30
CTRL_HORIZON = 3
CURR_TIME = 0
ts = np.array(data["Day"])
pv_names = ["IGG"]
pv_wts = np.array([1 / (1000) ** 2])
pv_sps = data[pv_names].values
mv_names = ["Cumulative_Normalized_Feed"]
mv_wts = np.array([1 / (0.01) ** 2])
constr = np.array([[0, 0.1]])  # feed

mv_matrix = data[mv_names].values

controller = Controller(
    controller_model=controller_model,
    bioreactor=bioreactor,
    ts=ts,
    pv_sps=pv_sps,
    pv_names=pv_names,
    pv_wts=pv_wts,
    mv_names=mv_names,
    mv_wts=mv_wts,
    pred_horizon=PRED_HORIZON,
    ctrl_horizon=CTRL_HORIZON,
    constr=constr,
)

# Simulate a DoE to Determine what factors to test in-silico

# Create a list of bioreactors for the DoE
NUM_REACTORS = 6

sim_bioreactors = [
    Bioreactor(
        vessel=f"BR{i}",
        process_model=simulation_model,
        data=control_data,
    )
    for i in range(1, NUM_REACTORS + 1)
]
controllers = [
    Controller(
        controller_model=controller_model,
        bioreactor=sim_bioreactors[count],
        ts=ts,
        pv_sps=pv_sps,
        pv_names=pv_names,
        pv_wts=pv_wts,
        mv_names=mv_names,
        mv_wts=mv_wts,
        pred_horizon=PRED_HORIZON,
        ctrl_horizon=CTRL_HORIZON,
        constr=constr,
    )
    for count, _ in enumerate(sim_bioreactors)
]

# Day 3-4, 6-7, and 9-10 pH and temp setpoints
DOE_setpoints = [[7.0, 30], [7.2, 34], [7.3, 33], [7.1, 35], [7.05, 36], [7.18, 34]]

# Change bioreactor data based on DoE setpoints
for count, bioreactor in enumerate(sim_bioreactors):
    bioreactor.data["pH_setpoint"].iloc[3:] = DOE_setpoints[count][0]
    bioreactor.data["Temperature"].iloc[3:] = DOE_setpoints[count][1]

    # # Change day 9's setpoints
    # bioreactor.data["pH_setpoint"].iloc[9] = DOE_setpoints[count][0]
    # bioreactor.data["Temperature"].iloc[9] = DOE_setpoints[count][1]

# Simulate all the bioreactors and each controller and get a dictionary output
DOE_dict = {}
for bioreactor, controller in zip(sim_bioreactors, controllers):
    for i in range(len(ts) - 1):
        controller.optimize()
        bioreactor.next_day()
    DOE_dict[bioreactor.vessel] = bioreactor.return_data()

# Plot the in-silico Simulations
br_plots = MPCVisualizer(sim_bioreactors, controllers)
# br_plots.plot_simulations()
print(br_plots.output_table())

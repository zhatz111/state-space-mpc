"""Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Standard Library Imports
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
SIM_FOLDER = "mpc-simulation"
SIM_REFERENCE_DATA = "control_data"

simulation_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/", SIM_FOLDER
)
reference_data = pd.read_csv(Path(simulation_path, rf"{SIM_REFERENCE_DATA}.csv"))

# Create output folder
batch_sheet_path = Path(simulation_path.expanduser(), SIM_REFERENCE_DATA)
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
bioreactor = Bioreactor(vessel="BR1-MPC", process_model=simulation_model, data=reference_data)

# Construct a controller object
PRED_HORIZON = 30
CTRL_HORIZON = 3
CURR_TIME = 0
ts = np.array(reference_data["Day"])
PV_NAMES = ["IGG"]
PV_WTS = np.array([1 / (1000) ** 2])
pv_sps = reference_data[PV_NAMES].values
MV_NAMES = ["Cumulative_Normalized_Feed"]
MV_WTS = np.array([1 / (0.01) ** 2])
MV_BOUNDS = np.array([[0, 0.1]])  # feed

mv_matrix = reference_data[MV_NAMES].values

controller = Controller(
    controller_model=controller_model,
    bioreactor=bioreactor,
    ts=ts,
    pv_sps=pv_sps,
    pv_names=PV_NAMES,
    pv_wts=PV_WTS,
    mv_names=MV_NAMES,
    mv_wts=MV_WTS,
    pred_horizon=PRED_HORIZON,
    ctrl_horizon=CTRL_HORIZON,
    constr=MV_BOUNDS,
)

# Simulate a DoE to Determine what factors to test in-silico

# Day 3-4, 6-7, and 9-10 pH and temp setpoints
DOE_FACTOR_LEVELS = [
    [18, 7.05, 33],
    [15, 7.35, 35],
    [15, 7.05, 33],
    [18, 7.20, 34],
    [12, 7.20, 34],
    [12, 7.35, 35],
    [12, 7.05, 33],
    [15, 7.35, 35],
    # [18, 7.35, 35],
]

# Create a list of bioreactors for the DoE
NUM_REACTORS = len(DOE_FACTOR_LEVELS)

sim_bioreactors = [
    Bioreactor(
        vessel=f"BR{i}",
        process_model=simulation_model,
        data=reference_data,
    )
    for i in range(1, NUM_REACTORS + 1)
]
controllers = [
    Controller(
        controller_model=simulation_model,
        bioreactor=sim_bioreactors[count],
        ts=ts,
        pv_sps=pv_sps,
        pv_names=PV_NAMES,
        pv_wts=PV_WTS,
        mv_names=MV_NAMES,
        mv_wts=MV_WTS,
        pred_horizon=PRED_HORIZON,
        ctrl_horizon=CTRL_HORIZON,
        constr=MV_BOUNDS,
    )
    for count, _ in enumerate(sim_bioreactors)
]

# Change bioreactor data based on DoE setpoints
for count, bioreactor in enumerate(sim_bioreactors):
    # bioreactor.data["pH_setpoint"].iloc[3:] = DOE_setpoints[count][0]
    # bioreactor.data["Temperature"].iloc[3:] = DOE_setpoints[count][1]

    # Change day 0's iVCC setpoints
    bioreactor.data["VCC"].iloc[0] = DOE_FACTOR_LEVELS[count][0]

    # Change day 3's pH and Temp setpoints
    bioreactor.data["pH_setpoint"].iloc[3] = DOE_FACTOR_LEVELS[count][1]
    bioreactor.data["Temperature"].iloc[3] = DOE_FACTOR_LEVELS[count][2]

    # Change day 9's pH and Temp setpoints
    bioreactor.data["pH_setpoint"].iloc[9] = DOE_FACTOR_LEVELS[count][1]
    bioreactor.data["Temperature"].iloc[9] = DOE_FACTOR_LEVELS[count][2]

    # Open loop simulation
    _, bioreactor.open_loop_df = bioreactor.sim_from_curr_day()

# Simulate all the bioreactors and each controller and get a dictionary output
DOE_dict = {}
for bioreactor, controller in zip(sim_bioreactors, controllers):
    for i in range(len(ts) - 1):
        controller.optimize(open_loop=False)
        bioreactor.next_day()
    DOE_dict[bioreactor.vessel] = bioreactor.return_data()

# Plot the in-silico Simulations
br_plots = MPCVisualizer(sim_bioreactors, controllers)
br_plots.plot_simulations()
# br_plots.output_table().to_clipboard()

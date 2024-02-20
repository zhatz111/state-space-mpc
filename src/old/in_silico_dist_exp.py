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
import datetime

# 3rd Party Library Imports
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# State-Space-Model Package Imports
from mpc.mpc_optimizer import Bioreactor, Controller
from models.ssm import StateSpaceModel

warnings.filterwarnings("ignore", category=UserWarning)

# Load an example dataset
BATCH_SHEET_FOLDER = "mpc-simulation"
BATCH_SHEET_NAME = "disturbance"
user_path = Path("~").expanduser()
user_name = user_path.parts[-1]
date_str = datetime.datetime.today().strftime("%y%m%d")
data_path = Path(user_path,"GSK","Biopharm Model Predictive Control - General","data")
simulation_path = Path(data_path, BATCH_SHEET_FOLDER)
data = pd.read_csv(Path(simulation_path, rf"{BATCH_SHEET_NAME}.csv"))

# Create output folder
batch_sheet_path = Path(simulation_path.expanduser(), BATCH_SHEET_NAME)
batch_sheet_path.mkdir(parents=True, exist_ok=True)

# Load the simulation model for bioreactor
SIM_MODEL_DIR = "CD96-Robustness_Sim_Model_Matrices"
SIM_MODEL_STATES = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
    "Osmo",
    "pCO2_at_Temp",
]
SIM_MODEL_INPUTS = [
    "Cumulative_Normalized_Feed",
    "Temperature",
    "pH_setpoint",
    "DO",
]

# Import both matrices and the scaler for the data
sim_model_scaler = joblib.load(Path(data_path,SIM_MODEL_DIR,"model_scaler.scl"))
sim_A_matrix = np.array(
    pd.read_csv(Path(data_path,SIM_MODEL_DIR,'A_Matrix.csv'),header=None)
)
sim_B_matrix = np.array(
    pd.read_csv(Path(data_path,SIM_MODEL_DIR,'B_Matrix.csv'),header=None)
)

# Instantiate the StateSpaceModel object
sim_model = StateSpaceModel(
    states=SIM_MODEL_STATES,
    inputs=SIM_MODEL_INPUTS,
    scaler=sim_model_scaler,
    a_matrix=sim_A_matrix,
    b_matrix=sim_B_matrix,
    name=SIM_MODEL_DIR
)

# Load the controller model for control
CTRL_MODEL_DIR = "CD96-Robustness_Sim_Model_Matrices"
CTRL_MODEL_STATES = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
    "Osmo",
    "pCO2_at_Temp",
]
CTRL_MODEL_INPUTS = [
    "Cumulative_Normalized_Feed",
    "Temperature",
    "pH_setpoint",
    "DO",
]
# Import both matrices and the scaler for the data
ctrl_model_scaler = joblib.load(Path(data_path,CTRL_MODEL_DIR,"model_scaler.scl"))
ctrl_A_matrix = np.array(
    pd.read_csv(Path(data_path,CTRL_MODEL_DIR,'A_Matrix.csv'),header=None)
)
ctrl_B_matrix = np.array(
    pd.read_csv(Path(data_path,CTRL_MODEL_DIR,'B_Matrix.csv'),header=None)
)

# Instantiate the StateSpaceModel object
ctrl_model = StateSpaceModel(
    states=CTRL_MODEL_STATES,
    inputs=CTRL_MODEL_INPUTS,
    scaler=ctrl_model_scaler,
    a_matrix=ctrl_A_matrix,
    b_matrix=ctrl_B_matrix,
    name=CTRL_MODEL_DIR
)

# Controller parameters
ts = np.array(data["Day"])
pv_names = ["IGG"]
pv_wts = np.array([1 / (1000) ** 2])
pv_sps = data[pv_names].values
mv_names = [
    "Cumulative_Normalized_Feed",
    # 'pH_setpoint'
]
mv_wts = np.array(
    [
        1 / (0.01) ** 2,
        # 1
    ]
)
constr = np.array(
    [
        [0, 0.1],  # feed
        # [7,   7.35]     # pH
    ]
)

PRED_HORIZON = 30
CTRL_HORIZON = 3

# Construct a bioreactor object
bioreactor = Bioreactor(vessel="BR1-MPC", process_model=sim_model, data=data,
                        plot_names=pv_names,plot_sps=pv_sps,plot_ts=ts)

# Construct an open-loop bioreactor object
bioreactor_open_loop = copy.deepcopy(bioreactor)
bioreactor_open_loop.vessel = "BR1-Open_Loop"

controller = Controller(
    controller_model=ctrl_model,
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

# Simulate trajectory without MPC
# bioreactor_open_loop.next_day()

# Reset
# bioreactor.reset()

# Simulate a process
for i in range(len(ts) - 1):
    controller.optimize(plot=True)
    # bioreactor.show_data()
    bioreactor_open_loop.next_day(plot=True)
    bioreactor.next_day()

plt.show()

for k, _ in enumerate(bioreactor.figs):
    fig = bioreactor.figs[j]
    fig.savefig(
        Path(batch_sheet_path, rf"{fig._suptitle.get_text()}_br-{date_str}_{user_name}.png")
    )  # pylint: disable=protected-access
    plt.close(fig)

for j, _ in enumerate(controller.figs):
    fig = controller.figs[j]
    fig.savefig(
        Path(batch_sheet_path, rf"{fig._suptitle.get_text()}_ct-{date_str}_{user_name}.png")
    )  # pylint: disable=protected-access
    plt.close(fig)

bioreactor_open_loop.show_data()
bioreactor.show_data()

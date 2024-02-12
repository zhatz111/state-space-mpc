"""Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2024-01-18
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Standard Library Imports
import sys
import warnings
from pathlib import Path
from datetime import datetime

# 3rd Party Library Imports
import numpy as np
import pandas as pd

# State-Space-Model Package Imports
from mpc.mpc_optimizer import Bioreactor, Controller
from visualization.visualize import MPCVisualizer
from models.ssm import StateSpaceModel, json_toscaler

warnings.filterwarnings("ignore")

# Store todays date and the top level directory in variables
todays_date = datetime.today().strftime("%Y-%m-%d")
top_dir = Path().absolute()

# -------------------------------------------------------------------------------------
# USER SPECIFIED DATA

# Specify the study number, current time and vessel
EXP_NUM = "AR24-005"
CURR_TIME = 3
VESSEL = 1  # want to eliminate this and add a for loop for all reactors

# Specify names for batch sheet parent folder and master sheet
SIM_FOLDER = "mpc-simulation"
SIM_REFERENCE_DATA = "master_sheet_example-D2"

# Specify batch sheet path and load the read-only "master" sheet
simulation_path = Path(
    "~/GSK/Biopharm Model Predictive Control - General/data/", SIM_FOLDER
)
reference_data_all = pd.read_csv(
    Path(top_dir, f"data/simulation/{EXP_NUM}", f"{SIM_REFERENCE_DATA}.csv")
)

# -------------------------------------------------------------------------------------
# LOAD MODEL DATA

# Import the MinMaxScaler Json file
json_scaler_path = Path(top_dir, f"data/parameters/{EXP_NUM}", "model_scaler.json")
sim_model_scaler = json_toscaler(json_file=json_scaler_path)

# Import the A and B matrix CSV files
sim_A_matrix = np.array(
    pd.read_csv(
        Path(top_dir, f"data/parameters/{EXP_NUM}", "A_Matrix.csv"),
        header=None,
    )
)

sim_B_matrix = np.array(
    pd.read_csv(
        Path(top_dir, f"data/parameters/{EXP_NUM}", "B_Matrix.csv"),
        header=None,
    )
)

# -------------------------------------------------------------------------------------
# DIRECTORY CREATION AND PARSING OF STATES & INPUTS

# Create output folder
batch_sheet_path = Path(simulation_path.expanduser(), SIM_REFERENCE_DATA)
batch_sheet_path.mkdir(parents=True, exist_ok=True)

# Parse the states from the reference data csv file
reference_data_this_vessel = reference_data_all.loc[
    reference_data_all["Bioreactor"] == VESSEL, :
]
contains_state_data = reference_data_this_vessel.columns.str.contains("--STATE_DATA")
contains_input = reference_data_this_vessel.columns.str.contains("--INPUT_DATA")

# store the states and inputs as a list
STATES = [
    x.split("--")[0] for x in reference_data_this_vessel.columns[contains_state_data]
]
INPUTS = [x.split("--")[0] for x in reference_data_this_vessel.columns[contains_input]]

# Parse the PV and MV names from the reference data csv file
PV_SUFFIX = "--STATE_SP"
MV_SUFFIX = "--INPUT_REF"
contains_PV = reference_data_this_vessel.columns.str.contains(PV_SUFFIX, case=False)
contains_MV = reference_data_this_vessel.columns.str.contains(MV_SUFFIX, case=False)

# Define the PV and MV names using the parsing from csv file
pv_names = [x.split("--")[0] for x in reference_data_this_vessel.columns[contains_PV]]
mv_names = [x.split("--")[0] for x in reference_data_this_vessel.columns[contains_MV]]

# Define the suffix after each MV name and index the matrix with these names
pv_sps = reference_data_this_vessel[[mv + PV_SUFFIX for mv in pv_names]].values
mv_matrix = reference_data_this_vessel[[mv + MV_SUFFIX for mv in mv_names]].values

# -------------------------------------------------------------------------------------
# MODEL, BIOREACTOR, & CONTROLLER CLASS INSTANTIATION

# Create a state space model for control (identical to bioreactor sim model)
controller_model = StateSpaceModel(
    states=STATES,
    inputs=INPUTS,
    scaler=sim_model_scaler,
    a_matrix=sim_A_matrix,
    b_matrix=sim_B_matrix,
)

# Construct a bioreactor object
bioreactor = Bioreactor(
    vessel=VESSEL, process_model=controller_model, data=reference_data_this_vessel
)

# Construct a controller object
PRED_HORIZON = 30
CTRL_HORIZON = 3
EST_HORIZON = 2
ts = np.array(reference_data_this_vessel["Day"])
PV_WTS = np.array([1 / (1000) ** 2])
MV_WTS = np.array([1 / (0.01) ** 2])
MV_BOUNDS = np.array([[0, 0.1]])  # feed
EST_WTS = np.array(
    [
        2.5e-07,  # IGG
        0.08,  # VCC
        0.03,  # Viability
        0.05,  # Lactate
        0.007,  # OSMO
        0.001,  # CO2
    ]
)

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
    filter_wt_on_data=0.75,
    est_wts=EST_WTS,
    est_horizon=EST_HORIZON,
)

# -------------------------------------------------------------------------------------
# MAIN MPC LOOP ESTIMATES & OPTIMIZES EACH BIOREACTOR

# Update the time cursor
bioreactor.curr_time = CURR_TIME

# Estimate CURR_DAY's state

# Update bioreactor.data>STATE_MOD (curr day)
controller.estimate()
# Update bioreactor.data>STATE_EST (curr day + 1)
bioreactor.next_day()
# Update bioreactor.data>STATE_PRED (curr day + 1:end of pred horizon)
controller.optimize(open_loop=False)

# -------------------------------------------------------------------------------------
# BIOREACTOR DATA SAVED

try:
    df_br_daily = pd.read_csv(
        batch_sheet_path / f"{EXP_NUM}-BR{bioreactor.vessel:02d}-daily_feed.csv"
    )
    if todays_date in df_br_daily["Simulation_Date"].unique():
        df_br_daily.loc[
            df_br_daily["Simulation_Date"] == todays_date, :
        ] = bioreactor.return_data(show_daily_feed=True, sim_date_col=True)
    else:
        df_new = bioreactor.return_data(show_daily_feed=False, sim_date_col=True)
        df_br_daily = pd.concat([df_br_daily, df_new])
    df_br_daily.to_csv(
        batch_sheet_path / f"{EXP_NUM}-BR{bioreactor.vessel:02d}-daily_feed.csv",
    )
except FileNotFoundError:
    bioreactor.return_data(show_daily_feed=True, sim_date_col=True).to_csv(
        batch_sheet_path / f"{EXP_NUM}-BR{bioreactor.vessel:02d}-daily_feed.csv"
    )

try:
    df_br_total = pd.read_csv(
        batch_sheet_path / f"{EXP_NUM}-BR{bioreactor.vessel:02d}-total_feed.csv"
    )
    if todays_date in df_br_total["Simulation_Date"].unique():
        df_br_total.loc[
            df_br_total["Simulation_Date"] == todays_date, :
        ] = bioreactor.return_data(show_daily_feed=False, sim_date_col=True)
    else:
        df_new = bioreactor.return_data(show_daily_feed=False, sim_date_col=True)
        df_br_total = pd.concat([df_br_total, df_new])
    df_br_total.to_csv(
        batch_sheet_path / f"{EXP_NUM}-BR{bioreactor.vessel:02d}-total_feed.csv"
    )
except FileNotFoundError:
    bioreactor.return_data(show_daily_feed=False, sim_date_col=True).to_csv(
        batch_sheet_path / f"{EXP_NUM}-BR{bioreactor.vessel:02d}-total_feed.csv"
    )

# -------------------------------------------------------------------------------------
# GENERATED PLOTS SAVED

# Plot the MPC Controller for each Bioreactor
br_plots = MPCVisualizer(bioreactor, controller)
br_plots.mpc_daily_plot(
    save_path=batch_sheet_path
    / f"BR{bioreactor.vessel:02d}_D{CURR_TIME}-{todays_date}.png",
    metadata={
        "Title": f"{EXP_NUM}-BR{bioreactor.vessel:02d}-D{CURR_TIME}",
        "Author": "Zach Hatzenbeller, Yu Luo",
        "Description": f"MPC plot for {EXP_NUM}. Developed within GSK R&D in BDSD",
        "Copyright": f"(c) GSK, R&D, BDSD {datetime.today().year}",
        "Creation Time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Software": f"Python v{sys.version}",
    },
)

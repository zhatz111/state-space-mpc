"""Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2024-02-26
"""

# Standard Library Imports
import sys
import glob
import warnings
from pathlib import Path
from datetime import datetime

# 3rd Party Library Imports
import yaml
import numpy as np
import pandas as pd
from InquirerPy.resolver import prompt

# State-Space-Model Package Imports
from src.mpc.mpc_optimizer import Bioreactor, Controller
from src.visualization.visualize import MPCVisualizer
from src.models.ssm import StateSpaceModel
from src.data.functions import json_toscaler


warnings.filterwarnings("ignore")

# Store todays date and the top level directory in variables
# todays_date = datetime.today().strftime("%Y-%m-%d")
todays_date = datetime.today().strftime("%y%m%d")
top_dir = Path().absolute()

# -------------------------------------------------------------------------------------
# USER SPECIFIED DATA

PARENT_FILE_PATH = Path(
    r"~\GSK\Biopharm Model Predictive Control - General\data"
).expanduser()
FOLDER_SEARCH_KEY = "Experiment"
matching_folders = [
    folder.name
    for folder in PARENT_FILE_PATH.iterdir()
    if folder.is_dir() and FOLDER_SEARCH_KEY in folder.name
]
questions = {
    "type": "list",
    "name": "folder",
    "message": "Which experiment are you running MPC for?",
    "choices": matching_folders,
}
answer = prompt(questions)
DATA_FOLDER_NAME = str(answer["folder"])

PATH_DIRECTORY = Path(PARENT_FILE_PATH, DATA_FOLDER_NAME)
yaml_files = glob.glob(str(Path(PATH_DIRECTORY, "*.yaml")))
yaml_data = open(yaml_files[0], "r", encoding="utf-8")
experiment_config = yaml.safe_load(yaml_data)
yaml_data.close()


# Specify the study number, measurement units, current time and vessel
# Units list contents must equal exactly the number of graphs being plotted
units_list = [
    "(mg/L)",
    "(MM cells/mL)",
    "(%)",
    "(g/L)",
    "(mOsm/kg)",
    "(mmHg)",
    "",
    "(\N{DEGREE SIGN}C)",
    "",
]

if len(experiment_config["Bioreactors"]) == 2 and experiment_config["Arange Bioreactors"]:
    # np.append(np.arange(1,25),999)  # [3,5,6,9,13,15,18,20] or np.arange(1,25)
    VESSELS = np.arange(
        experiment_config["Bioreactors"][0], experiment_config["Bioreactors"][1] + 1
    )
else:
    VESSELS = np.array(experiment_config["Bioreactors"])

data_path = Path(top_dir, f"data/simulation/{experiment_config['Experiment Number']}")
reference_data_all = pd.read_csv(Path(data_path, f"{experiment_config['Master Data File']}.csv"))

# -------------------------------------------------------------------------------------
# LOAD MODEL DATA

# Import the MinMaxScaler Json file
json_scaler_path = Path(
    top_dir,
    f"data/parameters/{experiment_config['Experiment Number']}",
    "model_scaler.json",
)
sim_model_scaler = json_toscaler(json_file=json_scaler_path)

# Import the A and B matrix CSV files
sim_a_matrix = np.array(
    pd.read_csv(
        Path(
            top_dir,
            f"data/parameters/{experiment_config['Experiment Number']}",
            "A_Matrix.csv",
        ),
        header=None,
    )
)

sim_b_matrix = np.array(
    pd.read_csv(
        Path(
            top_dir,
            f"data/parameters/{experiment_config['Experiment Number']}",
            "B_Matrix.csv",
        ),
        header=None,
    )
)

# -------------------------------------------------------------------------------------
# DIRECTORY CREATION AND PARSING OF STATES & INPUTS

# Create figure output folder
fig_path_top_dir = Path(PATH_DIRECTORY, experiment_config['Figures Folder'])
fig_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create a daily folder of all reactors
fig_path_lv2_day = Path(
    fig_path_top_dir.expanduser(), f"D{experiment_config['Current Day']}-{todays_date}"
)
fig_path_lv2_day.mkdir(parents=True, exist_ok=True)

for curr_vessel in VESSELS:
    # curr_vessel = 1  # want to eliminate this and add a for loop for all reactors

    # Create figure folder
    fig_path_lv2_BR = Path(fig_path_top_dir.expanduser(), f"BR{curr_vessel:02d}")
    fig_path_lv2_BR.mkdir(parents=True, exist_ok=True)

    # Parse the states from the reference data csv file
    reference_data_this_vessel = reference_data_all.loc[
        reference_data_all["Bioreactor"] == curr_vessel, :
    ]
    contains_state_data = reference_data_this_vessel.columns.str.contains(
        "--STATE_DATA"
    )
    contains_input = reference_data_this_vessel.columns.str.contains("--INPUT_DATA")

    # store the states and inputs as a list
    STATES = [
        x.split("--")[0]
        for x in reference_data_this_vessel.columns[contains_state_data]
    ]
    INPUTS = [
        x.split("--")[0] for x in reference_data_this_vessel.columns[contains_input]
    ]

    # Parse the PV and MV names from the reference data csv file
    PV_SUFFIX = "--STATE_SP"
    MV_SUFFIX = "--INPUT_REF"
    contains_PV = reference_data_this_vessel.columns.str.contains(PV_SUFFIX, case=False)
    contains_MV = reference_data_this_vessel.columns.str.contains(MV_SUFFIX, case=False)

    # Define the PV and MV names using the parsing from csv file
    pv_names = [
        x.split("--")[0] for x in reference_data_this_vessel.columns[contains_PV]
    ]
    mv_names = [
        x.split("--")[0] for x in reference_data_this_vessel.columns[contains_MV]
    ]

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
        a_matrix=sim_a_matrix,
        b_matrix=sim_b_matrix,
    )

    # Construct a bioreactor object
    bioreactor = Bioreactor(
        vessel=curr_vessel,
        process_model=controller_model,
        data=reference_data_this_vessel,
    )

    # Construct a controller object
    ts = np.array(reference_data_this_vessel["Day"])

    # 2024-02-28: original horizons
    PRED_HORIZON = 30
    CTRL_HORIZON = 3
    EST_HORIZON = 3

    # # 2024-02-25: original controller settings
    # PV_WTS = np.array([1 / (1000) ** 2])
    # MV_WTS = np.array([1 / (0.01) ** 2])
    # MV_BOUNDS = np.array([[0, 0.1]])  # feed

    # # 2024-02-28: increased the lower bound
    # PV_WTS = np.array([1 / (1000) ** 2])
    # MV_WTS = np.array([1 / (0.01) ** 2])
    # MV_BOUNDS = np.array([[0.002, 0.1]])  # feed

    # # 2024-02-29: decreased the upper bound
    # PV_WTS = np.array([1 / (1000) ** 2])
    # MV_WTS = np.array([1 / (0.01) ** 2])
    # MV_BOUNDS = np.array([[0.002, 0.03]])  # feed

    # # 2024-03-01: decreased the lower bound
    # PV_WTS = np.array([1 / (1000) ** 2])
    # MV_WTS = np.array([1 / (0.01) ** 2])
    # MV_BOUNDS = np.array([[0.001, 0.03]])  # feed

    # # 2024-03-02: brought the lower bound back to 0.002
    # PV_WTS = np.array([1 / (1000) ** 2])
    # MV_WTS = np.array([1 / (0.01) ** 2])
    # MV_BOUNDS = np.array([[0.002, 0.03]])  # feed

    # 2024-03-03: changed the upper bound to 0.05
    PV_WTS = np.array([1 / (1000) ** 2])
    MV_WTS = np.array([1 / (0.01) ** 2])
    MV_BOUNDS = np.array([[0.002, 0.05]])  # feed

    # 2024-02-22: original weights
    # EST_WTS = np.array(
    #     [
    #         2.5e-07,  # IGG
    #         0.08,  # VCC
    #         0.03,  # Viability
    #         0.05,  # Lactate
    #         0.007,  # OSMO
    #         0.001,  # CO2
    #     ]
    # )

    # 2024-02-23: increased lactate's weight
    # EST_WTS = np.array(
    #     [
    #         2.5e-07,  # IGG
    #         0.08,  # VCC
    #         0.03,  # Viability
    #         50,  # Lactate
    #         0.007,  # OSMO
    #         0.001,  # CO2
    #     ]
    # )

    # # 2024-02-25: increased IGG and VCC's weights
    # EST_WTS = np.array(
    #     [
    #         1e-06,  # IGG
    #         0.4,  # VCC
    #         0.03,  # Viability
    #         50,  # Lactate
    #         0.007,  # OSMO
    #         0.001,  # CO2
    #     ]
    # )

    # 2024-03-03: increased IGG and VCC's weights
    EST_WTS = np.array(
        [
            1e-05,  # IGG
            2,  # VCC
            10,  # Viability
            50,  # Lactate
            0.007,  # OSMO
            0.001,  # CO2
        ]
    )

    # # 2024-03-02: original weight
    # EST_FILTER_WT_ON_DATA = 0.75

    # 2024-03-03: increased weight
    EST_FILTER_WT_ON_DATA = 0.9

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
        filter_wt_on_data=EST_FILTER_WT_ON_DATA,
        est_wts=EST_WTS,
        est_horizon=EST_HORIZON,
    )

    

    # -------------------------------------------------------------------------------------
    # MAIN MPC LOOP ESTIMATES & OPTIMIZES EACH BIOREACTOR

    # Update the time cursor
    bioreactor.curr_time = experiment_config["Current Day"]

    # Estimate CURR_DAY's state

    # Update bioreactor.data>STATE_MOD (curr day)
    controller.estimate()
    # Update bioreactor.data>STATE_EST (curr day + 1)
    bioreactor.next_day()
    # Update bioreactor.data>STATE_PRED (curr day + 1:end of pred horizon)
    controller.optimize(open_loop=False)

    # -------------------------------------------------------------------------------------
    # BIOREACTOR DATA SAVED

    # Search if files are in directory
    filenames = [
        f"{experiment_config['Experiment Number']}-daily_feed.csv",
        f"{experiment_config['Experiment Number']}-total_feed.csv",
    ]
    dir_paths = [x.name for x in list(data_path.iterdir())]

    if all(item in filenames for item in dir_paths):
        # Read in CSV files
        df_br_daily = pd.read_csv(data_path / filenames[0])
        df_br_total = pd.read_csv(data_path / filenames[1])

        # Daily feed csv
        df_new_daily = bioreactor.return_data(show_daily_feed=True, exec_date=True)
        df_combined_daily = pd.concat([df_br_daily, df_new_daily], ignore_index=True)

        df_final_daily = df_combined_daily.drop_duplicates(
            subset=["Code_Run_Date", "Bioreactor", "Day"], keep="last"
        )
        df_final_daily.sort_values(by=["Code_Run_Date", "Bioreactor"], inplace=True)
        df_final_daily.to_csv(data_path / filenames[0], index=False)

        # Total feed csv
        df_new_total = bioreactor.return_data(show_daily_feed=False, exec_date=True)
        df_combined_total = pd.concat([df_br_total, df_new_total], ignore_index=True)

        df_final_total = df_combined_total.drop_duplicates(
            subset=["Code_Run_Date", "Bioreactor", "Day"], keep="last"
        )
        df_final_total.sort_values(by=["Code_Run_Date", "Bioreactor"], inplace=True)
        df_final_total.to_csv(data_path / filenames[1], index=False)
    else:
        # If no file exist currently
        bioreactor.return_data(show_daily_feed=True, exec_date=True).to_csv(
            data_path / filenames[0], index=False
        )
        bioreactor.return_data(show_daily_feed=False, exec_date=True).to_csv(
            data_path / filenames[1], index=False
        )

    # -------------------------------------------------------------------------------------
    # GENERATED PLOTS SAVED

    # Plot the MPC Controller for each Bioreactor
    br_plots = MPCVisualizer(bioreactor, controller)
    br_plots.mpc_daily_plot(
        save_path=fig_path_lv2_BR
        / f"BR{bioreactor.vessel:02d}_D{experiment_config['Current Day']}-{todays_date}.png",
        identifier=f"{experiment_config['Experiment Number']} \
        -MPC/BR{bioreactor.vessel:02d}/BR{bioreactor.vessel:02d} \
        _D{experiment_config['Current Day']}-{todays_date}",

        unit_list=units_list,
        metadata={
            "Title": f"{experiment_config['Experiment Number']}-D{experiment_config['Current Day']}",
            "Author": "Zach Hatzenbeller, Yu Luo",
            "Description": f"MPC plot for {experiment_config['Experiment Number']}. Developed within GSK R&D in BDSD",
            "Copyright": f"(c) GSK, R&D, BDSD {datetime.today().year}",
            "Creation Time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Software": f"Python v{sys.version}",
        },
        display=False,
    )
    br_plots.mpc_daily_plot(
        save_path=fig_path_lv2_day
        / f"BR{bioreactor.vessel:02d}_D{experiment_config['Current Day']}-{todays_date}.png",
        identifier=f"{experiment_config['Experiment Number']} \
        -MPC/BR{bioreactor.vessel:02d}/BR{bioreactor.vessel:02d} \
        _D{experiment_config['Current Day']}-{todays_date}",
        unit_list=units_list,
        metadata={
            "Title": f"{experiment_config['Experiment Number']}-D{experiment_config['Current Day']}",
            "Author": "Zach Hatzenbeller, Yu Luo",
            "Description": f"MPC plot for {experiment_config['Experiment Number']}. Developed within GSK R&D in BDSD",
            "Copyright": f"(c) GSK, R&D, BDSD {datetime.today().year}",
            "Creation Time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Software": f"Python v{sys.version}",
        },
        display=False,
    )

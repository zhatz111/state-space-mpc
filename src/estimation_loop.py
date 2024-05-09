"""Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2024-04-29
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
from src.data.functions import dict_toscaler, json_to_dict #, dict_to_json


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
    "",
    "(\N{DEGREE SIGN}C)",
    "",
    "",
]

if (
    len(experiment_config["Bioreactors"]) == 2
    and experiment_config["Arange Bioreactors"]
):
    # np.append(np.arange(1,25),999)  # [3,5,6,9,13,15,18,20] or np.arange(1,25)
    VESSELS = np.arange(
        experiment_config["Bioreactors"][0], experiment_config["Bioreactors"][1] + 1
    )
else:
    VESSELS = np.array(experiment_config["Bioreactors"])

reference_data_all = pd.read_csv(
    Path(PATH_DIRECTORY, f"{experiment_config['Master Data File']}.csv")
)

# -------------------------------------------------------------------------------------
# LOAD MODEL DATA

# Import the MinMaxScaler Json file
json_parameters_path = Path(
    PATH_DIRECTORY,
    f"{experiment_config['Experiment Number']}_model_parameters.json",
)

model_parameters = json_to_dict(json_parameters_path)

# Import the A and B matrix CSV files
sim_a_matrix = np.array(model_parameters["a_matrix"])
sim_b_matrix = np.array(model_parameters["b_matrix"])
# Create Scaler object from parameters
sim_model_scaler = dict_toscaler(dict_file=model_parameters["scaler"])

# -------------------------------------------------------------------------------------
# DIRECTORY CREATION AND PARSING OF STATES & INPUTS

# Create figure output folder
fig_path_top_dir = Path(PATH_DIRECTORY, experiment_config["Figures Folder"])
fig_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create a daily folder of all reactors
fig_path_lv2_day = Path(
    fig_path_top_dir.expanduser(), f"D{experiment_config['Current Day']}-{todays_date}"
)
fig_path_lv2_day.mkdir(parents=True, exist_ok=True)

for curr_vessel in VESSELS:
    for key, value in experiment_config["Controller Dictionary"].items():
        if curr_vessel in value:
            controller_key = key

    controller_config = json_to_dict(
        Path(PATH_DIRECTORY, "Controllers", f"{controller_key}.json")
    )

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
        states=list(map(str.upper,model_parameters["Model States"])),
        inputs=list(map(str.upper,model_parameters["Model Inputs"])),
        scaler=sim_model_scaler,
        a_matrix=sim_a_matrix,
        b_matrix=sim_b_matrix,
    )

    bioreactor_config = {
        "Batch Length": 13,
        "Manipulated Variables": "CUMULATIVE_NORMALIZED_FEED",
        # "Constraints": {
        #     "Viability": [80,100]
        # }
    }

    # Construct a bioreactor object
    bioreactor = Bioreactor(
        vessel=curr_vessel,
        process_model=controller_model,
        data=reference_data_this_vessel,
        config=bioreactor_config
    )

    # Verify dimensions (YL@2024-01-18)
    if len(np.array(controller_config["Estimation Weights"])) != len(STATES):
        raise ValueError("Wrong estimation weights dimension!")
    if len(np.array(controller_config["Process Variable Weights"])) != len(pv_names):
        raise ValueError("Wrong PV weights dimension!")
    if len(np.array(controller_config["Manipulated Variable Weights"])) != len(mv_names):
        raise ValueError("Wrong PV weights dimension!")
    if [x.upper() for x in sim_model_scaler.get_feature_names_out()] != STATES + INPUTS:
        raise ValueError("Model and CSV do not match!")

    controller = Controller(
        controller_model=controller_model,
        bioreactor=bioreactor,
        ts=np.array(controller_config["Time"]),
        pv_sps=np.array(controller_config["Process Variable Setpoints"]),
        pv_names=controller_config["Process Variables"],
        pv_wts=np.array(controller_config["Process Variable Weights"]),
        mv_names=controller_config["Manipulated Variables"],
        mv_wts=np.array(controller_config["Manipulated Variable Weights"]),
        pred_horizon=controller_config["Prediction Horizon"],
        ctrl_horizon=controller_config["Control Horizon"],
        est_horizon=controller_config["Estimation Horizon"],
        est_wts=np.array(controller_config["Estimation Weights"]),
        mv_constr=np.array(controller_config["Constraints"]),  # feed
        output_mods_user=np.array(controller_config["User Specified Modifiers"]),
        filter_wt_on_data=controller_config["Estimation Filter Weight on Data"],
        eor_names=controller_config["End of Run Names"],
        eor_constr=np.array(controller_config["End of Run Constraints"])
    )

    # Used to create a dictionary for the controller to serialize to json
    # Created 2024-04-29 by Zach Hatzenbeller

    # controller_dict = {
    #     "Time": ts.tolist(),
    #     "Process Variable Setpoints": pv_sps.tolist(),
    #     "Process Variables": pv_names,
    #     "Process Variable Weights": PV_WTS.tolist(),
    #     "Manipulated Variables": mv_names,
    #     "Manipulated Variable Weights": MV_WTS.tolist(),
    #     "Prediction Horizon": PRED_HORIZON,
    #     "Control Horizon": CTRL_HORIZON,
    #     "Estimation Horizon": EST_HORIZON,
    #     "Constraints": MV_BOUNDS.tolist(),
    #     "User Specified Modifiers": [],
    #     "Estimation Filter Weight on Data": EST_FILTER_WT_ON_DATA,
    #     "Estimation Weights": EST_WTS.tolist(),
    #     "Controller States": STATES,
    # }
    # dict_to_json(Path(PATH_DIRECTORY, "test_controller.json"), controller_dict)

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
    dir_paths = [x.name for x in list(PATH_DIRECTORY.iterdir())]

    if all(item in filenames for item in dir_paths):
        # Read in CSV files
        df_br_daily = pd.read_csv(PATH_DIRECTORY / filenames[0])
        df_br_total = pd.read_csv(PATH_DIRECTORY / filenames[1])

        # Daily feed csv
        df_new_daily = bioreactor.return_data(show_daily_feed=True, exec_date=True)
        df_combined_daily = pd.concat([df_br_daily, df_new_daily], ignore_index=True)

        df_final_daily = df_combined_daily.drop_duplicates(
            subset=["Code_Run_Date", "Bioreactor", "Day"], keep="last"
        )
        df_final_daily.sort_values(by=["Code_Run_Date", "Bioreactor"], inplace=True)
        df_final_daily.to_csv(PATH_DIRECTORY / filenames[0], index=False)

        # Total feed csv
        df_new_total = bioreactor.return_data(show_daily_feed=False, exec_date=True)
        df_combined_total = pd.concat([df_br_total, df_new_total], ignore_index=True)

        df_final_total = df_combined_total.drop_duplicates(
            subset=["Code_Run_Date", "Bioreactor", "Day"], keep="last"
        )
        df_final_total.sort_values(by=["Code_Run_Date", "Bioreactor"], inplace=True)
        df_final_total.to_csv(PATH_DIRECTORY / filenames[1], index=False)
    else:
        # If no file exist currently
        bioreactor.return_data(show_daily_feed=True, exec_date=True).to_csv(
            PATH_DIRECTORY / filenames[0], index=False
        )
        bioreactor.return_data(show_daily_feed=False, exec_date=True).to_csv(
            PATH_DIRECTORY / filenames[1], index=False
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
            "Description": f"MPC plot for {experiment_config['Experiment Number']}. \
            Developed within GSK R&D in BDSD",
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

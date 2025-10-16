"""
Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2025-08-27
"""

# Standard Library Imports
import sys
import warnings
from pathlib import Path
from datetime import datetime

# 3rd Party Library Imports
import numpy as np
import pandas as pd
from termcolor import colored
from InquirerPy.resolver import prompt

# State-Space-Model Package Imports
from mpc.mpc_optimizer import Bioreactor, Controller
from visualization.visualize import MPCVisualizer
from models.ssm import StateSpaceModel
from data.functions import dict_toscaler, read_config, json_to_dict

warnings.filterwarnings("ignore")

# Store todays date and the top level directory in variables
todays_date = datetime.today().strftime("%y%m%d")
top_dir = Path().absolute()

# -------------------------------------------------------------------------------------
# USER SPECIFIED DATA



# User specified current culture day: determined automatically if -1
CURR_TIME_USER = 0
SHOW_PLOT = True




# Use this path for experiment folders in GitHub repository
PARENT_FILE_PATH = top_dir / "data"

# Locate the experiment folder by searching for the KEY (case insensitive)
FOLDER_SEARCH_KEY = "Experiment"
matching_folders = [
    folder.name
    for folder in PARENT_FILE_PATH.iterdir()
    if folder.is_dir() and FOLDER_SEARCH_KEY.upper() in folder.name.upper()
]
questions = {
    "type": "list",
    "name": "folder",
    "message": "Which experiment are you running MPC for?",
    "choices": matching_folders,
}
answer = prompt(questions)
DATA_FOLDER_NAME = str(answer["folder"])

# Load config
PATH_DIRECTORY = Path(PARENT_FILE_PATH, DATA_FOLDER_NAME)
experiment_config = read_config(PATH_DIRECTORY)

# Set up vessels
vessels = np.array(experiment_config["Vessel ID"])

# Load the JSON file with all MPC payloads similar to vectors from kafka
if ".json" in experiment_config["Master Data File"]:
    master_sheet = json_to_dict(
        Path(PATH_DIRECTORY, experiment_config["Master Data File"]),
    )
else:
    raise ValueError("Wrong Master Sheet File extension!")

# -------------------------------------------------------------------------------------
# LOAD MODEL DATA

# Import the scaler
sim_model_scaler = dict_toscaler(
    dict_file=experiment_config["Model Parameters"]["scaler"]
)

# Create a state space model for control
controller_model = StateSpaceModel(
    model_parameters=experiment_config["Model Parameters"],
    scaler=sim_model_scaler,
)

# -------------------------------------------------------------------------------------
# ITERATE FROM DAY 0 TO THE CURRENT DAY

if "Inoc Date" in experiment_config and CURR_TIME_USER < 0:
    date_delta = (
        datetime.today().date()
        - datetime.strptime(experiment_config["Inoc Date"], "%Y-%m-%d").date()
    )
    CURR_TIME_END = np.min((experiment_config["Last Day"], date_delta.days))
else:
    CURR_TIME_END = CURR_TIME_USER

# Create figure output folder
fig_path_top_dir = Path(PATH_DIRECTORY, "figures")
fig_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create CSV data output folder
csv_path_top_dir = Path(PATH_DIRECTORY, "data")
csv_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create a daily figures folder of all reactors
fig_path_lv2_day = Path(
    fig_path_top_dir.expanduser(), f"D{CURR_TIME_END}-{todays_date}"
)
fig_path_lv2_day.mkdir(parents=True, exist_ok=True)

# Create bioreactor and controller instances
bioreactors = []
controllers = []
for curr_vessel in vessels:

    # The bioreactor controller will always be named "Controller"
    controller_key = "Controller"
    controller_config = experiment_config[controller_key]

    # Initialize a bioreactor instance
    bioreactor = Bioreactor(
        vessel=curr_vessel,
        process_model=controller_model,
        experiment_config=experiment_config,
        controller_config=controller_config,
    )
    bioreactors.append(bioreactor)

    # Initialize the controller for each bioreactor (2024-06-20)
    controller = Controller(
        controller_model=controller_model,
        bioreactor=bioreactor,
        controller_config=controller_config,
    )
    controllers.append(controller)

# Iterate the main code for each bioreactor
for count_vessel, curr_vessel in enumerate(vessels):
    # Create bioreactor-specific figure output folders
    if isinstance(curr_vessel, str):
        fig_path_lv2_BR = Path(fig_path_top_dir.expanduser(), curr_vessel)
    else:
        fig_path_lv2_BR = Path(fig_path_top_dir.expanduser(), f"BR{curr_vessel:02d}")
    fig_path_lv2_BR.mkdir(parents=True, exist_ok=True)

    # Create bioreactor-specific CSV output folders
    if isinstance(curr_vessel, str):
        csv_path_lv2_BR = Path(csv_path_top_dir.expanduser(), curr_vessel)
    else:
        csv_path_lv2_BR = Path(csv_path_top_dir.expanduser(), f"BR{curr_vessel:02d}")
    csv_path_lv2_BR.mkdir(parents=True, exist_ok=True)

    # Retrieve bioreactor and controller instances
    bioreactor = bioreactors[count_vessel]
    controller = controllers[count_vessel]

    print("")
    print("-" * 80)

    # Parse the states from the reference data csv file
    input_messages = {}
    input_message_dict = {}
    # Iterate from Day 0 to the current day
    for curr_time in range(0, CURR_TIME_END + 1):
        # -------------------------------------------------------------------------------------
        # MAIN MPC LOOP ESTIMATES & OPTIMIZES EACH BIOREACTOR

        # Ingest data from Input topic
        bioreactor.curr_time = curr_time
        vector_key = f"{curr_vessel}-{experiment_config["Batch ID"]}-day{curr_time}"
        input_message = master_sheet[vector_key]
        input_message_dict[bioreactor.curr_time] = input_message
        bioreactor.ingest_vectors(input_message_dict)

        # Estimate the current state
        controller.estimate()

        # Print current optimization result at the end of the time loop
        if curr_time == CURR_TIME_END:
            PRINT_PRED = True
        else:
            PRINT_PRED = False

        # Do not optimize at EoR
        if curr_time == experiment_config["Last Day"]:
            END_OF_RUN = True
        else:
            END_OF_RUN = False

        if curr_time < experiment_config["Last Day"]:
            controller.optimize(
                open_loop=False, print_pred=PRINT_PRED, end_of_run=END_OF_RUN
            )

    # Retrieve and print current feed rate (mL/min) for the feed pump
    result = bioreactor.get_result()
    print("")
    br_heading = f"{curr_vessel} on Day {curr_time}:"
    # print("." * len(br_heading))
    print(colored(br_heading, "green"))
    for key, value in result.items():
        print(key, end=None)
        print(colored(np.round(value, 4), "blue", attrs=["bold"]))

    # -------------------------------------------------------------------------------------
    # GENERATED PLOTS SAVED

    # Plot the MPC Controller for each Bioreactor
    # Use the last controller from the list which is the controller from the current day

    br_plots = MPCVisualizer(bioreactor, controller)

    if isinstance(curr_vessel, str):
        identifier = f"{bioreactor.vessel}_D{CURR_TIME_END}-{todays_date}"
    else:
        identifier = f"BR{bioreactor.vessel:02d}_D{CURR_TIME_END}-{todays_date}"

    br_plots.mpc_daily_plot(
        save_paths=(
            fig_path_lv2_BR / f"{identifier}.png",
            fig_path_lv2_day / f"{identifier}.png",
        ),
        identifier=f"{experiment_config['Batch ID']} \
        -MPC/{identifier}",
        unit_dict=experiment_config["Units Dictionary"],
        metadata={
            "Title": f"{experiment_config['Batch ID']}-D{CURR_TIME_END}",
            "Author": "Zach Hatzenbeller, Yu Luo",
            "Description": f"MPC plot for {experiment_config['Batch ID']}. \
            Developed within GSK R&D in BDSD",
            "Copyright": f"(c) GSK, R&D, BDSD {datetime.today().year}",
            "Creation Time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Software": f"Python v{sys.version}",
        },
        display=SHOW_PLOT,
    )

    # NEW CODE TO OUTPUT A SEPERATE CSV FILE EACH DAY FOR EACH REACTOR
    # DEVELOPED: 2024-06-06
    filenames = [
        f"{bioreactor.vessel}_D{CURR_TIME_END}-{todays_date}-daily_feed.csv",
        # f"{bioreactor.vessel}_D{curr_time_end}-{todays_date}-total_feed.csv",
    ]

    # If no file exist currently
    bioreactor.return_data(show_daily_inputs=True, exec_date=True).to_csv(
        csv_path_lv2_BR / filenames[0], index=False
    )

# Save and export the current config file
read_config(PATH_DIRECTORY, export=True)

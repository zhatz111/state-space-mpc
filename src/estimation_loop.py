"""
Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2024-05-13
"""

# Standard Library Imports
import sys
import glob
import warnings
import shutil
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
from src.data.functions import dict_toscaler

warnings.filterwarnings("ignore")

# Store todays date and the top level directory in variables
todays_date = datetime.today().strftime("%y%m%d")
top_dir = Path().absolute()

# -------------------------------------------------------------------------------------
# USER SPECIFIED DATA

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
def read_config(export=False):
    """
    The `read_config` function reads a YAML file containing experiment configuration data and returns
    the parsed configuration.
    
    Returns:
      The function `read_config` returns the experiment configuration data loaded from the first YAML
    file found in the specified directory.
    """
    yaml_files = glob.glob(str(Path(PATH_DIRECTORY, "*.yaml")))
    yaml_data = open(yaml_files[0], "r", encoding="utf-8")
    yaml_config = yaml.safe_load(yaml_data)
    yaml_data.close()

    if export:
        src_path = Path(yaml_files[0])
        dest_file = f"{src_path.stem}-{todays_date}{src_path.suffix}"
        csv_path_top_dir = Path(PATH_DIRECTORY, yaml_config["CSV Export Folder"])
        csv_path_top_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path,Path(csv_path_top_dir,dest_file))

    return yaml_config

experiment_config = read_config()

# Set up vessels
if (
    len(experiment_config["Bioreactors"]) == 2
    and experiment_config["Arange Bioreactors"]
):
    vessels = np.arange(
        experiment_config["Bioreactors"][0], experiment_config["Bioreactors"][1] + 1
    )
else:
    vessels = np.array(experiment_config["Bioreactors"])

# Load the master data sheet (mock INPUT TOPIC storing table)
if '.xlsx' in experiment_config['Master Data File']:
    master_sheet = pd.read_excel(
        Path(PATH_DIRECTORY, experiment_config['Master Data File']),
        skiprows=[0],
    ).rename(
        columns={
            "Batch": "Bioreactor",
        }
    )
elif '.csv' in experiment_config['Master Data File']:
    master_sheet = pd.read_csv(
        Path(PATH_DIRECTORY, experiment_config['Master Data File'])
    )    
else:
    raise ValueError('Wrong master sheet file extension!')

# -------------------------------------------------------------------------------------
# LOAD MODEL DATA

# Import the A and B matrices and the scaler
sim_a_matrix = np.array(experiment_config["Model Parameters"]["a_matrix"])
sim_b_matrix = np.array(experiment_config["Model Parameters"]["b_matrix"])
sim_model_scaler = dict_toscaler(
    dict_file=experiment_config["Model Parameters"]["scaler"]
)

# Create a state space model for control
controller_model = StateSpaceModel(
    states=list(
        map(str.upper, experiment_config["Model Parameters"]["Model States"])
    ),
    inputs=list(
        map(str.upper, experiment_config["Model Parameters"]["Model Inputs"])
    ),
    scaler=sim_model_scaler,
    a_matrix=sim_a_matrix,
    b_matrix=sim_b_matrix,
)

# -------------------------------------------------------------------------------------
# ITERATE FROM DAY 0 TO THE CURRENT DAY

# User specified current culture day: determined automatically if -1
CURR_TIME_USER = -1
if "Inoc Date" in experiment_config and CURR_TIME_USER < 0:
    date_delta = datetime.today().date() - datetime.strptime(experiment_config["Inoc Date"],"%Y-%m-%d").date()
    curr_time_end = np.min((experiment_config["Last Day"],date_delta.days))
else:
    curr_time_end = CURR_TIME_USER

# Create figure output folder
fig_path_top_dir = Path(PATH_DIRECTORY, experiment_config["Figures Folder"])
fig_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create CSV data output folder
csv_path_top_dir = Path(PATH_DIRECTORY, experiment_config["CSV Export Folder"])
csv_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create a daily figures folder of all reactors
fig_path_lv2_day = Path(
    fig_path_top_dir.expanduser(), f"D{curr_time_end}-{todays_date}"
)
fig_path_lv2_day.mkdir(parents=True, exist_ok=True)

# Create bioreactor and controller instances
bioreactors = []
controllers = []
for curr_vessel in vessels:

    # Locate the bioreactor-specific controller setting
    for key, value in experiment_config["Controller Dictionary"].items():
        if curr_vessel in value:
            controller_key = key

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
        controller_config=controller_config
        )
    controllers.append(controller)

# Iterate the main code for each bioreactor
for count_vessel, curr_vessel in enumerate(vessels):

    # Create bioreactor-specific figure output folders
    if isinstance(curr_vessel,str):
        fig_path_lv2_BR = Path(fig_path_top_dir.expanduser(), curr_vessel)
    else:
        fig_path_lv2_BR = Path(fig_path_top_dir.expanduser(), f"BR{curr_vessel:02d}")
    fig_path_lv2_BR.mkdir(parents=True, exist_ok=True)

    # Create bioreactor-specific CSV output folders
    if isinstance(curr_vessel,str):
        csv_path_lv2_BR = Path(csv_path_top_dir.expanduser(), curr_vessel)
    else:
        csv_path_lv2_BR = Path(csv_path_top_dir.expanduser(), f"BR{curr_vessel:02d}")
    csv_path_lv2_BR.mkdir(parents=True, exist_ok=True)

    # Retrieve bioreactor and controller instances
    bioreactor = bioreactors[count_vessel]
    controller = controllers[count_vessel]

    # Iterate from Day 0 to the current day
    for curr_time in range(0,curr_time_end + 1):

        # Parse the states from the reference data csv file
        input_messages = master_sheet.loc[
            master_sheet["Bioreactor"] == curr_vessel, :
        ]

        # -------------------------------------------------------------------------------------
        # MAIN MPC LOOP ESTIMATES & OPTIMIZES EACH BIOREACTOR

        # Ingest data from Input topic
        bioreactor.curr_time = curr_time
        input_message = input_messages.loc[input_messages["Day"] == curr_time,:].squeeze()
        bioreactor.ingest_vector(input_message)

        # Estimate the current state
        controller.estimate()

        # Print current optimization result at the end of the time loop
        if curr_time == curr_time_end:
            print_pred = True
        else:
            print_pred = False

        # Do not optimize at EoR
        if curr_time == experiment_config["Last Day"]:
            end_of_run = True
        else:
            end_of_run = False            

        controller.optimize(open_loop=False,print_pred=print_pred,end_of_run=end_of_run)

    # Retrieve and print current feed rate (mL/min) for the feed pump
    result = bioreactor.get_result()
    print(f"Day {curr_time}'s feed rate (mL/min): {result['FeedRate_mL_min']}")

    # -------------------------------------------------------------------------------------
    # GENERATED PLOTS SAVED

    # Plot the MPC Controller for each Bioreactor
    # Use the last controller from the list which is the controller from the current day
    br_plots = MPCVisualizer(bioreactor, controller)

    if isinstance(curr_vessel,str):
        identifier = f"{bioreactor.vessel}_D{curr_time_end}-{todays_date}"
    else:
        identifier = f"BR{bioreactor.vessel:02d}_D{curr_time_end}-{todays_date}"

    br_plots.mpc_daily_plot(
        save_paths=(
            fig_path_lv2_BR / f"{identifier}.png",
            fig_path_lv2_day / f"{identifier}.png",
            ),
        identifier=f"{experiment_config['Experiment Number']} \
        -MPC/{identifier}",
        unit_dict=experiment_config["Units Dictionary"],
        metadata={
            "Title": f"{experiment_config['Experiment Number']}-D{curr_time_end}",
            "Author": "Zach Hatzenbeller, Yu Luo",
            "Description": f"MPC plot for {experiment_config['Experiment Number']}. \
            Developed within GSK R&D in BDSD",
            "Copyright": f"(c) GSK, R&D, BDSD {datetime.today().year}",
            "Creation Time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Software": f"Python v{sys.version}",
        },
        display=True,
    )
    
    # NEW CODE TO OUTPUT A SEPERATE CSV FILE EACH DAY FOR EACH REACTOR
    # DEVELOPED: 2024-06-06
    filenames = [
        f"{bioreactor.vessel}_D{curr_time_end}-{todays_date}-daily_feed.csv",
        # f"{bioreactor.vessel}_D{curr_time_end}-{todays_date}-total_feed.csv",
    ]

    # If no file exist currently
    bioreactor.return_data(show_daily_feed=True, exec_date=True).to_csv(
        csv_path_lv2_BR / filenames[0], index=False
    )

# Save and export the current config file
read_config(export=True) 
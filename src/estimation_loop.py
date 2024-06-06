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

# Use this path for experiment folders in MPC teams site
# PARENT_FILE_PATH = Path(
#     r"~\GSK\Biopharm Model Predictive Control - General\data"
# ).expanduser()

# Use this path for experiment folders in GitHub repository
PARENT_FILE_PATH = top_dir / "data"

# Locate the experiment folder by searching for the KEY
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

# Load config
PATH_DIRECTORY = Path(PARENT_FILE_PATH, DATA_FOLDER_NAME)
def read_config():
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
    return yaml_config

experiment_config = read_config()

# Specify the study number, measurement units, current time and vessel
# Units list contents must equal exactly the number of graphs being plotted

if (
    len(experiment_config["Bioreactors"]) == 2
    and experiment_config["Arange Bioreactors"]
):
    VESSELS = np.arange(
        experiment_config["Bioreactors"][0], experiment_config["Bioreactors"][1] + 1
    )
else:
    VESSELS = np.array(experiment_config["Bioreactors"])

# Load the master data sheet
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
# ITERATE FROM DAY 0 TO THE CURRENT DAY (SIMULATION)

# Mock current time (end of time iteration)
curr_time_user = -1
if "Inoc Date" in experiment_config and curr_time_user < 0:
    date_delta = datetime.today().date() - datetime.strptime(experiment_config["Inoc Date"],"%Y-%m-%d").date()
    curr_time_end = date_delta.days
else:
    curr_time_end = curr_time_user

# Create figure output folder
fig_path_top_dir = Path(PATH_DIRECTORY, experiment_config["Figures Folder"])
fig_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create a daily folder of all reactors
fig_path_lv2_day = Path(
    fig_path_top_dir.expanduser(), f"D{curr_time_end}-{todays_date}"
)
fig_path_lv2_day.mkdir(parents=True, exist_ok=True)

# Iterate bioreactors to initialize a bioreactor instance
bioreactors = []
for curr_vessel in VESSELS:

    # Locate the bioreactor-specific controller setting
    for key, value in experiment_config["Controller Dictionary"].items():
        if curr_vessel in value:
            controller_key = key

    controller_config = experiment_config[controller_key]

    # Initialize a bioreactor instance
    bioreactor = Bioreactor(
        vessel=curr_vessel,
        process_model=controller_model,
        # data=reference_data_this_vessel,
        experiment_config=experiment_config,
        controller_config=controller_config,
    )
    bioreactors.append(bioreactor)

# Iterate bioreactors
for count_vessel, curr_vessel in enumerate(VESSELS):

    # Create bioreactor-specific output folders
    if isinstance(curr_vessel,str):
        fig_path_lv2_BR = Path(fig_path_top_dir.expanduser(), curr_vessel)
    else:
        fig_path_lv2_BR = Path(fig_path_top_dir.expanduser(), f"BR{curr_vessel:02d}")
    fig_path_lv2_BR.mkdir(parents=True, exist_ok=True)

    # store Controller objects in list to use last one for plotting
    controller_list = []

    # Iterate times
    for curr_time in range(0,curr_time_end + 1):

        # Read config again every day for tuning parameter changes
        experiment_config = read_config()

        # Locate the bioreactor-specific controller setting
        for key, value in experiment_config["Controller Dictionary"].items():
            if curr_vessel in value:
                controller_key = key

        controller_config = experiment_config[controller_key]

        # Parse the states from the reference data csv file
        input_messages = master_sheet.loc[
            master_sheet["Bioreactor"] == curr_vessel, :
        ]

        # # Construct a bioreactor object
        # bioreactor = Bioreactor(
        #     vessel=curr_vessel,
        #     process_model=controller_model,
        #     # data=reference_data_this_vessel,
        #     experiment_config=experiment_config,
        #     controller_config=controller_config,
        # )

        # Create/update a controller instance
        bioreactor = bioreactors[count_vessel]
        controller = Controller(
            controller_model=controller_model,
            bioreactor=bioreactor,
            controller_config=controller_config
        )
        controller_list.append(controller)

        # -------------------------------------------------------------------------------------
        # MAIN MPC LOOP ESTIMATES & OPTIMIZES EACH BIOREACTOR

        # Ingest data from Input topic
        bioreactor.curr_time = curr_time
        input_message = input_messages.loc[input_messages["Day"] == curr_time,:].squeeze()
        bioreactor.ingest_vector(input_message)

        # Update bioreactor.data>STATE_MOD and STATE_EST (curr day)
        controller.estimate()

        # Update bioreactor.data>STATE_PRED (curr day to end of pred horizon)
        if curr_time == curr_time_end:
            PRINT_PRED = True
        else:
            PRINT_PRED = False

        controller.optimize(open_loop=False,print_pred=PRINT_PRED)

        result = bioreactor.get_result()

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
    # Use the last controller from the list which is the controller from the current day
    br_plots = MPCVisualizer(bioreactor, controller_list[-1])
    
    if isinstance(curr_vessel,str):
        identifier = f"{bioreactor.vessel}_D{curr_time_end}-{todays_date}"
    else:
        identifier = f"BR{bioreactor.vessel:02d}_D{curr_time_end}-{todays_date}"
    
    br_plots.mpc_daily_plot(
        save_path=fig_path_lv2_BR
        / f"{identifier}.png",
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
    br_plots.mpc_daily_plot(
        save_path=fig_path_lv2_day
        / f"{identifier}.png",
        identifier=f"{experiment_config['Experiment Number']} \
        -MPC/{identifier}",
        unit_dict=experiment_config["Units Dictionary"],
        metadata={
            "Title": f"{experiment_config['Experiment Number']}-D{curr_time_end}",
            "Author": "Zach Hatzenbeller, Yu Luo",
            "Description": f"MPC plot for {experiment_config['Experiment Number']}. Developed within GSK R&D in BDSD",
            "Copyright": f"(c) GSK, R&D, BDSD {datetime.today().year}",
            "Creation Time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Software": f"Python v{sys.version}",
        },
        display=False,
    )

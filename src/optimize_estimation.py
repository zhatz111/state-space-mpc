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
from data.functions import dict_toscaler, read_config
from scipy.optimize import minimize

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
experiment_config = read_config(PATH_DIRECTORY)

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
if ".xlsx" in experiment_config["Master Data File"]:
    master_sheet = pd.read_excel(
        Path(PATH_DIRECTORY, experiment_config["Master Data File"]),
        skiprows=[0],
    ).rename(
        columns={
            "Batch": "Bioreactor",
        }
    )
elif ".csv" in experiment_config["Master Data File"]:
    master_sheet = pd.read_csv(
        Path(PATH_DIRECTORY, experiment_config["Master Data File"])
    )
else:
    raise ValueError("Wrong master sheet file extension!")

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

# User specified current culture day: determined automatically if -1
CURR_TIME_USER = -1
SHOW_PLOT = False
if "Inoc Date" in experiment_config and CURR_TIME_USER < 0:
    date_delta = (
        datetime.today().date()
        - datetime.strptime(experiment_config["Inoc Date"], "%Y-%m-%d").date()
    )
    CURR_TIME_END = np.min((experiment_config["Last Day"], date_delta.days))
else:
    CURR_TIME_END = CURR_TIME_USER

# Create figure output folder
fig_path_top_dir = Path(PATH_DIRECTORY, experiment_config["Figures Folder"])
fig_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create CSV data output folder
csv_path_top_dir = Path(PATH_DIRECTORY, experiment_config["CSV Export Folder"])
csv_path_top_dir.mkdir(parents=True, exist_ok=True)

# Create a daily figures folder of all reactors
fig_path_lv2_day = Path(
    fig_path_top_dir.expanduser(), f"D{CURR_TIME_END}-{todays_date}"
)
fig_path_lv2_day.mkdir(parents=True, exist_ok=True)

def run_mpc_simulation(experiment_config, show_plot):
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
            controller_config=controller_config,
        )
        controllers.append(controller)

    # Iterate the main code for each bioreactor
    worst_estimates = np.zeros(len(vessels), dtype=float)

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

        # print("")
        # print("-" * 80)

        # Parse the states from the reference data csv file
        input_messages = master_sheet.loc[master_sheet["Bioreactor"] == curr_vessel, :]
        input_message_dict = {}
        # Iterate from Day 0 to the current day
        for curr_time in range(0, CURR_TIME_END + 1):
            # -------------------------------------------------------------------------------------
            # MAIN MPC LOOP ESTIMATES & OPTIMIZES EACH BIOREACTOR

            # Ingest data from Input topic
            bioreactor.curr_time = curr_time
            input_message = (
                input_messages.loc[input_messages["Day"] == curr_time, :]
                .squeeze()
                .to_dict()
            )
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

        # # Retrieve and print current feed rate (mL/min) for the feed pump
        # result = bioreactor.get_result()
        # print("")
        # br_heading = f"{curr_vessel} on Day {curr_time}:"
        # print(colored(br_heading, "green"))
        # for key, value in result.items():
        #     print(key, end=None)
        #     print(colored(np.round(value, 4), "blue", attrs=["bold"]))

        # -------------------------------------------------------------------------------------
        # GENERATED PLOTS SAVED

        # Plot the MPC Controller for each Bioreactor
        # Use the last controller from the list which is the controller from the current day

        # br_plots = MPCVisualizer(bioreactor, controller)

        # if isinstance(curr_vessel, str):
        #     identifier = f"{bioreactor.vessel}_D{CURR_TIME_END}-{todays_date}"
        # else:
        #     identifier = f"BR{bioreactor.vessel:02d}_D{CURR_TIME_END}-{todays_date}"

        # worst_estimates[count_vessel] = br_plots.mpc_daily_plot(
        #     save_paths=(
        #         fig_path_lv2_BR / f"{identifier}.png",
        #         fig_path_lv2_day / f"{identifier}.png",
        #     ),
        #     identifier=f"{experiment_config['Experiment Number']} \
        #     -MPC/{identifier}",
        #     unit_dict=experiment_config["Units Dictionary"],
        #     metadata={
        #         "Title": f"{experiment_config['Experiment Number']}-D{CURR_TIME_END}",
        #         "Author": "Zach Hatzenbeller, Yu Luo",
        #         "Description": f"MPC plot for {experiment_config['Experiment Number']}. \
        #         Developed within GSK R&D in BDSD",
        #         "Copyright": f"(c) GSK, R&D, BDSD {datetime.today().year}",
        #         "Creation Time": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        #         "Software": f"Python v{sys.version}",
        #     },
        #     display=show_plot,
        # )

    #     # NEW CODE TO OUTPUT A SEPERATE CSV FILE EACH DAY FOR EACH REACTOR
    #     # DEVELOPED: 2024-06-06
    #     filenames = [
    #         f"{bioreactor.vessel}_D{CURR_TIME_END}-{todays_date}-daily_feed.csv",
    #     ]

    #     # If no file exist currently
    #     bioreactor.return_data(show_daily_inputs=True, exec_date=True).to_csv(
    #         csv_path_lv2_BR / filenames[0], index=False
    #     )

    # # Save and export the current config file
    # read_config(PATH_DIRECTORY, export=True)

    # Return the worst estimate
    return worst_estimates

# Call the function
worst_estimates = run_mpc_simulation(experiment_config=experiment_config,show_plot=False)

# Extract keys and values for 'Offset Integral Gain' from Controller_1's State Variables
state_variables = experiment_config["Controller_1"]["State Variables"]
keys_offset_integral_gain = list(state_variables.keys())
values_offset_integral_gain = [
    state_variables[key]["Offset Integral Gain"] for key in keys_offset_integral_gain
]
print(keys_offset_integral_gain)
print(values_offset_integral_gain)

def update_offset_integral_gain(keys, values):
    if len(keys) != len(values):
        raise ValueError("Keys and values arrays must have the same length.")
    
    for key, value in zip(keys, values):
        if key in experiment_config["Controller_1"]["State Variables"]:
            experiment_config["Controller_1"]["State Variables"][key]["Offset Integral Gain"] = value
        else:
            raise KeyError(f"Key '{key}' not found in State Variables.")
        
    return experiment_config

def est_optim_obj(x):
    experiment_config = update_offset_integral_gain(keys=keys_offset_integral_gain, values=x)
    worst_estimates = run_mpc_simulation(experiment_config=experiment_config,show_plot=False)
    return max(worst_estimates)

# Define bounds for each value in x (0 to 1.5)
bounds = [(0, 1.5) for _ in values_offset_integral_gain]

# # Perform optimization
# # Wrap the objective function to display progress
# class ProgressCallback:
#     def __init__(self, total_iterations):
#         self.pbar = tqdm(total=total_iterations, desc="Optimization Progress")
#         self.iteration = 0

#     def __call__(self, xk):
#         self.iteration += 1
#         self.pbar.update(1)

#     def close(self):
#         self.pbar.close()

# # Estimate the number of iterations (this is a rough estimate for progress tracking)
max_iterations = 100  # Adjust based on expected optimization complexity
# progress = ProgressCallback(max_iterations)

print("BEFORE:")
print(keys_offset_integral_gain)
print(values_offset_integral_gain)
print(worst_estimates)

# try:
result = minimize(
    est_optim_obj,
    x0=values_offset_integral_gain,
    bounds=bounds,
    method="L-BFGS-B",
    # callback=progress,
    options={"maxiter": max_iterations},
)
# finally:
#     progress.close()

# # Print the optimization result
# print("Optimization Result:")
# print(result)
experiment_config = update_offset_integral_gain(keys=keys_offset_integral_gain,values=result.x)

print("\n")
print("AFTER:")
print(keys_offset_integral_gain)
print(result.x)
print(run_mpc_simulation(experiment_config=experiment_config,show_plot=False))

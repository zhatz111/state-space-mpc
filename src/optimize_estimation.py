"""
Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2025-08-27
"""

# Standard Library Imports
import warnings
from pathlib import Path
from datetime import datetime

# 3rd Party Library Imports
import numpy as np
from InquirerPy.resolver import prompt

# State-Space-Model Package Imports
from mpc.mpc_optimizer import Bioreactor, Controller
from models.ssm import StateSpaceModel
from data.functions import dict_toscaler, read_config, json_to_dict
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Store todays date and the top level directory in variables
todays_date = datetime.today().strftime("%y%m%d")
top_dir = Path().absolute()


CONTROLLER_NAME = "Controller"


# -------------------------------------------------------------------------------------
# USER SPECIFIED DATA


# User specified current culture day: determined automatically if -1
CURR_TIME_USER = 14
SHOW_PLOT = False
max_iterations = 5  # Adjust based on expected optimization complexity

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
    date_delta = datetime.today().date() - experiment_config["Inoc Date"]
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


def run_mpc_simulation(experiment_config, show_plot):
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

    worst_estimates = np.zeros((len(vessels), CURR_TIME_END+1), dtype=float)

    # Iterate the main code for each bioreactor
    for count_vessel, curr_vessel in enumerate(vessels):

        # Retrieve bioreactor and controller instances
        bioreactor = bioreactors[count_vessel]
        controller = controllers[count_vessel]

        # Parse the states from the reference data csv file
        input_message_dict = {}
        # Iterate from Day 0 to the current day
        for curr_time in range(0, CURR_TIME_END + 1):
            # -------------------------------------------------------------------------------------
            # MAIN MPC LOOP ESTIMATES & OPTIMIZES EACH BIOREACTOR

            # Ingest data from Input topic
            bioreactor.curr_time = curr_time
            vector_key = f"{curr_vessel}-{experiment_config['Batch ID']}-day{curr_time}"
            input_message = master_sheet[vector_key]
            input_message_dict[bioreactor.curr_time] = input_message
            bioreactor.ingest_vectors(input_message_dict)

            # Estimate the current state
            controller.estimate()

            # Do not optimize at EoR
            if curr_time == experiment_config["Last Day"]:
                END_OF_RUN = True
            else:
                END_OF_RUN = False

            if curr_time < experiment_config["Last Day"]:
                controller.optimize(
                    open_loop=False,
                    print_pred=False,
                    end_of_run=END_OF_RUN,
                    disp=False,
                )

            worst_estimates[count_vessel, curr_time] = bioreactor.estimation_error()

    return worst_estimates


# Call the function
worst_estimates = run_mpc_simulation(
    experiment_config=experiment_config, show_plot=False
)
sum_row_estimate = np.nansum(worst_estimates, axis=1)
max_reactor_estimate = np.nanmax(sum_row_estimate)


tuning_params = {}

for key, data in experiment_config[CONTROLLER_NAME]["State Variables"].items():
    prop_key = f"State Variables--{key}--Offset Proportional Gain"
    inte_key = f"State Variables--{key}--Offset Integral Gain"
    tuning_params[prop_key] = data["Offset Proportional Gain"]
    tuning_params[inte_key] = data["Offset Integral Gain"]

tuning_params["Prediction Horizon"] = experiment_config[CONTROLLER_NAME][
    "Prediction Horizon"
]
tuning_params["Control Horizon"] = experiment_config[CONTROLLER_NAME]["Control Horizon"]

tuning_params["Estimation Horizon"] = experiment_config[CONTROLLER_NAME][
    "Estimation Horizon"
]
tuning_params["Estimation Filter Weight on Data"] = experiment_config[CONTROLLER_NAME][
    "Estimation Filter Weight on Data"
]
# tuning_params["Overshoot Weight"] = experiment_config[CONTROLLER_NAME][
#     "Overshoot Weight"
# ]
# tuning_params["Undershoot Weight"] = experiment_config[CONTROLLER_NAME][
#     "Undershoot Weight"
# ]
# tuning_params["Trajectory Discount Weight"] = experiment_config[CONTROLLER_NAME][
#     "Trajectory Discount Weight"
# ]



# Extract keys and values for 'Offset Integral Gain' from the controllers State Variables
state_variables = experiment_config[CONTROLLER_NAME]["State Variables"].keys()
tuning_values = [value for _, value in tuning_params.items()]
tuning_keys = list(tuning_params.keys())


def update_tuning_parameters(keys, values):
    if len(keys) != len(values):
        raise ValueError("Keys and values arrays must have the same length.")

    for dict_key, value in zip(keys, values):
        tuning_params[dict_key] = value
        key_list = dict_key.split("--")
        if len(key_list) == 3:
            experiment_config[CONTROLLER_NAME][key_list[0]][key_list[1]][
                key_list[2]
            ] = value
        else:
            if key_list[0] in [
                "Prediction Horizon",
                "Control Horizon",
                "Estimation Horizon",
            ]:
                experiment_config[CONTROLLER_NAME][key_list[0]] = int(value)
            else:
                experiment_config[CONTROLLER_NAME][key_list[0]] = value

    return experiment_config


class TuningOptimizer:
    def __init__(self, initial_x):
        self.best_worst = np.inf
        self.best_x = initial_x
        self.iters = 0

    def est_optim_tuning_obj(self, x):
        experiment_config = update_tuning_parameters(
            keys=list(tuning_params.keys()), values=x
        )
        worst_estimates = run_mpc_simulation(
            experiment_config=experiment_config, show_plot=False
        )
        sum_row_estimate = np.nansum(worst_estimates, axis=1)
        max_reactor_estimate = np.nanmax(sum_row_estimate)
        if self.iters % 5 == 0:
            print(f"Iteration: {self.iters}, Tuning Error: {max_reactor_estimate}")
        if max_reactor_estimate < self.best_worst:
            self.best_worst = max_reactor_estimate
            self.best_x = x

        self.iters += 1
        return max_reactor_estimate


tuner = TuningOptimizer(initial_x=tuning_values)

# Define bounds for each value in x (0 to 1.5)
state_bounds = [(0, 5.0) for _ in range(len(state_variables) * 2)]
horizon_bounds = [
    (2, 20) for _ in ["Estimation Horizon", "Control Horizon", "Prediction Horizon",] #  
]
estimate_bounds = [(1e-10, 1.0)]
# control_bounds = [
#     (0.5, 5.0) for _ in ["Overshoot Weight", "Undershoot Weight", "Trajectory Discount Weight"]
# ]
bounds = state_bounds + horizon_bounds + estimate_bounds # + control_bounds

print("BEFORE Optimization:")
print(f"Tuning Error: {max_reactor_estimate}")
width = 55
for key, data in tuning_params.items():
    print(f"{key:<{width}} : {data}")
print("")


# try:
def display_progress(xk):
    current_obj_value = tuner.est_optim_tuning_obj(xk)
    print(f"Current Objective Value: {current_obj_value}")


result = minimize(
    tuner.est_optim_tuning_obj,
    x0=tuning_values,
    bounds=bounds,
    method="SLSQP",  # "L-BFGS-B",
    options={"maxiter": max_iterations, "disp": True},
)

experiment_config = update_tuning_parameters(keys=tuning_keys, values=tuner.best_x)

print("")
print("AFTER Optimization:")
print(f"Tuning Error: {tuner.best_worst}")
for key, data in tuning_params.items():
    print(f"{key:<{width}}: {data}")

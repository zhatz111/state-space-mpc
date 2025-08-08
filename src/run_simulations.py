"""Main code for running in-silico experiments with an MPC model
Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
Created: 2025-08-07
Modified: 2025-08-08
"""

# Imports from Standard Library
import glob
import random
import warnings
from pathlib import Path
from datetime import datetime

# Imports from third party
import numpy as np
import pandas as pd
from InquirerPy.resolver import prompt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Imports from within repository
from data.make_dataset import ModelData
from models.train_model import ModelTraining
from mpc.simulations import ModelSimulations
from data.functions import (
    json_to_dict,
    dict_to_json,
    dict_toscaler,
    scaler_todict,
)

# suppress warnings
warnings.filterwarnings("ignore")

top_dir = Path().absolute()
PARENT_FILE_PATH = top_dir / "models"

FOLDER_SEARCH_KEY = "Model"
matching_folders = [
    folder.name
    for folder in PARENT_FILE_PATH.iterdir()
    if folder.is_dir() and FOLDER_SEARCH_KEY in folder.name
]
questions = {
    "type": "list",
    "name": "folder",
    "message": "What folder to use for training?",
    "choices": matching_folders,
}
answer = prompt(questions)
MODELING_DATA_FOLDER_NAME = str(answer["folder"])

PATH_DIRECTORY = Path(PARENT_FILE_PATH, MODELING_DATA_FOLDER_NAME)
json_files = glob.glob(str(Path(PATH_DIRECTORY, "*.json")))
csv_files = glob.glob(str(Path(PATH_DIRECTORY, "*.csv")))

for file in csv_files:
    file_ = Path(file)

    if "simulation" in file_.name:
        df_simulations = pd.read_csv(file_)

for file in json_files:
    file_ = Path(file)

    if "model" in file_.name:
        model_config = json_to_dict(file_)

    if "simulation" in file_.name:
        simulations_config = json_to_dict(file_)


def main():
    """
    The main function reads data, preprocesses it, evaluates the DoE conditions, and saves
    the predictions from the model.
    """

    scaler_name = model_config["Scaler"]
    scaler_experiment = dict_toscaler(
        model_config["scaler"], scaler_class=scaler_name
    )

    a_matrix = np.array(model_config["a_matrix"])
    b_matrix = np.array(model_config["b_matrix"])
    af_col_matrix = np.array(model_config["af_col"])
    af_row_matrix = np.array(model_config["af_row"])
    bf_row_matrix = np.array(model_config["bf_row"])

    simulation_data = {}
    for name, batch in df_simulations.groupby("Batch"):
        batch = batch.reset_index(drop=True)
        batch[model_config["Model States"]] = np.nan

        if simulations_config["ivcc_initial_condition"]:
            for key, value in simulations_config["initial_conditions"].items():
                batch.loc[0, key] = random.choice(value)
            batch.loc[0, "VCC"] = batch.loc[0, "iVCC"]
        else:
            batch.loc[0] = simulations_config["initial_conditions"]

        simulation_data[name] = batch

    df_sim = pd.concat(simulation_data)[
        ["Batch", "Day"] + model_config["Model States"] + model_config["Model Inputs"]
    ].reset_index(drop=True)

    sim_data_modifier = ModelData(
        raw_data=df_sim,
        group="Batch",
        scaler=scaler_experiment,
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
    )

    sim_data_modifier.feature_scaling(
        data=df_sim, scaler=scaler_experiment, new_scaler=False
    )

    simulator = ModelSimulations(
        simulation_data=sim_data_modifier.df,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
        num_days=model_config["Process Time"],
        scaler=scaler_experiment,
        hidden_state=True,
        rho=model_config["rho"],
        af_col=af_col_matrix,
        af_row=af_row_matrix,
        bf_row=bf_row_matrix,
    )
    time = datetime.now().strftime("%Y-%m-%d_%H%M")

    simulator.simulate(
        file_save_path=Path(PATH_DIRECTORY, "Simulation Data", f"{model_config["Asset"]}_simulation_{time}.csv"),
        target_label=model_config["Target Plotting Label"],
        ylim=9000,
    )


if __name__ == "__main__":
    main()

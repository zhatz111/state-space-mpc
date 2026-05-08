"""
Main code for training an model for MPC
    Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2025-08-27
"""

# Imports from Standard Library
import glob
import warnings
from pathlib import Path
from datetime import datetime

# Imports from third party
import numpy as np
import pandas as pd
from InquirerPy.resolver import prompt
from sklearn.preprocessing import MinMaxScaler

# Imports from within repository
from data.make_dataset import ModelData
from models.train_model import ModelTraining
from data.functions import (
    json_to_dict,
    dict_to_json,
    dict_toscaler,
    scaler_todict,
)

# suppress warnings
warnings.filterwarnings("ignore")


EVALUATION_TYPE = "evaluate"  # "train" or "evaluate"


# Use this path for experiment folders in MPC teams site
# PARENT_FILE_PATH = Path(
#     r"~\GSK\Biopharm Model Predictive Control - General\data"
# ).expanduser()

# Use this path for experiment folders in GitHub repository
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
for file in json_files:
    file_ = Path(file)

    if "model" in file_.name:
        model_config = json_to_dict(file_)


def main():
    """
    The main function reads data, preprocesses it, trains a model, and saves
    the model scaler and matrices.
    """
    partition_data = model_config["Partitions"]

    if model_config["Scaler"] == "MinMaxScaler":
        scaler_train = MinMaxScaler()
    else:
        scaler_train = dict_toscaler(
            model_config["scaler"], scaler_class="MinMaxScaler"
        )

    a_matrix = np.array(model_config["a_matrix"])
    b_matrix = np.array(model_config["b_matrix"])

    # If num_partitions changed, preserve existing matrices and randomly init any new ones
    if partition_data:
        n_p = partition_data["num_partitions"]
        n_states = len(model_config["Model States"])
        n_inputs = len(model_config["Model Inputs"])
        if a_matrix.shape[0] != n_p:
            new_a = np.random.uniform(-0.5, 0.5, (n_p, n_states, n_states))
            new_b = np.random.uniform(-0.5, 0.5, (n_p, n_states, n_inputs))
            n_keep = min(a_matrix.shape[0], n_p)
            new_a[:n_keep] = a_matrix[:n_keep]
            new_b[:n_keep] = b_matrix[:n_keep]
            a_matrix, b_matrix = new_a, new_b
            print(
                f"num_partitions changed to {n_p}. Preserved {n_keep} existing matrices; new partitions randomly initialized."
            )

    data = pd.read_csv(
        Path(PATH_DIRECTORY, f"{model_config['Modeling Data File Name']}.csv")
    )

    dataframe = ModelData(
        raw_data=data,
        scaler=scaler_train,
        group="Batch",
        discard=model_config["Batches to Discard"],
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
        column_types=model_config["Data Column Types"],
    )

    # Class method to clean up all the data
    # this includes interpolation to start, spline smoothing, splitting
    # the data into training and testing sets
    # and finally feature scaling using the scaler of choice
    train_data, test_data, train_list, test_list = dataframe.clean(
        metadata_columns=model_config["Include Data Columns"],
        # smoothing_list=model_config["Data Smoothing List"],
        test_size=model_config["Testing Data Size"],
        n_splits=model_config["Number Cross Val Splits"],
        random_state=model_config["Random Seed"],
        win_len=model_config["Window Length"],
        n_points=model_config["Spline Points"],
        bolus_tau=model_config.get("Bolus Decay Tau", 1.0),
    )

    # dataframe.graph_train_data(test_label="IGG")
    # dataframe.graph_smoothed_unsmoothed_data(test_label="IGG")

    model_config["scaler"] = scaler_todict(scaler=scaler_train)

    model_train_obj = ModelTraining(
        train_data=train_data,
        test_data=test_data,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
        pv_wghts=model_config["Process Variable Weights"],
        instability_weights=model_config["Instability Weights"],
        num_days=model_config["Process Time"],
        scaler=scaler_train,
        partitions_data=partition_data,
        algorithm="basinhopping",
    )
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    new_directory = Path(
        PATH_DIRECTORY, f"{model_config['A & B Matrices Folder Name']}"
    )
    new_directory.mkdir(
        parents=True, exist_ok=True
    )  # Creates parent directories if they don't exist

    if EVALUATION_TYPE.lower() == "train":
        model_train_obj.train_test_model(
            save_path=new_directory,
            test_label=model_config["Target Plotting Label"],
            iterations=model_config["Training Iterations"],
            basin_temp=model_config["BasinHopping Temperature"],
            first_train=False,
            show_plots=False,
        )
    else:
        model_train_obj.plot_train_data(
            test_label=model_config["Target Plotting Label"], ylim=None
        )

        model_train_obj.plot_test_data(
            test_label=model_config["Target Plotting Label"], ylim=None
        )

    # Update the model config file
    if not partition_data:
        model_config["a_matrix"] = [model_train_obj.a_matrix.tolist()]
        model_config["b_matrix"] = [model_train_obj.b_matrix.tolist()]
    else:
        n_p = partition_data["num_partitions"]
        model_config["a_matrix"] = [
            model_train_obj.a_matrices[i].tolist() for i in range(n_p)
        ]
        model_config["b_matrix"] = [
            model_train_obj.b_matrices[i].tolist() for i in range(n_p)
        ]

    model_config["Iterations"] += model_train_obj.iters
    model_config["Training Batches"] = train_list
    model_config["Testing Batches"] = test_list

    if model_train_obj.model_error != 0:
        model_config["Model RMSE"] = model_train_obj.lowest_model_error
        model_config["States RMSE"] = model_train_obj.lowest_model_error_dict
        model_config["Last Model Training"] = time
        model_config["Matrix Stability Penalties"] = (
            model_train_obj.stability_error_dict
        )

    # Export updated data back to JSON file
    save_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dict_to_json(json_files[0], model_config)
    dict_to_json(
        new_directory / f"{Path(json_files[0]).stem}-{save_time}.json", model_config
    )


if __name__ == "__main__":
    main()

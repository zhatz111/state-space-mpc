"""Main code for training an model for MPC
    Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2023-10-05
    Modified: 2024-04-23
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
from src.data.make_dataset import ModelData
from src.models.train_model import ModelTraining
from src.data.functions import (
    json_to_dict,
    dict_to_json,
    dict_toscaler,
    scaler_todict,
)

# suppress warnings
warnings.filterwarnings("ignore")

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
model_config = json_to_dict(json_files[0])


def main():
    """
    The main function reads data, preprocesses it, trains a model, and saves
    the model scaler and matrices.
    """
    if model_config["First Training"]:
        scaler_train = MinMaxScaler()
    else:
        scaler_train = dict_toscaler(model_config["scaler"])

    a_matrix = np.array(model_config["a_matrix"])
    b_matrix = np.array(model_config["b_matrix"])

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
    )

    # Class method to clean up all the data
    # this includes interpolation to start, spline smoothing, splitting
    # the data into training and testing sets
    # and finally feature scaling using the scaler of choice
    train_data, test_data = dataframe.clean(
        metadata_columns=model_config["Include Data Columns"],
        smoothing_list=model_config["Data Smoothing List"],
        test_size=model_config["Testing Data Size"],
        n_splits=model_config["Number Cross Val Splits"],
        random_state=model_config["Random Seed"],
        win_len=model_config["Window Length"],
    )

    model_config["scaler"] = scaler_todict(scaler=scaler_train)

    model_train_obj = ModelTraining(
        train_data=train_data,
        test_data=test_data,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
        pv_wghts=model_config["Process Variable Weights"],
        num_days=model_config["Process Time"],
        scaler=scaler_train,
        algorithm="basin",
    )
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    model_train_obj.train_test_model(
        save_path=Path(PATH_DIRECTORY, f"{model_config['A & B Matrices Folder Name']}"),
        test_label=model_config["Target Plotting Label"],
        iterations=model_config["Training Iterations"],
        first_train=False,
    )

    # Update the model config file
    model_config["a_matrix"] = model_train_obj.a_matrix.tolist()
    model_config["b_matrix"] = model_train_obj.b_matrix.tolist()
    model_config["Iterations"] += model_train_obj.iters
    model_config["Model RMSE"] = model_train_obj.model_error
    model_config["States RMSE"] = model_train_obj.true_model_error_dict
    model_config["Last Model Training"] = time

    # Export updated data back to JSON file
    dict_to_json(json_files[0], model_config)


if __name__ == "__main__":
    main()

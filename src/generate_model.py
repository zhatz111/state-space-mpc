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
import json
import yaml
import numpy as np
import pandas as pd
from InquirerPy.resolver import prompt
from sklearn.preprocessing import MinMaxScaler

# Imports from within repository
from src.data.make_dataset import ModelData
from src.models.train_model import ModelTraining
from src.data.functions import (
    scaler_tojson,
    json_toscaler,
    update_json,
    json_to_dict,
    dict_to_json,
)

# suppress warnings
warnings.filterwarnings("ignore")

PARENT_FILE_PATH = Path(
    r"~\GSK\Biopharm Model Predictive Control - General\data"
).expanduser()
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
yaml_files = glob.glob(str(Path(PATH_DIRECTORY, "*.yaml")))
yaml_data = open(yaml_files[0], "r", encoding="utf-8")
model_config = yaml.safe_load(yaml_data)
yaml_data.close()

METADATA_TEMPLATE = {
    "Asset": model_config["Asset"],
    "Training Data Study": model_config["Training Data Study"],
    "Iterations": 0,
    "Model RMSE": 0,
    "States RMSE": {},
    "Last Model Training": "",
}


def main():
    """
    The main function reads data, preprocesses it, trains a model, and saves 
    the model scaler and matrices.
    """
    if f"{model_config['Training Data Study']}_scaler.json" in list(
        f.name
        for f in Path(
            PATH_DIRECTORY, model_config["A & B Matrices Folder Name"]
        ).iterdir()
    ):
        scaler_train = json_toscaler(
            Path(
                PATH_DIRECTORY,
                f"{METADATA_TEMPLATE['Asset']}_scaler.json",
            )
        )
    else:
        scaler_train = MinMaxScaler()

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

    a_matrix = np.array(
        pd.read_csv(
            Path(
                PATH_DIRECTORY,
                f"{model_config['A & B Matrices Folder Name']}/A_Matrix.csv",
            ),
            header=None,
        )
    )

    b_matrix = np.array(
        pd.read_csv(
            Path(
                PATH_DIRECTORY,
                f"{model_config['A & B Matrices Folder Name']}/B_Matrix.csv",
            ),
            header=None,
        )
    )

    scaler_tojson(
        scaler=scaler_train,
        save_path=Path(
            PATH_DIRECTORY,
            f"{model_config['A & B Matrices Folder Name']}/{model_config['Asset']}_scaler.json",
        ),
    )

    model_train_obj = ModelTraining(
        train_data,
        test_data,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
        pv_wghts=model_config["Process Variable Weights"],
        num_days=model_config["Process Time"],
        scaler=scaler_train,
    )
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    model_train_obj.train_test_model(
        Path(PATH_DIRECTORY, f"{model_config['A & B Matrices Folder Name']}"),
        test_label=model_config["Target Plotting Label"],
        iterations=model_config["Training Iterations"],
        first_train=model_config["First Training"],
    )

    post_training_metadata = {
        "Iterations": model_train_obj.iters,
        "Model RMSE": model_train_obj.model_error,
        "States RMSE": model_train_obj.model_error_dict,
        "Last Model Training": time,
    }
    json_path = Path(
        PATH_DIRECTORY,
        f"{METADATA_TEMPLATE['Asset']}_model_metadata.json",
    )
    if f"{METADATA_TEMPLATE['Asset']}_model_metadata.json" in list(
        f.name for f in PATH_DIRECTORY.iterdir()
    ):
        post_training_metadata = update_json(
            json_path,
            post_training_metadata,
        )
    else:
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(METADATA_TEMPLATE, file)
        post_training_metadata = update_json(
            json_path,
            post_training_metadata,
        )

    # generate a json file to import model parameters into mpc script
    scaler_dict = json_to_dict(
        Path(
            PATH_DIRECTORY,
            model_config["A & B Matrices Folder Name"],
            f"{METADATA_TEMPLATE['Asset']}_scaler.json",
        )
    )
    params = {
        "States": model_config["Model States"],
        "Inputs": model_config["Model Inputs"],
        "a_matrix": a_matrix.tolist(),
        "b_matrix": b_matrix.tolist(),
        "scaler": scaler_dict,
    }

    model_parameters = {**params, **post_training_metadata}
    dict_to_json(
        Path(
            PATH_DIRECTORY,
            f"{METADATA_TEMPLATE['Asset']}_model_parameters.json",
        ),
        data=model_parameters
    )

if __name__ == "__main__":
    main()

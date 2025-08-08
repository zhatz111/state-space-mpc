"""Main code for training an model for MPC
    Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
    Created: 2024-04-24
    Modified: 2025-08-08
"""

# Imports from Standard Library
import glob
import warnings
from pathlib import Path

# Imports from third party
import numpy as np
import pandas as pd
from InquirerPy.resolver import prompt
from sklearn.preprocessing import MinMaxScaler

# Imports from within repository
from data.make_dataset import ModelData
from models.train_model import ModelTraining
from visualization.pdf_report import generate_report
from data.functions import json_toscaler, json_to_dict

# suppress warnings
warnings.filterwarnings("ignore")

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
    "message": "Generate Report for:",
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
    if f"{model_config['Training Data Study']}_scaler.json" in list(
        f.name
        for f in Path(
            PATH_DIRECTORY, model_config["A & B Matrices Folder Name"]
        ).iterdir()
    ):
        scaler_train = json_toscaler(
            Path(
                PATH_DIRECTORY,
                f"{model_config['Asset']}_scaler.json",
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

    a_matrix = np.array(model_config["a_matrix"])
    b_matrix = np.array(model_config["b_matrix"])
    af_col_matrix = np.array(model_config["af_col"])
    af_row_matrix = np.array(model_config["af_row"])
    bf_row_matrix = np.array(model_config["bf_row"])

    model_report_obj = ModelTraining(
        train_data,
        test_data,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
        pv_wghts=model_config["Process Variable Weights"],
        num_days=model_config["Process Time"],
        scaler=scaler_train,
        hidden_state=True,
        rho=model_config["rho"],
        af_col=af_col_matrix,
        af_row=af_row_matrix,
        bf_row=bf_row_matrix,
    )

    report_metadata = json_to_dict(Path(PATH_DIRECTORY, f"{model_config['Asset']}_model_parameters.json"))
    generate_report(
        model_report_obj=model_report_obj,
        output_folder=Path(PATH_DIRECTORY, f"{model_config['Asset']} Report"),
        metadata = report_metadata
    )

if __name__ == "__main__":
    main()

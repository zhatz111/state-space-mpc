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
    HIDDEN_STATE = model_config["Hidden State"]

    if model_config["Scaler"] == "MinMaxScaler":
        scaler_train = MinMaxScaler()
    else:
        scaler_train = dict_toscaler(model_config["scaler"], scaler_class="MinMaxScaler")

    a_matrix = np.array(model_config["a_matrix"])
    b_matrix = np.array(model_config["b_matrix"])
    af_col_matrix = np.array(model_config["af_col"])
    af_row_matrix = np.array(model_config["af_row"])
    bf_row_matrix = np.array(model_config["bf_row"])

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
    train_data, test_data, train_list, test_list = dataframe.clean(
        metadata_columns=model_config["Include Data Columns"],
        smoothing_list=model_config["Data Smoothing List"],
        test_size=model_config["Testing Data Size"],
        n_splits=model_config["Number Cross Val Splits"],
        random_state=model_config["Random Seed"],
        win_len=model_config["Window Length"],
    )

    # dataframe.graph_smoothed_unsmoothed_data(model_config["Data Smoothing List"], test_label="IGG")

    model_config["scaler"] = scaler_todict(scaler=scaler_train)

    model_train_obj = ModelTraining(
        train_data=train_data,
        test_data=test_data,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        states=model_config["Model States"],
        inputs=model_config["Model Inputs"],
        pv_wghts=model_config["Process Variable Weights"],
        instability_weight=model_config["Instability Weight"],
        num_days=model_config["Process Time"],
        scaler=scaler_train,
        algorithm="basinhopping",
        hidden_state=HIDDEN_STATE,
        rho=model_config["rho"],
        af_col=af_col_matrix,
        af_row=af_row_matrix,
        bf_row=bf_row_matrix,
    )
    time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    new_directory = Path(PATH_DIRECTORY, f"{model_config['A & B Matrices Folder Name']}")
    new_directory.mkdir(parents=True, exist_ok=True) # Creates parent directories if they don't exist

    model_train_obj.train_test_model(
        save_path=new_directory,
        test_label=model_config["Target Plotting Label"],
        iterations=model_config["Training Iterations"],
        basin_temp=model_config["BasinHopping Temperature"],
        first_train=False,
    )

    # model_train_obj.plot_train_data(
    #     test_label=model_config["Target Plotting Label"]
    # )

    # model_train_obj.plot_test_data(
    #     test_label=model_config["Target Plotting Label"]
    # )

    # Update the model config file
    model_config["Hidden State"] = HIDDEN_STATE
    model_config["a_matrix"] = model_train_obj.a_matrix.tolist()
    model_config["b_matrix"] = model_train_obj.b_matrix.tolist()

    if HIDDEN_STATE:
        model_config["af_col"] = model_train_obj.af_col.tolist()
        model_config["af_row"] = model_train_obj.af_row.tolist()
        model_config["bf_row"] = model_train_obj.bf_row.tolist()
        model_config["rho"] = model_train_obj.rho

    model_config["Iterations"] += model_train_obj.iters
    model_config["Training Batches"] = train_list
    model_config["Testing Batches"] = test_list

    if model_train_obj.model_error != 0:
        model_config["Model RMSE"] = model_train_obj.lowest_model_error
        model_config["States RMSE"] = model_train_obj.lowest_model_error_dict
        model_config["Last Model Training"] = time

    # Export updated data back to JSON file
    dict_to_json(json_files[0], model_config)


if __name__ == "__main__":
    main()

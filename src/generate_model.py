"""_summary_
"""

# Imports from Standard Library
from pathlib import Path
from datetime import datetime

# Imports from third party
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Imports from within repository
from src.data.make_dataset import ModelData
from src.models.train_model import ModelTraining
from src.models.ssm import scaler_tojson, json_toscaler, update_json

# suppress warnings
warnings.filterwarnings("ignore")

# CREATE A CONFIG FILE THAT CAN TAKE THE PLACE OF ALL THESE PARAMETERS THAT I SPECIFY
# THIS WAY I CAN USE THIS SCRIPT REALLY EASY FOR MULTIPLE ASSETS BY JUST CHANGING THE CONFIG FILE
# MAKE THE CONFIG FILE A JSON


METADATA_TEMPLATE = {
    "Asset": "aIL33-LA",
    "Training Data Study": "AR24-012",
    "Iterations": 0,
    "Model RMSE": 0,
    "States RMSE": {},
    "Last Model Training": "",
}
FILE_PATH = Path(r"~\GSK\Biopharm Model Predictive Control - General\data").expanduser()
DATA_FOLDER_EXT = "2024-04-12 aIL33-LA Data"
DATA_FILE_EXT = "AR24-012 Feed-pH-Temp Model Data"
MATRIX_FOLDER_EXT = "AR24-012 Matrices"
PDF_PLOT_FILENAME = "model_test_report"

TARGET_LABEL = "IGG"
PROCESS_TIME = 13  # should be length of days + 1
ITERATIONS = 100
FIRST_TRAIN = False
pv_wghts = [
    8.0,
    1.0,
    1.0,
    1.0,
    1.0,
]

# Make sure to check the window length for smoothing with moving average

include_columns = [
    "Batch",
    "Condition",
    "Day",
]

STATES = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
    "pH_at_temp",
]

INPUTS = [
    "Cumulative_Normalized_Feed",
    "Temperature",
    "pH_setpoint",
    "DO",
]

SMOOTHE_LIST = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
]

DISCARD = [
    # "AR23-067-005P",
    # "AR23-067-011P",
    # "AR23-019-001P",
    # "AR23-019-004P",
    # "AR23-019-007P",
    # "AR23-019-014P",
]


def main():
    """
    The main function reads data, preprocesses it, trains a model, and saves the model scaler and
    matrices.
    """
    directory = Path(FILE_PATH, DATA_FOLDER_EXT)
    if f"{METADATA_TEMPLATE['Training Data Study']}_scaler.json" in list(
        f.name for f in directory.iterdir()
    ):
        scaler_train = json_toscaler(
            Path(directory, f"{METADATA_TEMPLATE['Training Data Study']}_scaler.json")
        )
    else:
        scaler_train = MinMaxScaler()

    data = pd.read_csv(Path(directory, f"{DATA_FILE_EXT}.csv"))

    dataframe = ModelData(
        raw_data=data,
        scaler=scaler_train,
        group="Batch",
        discard=DISCARD,
        states=STATES,
        inputs=INPUTS,
    )

    # Class method to clean up all the data
    # this includes interpolation to start, spline smoothing, splitting the data into training and testing sets
    # and finally feature scaling using the scaler of choice
    train_data, test_data = dataframe.clean(
        metadata_columns=include_columns,
        smoothing_list=SMOOTHE_LIST,
        test_size=0.20,
        n_splits=2,
        random_state=1,
        win_len=2,
    )

    a_matrix = np.array(
        pd.read_csv(
            Path(directory, f"{MATRIX_FOLDER_EXT}/A_Matrix.csv"),
            header=None,
        )
    )

    b_matrix = np.array(
        pd.read_csv(
            Path(directory, f"{MATRIX_FOLDER_EXT}/B_Matrix.csv"),
            header=None,
        )
    )

    scaler_tojson(
        scaler=scaler_train,
        save_path=Path(
            directory,
            f"{MATRIX_FOLDER_EXT}/{METADATA_TEMPLATE['Training Data Study']}_scaler.json",
        ),
    )

    model_train_obj = ModelTraining(
        train_data,
        test_data,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        states=STATES,
        inputs=INPUTS,
        pv_wghts=pv_wghts,
        num_days=PROCESS_TIME,
        scaler=scaler_train,
    )
    time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # UNCOMMENT THIS CODE TO TRAIN THE MODEL ON THE DATA
    model_train_obj.train_test_model(
        Path(directory, f"{MATRIX_FOLDER_EXT}"),
        test_label=TARGET_LABEL,
        iterations=ITERATIONS,
        first_train=FIRST_TRAIN,
    )

    post_training_metadata = {
        "Iterations": model_train_obj.iters,
        "Model RMSE": model_train_obj.model_error,
        "States RMSE": model_train_obj.model_error_dict,
        "Last Model Training": time,
    }
    json_path = Path(
        directory, f"{METADATA_TEMPLATE['Training Data Study']}_model_metadata.json"
    )
    if f"{METADATA_TEMPLATE['Training Data Study']}_model_metadata.json" in list(
        f.name for f in directory.iterdir()
    ):
        update_json(
            json_path,
            post_training_metadata,
        )
    else:
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(METADATA_TEMPLATE, file)
        update_json(
            json_path,
            post_training_metadata,
        )

if __name__ == "__main__":
    main()

"""_summary_
"""

# Imports from Standard Library
import glob
from pathlib import Path
from datetime import datetime

# Imports from third party
import warnings
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Imports from within repository
from src.data.make_dataset import ModelData
from src.models.train_model import ModelTraining
from src.models.ssm import scaler_tojson, json_toscaler, update_json

# suppress warnings
warnings.filterwarnings("ignore")


# FILL THIS OUT BEFORE RUNNING SCRIPT
# ------------------------------------------------------------
PARENT_FILE_PATH = Path(
    r"~\GSK\Biopharm Model Predictive Control - General\data"
).expanduser()
MODELING_DATA_FOLDER_NAME = "2024-04-16 aCD96 Robustness Data"
# ------------------------------------------------------------



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
    The main function reads data, preprocesses it, trains a model, and saves the model scaler and
    matrices.
    """
    if f"{METADATA_TEMPLATE['Training Data Study']}_scaler.json" in list(
        f.name for f in PATH_DIRECTORY.iterdir()
    ):
        scaler_train = json_toscaler(
            Path(
                PATH_DIRECTORY,
                f"{METADATA_TEMPLATE['Training Data Study']}_scaler.json",
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
    # this includes interpolation to start, spline smoothing, splitting the data into training and testing sets
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
                f"{model_config['A & B Matrices Folder Extension']}/A_Matrix.csv",
            ),
            header=None,
        )
    )

    b_matrix = np.array(
        pd.read_csv(
            Path(
                PATH_DIRECTORY,
                f"{model_config['A & B Matrices Folder Extension']}/B_Matrix.csv",
            ),
            header=None,
        )
    )

    scaler_tojson(
        scaler=scaler_train,
        save_path=Path(
            PATH_DIRECTORY,
            f"{model_config['A & B Matrices Folder Extension']}/{model_config['Asset']}_scaler.json",
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
    time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    model_train_obj.train_test_model(
        Path(PATH_DIRECTORY, f"{model_config['A & B Matrices Folder Extension']}"),
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

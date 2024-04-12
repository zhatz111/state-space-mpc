"""_summary_
"""

# Imports from third party
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Imports from within repository
from src.data.make_dataset import ModelData

# suppress warnings
warnings.filterwarnings("ignore")

DATA_FOLDER_EXT = "2024-04-12 aIL33-LA Data"
DATA_FILE_EXT = "AR24-012 Feed-pH-Temp Model Data"

# Make sure to check the window length for smoothing with moving average
# This small script will interpolate and smooth your data then save it as
# a CSV file in the same folder as the uncleaned data.

# I use this script to then graph the clean and unclean data in spotfire to quickly
# check the quality of the smoothing.

# The states list will also smooth all of those columns

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
    "Glucose",
    "Ammonium",
    "Osmo",
    "pH_at_temp",
]

INPUTS = [
    "Cumulative_Normalized_Feed",
    "Temperature",
    "pH_setpoint",
    "DO",
]

DISCARD = []


def main():
    """
    The main function reads data, preprocesses it, trains a model, and saves the model scaler and
    matrices.
    """
    scaler_train = MinMaxScaler()

    data = pd.read_csv(
        rf"~\GSK\Biopharm Model Predictive Control - General\data\{DATA_FOLDER_EXT}\{DATA_FILE_EXT}.csv"
    )

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
    cleaned_unscaled_data = dataframe.moving_average_smoother(
        smoothing_list=STATES, win_len=2
    )
    cleaned_unscaled_data.to_csv(
        rf"~\GSK\Biopharm Model Predictive Control - General\data\{DATA_FOLDER_EXT}\{DATA_FILE_EXT}_cleaned.csv"
    )

if __name__ == "__main__":
    main()

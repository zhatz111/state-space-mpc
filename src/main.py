import sys
sys.path.insert(0, r'C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model')

import joblib
import pandas as pd
import numpy as np
# import data.make_dataset
# import models
from sklearn.preprocessing import MinMaxScaler
from src.data.make_dataset import ModelData
from src.models.train_model import ModelTraining

STATES = [
    # "TCC",
    "VCC",
    # "Lact",
    # "Osmo",
    "Gluc",
    # "Ammonium",
    # "IGG",
]

INPUTS = [
    "Daily_Feed_Normalized",
    "Daily_Glucose_Normalized",
]

SPLINES = [
    # "TCC",
    "VCC",
    # "Lact",
    # "Osmo",
    # "Ammonium",
    # "IGG",
    "Daily_Feed_Normalized",
]

DISCARD = [
    "AR21-048-001",
    "AR21-048-003",
    "AR21-048-009",
    "AR22-001-001",
]

# DISCARD = []

feature_minmaxscaling_exclusion = [
    "Batch",
    "Day",
    "Condition",
    "Sample Time",
    "Volume",
]

post_scaling_exclusion = [
    "F30 Feed Amount (mL)",
    "Glucose Added (mL)",
    "Volume",
]

# scaler_train = MinMaxScaler()
scaler_train = joblib.load("scaler_train.scale")
data = pd.read_csv(r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\raw\AR22-001-Model-Data.csv")

#AR22-001-Model-Data
#DR-Model-Data-PVRIG

dataframe = ModelData(
    df=data,
    scaler_train=scaler_train,
    group="Batch",
    discard=DISCARD,
)

# Class method to clean up all the data
# this includes interpolation to start, spline smoothing, train and test set splitting
# and finally feature scaling using the scaler of choice
train_data, test_data = dataframe.clean(
    feature_minmaxscaling_exclusion,
    spline_list=SPLINES,
    post_scale_exclusion=post_scaling_exclusion,
    test_size = 0.10,
    n_splits = 2,
    random_state = 1,
)

# Save the Scaler for both the training and test sets to rescale in the future
# dataframe.save_scaler("scaler_train", scaler=scaler_train)
A_Matrix = np.loadtxt(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR22-001-Kalman-Filter\A_Matrix.csv", delimiter=',')
B_Matrix = np.loadtxt(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR22-001-Kalman-Filter\B_Matrix.csv", delimiter=',')

# "C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\current\Model 1\A_Matrix.csv"

# Number of days is always equal to the last day number + 1, so 12 day culture duration will equal 13 days
first_model_train = ModelTraining(
    train_data,
    test_data,
    a_matrix=A_Matrix,
    b_matrix=B_Matrix,
    states=STATES,
    inputs=INPUTS,
    num_days=13
)

first_model_train.train_test_model(
    r"M:\Zach Hatzenbeller\State-Space-Matrices\AR22-001-Kalman-Filter",
    test_label="Gluc",
    iterations=40,
    first_train=False,
)

# first_model_train.evaluate(
#     test_label="Gluc",
# )

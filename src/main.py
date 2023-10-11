"""_summary_
"""
# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

# Imports from third party
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Imports from within repository
from data.make_dataset import ModelData
from models.train_model import ModelTraining
from models.optimize_model import ModelOptimizer

#suppress warnings
warnings.filterwarnings('ignore')

DATA_FOLDER_EXT = "aCD96-Robustness-ambrs"
DATA_FILE_EXT = "AR21-042_AR23-019_067-Model-Data"
MATRIX_FOLDER_EXT = "CD96-Robustness"
PROCESS_TIME = 11
VOLUME = 200

# Make sure to check the window length for smoothing with moving average

STATES = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
    "Osmo",
    "pCO2_at_Temp",
]

INPUTS = [
    "Cumulative_Normalized_Feed",
    # "Cumulative_Normalized_Glucose",
    "Temperature",
    "pH_setpoint",
    "DO",
]

SMOOTHE_LIST = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
    "Osmo",
    "pCO2_at_Temp",
]

DISCARD = [
    # "AR23-019-009", # Reason: Lowest titer by far on day 14 and out of range
    # "AR23-019-011", # Reason: Titer and VCC predictions were very far off for this one batch
    # "AR23-019-003", # Reason: Second highest titer, very far out of range of most
    # "AR23-067-005", # Reason: Highest titer and well out of range of most
]

column_inclusion = [
    "Batch",
    "Condition",
    "Day",
]

scaler_train = MinMaxScaler()

data = pd.read_csv(
    fr"~\GSK\Biopharm Model Predictive Control - General\data\{DATA_FOLDER_EXT}\{DATA_FILE_EXT}.csv"
    )

dataframe = ModelData(
    raw_data=data,
    scaler_train=scaler_train,
    group="Batch",
    discard=DISCARD,
    states=STATES,
    inputs=INPUTS,
)

# Class method to clean up all the data
# this includes interpolation to start, spline smoothing, train and test set splitting
# and finally feature scaling using the scaler of choice
train_data, test_data = dataframe.clean(
    column_inclusion=column_inclusion,
    smoothing_list=SMOOTHE_LIST, # is smoothing list is empty, no data will be smoothed
    test_size = 0.20,
    n_splits = 2,
    random_state = 1,
    win_len=2,
)

# dataframe.graph_train_data(
#     smoothing_list=SMOOTHE_LIST,
#     test_label="VCC",
# )

# dataframe.graph_smoothed_unsmoothed_data(
#     smoothing_list=SMOOTHE_LIST,
#     test_label="VCC",
# )

with open(
    fr"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\A_Matrix.csv", 
    encoding="utf-8"
    ) as a_matrix:
    A_Matrix = np.loadtxt(a_matrix, delimiter=',')[:len(STATES),:len(STATES)]

with open(
    fr"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\B_Matrix.csv", 
    encoding="utf-8"
    ) as b_matrix:
    B_Matrix = np.loadtxt(b_matrix, delimiter=',')
    B_Matrix = np.c_[B_Matrix][:len(STATES),:len(INPUTS)]

# Number of days is always equal to the last day number + 1, so 12
# day culture duration will equal 13 days
scaler_dict = {}
for count, name in enumerate(scaler_train.get_feature_names_out()):
    scaler_dict[name] = [scaler_train.min_[count], scaler_train.scale_[count]]

joblib.dump(scaler_train, fr"M:\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\model_scaler.scl")

# UNCOMMENT TO PRINT OUT MODLE SCALING PARAMETERS FROM DICTIONARY

# for key, value in scaler_dict.items():
#     print(key)
#     print("scale:","",round(value[0],6))
#     print("min_:","",round(value[1],6))
#     print()

# Dictionary of constraints for the constraints needed in the optimized fucntion
constraint_dict = {
    "Volume": 150,
    "Sample_vol": 1,
    "Ammonium": 15,
    "Lactate": 5,
    "VCC": 30,
    "Max_feed_volume": 30,
}

# initial starting condition for your states in the model
initial_condition = np.array(
    test_data[test_data["Batch"]==test_data["Batch"].values[0]].filter(STATES)
    )[0,:]

feed_setpoint = test_data[test_data["Batch"]==test_data["Batch"].unique()[0]] \
    ["Cumulative_Normalized_Feed"].tolist()

setpoints = feed_setpoint

first_model_train = ModelTraining(
    train_data,
    test_data,
    a_matrix=A_Matrix,
    b_matrix=B_Matrix,
    states=STATES,
    inputs=INPUTS,
    num_days=PROCESS_TIME,
    scaler=scaler_train,
)

model_optimize = ModelOptimizer(
    target_label="IGG",
    a_matrix=A_Matrix,
    b_matrix=B_Matrix,
    states=STATES,
    inputs=INPUTS,
    scaler=scaler_train,
    constraint_dict=constraint_dict,
    initial_input=setpoints,
    initial_condition=initial_condition,
    days=PROCESS_TIME,
    volume=VOLUME,
    max_iters=100,
    scaler_dict=scaler_dict
)

# UNCOMMENT THIS CODE TO RUN OPTIMIZATION

# model_optimize.optimize()
# model_optimize.plot_inputs()
# model_optimize.plot_states()

# UNCOMMENT THIS CODE TO TRAIN THE MODEL ON THE DATA
# first_model_train.train_test_model(
#     fr"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}",
#     test_label="IGG",
#     iterations=50,
#     first_train=False,
# )

# first_model_train.plot_test_data(
#     test_label="IGG",
# )

# first_model_train.plot_train_data(
#     test_label="IGG",
#     random_plots=True,
# )

# first_model_train.plot_train_data(
#     test_label="VCC",
# )

# r2 = first_model_train.get_r2_table()
# print(r2)
# pd.DataFrame(r2).to_clipboard()

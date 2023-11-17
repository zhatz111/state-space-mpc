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

# suppress warnings
warnings.filterwarnings("ignore")

DATA_FOLDER_EXT = "aCD96-Robustness-ambrs"
DATA_FILE_EXT = "AR21-042_AR23-019_067-Model-Data"
MATRIX_FOLDER_EXT = "CD96-Robustness_Control_Model"
PDF_PLOT_FILENAME = "model2_report"
TARGET_LABEL = "IGG"
PROCESS_TIME = 11
VOLUME = 200

# Make sure to check the window length for smoothing with moving average

STATES = [
    "IGG",
    "VCC",
    "Viability",
    "Lactate",
    "Osmo",
    "pCO2_at_temp",
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
    "Osmo",
    "pCO2_at_temp",
]

DISCARD = ["AR23-067-005P"]

column_inclusion = [
    "Batch",
    "Condition",
    "Day",
]

scaler_train = MinMaxScaler()

data = pd.read_csv(
    rf"~\GSK\Biopharm Model Predictive Control - General\data\{DATA_FOLDER_EXT}\{DATA_FILE_EXT}.csv"
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
    smoothing_list=SMOOTHE_LIST,  # is smoothing list is empty, no data will be smoothed
    test_size=0.20,
    n_splits=2,
    random_state=1,
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

A_Matrix = np.array(
    pd.read_csv(
        rf"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\A_Matrix.csv",
        header=None,
    )
)

B_Matrix = np.array(
    pd.read_csv(
        rf"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\B_Matrix.csv",
        header=None,
    )
)

# Number of days is always equal to the last day number + 1, so 12
# day culture duration will equal 13 days
scaler_dict = {}
for count, name in enumerate(scaler_train.get_feature_names_out()):
    scaler_dict[name] = {
        "Label": name[-15:],
        "min_": round(scaler_train.min_[count], 5),
        "scale_": round(scaler_train.scale_[count], 5),
    }
scaler_table = pd.DataFrame.from_dict(scaler_dict).T.reset_index(drop=True)

joblib.dump(
    scaler_train,
    rf"M:\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\model_scaler.scl",
)

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
    test_data[test_data["Batch"] == test_data["Batch"].values[0]].filter(STATES)
)[0, :]

feed_setpoint = test_data[test_data["Batch"] == test_data["Batch"].unique()[0]][
    "Cumulative_Normalized_Feed"
].tolist()

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
    scaler_dict=scaler_dict,
)

# UNCOMMENT THIS CODE TO RUN OPTIMIZATION

# model_optimize.optimize()
# model_optimize.plot_inputs()
# model_optimize.plot_states()

# UNCOMMENT THIS CODE TO TRAIN THE MODEL ON THE DATA
first_model_train.train_test_model(
    fr"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}",
    test_label="IGG",
    iterations=25,
    first_train=False,
)

# first_model_train.plot_test_data(
#     test_label=TARGET_LABEL,
# )

# first_model_train.plot_train_data(
#     test_label=TARGET_LABEL,
#     random_plots=True,
# )

# first_model_train.generate_report(
#     output_pdf=fr"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\{PDF_PLOT_FILENAME}.pdf",
#     scaler_df=scaler_table,
#     metadata_path=r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\reports\report_info\metadata.txt",
#     figures_filepath=r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\reports\test_report",
#     logo_filepath=r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\reports\report_info\GSK_logo_2022.png",
#     xlim=15
# )

# first_model_train.plot_train_data(
#     test_label="VCC",
# )

# rmse = first_model_train.get_rmse_table()
# print(rmse)
# pd.DataFrame(rmse).to_clipboard()

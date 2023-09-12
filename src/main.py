
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.make_dataset import ModelData
from models.train_model import ModelTraining
from models.optimize_model import ModelOptimizer

#suppress warnings
warnings.filterwarnings('ignore')

DATA_FOLDER_EXT = "aCD96-ar21-023-042"
MATRIX_FOLDER_EXT = "CD96-AR21-042"
FILE_EXT = "AR21-042-Model-Data"

STATES = [
    "IGG",
    "VCC",
    "Ammonium",
    "Lactate",
]

INPUTS = [
    "Normalized_Feed_Percent",
]

SMOOTHE_LIST = [
    "VCC",
    "Ammonium",
    "Lactate",
]

DISCARD = []

column_inclusion = [
    "Batch",
    "Condition",
    "Day",
]

scaler_train = MinMaxScaler()

data = pd.read_csv(fr"~\GSK\Biopharm Model Predictive Control - General\data\{DATA_FOLDER_EXT}\{FILE_EXT}.csv")

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
    win_len=7,
    poly_order=3,
)

with open(fr"M:\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\A_Matrix.csv", encoding="utf-8") as a_matrix:
    A_Matrix = np.loadtxt(a_matrix, delimiter=',')

with open(fr"M:\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\B_Matrix.csv", encoding="utf-8") as b_matrix:
    B_Matrix = np.loadtxt(b_matrix, delimiter=',')
    B_Matrix = np.c_[B_Matrix]


# Number of days is always equal to the last day number + 1, so 12 day culture duration will equal 13 days
scaler_dict = {}
for count, name in enumerate(scaler_train.get_feature_names_out()):
    scaler_dict[name] = [scaler_train.min_[count], scaler_train.scale_[count]]
df_Scaler = pd.DataFrame.from_dict(scaler_dict, orient="index").reset_index()

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
initial_condition = np.array(test_data[test_data["Batch"]==test_data["Batch"].values[0]].filter(STATES))[0,:]
volume = 200


feed_setpoint = test_data[test_data["Batch"]==test_data["Batch"].unique()[0]]["Normalized_Feed_Percent"].tolist()
setpoints = feed_setpoint

first_model_train = ModelTraining(
    train_data,
    test_data,
    a_matrix=A_Matrix,
    b_matrix=B_Matrix,
    states=STATES,
    inputs=INPUTS,
    num_days=13,
    scaler_dict=scaler_dict,
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
    days=13,
    volume=volume,
    max_iters=100,
    scaler_dict=scaler_dict
)

# UNCOMMENT THIS CODE TO RUN OPTIMIZATION

# model_optimize.optimize()
# model_optimize.plot_inputs()
# model_optimize.plot_states()

# dataframe.graph_train_data(
#     smoothing_list=SMOOTHE_LIST,
#     test_label="Ammonium",
# )

# UNCOMMENT THIS CODE TO TRAIN THE MODEL ON THE DATA

# first_model_train.train_test_model(
#     fr"M:\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}", # \Experimental_matrices
#     test_label="IGG",
#     iterations=50,
#     first_train=False,
# )

# first_model_train.evaluate(
#     test_label="Lactate",
# )

first_model_train.test_model(
    test_label="Lactate",
)

# r2 = first_model_train.get_r2_table()
# print(r2)
# pd.DataFrame(r2).to_clipboard()

# SINGLE BATCH TEST

# data_test = pd.read_csv(r"~\GSK\Biopharm Model Predictive Control - General\data\aPVRIG-ar23-029\MR23-045_Flu_mAb_Test_Batch.csv")

# dataframe_test = ModelData(
#     df=data_test,
#     scaler_train=scaler_train,
#     group="Batch",
#     discard=[],
#     states=STATES,
#     inputs=INPUTS,
# )

# smoothed_data = dataframe_test.spline_smoothing(
#     smoothing_list=SMOOTHE_LIST,
#     win_len=7,
#     poly_order=3,
# )

# testing_data = dataframe_test.feature_scaling(
#     scaler=scaler_train,
#     data=smoothed_data,
#     new_scaler=False,
# )

# single_batch_test = ModelTraining(
#     train_data,
#     testing_data,
#     a_matrix=A_Matrix,
#     b_matrix=B_Matrix,
#     states=STATES,
#     inputs=INPUTS,
#     num_days=15,
#     scaler_dict=scaler_dict,
# )

# single_batch_test.single_batch_test(
#     test_label="IGG",
# )

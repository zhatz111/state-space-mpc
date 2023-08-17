import sys
sys.path.insert(0, r'C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model')

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data.make_dataset import ModelData
from src.models.train_model import ModelTraining
from src.models.optimize_model import ModelOptimizer

#suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Current model error (includes glucose input): 8.07
# Current model error (excludes glucose input): 11.9

folder_ext = "AR23-029_MR23-045"
file_ext = "AR23-029_MR23_045-Model-Data"

STATES = [
    # "IVC",
    "VCC",
    # "ILAC",
    # "Osmo",
    "Ammonium",
    "IGG",
]

INPUTS = [
    "Daily_Feed_Normalized",
    "Post_Glucose_Conc",
    "Temperature",
    # "Post_Gluose_Conc",
]

SMOOTHE_LIST = [
    "VCC",
    # "Osmo",
    "Ammonium",
    "IGG",
    # "IVC",
    # "ILAC",
]

DISCARD = [
    # Discarding these batches because IGG r2 value was 0.55 for both ar23-029 batches, while
    # all other batches were above 0.85, the ar23-014 batch is being discard because it has a negative
    # r2 value for lactate
    "AR23-014-007", # lactate r2 value is -0.09
    "AR23-014-020", # lactate r2 value is negative
    "AR23-029-015", # lactate r2 value is -0.091992
    "AR23-029-023", # IGG r2 value is less than 0.6
    "AR23-029-024", # IGG r2 value is less than 0.6
    "MR23-035-724",
]

column_inclusion = [
    "Batch",
    "Condition",
    "Day",
]

scaler_train = MinMaxScaler()
# scaler_train = joblib.load("./models/AR23-014_029/data/scaler_train_AR23-014_029.scale")

data = pd.read_csv(fr"~\GSK\Biopharm Model Predictive Control - General\data\aPVRIG-ar23-029\{file_ext}.csv")

dataframe = ModelData(
    df=data,
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
    smoothing_list=SMOOTHE_LIST,
    test_size = 0.20,
    n_splits = 2,
    random_state = 1,
    win_len=7,
    poly_order=3,
)

# Save the Scaler for both the training and test sets to rescale in the future
# with open("./models/AR23-014_029/data/scaler_train_AR23-014_029", encoding="utf-8") as scaler:
#     dataframe.save_scaler(scaler, scaler=scaler_train)

# A_Matrix = pd.read_csv(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\A_Matrix.csv").values
# B_Matrix = pd.read_csv(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\B_Matrix.csv").values

# print(A_Matrix)
# print(B_Matrix)

with open(fr"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\current\{folder_ext}\A_Matrix.csv", encoding="utf-8") as a_matrix:
    A_Matrix = np.loadtxt(a_matrix, delimiter=',')

with open(fr"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\current\{folder_ext}\B_Matrix.csv", encoding="utf-8") as b_matrix:
    B_Matrix = np.loadtxt(b_matrix, delimiter=',')

# with open(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\Experimental_matrices\A_Matrix.csv", encoding="utf-8") as a_matrix:
#     A_Matrix = np.loadtxt(a_matrix, delimiter=',')

# with open(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\Experimental_matrices\B_Matrix.csv", encoding="utf-8") as b_matrix:
#     B_Matrix = np.loadtxt(b_matrix, delimiter=',')


# Number of days is always equal to the last day number + 1, so 12 day culture duration will equal 13 days
scaler_dict = {}
for count, name in enumerate(scaler_train.get_feature_names_out()):
    scaler_dict[name] = [scaler_train.scale_[count], scaler_train.min_[count]]


# UNCOMMENT TO PRINT OUT MODLE SCALING PARAMETERS FROM DICTIONARY

# for key, value in scaler_dict.items():
#     print(key)
#     print("scale:","",round(value[0],6))
#     print("min_:","",round(value[1],6))
#     print()

df_Scaler = pd.DataFrame.from_dict(scaler_dict, orient="index").reset_index()

# Dictionary of constraints for the constraints needed in the optimized fucntion
constraint_dict = {
    "Volume": 150,
    "Sample_vol": 1,
    "Ammonium": 15,
    "Lactate": .5,
    "Glucose": 1.0,
    "IGG": 7000,
    "VCC": 30,
    "IVC": 300,
    "ILAC": 10,
    "Max_feed_volume": 40,
}

# input length x day length matrix for the inputs into your model optimizer
# glucose_input = np.array(test_data[test_data["Batch"]==test_data.sample()["Batch"].values[0]].filter(like="Daily_Glucose_Normalized"))
# glucose_input = np.array(test_data.groupby("Day")["Daily_Glucose_Normalized"].mean())

# feed = np.array(test_data[test_data["Batch"]=="AR23-029-004"].filter(like="Daily_Feed_Normalized"))
# feed_polynomial = np.poly1d(np.polyfit(np.arange(0,14),feed.ravel()[:-1],deg=3))

# print((feed-scaler_dict["Daily_Feed_Normalized"][1])/scaler_dict["Daily_Feed_Normalized"][0])

# initial starting condition for your states in the model
initial_condition = np.array(test_data[test_data["Batch"]==test_data["Batch"].values[0]].filter(STATES))[0,:]
volume = 150

# print(test_data["Batch"].values[0])
# # initial_condition[0] = 0.980477
# # initial_condition[0] = initial_condition[0]*2
# print(initial_condition)

# Setpoints are: Daily Feed %, pH setpoint, temp start, temp end, temp shift day
# setpoints = list(feed_polynomial.coef) + [7.05,36.5,31.,5.]

# post_glucose_setpoint = np.array(test_data[test_data["Batch"]==test_data.sample()["Batch"].values[0]].filter(like="Post_Glucose_Conc"))

# feed_setpoint = [0.03]*14
# post_glucose_setpoint = [0]*15
feed_setpoint = test_data[test_data["Batch"]=="MR23-045-718"]["Daily_Feed_Normalized"].tolist()[:14]
post_glucose_setpoint = test_data[test_data["Batch"]=="MR23-045-718"]["Post_Glucose_Conc"].tolist()

# post_glucose_setpoint = np.array(test_data[test_data["Batch"]==test_data.sample()["Batch"].values[0]].filter(like="Post_Glucose_Conc"))
setpoints = feed_setpoint + post_glucose_setpoint + [36.5,31.,5.]

# print(initial_condition)
# print()
# print(setpoints)
# print()
# print(glucose_input)
# print()

first_model_train = ModelTraining(
    train_data,
    test_data,
    a_matrix=A_Matrix,
    b_matrix=B_Matrix,
    states=STATES,
    inputs=INPUTS,
    num_days=15,
    scaler_dict=scaler_dict,
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
    days=15,
    max_iters=100,
    scaler_dict=scaler_dict
)

# UNCOMMENT THIS CODE TO RUN OPTIMIZATION

# model_optimize.glucose = post_glucose_setpoint
model_optimize.optimize()
model_optimize.plot_inputs()
model_optimize.plot_states()
# pd.DataFrame(model_optimize.result).to_clipboard()

# dataframe.graph_train_data(
#     smoothing_list=SMOOTHE_LIST,
#     test_label="Post_Gluose_Conc",
#     ylim=8,
# )

# UNCOMMENT THIS CODE TO TRAIN THE MODEL ON THE DATA

# first_model_train.train_test_model(
#     fr"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\current\{folder_ext}", # \Experimental_matrices
#     test_label="IGG",
#     iterations=50,
#     first_train=False,
# )

# first_model_train.evaluate(
#     test_label="IGG",
#     ylim=6000,
# )

# first_model_train.evaluate(
#     test_label="VCC",
#     ylim=35,
# )

# first_model_train.evaluate(
#     test_label="Ammonium",
#     ylim=16,
# )

# first_model_train.test_model(
#     test_label="Lactate",
#     ylim=3,
# )

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
#     test_label="Osmo",
# )

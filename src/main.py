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

STATES = [
    "VCC",
    "Lactate",
    "IGG",
]

INPUTS = [
    "Daily_Feed_Normalized",
    "Daily_Glucose_Normalized",
    "pH_Setpoint",
    "Temperature",
]

SMOOTHE_LIST = [
    "VCC",
    "Lactate",
    "IGG",
]

DISCARD = [
    # Discarding these batches because IGG r2 value was 0.55 for both ar23-029 batches, while
    # all other batches were above 0.85, the ar23-014 batch is being discard because it has a negative
    # r2 value for lactate
    "AR23-014-007", # lactate r2 value is -0.09
    "AR23-014-020", # lactate r2 value is negative
    "AR23-029-023", # IGG r2 value is less than 0.6
    "AR23-029-024", # IGG r2 value is less than 0.6
]

column_inclusion = [
    "Batch",
    "Condition",
    "Day",
]

scaler_train = MinMaxScaler()
# scaler_train = joblib.load("./models/AR23-014_029/data/scaler_train_AR23-014_029.scale")

data = pd.read_csv(r"~\GSK\Biopharm Model Predictive Control - General\data\aPVRIG-ar23-029\AR23-014_029-Model-Data.csv")

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
    poly_order=2,
)

# Save the Scaler for both the training and test sets to rescale in the future
# with open("./models/AR23-014_029/data/scaler_train_AR23-014_029", encoding="utf-8") as scaler:
#     dataframe.save_scaler(scaler, scaler=scaler_train)

# A_Matrix = pd.read_csv(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\A_Matrix.csv").values
# B_Matrix = pd.read_csv(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\B_Matrix.csv").values

# print(A_Matrix)
# print(B_Matrix)


with open(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\A_Matrix.csv", encoding="utf-8") as a_matrix:
    A_Matrix = np.loadtxt(a_matrix, delimiter=',')

with open(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029\B_Matrix.csv", encoding="utf-8") as b_matrix:
    B_Matrix = np.loadtxt(b_matrix, delimiter=',')

# with open("./models/AR23-014_029/data/B_Matrix.csv", encoding="utf-8") as b_matrix:
#     B_Matrix = np.loadtxt(b_matrix, delimiter=',')

# Number of days is always equal to the last day number + 1, so 12 day culture duration will equal 13 days
scaler_dict = {}
for count, name in enumerate(scaler_train.get_feature_names_out()):
    scaler_dict[name] = [scaler_train.scale_[count], scaler_train.min_[count]]

df_Scaler = pd.DataFrame.from_dict(scaler_dict, orient="index").reset_index()

# Dictionary of constraints for the constraints needed in the optimized fucntion
constraint_dict = {
    "Ammonium": 15,
    "Lactate": 0.2,
    "Glucose": 1.0,
    "IGG": 7000,
    "VCC": 15,
}
# input length x day length matrix for the inputs into your model optimizer
glucose_input = np.array(test_data[test_data["Batch"]=="AR23-029-004"].filter(like="Daily_Glucose_Normalized"))
feed = np.array(test_data[test_data["Batch"]=="AR23-029-004"].filter(like="Daily_Feed_Normalized"))
feed_polynomial = np.poly1d(np.polyfit(np.arange(0,14),feed.ravel()[:-1],deg=3))

# initial starting condition for your states in the model
initial_condition = np.array(test_data[test_data["Batch"]=="AR23-029-004"].filter(STATES))[0,:]
volume = 200

# Setpoints are: Daily Feed %, pH setpoint, temp start, temp end, temp shift day
setpoints = list(feed_polynomial.coef) + [7.05,36.5,31.,5.]

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
    max_iters=10,
    scaler_dict=scaler_dict
)

# UNCOMMENT THIS CODE TO RUN OPTIMIZATION

# model_optimize.glucose = glucose_input
# model_optimize.optimize()
# model_optimize.plot_inputs()
# model_optimize.plot_states()
# pd.DataFrame(model_optimize.result).to_clipboard()


# UNCOMMENT THIS CODE TO TRAIN THE MODEL ON THE DATA

# first_model_train.train_test_model(
#     r"M:\Zach Hatzenbeller\State-Space-Matrices\AR23-014_029",
#     test_label="IGG",
#     iterations=50,
#     first_train=False,
# )

first_model_train.evaluate(
    test_label="IGG",
    ylim=6500,
)

# r2 = first_model_train.get_r2_table()
# pd.DataFrame(r2).to_clipboard()
# print(r2)

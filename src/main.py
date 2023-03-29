import sys
sys.path.insert(0, r'C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model')

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data.make_dataset import ModelData
from src.models.train_model import ModelTraining
from src.models.optimize_model import ModelOptimizer

STATES = [
    # "TCC",
    "VCC",
    # "Viability",
    # "Glucose",
    "Glutamate",
    "Lactate",
    # "Osmo",
    "Ammonium",
    "IGG",
    # "pH_at_temp",
]

INPUTS = [
    "Daily_Feed_Normalized",
    "Daily_Glucose_Normalized",
    # "Controller_pH",
    "pH_Setpoint",
    # "Daily_CO2_Volume_Normalized",
    "Temperature",
]

SPLINES = [
    # "TCC",
    "VCC",
    # "Viability",
    # "Glucose",
    "Glutamate",
    "Lactate",
    # "Osmo",
    "Ammonium",
    "IGG",
    # "pH_at_temp",
]

DISCARD = [
    # "AR23-014-007",
    # "AR23-014-017",
    # "AR23-014-023",
    # "AR23-014-024",
]

column_inclusion = [
    "Batch",
    "Condition",
    "Day",
]

scaler_train = MinMaxScaler()
# scaler_train = joblib.load("scaler_train.scale")

data = pd.read_csv(r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\raw\AR23-014-Model-Data.csv")

#AR22-001-Model-Data
#DR-Model-Data-PVRIG

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
    spline_list=SPLINES,
    test_size = 0.15,
    n_splits = 2,
    random_state = 1,
)

# Save the Scaler for both the training and test sets to rescale in the future
dataframe.save_scaler("./models/Model 2/scaler_train_AR23-014", scaler=scaler_train)

# A_Matrix = np.loadtxt(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR22-001-Kalman-Filter\A_Matrix.csv", delimiter=',')
# B_Matrix = np.loadtxt(r"M:\Zach Hatzenbeller\State-Space-Matrices\AR22-001-Kalman-Filter\B_Matrix.csv", delimiter=',')

A_Matrix = np.loadtxt("./models/Model 2/data/A_Matrix.csv", delimiter=',')
B_Matrix = np.loadtxt("./models/Model 2/data/B_Matrix.csv", delimiter=',')

# "C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\current\Model 1\A_Matrix.csv"

# Number of days is always equal to the last day number + 1, so 12 day culture duration will equal 13 days
scaler_dict = {}
for count, name in enumerate(scaler_train.get_feature_names_out()):
    scaler_dict[name] = [scaler_train.scale_[count], scaler_train.min_[count]]
# Dictionary of constraints for the constraints needed in the optimized fucntion
constraint_dict = {
    "Ammonium": 15,
    "Lactate": 3.0,
    "Glucose": 1.0,
    "IGG": 7000,
    "VCC": 30,
}
# input length x day length matrix for the inputs into your model optimizer
glucose_input = np.array(test_data[test_data["Batch"]=="AR23-014-004"].filter(like="Daily_Glucose_Normalized"))
# initial starting condition for your states in the model
initial_condition = np.array(test_data[test_data["Batch"]=="AR23-014-004"].filter(STATES))[0,:]
volume = 200
# Setpoints are: Daily Feed %, pH setpoint, temp start, temp end, temp shift day
setpoints = np.array([[3.],[7.05],[36.5],[31.],[5.]])
# initial_input = np.vstack([setpoints,glucose_input])

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

model_optimize.glucose = glucose_input
model_optimize.optimize()
model_optimize.plot_inputs()
model_optimize.plot_states()

# first_model_train.train_test_model(
#     r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\models\Model 2\data",
#     test_label="IGG",
#     iterations=100,
#     first_train=False,
# )

# first_model_train.evaluate(
#     test_label="IGG",
#     # ylim=5500,
# )

# rmse = first_model_train.get_rmse_table()
# rmse.to_clipboard()
# print(rmse)

r2 = first_model_train.get_r2_table()
r2.to_clipboard()
print(r2)

# corr = first_model_train.get_corrcoef_table()
# corr.to_clipboard()
# print(corr)

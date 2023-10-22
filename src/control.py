"""Main code for simulating closed-loop MPC
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from mpc.mpc_optimizer import *
from models.ssm import *
warnings.filterwarnings('ignore',category=UserWarning)

# Load an example dataset
data = pd.read_csv(
    fr"~\GSK\Biopharm Model Predictive Control - General\data\batch-record-template\data_sp_mv_only.csv"
    )

# Load the model
MATRIX_FOLDER_EXT = "mpc-final-matrices"
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
    "Temperature",
    "pH_setpoint",
    "DO",
]

# Import both matrices and the scaler for the data
model_scaler = joblib.load(
    rf"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\model_scaler.scl"
)
A_matrix = np.array(
    pd.read_csv(
        rf"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\A_Matrix.csv",
        header=None,
    )
)
B_matrix = np.array(
    pd.read_csv(
        rf"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\B_Matrix.csv",
        header=None,
    )
)

# Instantiate the StateSpaceModel object
robustness_model = StateSpaceModel(
    states=STATES,
    inputs=INPUTS,
    scaler=model_scaler,
    a_matrix=A_matrix,
    b_matrix=B_matrix,
)

# Construct a bioreactor object
bioreactor = Bioreactor(
    vessel=1,
    process_model=robustness_model,
    data=data)

# Construct an open-loop bioreactor object
bioreactor_open_loop = copy.deepcopy(bioreactor)

# Construct a controller object
ts = data['Day'].values
pv_names = np.array(['IGG'])
pv_wts = np.array([1])
pv_sps = data[pv_names].values
mv_names = np.array([
    'Cumulative_Normalized_Feed',
    # 'pH_setpoint'
    ])
mv_wts = np.array([
    1,
    # 1
    ])
constr = np.array([
    [0,     0.1],   # feed
    # [7,   7.35]     # pH
    ])
mv_matrix = data[mv_names].values
pred_horizon = 30
ctrl_horizon = 3

curr_time = 0

controller = Controller(
    controller_model=robustness_model,
    bioreactor=bioreactor,
    ts=ts,
    pv_sps=pv_sps,
    pv_names=pv_names,
    pv_wts=pv_wts,
    mv_names=mv_names,
    mv_wts=mv_wts,
    pred_horizon=pred_horizon,
    ctrl_horizon=ctrl_horizon,
    constr=constr,
    curr_time=curr_time
)

# Simulate trajectory without MPC
# bioreactor_open_loop.next_day()

# # Reset
# bioreactor.reset()

# Simulate a process
for i in range(len(ts) - 1):
    controller.optimize(plot=True)
    # bioreactor.show_data()
    bioreactor_open_loop.next_day()
    bioreactor.next_day()
    
plt.show()
bioreactor_open_loop.show_data()   
bioreactor.show_data()

# x_out = bioreactor.next_day()
# print(x_out)
# x_out = bioreactor.next_day()
# print(x_out)

# print(bioreactor.data)


# # mv_array = mv_matrix[data['Day'] >= curr_time,:].flatten() + 1
# # controller.obj_func_wrapper(mv_array=mv_array)

# controller.optimize()



# ts = data.loc[:,'Day'].values
# pv_names = np.array(['IGG'])
# pv_wts = np.array([1])
# pv_sps = data.loc[:,pv_names].values
# mv_names = np.array(['Cumulative_Normalized_Feed','pH_setpoint'])
# mv_wts = np.array([1,1])
# mv_matrix = data.loc[:,mv_names].values
# pred_horizon = 30
# ctrl_horizon = 3
# constr = np.array([[0,7],[0.1,7.35]])
# curr_time = 3

# controller = Controller(
#     controller_model=robustness_model,
#     bioreactor=bioreactor,
#     ts=ts,
#     pv_sps=pv_sps,
#     pv_names=pv_names,
#     pv_wts=pv_wts,
#     mv_names=mv_names,
#     mv_wts=mv_wts,
#     pred_horizon=pred_horizon,
#     ctrl_horizon=ctrl_horizon,
#     constr=constr,
#     curr_time=curr_time
# )


# # Libraries
# import numpy as np
# import cvxpy as cp  # For optimization (optional)
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from scipy.optimize import minimize

# # Constants
# DATA_FOLDER_EXT = "aCD96-Robustness-ambrs"
# DATA_FILE_EXT = "AR23-019_067-Model-Data"
# MATRIX_FOLDER_EXT = "CD96-Robustness"
# PROCESS_TIME = 11
# VOLUME = 200
# STATES = [
#     "IGG",
#     "VCC",
#     "Lactate",
#     "Ammonium",
# ]
# INPUTS = [
#     "Cumulative_Normalized_Feed",
#     "Temperature",
#     "pH_setpoint",
#     "DO",
# ]

# # Define the model
# with open(
#     fr"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\A_Matrix.csv", 
#     encoding="utf-8"
#     ) as a_matrix:
#     A_Matrix = np.loadtxt(a_matrix, delimiter=',')[:len(STATES),:len(STATES)]

# with open(
#     fr"\\kopdsntp006\SA199800263\Zach Hatzenbeller\State-Space-Matrices\{MATRIX_FOLDER_EXT}\B_Matrix.csv", 
#     encoding="utf-8"
#     ) as b_matrix:
#     B_Matrix = np.loadtxt(b_matrix, delimiter=',')
#     B_Matrix = np.c_[B_Matrix][:len(STATES),:len(INPUTS)]
# # A = np.array([[-28.42917581, .43123473, -13.97685655, -65.66274965],
# #               [153.9119679, -18.73903529, -5.175464382, -20.9226924],
# #               [42.69029844, 23.34400248, -93.96171558, -2.447073457],
# #               [-15.02232579, 9.127941243, 0.353058181, -44.80103775]])
# # B = np.array([[80.65505836,-12.30935346, 0.741227919, 1.271968107],
# #               [-98.87239294, -35.59799158, -4.688637079, 3.903624997],
# #               [-51.83706735, -2.714547051, 6.779817164, 28.81280986],
# #               [53.33622907, -11.46735006, -0.20281312, -1.754946275]])

# # Define the controller
# N = 10  # Prediction horizon
# M = 3   # Control horizon
# Q = np.diag([1.0, 0, 0, 0])  # State cost
# R = np.diag([0.1, 0.1, ])       # Control cost
# x0 = np.array([0.0, 0.0])  # Initial state

# # Define decision variables
# u = cp.Variable((M, 1))

# # Define the cost function
# cost = cp.quad_form(x0, Q)  # Initial state cost

# for k in range(N):
#     x_kp1 = A @ x0 + B @ u[0]
#     cost += cp.quad_form(x_kp1, Q)
#     cost += cp.quad_form(u[0], R)
#     x0 = x_kp1  # Update the state for the next time step

# # Define constraints
# constraints = [u >= 0, u <= 1]  # Control input constraints (optional)

# # Create the optimization problem
# prob = cp.Problem(cp.Minimize(cost), constraints)

# # Solve
# prob.solve()

# # Results
# optimal_u = u.value[0, 0]
# print(optimal_u)


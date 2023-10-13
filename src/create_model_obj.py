"""_summary_
"""

# pylint: disable=locally-disabled, multiple-statements, fixme, no-name-in-module
# pylint: disable=locally-disabled, multiple-statements, fixme, import-error

import joblib
import numpy as np
import pandas as pd

# Imports from within repository
from models.ssm import StateSpaceModel

# These three varibles are constant
MATRIX_FOLDER_EXT = "CD96-Robustness"
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

data = pd.read_csv(
    fr"~\GSK\Biopharm Model Predictive Control - General\data\batch-record-template\data.csv"
    )

x0 = np.array(data.loc[0,STATES].values)
U = np.array(data.loc[0:0,INPUTS].values)
time = np.arange(0,len(U),1)

xHats = robustness_model.ssm_lsim(x0,U,time)
print(xHats)
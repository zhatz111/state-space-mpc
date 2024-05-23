"""This program is designed to simulate a bioreactor run through the use of
experimental data. The program tries to model a bioreactor by using the State-
space representation. This take in states and inputs and uses them along with
and A and B matrix to determine the future state of the system.

Returns:
    1D Matrix: This program ultimately returns a 1D matrix with the optimal CSFR
    values to maximize the target you are looking at.
"""
# Import Data manipulation and Matrices Libraries
import numpy as np
import pandas as pd

# Matplotlib Packages
from pylab import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

# Scipy Packages
from scipy import signal
from scipy import optimize
from sklearn.metrics import r2_score
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# END OF IMPORTS

# ------------------------------------------------------------------------------------

# START OF SCRIPT

print("Initializing Variables...")
# Set the random seed to achieve the same randomness everytime the script is run
# This helps with having reproducible results for the same inputs each time
np.random.seed(10)

# Variables to initialize before running this script
# VOLUME_ERROR was determined through a minimization function on the error
# between the calculated volume and the actual volume for each batch. This represents the volume
# change due to sampling, other additions (base & acid) and evaporation (ambr scale only)
DAYS = 13
NUMBER_STATES = 7
NUMBER_INPUTS = 2
VOLUME_ERROR = 1.431  # Should be in mL
TOTAL = NUMBER_STATES + NUMBER_INPUTS

# Constraint variables in the CSFR optimization function
MAX_CSFR = 0.0005
MIN_CSFR = 0.0001
MAX_LACTATE = 2
MAX_AMMONIUM = 40
GLUC_SETPOINT = 1.5
MAX_FEED_PERCENT = 20
VIABILITY_THRESHOLD = 0.5

# SPLINE_FIT: Smooth data with a spline fit for all states specified in the spline list
# MINIMIZE_ITERS: Determines how long you want the error reduction function to run for
#                 (ex: 100 ~~ 10-15 minutes)
# ERROR_REDUCTION: If set to true the minimize function will be used to reduce the A and B
#                  matrix error
# OPTIMIZED_ARRAYS: If set to true then the A and B matrix will be the ones you defined
#                   If set to false the A and B matrix will be from the lienar regression
# INITIAL_REGRESSION: Set this to true if this is the first time running the script and no
#                     pre-defined matrices have been found
# ALL_INPUT_OPTIMIZATION: If set to true all the inputs will be optimized, if set to false
#                         only the first input will be optimized the rest will be set to
#                         average values based on the batches values
# BATCH: Decide which batch you want to evaluate the optimized matrices on, can either be
#        in test or train set
# TEST_BATCHES: Decide which batches you want to include as the test batches for evaluating
#               on new data after A and B are determined
# PLOT_LIST: Tell the program what data you want plotted at the end to compare the simulation
#            and experimental results to one another
SPLINE_FIT = True
MINIZE_ITERS = 30
ERROR_REDUCTION = False
OPTIMIZED_ARRAYS = True
INITIAL_REGRESSION = False
ALL_INPUT_OPTIMIZATION = True
BATCH = "AR22-001-003"
PLOT_LIST = ["VCC", "Lact", "Gluc", "IGG"]
TEST_BATCHES = ["AR22-001-019", "AR22-001-003"]

# Plot Parameters for nice looking plots
# Edit the linewidth, markerwidth, optwidth, and markersize to get plot to
# your specification
# Edit the font, font size, and axes width of the figure and colors of the
# simulated/experimental data graphs
OPTWIDTH = 1
LINEWIDTH = 1
MARKERSIZE = 7
MARKERWIDTH = 1
SIMULATION_COLOR = "#346beb"
EXPERIMENTAL_COLOR = "#eb4034"
mpl.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.5

# COLUMN_ORDER: List of columns in excel file which you wish to use in this modelling.
#               These should be in the order as such: {"Day", STATES, INPUTS}.
#               "Day" column must be included for this program to run properly.
# STATES: Tell the program what states and inputs you will be using for your analysis
#         Based on the number of states and inputs you specify at the top, the program
#         will seperate the list into the appropriate format.
# SPLINES: Tell the program which states and inputs you want smoothed using a spline.
#          This can be very beneficial to tease out noisy data or data that you believe has lots
#          of variability with its measurements. This should help to make your A and B matrices
#          more accurate in most cases.
# DISCARD: You can add batches to this list that you do not want to use in the analysis.
#          This is useful if you want to filter out some data that may not be representative.
COLUMN_ORDER = [
    "Day",
    "TCC",
    "VCC",
    "Lact",
    "Osmo",
    "Gluc",
    "Ammonium",
    "IGG",
    "Volume",
    "F30 Feed Amount (mL)",
    "Glucose Added (mL)",
]

STATES_INPUTS = [
    "TCC",
    "VCC",
    "Lact",
    "Osmo",
    "Gluc",
    "Ammonium",
    "IGG",
    "Daily Feed Normal (mL)",
    "Daily Glucose Normal (mL)",
]

SPLINES = [
    "TCC",
    "VCC",
    "Lact",
    "Osmo",
    "Ammonium",
    "IGG",
    "F30 Feed Amount (mL)",
]

DISCARD = [
    "AR21-048-001",
    "AR21-048-003",
    "AR21-048-009",
    "AR22-001-001",
]

if ALL_INPUT_OPTIMIZATION:
    bounds = np.empty(((DAYS * NUMBER_INPUTS), NUMBER_INPUTS))
    bounds[:, 0] = 0
    bounds[:, 1] = 2
else:
    bounds = np.empty((DAYS, 1))
    bounds[:, 0] = 0
    bounds[:, 1] = 2

# ------------------------------------------------------------------------------------

# THIS IS THE START OF THE BULK OF THE SCRIPT, NO MORE INPUTS NEEDED AFTER THIS POINT
# Read the excel file that contains the data to model and discard any batches that may be outliers
print("Reading Data File...")
df = pd.read_csv("./models/Model 1/data/Model_1_Data_AR22-001.csv")
df = df[~df["Batch"].str.contains("|".join(DISCARD))]

# Build a dictionary for each batch in the data file and interpolate any null values for
# continuous data
# Linear interpolation was selected after evaluating multiple other avenues
# Re-built a dataframe from the interpolated dictionary for easier use in future steps
print("Interpolating missing data points...")
batch_dict = {}
x = np.linspace(0, DAYS - 1, DAYS)
for i in df["Batch"].unique():
    df_batch = df[df["Batch"] == i]
    df_batch = df_batch[COLUMN_ORDER]
    df_batch.interpolate(method="linear", inplace=True)
    df_batch.interpolate(
        method="backfill", inplace=True, limit_direction="backward", limit=2
    )

    # This code is specific to PVRIG project and can be discarded during templating
    df_batch["Daily Feed Fed (mL)"] = np.diff(
        df_batch["F30 Feed Amount (mL)"], n=1, prepend=0
    )
    df_batch["Daily Glucose Fed (mL)"] = np.diff(
        df_batch["Glucose Added (mL)"], n=1, prepend=0
    )

    # Make scale dependent variables independent
    df_batch["Daily Feed Normal (mL)"] = df_batch["Daily Feed Fed (mL)"] / df_batch['Volume']
    df_batch["Daily Glucose Normal (mL)"] = df_batch["Daily Glucose Fed (mL)"] / df_batch['Volume']

    spl_dict = {}
    for w in SPLINES:
        spl = UnivariateSpline(x, df_batch[w])
        spl_dict[w] = spl
    if SPLINE_FIT:
        for w in SPLINES:
            df_batch[w] = spl_dict[w](x)

    # This code is specific to PVRIG project and can be discarded during templating
    df_batch["CSFR"] = (df_batch["Daily Feed Fed (mL)"] / (24 * 60)) / (
        df_batch["VCC"] * (df_batch['Volume']/1000)
    )
    df_batch = df_batch.replace(np.nan, 0)
    batch_dict[i] = df_batch

interpolated_df = pd.concat(batch_dict).reset_index()

# Scale all the data with the min-max scaler to avoid large values in certain columns from
# dominating the model
# This scaler will scale the data between 0 and 1
print("Scaling data...")
pre_scaled = interpolated_df.iloc[:, 3:]

# This code is specific to PVRIG project and can be discarded during templating
pre_scaled = pre_scaled.drop([
    "F30 Feed Amount (mL)",
    "Glucose Added (mL)",
    "Daily Feed Fed (mL)",
    "Daily Glucose Fed (mL)",
    "CSFR",
    "Volume",
    ], axis=1
)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(pre_scaled)
features = scaler.get_feature_names_out()
df_scaled = pd.DataFrame(scaled, columns=features)
df_scaled["Day"] = interpolated_df.iloc[:, 2].values
df_scaled["Batch"] = interpolated_df.iloc[:, 0].values


# Seperate data into train and test batches
df_train = df_scaled[~df_scaled["Batch"].str.contains("|".join(TEST_BATCHES))]
df_test = df_scaled[df_scaled["Batch"].str.contains("|".join(TEST_BATCHES))]

# Initial determination of the A and B matrices for the state space model
# This uses linear regression to correlate each state with one another and provide an intial guess
# before optimization
print("Determing A and B matrices through linear regression...")
x = np.array(
    df_train[(df_train["Day"] >= 0) & (df_train["Day"] <= 11)].iloc[:, 0:TOTAL]
)
y = np.array(
    df_train[(df_train["Day"] >= 1) & (df_train["Day"] <= 12)].iloc[:, 0:NUMBER_STATES]
)
regression_matrix = np.zeros([NUMBER_STATES, TOTAL])
for j in range(NUMBER_STATES):
    reg = LinearRegression().fit(x, y[:, j])
    regression_matrix[j, :] = reg.coef_

# Define the matrices to be used in the state space model
# The A and B in this case are from the initial guess
A_Matrix = regression_matrix[:NUMBER_STATES, :NUMBER_STATES]
B_Matrix = regression_matrix[:NUMBER_STATES, NUMBER_STATES:]
C = np.identity(NUMBER_STATES)
D = np.zeros([NUMBER_STATES, NUMBER_INPUTS])

# This is the optimization function that takes in the A and B matrices
# It reshapes them to be a column vector then tries to minimize the column vector
# to decrease the squared sum of errors from the objective funtion
# Make sure to specify the maximum iterations to ensure this function will stop
# Optimize the A and B matrix to decrease the error

START = 0
STOP = DAYS
STEP = 1
t = np.arange(START, STOP, STEP)

# ------------------------------------------------------------------------------------

# START OF ERROR REDUCTION OPTIMIZATION FUNCTION FOR A AND B MATRICES
# If set to false it will use the hard coded matrix you have above
# If set to true it will use the matrix that was determined through linear regression
if INITIAL_REGRESSION:
    A_Matrix = regression_matrix[:NUMBER_STATES, :NUMBER_STATES]
    B_Matrix = regression_matrix[:NUMBER_STATES, NUMBER_STATES:]
else:
    A_Matrix = np.loadtxt(r".\data\current\Model 1\A_Matrix.csv", delimiter=',')
    B_Matrix = np.loadtxt(r".\data\current\Model 1\B_Matrix.csv", delimiter=',')

C = np.identity(NUMBER_STATES)
D = np.zeros([NUMBER_STATES, NUMBER_INPUTS])

# Needs to reshape the matrix to a column vector
a_sim = A_Matrix.reshape(-1, 1)
b_sim = B_Matrix.reshape(-1, 1)
combined_mat = np.vstack([a_sim, b_sim])

if ERROR_REDUCTION:
    print("Performing A and B Matrix Optimization...")
    # The objective function is what spits out the error of the matrices and
    # tries to reduce this error
    # Define non-dependent variables for use in objective function
    def objective_func(mat, info):
        """Objective function to minimize the error of that the A and B matrices
           have when being used in the State Space model.

        Args:
            mat (1D Matrix): This is the u matrix reshaped into a 1D column vector
            info (int): This takes in the minize functions iteration number for keeping
            track of how long that function runs for

        Returns:
            error: This function returns the sum of squared errors
        """
        y_sim_all = np.zeros([DAYS, NUMBER_STATES])
        y_actual_all = np.zeros([DAYS, NUMBER_STATES])
        iter_counter = 0
        for unique_batch in df_train["Batch"].unique():
            df_sim = df_train[df_train["Batch"] == unique_batch].sort_values("Day")
            x_sim_er = np.array(df_sim.iloc[0, 0:NUMBER_STATES])
            u_sim = np.array(df_sim.iloc[:, NUMBER_STATES:TOTAL])
            y_actual = np.array(df_sim.iloc[:, :NUMBER_STATES])
            a_matrix = mat[: (NUMBER_STATES**2)].reshape(NUMBER_STATES, NUMBER_STATES)
            b_matrix = mat[(NUMBER_STATES**2) :].reshape(NUMBER_STATES, NUMBER_INPUTS)
            c_matrix = np.identity(NUMBER_STATES)
            d_matrix = np.zeros([NUMBER_STATES, NUMBER_INPUTS])
            state = signal.StateSpace(a_matrix, b_matrix, c_matrix, d_matrix, dt=1)
            _, y_out_er, _ = signal.dlsim(state, u_sim, t, x_sim_er)
            if iter_counter == 0:
                y_sim_all = y_out_er
                y_actual_all = y_actual
            else:
                y_sim_all = np.vstack([y_sim_all, y_out_er])
                y_actual_all = np.vstack([y_actual_all, y_actual])
            iter_counter += 1
        if info["Nfeval"] % 100 == 0:
            print("Iteration: ", info["Nfeval"])
            print("Error: ", ((y_actual_all - y_sim_all) ** 2).sum())
        info["Nfeval"] += 1
        return ((y_actual_all - y_sim_all) ** 2).sum()

    res = optimize.minimize(
        objective_func,
        combined_mat,
        args=({"Nfeval": 0},),
        options={"maxiter": MINIZE_ITERS},
    )
    opt_matrix = res.x

    # This returns the matrix in the correct shape for use later on in the evaluation
    A_Matrix = opt_matrix[: (NUMBER_STATES**2)].reshape(NUMBER_STATES, NUMBER_STATES)
    B_Matrix = opt_matrix[(NUMBER_STATES**2) :].reshape(NUMBER_STATES, NUMBER_INPUTS)

    print("A-Matrix:")
    # print the A matrix in a form that can be copied easily
    print(repr(A_Matrix))
    print(" ")
    print("B-Matrix:")
    # print the B matrix in a form that can be copied easily
    print(repr(B_Matrix))
    print(" ")
    # Save the matrices in a csv file to easily access them in the future
    # Good practice would be to create a master excel file for permanent storage of certain
    # matrices so they do not get overwritten everytime
    np.savetxt(r".\Current\A_Matrices\A_Matrix.csv", A_Matrix, delimiter=',')
    np.savetxt(r".\Current\B_Matrices\B_Matrix.csv", B_Matrix, delimiter=',')

#THIS IS THE END OF THE ERROR REDUCTION FUNCTION

# ------------------------------------------------------------------------------------

# THIS SECTION TAKES THE NEW MATRICES FROM ERROR REDUCTION AND GENERATES THE SIMULATED
# VALUES THEN STORES THEM TO LATER MOVE INTO THE OPTIMIZATION FUNCTION AND CREATE GRAPHS

# Determines where the batch is pulled from if it is in the train or test set
if BATCH in TEST_BATCHES:
    x0 = np.array(
        df_test[df_test["Batch"] == BATCH].sort_values("Day").iloc[0, 0:NUMBER_STATES]
    )
    u0 = np.array(
        df_test[df_test["Batch"] == BATCH]
        .sort_values("Day")
        .iloc[:, NUMBER_STATES:TOTAL]
    )
else:
    x0 = np.array(
        df_train[df_train["Batch"] == BATCH].sort_values("Day").iloc[0, 0:NUMBER_STATES]
    )
    u0 = np.array(
        df_train[df_train["Batch"] == BATCH]
        .sort_values("Day")
        .iloc[:, NUMBER_STATES:TOTAL]
    )

# This defines the state space model from scipy
print("Initializing State Space Model...")
bioreactor = signal.StateSpace(A_Matrix, B_Matrix, C, D, dt=1)
t_out, y_out, x_out = signal.dlsim(bioreactor, u0, t, x0)

state_arrays = {}
states = STATES_INPUTS

# This adds the states data into the state arrays
j = 0
for i in states[:NUMBER_STATES]:
    state_matrix = y_out[:, j]
    state_arrays[i] = state_matrix
    j += 1

# This adds on the inputs data into the state arrays
j = 0
for x in states[NUMBER_STATES:]:
    state_arrays[x] = u0[:, j]
    j += 1

def calculate_volume(feed_arr, glucose_arr, volume_initial, sample_volume):
    """_summary_

    Args:
        feed_arr (_type_): _description_
        glucose_arr (_type_): _description_
        volume_initial (_type_): _description_
        sample_volume (_type_): _description_

    Returns:
        _type_: _description_
    """
    volume_arr = []
    feed_scaled = []
    gluc_scaled = []

    for count, _ in enumerate(feed_arr):
        if count == 0:
            volume = volume_initial
            feed = feed_arr[0] * volume_initial
            gluc = glucose_arr[0] * volume_initial
        else:
            volume = volume_arr[count-1] + feed_scaled[count-1] + gluc_scaled[count-1] - sample_volume
            feed = feed_arr[count] * volume
            gluc = glucose_arr[count] * volume

        volume_arr.append(volume)
        feed_scaled.append(feed)
        gluc_scaled.append(gluc)

    return np.array(volume_arr), np.array(feed_scaled), np.array(gluc_scaled)

#THIS IS THE END OF THE DATA WRANGLING POST ERROR REDUCTION

# ------------------------------------------------------------------------------------

# START OF THE OPTIMIZATION FUNCTION TO MINIMIZE/MAXIMIZE THE TARGET VALUE

# Optimization to determine what the best feed scheudle is for the model to maximize titer
# This generates an intiial condition that is the average of all the initial conditions
# for each batch
# This is to help generalize the starting condition so that it can spit out a optimized solution
# This code is specific to PVRIG feed optimization and can be changed when templating
initial_arr = np.zeros([NUMBER_STATES])
initial_arr_glc = np.zeros([DAYS, 1])
initial_volume =  []
j = 0
for i in df_train["Batch"].unique():
    initial_cons = np.array(df_train[df_train["Batch"] == i].iloc[0, 0:NUMBER_STATES])
    vol = df[df["Batch"] == i]["Volume"].iloc[0]
    initial_glc = np.array(
        df_train[df_train["Batch"] == i].iloc[:, (NUMBER_STATES + 1)]
    )
    if j == 0:
        intial_arr = initial_cons
        initial_arr_glc = initial_glc
    else:
        initial_arr = np.vstack([initial_arr, initial_cons])
        initial_arr_glc = np.vstack([initial_arr_glc, initial_glc])
    initial_volume.append(vol)
    j += 1

average_cons = initial_arr[1:, :].mean(axis=0)
average_glc = initial_arr_glc.mean(axis=0)
average_vol = np.mean(initial_volume)

# Start of the optimization function with all the constraints to avoid unwanted solutions
x_sim = average_cons
t_sim = np.arange(START, STOP, STEP)

if ALL_INPUT_OPTIMIZATION:
    if BATCH in TEST_BATCHES:
        initial_input = np.array(
            df_test[df_test["Batch"] == BATCH]
            .sort_values("Day")
            .iloc[:, NUMBER_STATES:TOTAL]
        ).reshape(-1, 1)
    else:
        initial_input = np.array(
            df_train[df_train["Batch"] == BATCH]
            .sort_values("Day")
            .iloc[:, NUMBER_STATES:TOTAL]
        ).reshape(-1, 1)
else:
    if BATCH in TEST_BATCHES:
        initial_input = np.array(
            df_test[df_test["Batch"] == BATCH]
            .sort_values("Day")
            .iloc[:, NUMBER_STATES : (NUMBER_STATES + 1)]
        ).reshape(-1, 1)
    else:
        initial_input = np.array(
            df_train[df_train["Batch"] == BATCH]
            .sort_values("Day")
            .iloc[:, NUMBER_STATES : (NUMBER_STATES + 1)]
        ).reshape(-1, 1)

print("Performing Feed Strategy Optimization...")

def optimization_output(input_matrix):
    """This function generates the simulation values using the State-space representation
    and intial conditions.

    Args:
        input_matrix (1d Matrix): Matrix of initial conditions to start the simulation

    Returns:
        NxM Matrix: Output matrix of all the simulated data
    """
    a_matrix = A_Matrix
    b_matrix = B_Matrix
    c_matrix = np.identity(NUMBER_STATES)
    d_matrix = np.zeros([NUMBER_STATES, NUMBER_INPUTS])

    if ALL_INPUT_OPTIMIZATION:
        u_sim = input_matrix.reshape(DAYS, NUMBER_INPUTS)
    else:
        u_sim = np.hstack([input_matrix.reshape(DAYS, 1), average_glc.reshape(DAYS, 1)])

    state = signal.StateSpace(a_matrix, b_matrix, c_matrix, d_matrix, dt=1)
    _, y_func, _ = signal.dlsim(state, u_sim, t_sim, x_sim)
    return y_func, u_sim

def inverse_scaler(y_output, days, number_inputs, u_sim, sim = True):
    """This function scales each data column back to its original scale. This is mainly
    used to constrain the simulation outputs to within reasonable bounds.

    Args:
        y_output (NxM Matrix): Matrix of simulated output values
        days (int): Number of days to simulate
        number_inputs (int): Number of inputs for simulation
        u_sim (NxM Matrix): Matrix of inputs for each day
        sim (bool, optional): Determines whether to output scaled data using the u_sim data or
        just a matrix of zeros. Defaults to True.

    Returns:
        _type_: _description_
    """
    if sim:
        scale_data = scaler.inverse_transform(np.hstack([y_output, np.zeros([days, number_inputs])]))
    else:
        scale_data = scaler.inverse_transform(np.hstack([y_output, u_sim]))
    return scale_data

def opt_objective_func(input_matrix, info):
    """_summary_

    Args:
        input_matrix (_type_): _description_
        info (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, _ = optimization_output(input_matrix)
    # display information
    if info["Nfeval"] % 100 == 0:
        print("Iteration: ", info["Nfeval"])
        print("Error: ", y_out_opt[DAYS - 1, 6])
    info["Nfeval"] += 1
    return 0 - y_out_opt[DAYS - 1, 6]

def minzero_constraint(input_matrix):  # Nothing can be less than 0
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, _ = optimization_output(input_matrix)
    return min(y_out_opt.reshape(-1, 1))

def tcc_constraint(input_matrix):  # VCC cannot be greater than TCC  
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, _ = optimization_output(input_matrix)
    tcc = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, _)[:,0]
    vcc = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, _)[:,1]
    return min(tcc - vcc)

def viability_constraint(input_matrix):  # Viability Constraint
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, _ = optimization_output(input_matrix)
    tcc = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, _)[:,0]
    vcc = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, _)[:,1]
    return 1 - max(vcc / tcc)

def ammonium_constraint(input_matrix):  # Ammonium constraint
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, _ = optimization_output(input_matrix)
    amm = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, _)[:,5]
    return MAX_AMMONIUM - amm

def lactate_constraint(input_matrix):  # Lactate Constraint
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, _ = optimization_output(input_matrix)
    lac = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, _)[:,2]
    return MAX_LACTATE - lac

def feedpercent_constraint(input_matrix):  # feed percent constraint
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, u_sim = optimization_output(input_matrix)
    scaled_matrix = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, u_sim, sim=False)
    _, feed, _ = calculate_volume(scaled_matrix[: (DAYS - 1), NUMBER_STATES], scaled_matrix[: (DAYS - 1), NUMBER_STATES+1],average_vol,VOLUME_ERROR)
    feed_percent = (np.sum(feed) / average_vol)*100
    return MAX_FEED_PERCENT - feed_percent

def maxcsfr_constraint(input_matrix):  # CSFR constraint
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, u_sim = optimization_output(input_matrix)
    scaled_matrix = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, u_sim, sim=False)
    volume, feed, _ = calculate_volume(scaled_matrix[: (DAYS - 1), NUMBER_STATES], scaled_matrix[: (DAYS - 1), NUMBER_STATES+1],average_vol,VOLUME_ERROR)
    csfr = (feed / (24 * 60)) / (
        scaled_matrix[: (DAYS - 1), 1] * (volume/1000)
    )
    return MAX_CSFR - max(csfr)

def mincsfr_constraint(input_matrix):  # CSFR constraint
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, u_sim = optimization_output(input_matrix)
    scaled_matrix = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, u_sim, sim=False)
    volume, feed, _ = calculate_volume(scaled_matrix[: (DAYS - 1), NUMBER_STATES], scaled_matrix[: (DAYS - 1), NUMBER_STATES+1],average_vol,VOLUME_ERROR)
    csfr = (feed / (24 * 60)) / (
        scaled_matrix[: (DAYS - 1), 1] * (volume/1000)
    )
    return min(csfr) - MIN_CSFR

def glucose_constraint(input_matrix):  # Glucose constraint
    """_summary_

    Args:
        input_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_out_opt, u_sim = optimization_output(input_matrix)
    scaled_matrix = inverse_scaler(y_out_opt, DAYS, NUMBER_INPUTS, u_sim, sim=False)
    return min(scaled_matrix[:, 4]) - GLUC_SETPOINT

constraints = [
    {"type": "ineq", "fun": minzero_constraint},
    # {"type": "ineq", "fun": tcc_constraint},
    {"type": "ineq", "fun": viability_constraint},
    {"type": "ineq", "fun": ammonium_constraint},
    {"type": "ineq", "fun": lactate_constraint},
    {"type": "ineq", "fun": feedpercent_constraint},
    {"type": "ineq", "fun": maxcsfr_constraint},
    {"type": "ineq", "fun": mincsfr_constraint},
    {"type": "ineq", "fun": glucose_constraint},
]

res = optimize.minimize(
    opt_objective_func,
    initial_input,
    args=({"Nfeval": 0},),
    constraints=constraints,
    bounds=bounds,
)

if ALL_INPUT_OPTIMIZATION is True:
    opt_input = res.x.reshape(DAYS, NUMBER_INPUTS)
else:
    opt_feed = res.x.reshape(DAYS, 1)
    opt_input = np.hstack([opt_feed, average_glc.reshape(DAYS, 1)])

print("Input Matrix:")
# Make the output Matrix more readable and easier to copy
print(repr(opt_input))

#THIS IS THE END OF THE OPTIMIZATION FUNCTION

# ------------------------------------------------------------------------------------

# THIS IS THE START OF THE POST OPTIMIZATION DATA WRANGLING
# THIS SECTION TAKES THE NEWLY OPTIMIZED B MATRIX AND EVALUATES IT TO GENERATE SIMULATION
# VALUES AND CREATE GRAPHS FOR VISUAL REPRESENTATION

# Evaluate the optimized solution in the state space model
# Store the returned values into a dictionary
bioreactor = signal.StateSpace(A_Matrix, B_Matrix, C, D, dt=1)
t_out, y_opt, x_out = signal.dlsim(bioreactor, opt_input, t, average_cons)

optimize_arrays = {}
states_postopt = STATES_INPUTS
j = 0
for i in states_postopt[:NUMBER_STATES]:
    state_matrix = y_opt[:, j]
    optimize_arrays[i] = state_matrix
    j += 1

# This adds on the inputs data into the state_postopt arrays
j = 0
for x in states_postopt[NUMBER_STATES:]:
    optimize_arrays[x] = opt_input[:, j]
    j += 1


# Re-scale the output values from the model and store them in a DataFrame
df_optimize = pd.DataFrame.from_dict(optimize_arrays, orient="columns")
rescaled_opt = scaler.inverse_transform(df_optimize)
df_rescaled_opt = pd.DataFrame(rescaled_opt, columns=features)

# These are specific to the PVRIG use case and don't need to be carried forward to template
df_rescaled_opt["Viability"] = (df_rescaled_opt["VCC"] / df_rescaled_opt["TCC"]) * 100
Volume, Daily_Feed, Daily_Glucose = calculate_volume(
    df_rescaled_opt["Daily Feed Normal (mL)"],
    df_rescaled_opt["Daily Glucose Normal (mL)"],
    average_vol,
    VOLUME_ERROR,
)
print(df_rescaled_opt)
print(Volume)
df_rescaled_opt["CSFR"] = (Daily_Feed / (24 * 60)) / (
    df_rescaled_opt["VCC"] * (Volume/1000)
)
# df_rescaled_opt["Daily Feed Normal (mL)"] = Daily_Feed
# df_rescaled_opt["Daily Glucose Normal (mL)"] = Daily_Glucose
df_rescaled_opt['Volume'] = Volume



# Calcualte the Model Metrics to understand how well it does (R2)

y_output_train = []
train_titer = []
train_r2 = []
for unique_batch in df_train["Batch"].unique():
    df_sim = df_train[df_train["Batch"] == unique_batch].sort_values("Day")
    x_sim_er = np.array(df_sim.iloc[0, 0:NUMBER_STATES])
    u_sim = np.array(df_sim.iloc[:, NUMBER_STATES:TOTAL])
    y_actual = np.array(df_sim.iloc[:, :NUMBER_STATES])
    a_matrix = A_Matrix
    b_matrix = B_Matrix
    c_matrix = np.identity(NUMBER_STATES)
    d_matrix = np.zeros([NUMBER_STATES, NUMBER_INPUTS])
    state = signal.StateSpace(a_matrix, b_matrix, c_matrix, d_matrix, dt=1)
    _, y_out_train, _ = signal.dlsim(state, u_sim, t, x_sim_er)
    y_output_train.extend(y_out_train[:, 6])
    train_titer.extend(df_sim['IGG'])


y_output_test = []
test_titer = []
test_r2 = []
for unique_batch in df_test["Batch"].unique():
    df_sim = df_test[df_test["Batch"] == unique_batch].sort_values("Day")
    x_sim_er = np.array(df_sim.iloc[0, 0:NUMBER_STATES])
    u_sim = np.array(df_sim.iloc[:, NUMBER_STATES:TOTAL])
    y_actual = np.array(df_sim.iloc[:, :NUMBER_STATES])
    a_matrix = A_Matrix
    b_matrix = B_Matrix
    c_matrix = np.identity(NUMBER_STATES)
    d_matrix = np.zeros([NUMBER_STATES, NUMBER_INPUTS])
    state = signal.StateSpace(a_matrix, b_matrix, c_matrix, d_matrix, dt=1)
    _, y_out_test, _ = signal.dlsim(state, u_sim, t, x_sim_er)
    y_output_test.extend(y_out_train[:, 6])
    test_titer.extend(df_sim['IGG'])

print(r2_score(y_output_train,train_titer))
print(r2_score(y_output_test,test_titer))




# Plot the optimzed CSFR path that the model predicts will generate the most titer
# Plotting Parameters
m = ["o", "s", "v", "p", "P", "H", "D"]
cmap = cm.get_cmap("tab20")
color_map = cmap(np.linspace(0, 1, len(interpolated_df["level_0"].unique())))

# Data for plotting the simulation and experimnetal values but not the optimized function values
if BATCH in TEST_BATCHES:
    df_plot = (
        df_test[df_test["Batch"] == BATCH].sort_values("Day").iloc[:, :NUMBER_STATES]
    )
else:
    df_plot = (
        df_train[df_train["Batch"] == BATCH].sort_values("Day").iloc[:, :NUMBER_STATES]
    )


# These plots are to compare the optimal strategy to the other batches strategies
fig_opt, axs = plt.subplots(ncols=3, nrows=3, figsize=(15, 6), sharex=True)
fig_opt.patch.set_facecolor("xkcd:white")

# gs = axs[1, 3].get_gridspec()
# axbig = fig_opt.add_subplot(gs[:, 0])
# for ax in axs[:, 0]:
#     ax.remove()



# This plots all the other states in a for loop with the optimal data and all the batch data
for ax, p in zip(axs[:, :].flatten(), STATES_INPUTS[:9]):
    ax.plot(
        df_rescaled_opt.index,
        df_rescaled_opt[p],
        linewidth=OPTWIDTH,
        marker="p",
        markerfacecolor="white",
        markersize=MARKERSIZE,
        markeredgewidth=OPTWIDTH,
        color="black",
    )
    for i in interpolated_df["level_0"].unique()[1::2]:
        plotter = interpolated_df[interpolated_df["level_0"] == i]
        ax.plot(
            plotter["Day"],
            plotter[p],
            linewidth=LINEWIDTH,
            marker="o",
            markerfacecolor="white",
            markersize=MARKERSIZE,
            markeredgewidth=MARKERWIDTH,
            label=i,
        )
    ax.set_xlabel("Day", labelpad=5)
    ax.set_title(f"Optimized {p}", pad=5)

print(df_rescaled_opt["IGG"])

# This plot is the test plot that compared the simulated results to the actual results to
# determine if the simulation can do a good job prediciting

# for i, j in enumerate(PLOT_LIST):
#     axbig.plot(
#         t_out,
#         state_arrays[j],
#         linewidth=LINEWIDTH,
#         marker=m[i],
#         markerfacecolor="white",
#         markersize=MARKERSIZE + 2,
#         markeredgewidth=MARKERWIDTH,
#         color=SIMULATION_COLOR,
#         label=f"Simulated {j}",
#     )
#     axbig.plot(
#         t_out,
#         df_plot[j],
#         linewidth=LINEWIDTH,
#         marker=m[i],
#         markerfacecolor="white",
#         markersize=MARKERSIZE + 2,
#         markeredgewidth=MARKERWIDTH,
#         color=EXPERIMENTAL_COLOR,
#         label=f"Experimental {j}",
#     )
# axbig.set_xlabel("Day", labelpad=5)
# axbig.set_ylabel("Scaled Values", labelpad=5)
# axbig.set_title("Simulated vs Experimental Data", pad=5)
# axbig.legend(
#     # bbox_to_anchor=(1.04, 1),
#     # loc="upper left",
#     borderaxespad=2,
#     fontsize="medium",
#     markerscale=0.5,
#     edgecolor="inherit",
#     fancybox=False,
#     borderpad=0.7,
# )

fig_test = plt.figure(figsize=(8, 5))
fig_test.patch.set_facecolor("xkcd:white")

plt.plot(
    df_rescaled_opt.index,
    df_rescaled_opt["CSFR"],
    linewidth=OPTWIDTH,
    marker="p",
    markerfacecolor="white",
    markersize=MARKERSIZE,
    markeredgewidth=OPTWIDTH,
    color="black",
)
for i in interpolated_df["level_0"].unique()[1::2]:
    plotter = interpolated_df[interpolated_df["level_0"] == i]
    plt.plot(
        plotter["Day"],
        plotter["CSFR"],
        linewidth=LINEWIDTH,
        marker="o",
        markerfacecolor="white",
        markersize=MARKERSIZE,
        markeredgewidth=MARKERWIDTH,
        label=i,
    )
plt.xlabel("Day", labelpad=5)
plt.ylabel("CSFR", labelpad=5)
plt.title("Optimized CSFR", pad=5)

fig_2 = plt.figure(figsize=(8, 5))

for i, j in enumerate(PLOT_LIST):
    plt.plot(
        t_out,
        state_arrays[j],
        linewidth=LINEWIDTH,
        marker=m[i],
        markerfacecolor="white",
        markersize=MARKERSIZE + 2,
        markeredgewidth=MARKERWIDTH,
        color=SIMULATION_COLOR,
        label=f"Simulated {j}",
    )
    plt.plot(
        t_out,
        df_plot[j],
        linewidth=LINEWIDTH,
        marker=m[i],
        markerfacecolor="white",
        markersize=MARKERSIZE + 2,
        markeredgewidth=MARKERWIDTH,
        color=EXPERIMENTAL_COLOR,
        label=f"{j}",
    )
plt.xlabel("Day", labelpad=5)
plt.ylabel("Scaled Values", labelpad=5)
plt.title("Simulated vs Experimental Data", pad=5)
plt.legend(
    bbox_to_anchor=(1.04, 1),
    # loc="upper left",
    borderaxespad=2,
    fontsize="medium",
    markerscale=0.5,
    edgecolor="inherit",
    fancybox=False,
    borderpad=0.7,
)

feed_percent = (np.sum(Daily_Feed) / average_vol)*100
print(feed_percent)

df_rescaled_opt["CSFR"].to_clipboard()

fig_test.tight_layout()
fig_opt.tight_layout()
fig_2.tight_layout()
plt.show()

# AR22-075 Model Parameters

## Description: 
### These are the models parameters used in the code written for model 1. To recreate the optimized CSFR path used in AR22-075 these values must be used.

## Polynomial Model Parameters:

```python
# Set the random seed to achieve the same randomness everytime the script is run
# This helps with have reproducible results for the same inputs each time
np.random.seed(10)

# Variables to initialize before running this script
# VOLUME_ERROR was determined through a minimization function on the error
# between the calculated volume and the actual volume for each batch. This represents the volume
# change due to sampling, other additions (base & acid) and evaporation (ambr scale only)
DAYS = 13
DEGREE = 3
NUMBER_STATES = 7
NUMBER_INPUTS = 2
VOLUME_ERROR = 1.431  # Should be in mL
TOTAL = NUMBER_STATES + NUMBER_INPUTS


# Constraint variables in the CSFR optimization function
MAX_CSFR = 0.0005
MIN_CSFR = 0.0001
MAX_LACTATE = 2
MAX_AMMONIUM = 40
GLUC_SETPOINT = 2
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
```

## Standard Model Parameters:

```python
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
```
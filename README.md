# State Space Dynamic Modeling

The purpose of this project is to try and utlize the state space equations to model the dynamics of cell culture.

The main reasons for starting this project are:
- Optimizing process development parameters (ex: feeding, pH, Temperature)
- Prediciting future cell dynamics based on previous data
- Generating process knowledge for future experimentation

## Changelog

- 2024-01-29
  - mpc_optimizer.py: change the names/values of both the --INPUT_DATA and --INPUT_REF columns of the feed
- 2024-01-24
  - estimation.py: adjusted the weights on estimated states 
  - mpc_optimizer.py: state() now has an optional argument to retrieve the state on a specific day; sim_from_day() simulates future state trajectory from a specific day or from the current day if the optional argument is not provided; new attribute filter_wt_on_data for Controller to put a weight on measurement and 1 - weight on model predicted states; added an extra step to filter states based on data before estimating the output modifiers; added a column-wise average before sum to account for missing data
- 2024-01-18
  - estimation.py: specify CURR_TIME and VESSEL in the beginning of the code; use the master sheet's file name as output folder's name; only use the correct vessel/bioreactor in the reference data; same model for bioreactor and controller; added verification checks to compare weights and pvs/mvs lengths; user can specify and override the mod values; follow the estimate, next_day, optimize, return_data order
  - ssm.py: renamed delta_p as output_mods 
  - mpc_optimizer.py: added a boolean to return either total or daily feed in return_data(); removed log_sample(); use data as estimated state on Day 0 in state(); separated x and y in sim outputs; corrected a few cases where STATE_DATA was used instead of STATE_EST or STATE_PRED; added est_horizon as a controller attribute; in estimate(), use existing estimated state without re-running simulation; fixed an error where cost was calculated incorrectly (arrays were added instead of appended)

## Table of Contents

- [State Space Dynamic Modeling](#state-space-dynamic-modeling)
  - [Table of Contents](#table-of-contents)
  - [Model Development History](#model-development-history)
    - [Model 1: AR22-075 Process Model](#model-1-ar22-075-process-model)
  - [Setting up a Python Virtual Environment](#setting-up-a-python-virtual-environment)
  - [Project Organization](#project-organization)


## Model Development History

### Model 1: AR22-075 Process Model
- Description: This was the first model developed. The goal was to optimize the feeding strategy for PVRIG process 2 development. The model was designed to optimize the feeding strategy by maximizing the titer. Multiple iterations that are not documented were developed prior to this first model.
- Deployment Date: 18Oct22
- Links: 
  - [Model Parameters](https://mygithub.gsk.com/gsk-tech/state-space-model/blob/main/models/Model%201/Model-Parameters.md)
  - [Code](https://mygithub.gsk.com/gsk-tech/state-space-model/tree/main/models/Model%201/Code)
  - [Figures](https://mygithub.gsk.com/gsk-tech/state-space-model/tree/main/models/Model%201/Figures)


## Setting up a Python Virtual Environment

- Install the "virtualenv" python package using terminal with the command below
```shell
pip install virtualenv
```
- Load IDE of choice (VS Code for myself), start a new terminal within your GitHub project directory and enter command below (may need to use python3 instead).
  - The ".venv" is just the name of the virtual environment, this can be changed based on preference. The ".venv" is a generally accepted naming convention.
```shell
python -m virtualenv .venv
```
- After package installation use pipreqs to generate requirements.txt.
  - To use pipreqs properly you need to cd into the "src" folder (from data-science cookiecutter) to run it properly.
  - This will generate a requirements.txt file in the "src" folder directory which should be moved to the main folder directory.
```shell
pip install pipreqs
cd Documents/GitHub/{repository_name}/src
pipreqs
```


## Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

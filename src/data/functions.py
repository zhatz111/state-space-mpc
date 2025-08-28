"""
Code for useful functions needed in other scripts
  Created by Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
  Created: 2024-04-19
  Modified: 2025-08-27
"""

# Standard library imports
import glob
import shutil
from typing import Union
from pathlib import Path
from datetime import datetime

# Third party library imports
import json
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from models.ssm import StateSpaceModel

# Create type hint for the scaler object being passed to SSM class
ScalerType = Union[MinMaxScaler, StandardScaler]


def daily_to_cumulative(
    model: StateSpaceModel, input_variables: list, u_matrix_daily: np.ndarray
):
    """
    The function `daily_to_cumulative` converts a U matrix's daily variable
    column to a cumulative variable for lsim.

    Args:
      model: The "model" parameter is a variable that represents a model object.
      It is not specified in
    the code snippet provided, so its exact definition and usage would depend on
    the context in which this function is being used.
      input_variable: The variable target to convert from daily to cumulative
      u_matrix_daily: The u_matrix_daily parameter is a numpy array representing
      the U matrix with daily
    feed values.

    Returns:
      the modified U matrix with the daily feed column converted to cumulative
      feed.
    """
    u_matrix_cumulative = np.copy(u_matrix_daily)
    for var in input_variables:
      cumulative_loc = np.where(np.isin(model.inputs, var))[0]
      u_matrix_cumulative[:, cumulative_loc] = np.append(
          0, np.cumsum(u_matrix_cumulative[:-1, cumulative_loc])
      ).reshape([-1, 1])
    return u_matrix_cumulative


def update_json(json_path: Union[str, Path], values_dict: dict):
    """
    The function `update_json` updates a JSON file with values from a
    dictionary.

    Args:
      json_path (Union[str, Path]): The `json_path` parameter in the
      `update_json` function is the path
    to the JSON file that you want to update. It can be either a string
    representing the file path or a `Path` object from the `pathlib` module.
      values_dict (dict): The `values_dict` parameter in the `update_json`
      function is a dictionary that
    contains key-value pairs where the key is a string representing the key in
    the JSON file that you want to update, and the value is the new value that
    you want to set for that key in the JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for key, value in values_dict.items():
        if key == "Iterations":
            data[key] += value
        else:
            data[key] = value

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    return data


def scaler_tojson(scaler: MinMaxScaler, save_path: Union[str, Path]):
    """
    The function `scaler_tojson` takes a scaler object and saves its attributes
    to a JSON file.

    Args:
      scaler: The `scaler` parameter is an instance of a scaler object. It could
      be any scaler object
    from a machine learning library, such as `StandardScaler` from scikit-learn.
    The scaler object is used to scale or normalize data.
      save_path: The `save_path` parameter is the file path where the JSON file
      will be saved. It should
    include the file name and extension. For example, if you want to save the
    JSON file as "scaler_attributes.json" in the current directory, you can set
    `save_path` as "sc
    """
    # Prepare a dictionary to hold the scaler's attributes
    scaler_attributes = {
        attr_name: getattr(scaler, attr_name).tolist()
        if hasattr(getattr(scaler, attr_name), "tolist")
        else getattr(scaler, attr_name)
        for attr_name in vars(scaler)
    }

    # Save the attributes to a JSON file
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(scaler_attributes, file, indent=4)


def json_toscaler(json_file: Union[str, Path], minmaxscaler=True):
    """
    The function `json_toscaler` takes a JSON file containing attributes of a
    scaler and reconstructs the scaler object using the loaded attributes.

    Args:
      json_file: The path to the JSON file that contains the attributes of the
      scaler object. minmaxscaler: The `minmaxscaler` parameter is a boolean
      flag that determines whether to use the
    `MinMaxScaler` class for scaling the data. If `minmaxscaler` is set to
    `True`, the function will use `MinMaxScaler` for scaling the data. If it is
    set to `. Defaults to True

    Returns:
      a reconstructed scaler object.
    """
    # Load the attributes from the JSON file
    with open(json_file, "r", encoding="utf-8") as file:
        loaded_attributes = json.load(file)

    # Initialize a new MinMaxScaler instance
    if minmaxscaler:
        reconstructed_scaler = MinMaxScaler()
    else:
        raise ValueError("This method currently only works for the MinMaxScaler")

    # Set the loaded attributes back to the scaler
    for attr_name, attr_value in loaded_attributes.items():
        setattr(
            reconstructed_scaler,
            attr_name,
            np.array(attr_value) if isinstance(attr_value, list) else attr_value,
        )

    return reconstructed_scaler


def dict_toscaler(dict_file: dict, scaler_class="MinMaxScaler"):
    """
    The function `json_toscaler` takes a JSON file containing attributes of a
    scaler and reconstructs the scaler object using the loaded attributes.

    Args:
      json_file: The path to the JSON file that contains the attributes of the
      scaler object. minmaxscaler: The `minmaxscaler` parameter is a boolean
      flag that determines whether to use the
    `MinMaxScaler` class for scaling the data. If `minmaxscaler` is set to
    `True`, the function will use `MinMaxScaler` for scaling the data. If it is
    set to `. Defaults to True

    Returns:
      a reconstructed scaler object.
    """
    # Initialize a new MinMaxScaler instance
    if scaler_class == "MinMaxScaler":
        reconstructed_scaler = MinMaxScaler()
    else:
        raise ValueError(
            "This method currently only works for the MinMaxScaler and StandardScaler"
        )

    # Set the loaded attributes back to the scaler
    for attr_name, attr_value in dict_file.items():
        setattr(
            reconstructed_scaler,
            attr_name,
            np.array(attr_value) if isinstance(attr_value, list) else attr_value,
        )

    return reconstructed_scaler


def scaler_todict(scaler: Union[MinMaxScaler, StandardScaler, RobustScaler]):
    """
    The function `scaler_tojson` takes a scaler object and saves its attributes
    to a JSON file.

    Args:
      scaler: The `scaler` parameter is an instance of a scaler object. It could
      be any scaler object
    from a machine learning library, such as `StandardScaler` from scikit-learn.
    The scaler object is used to scale or normalize data.
      save_path: The `save_path` parameter is the file path where the JSON file
      will be saved. It should
    include the file name and extension. For example, if you want to save the
    JSON file as "scaler_attributes.json" in the current directory, you can set
    `save_path` as "sc
    """
    # Prepare a dictionary to hold the scaler's attributes
    scaler_attributes = {
        attr_name: getattr(scaler, attr_name).tolist()
        if hasattr(getattr(scaler, attr_name), "tolist")
        else getattr(scaler, attr_name)
        for attr_name in vars(scaler)
    }

    return scaler_attributes


def json_to_dict(json_file_path: Union[str, Path]):
    """
    The function `json_to_dict` reads a JSON file and returns its contents as a
    dictionary.

    Args:
      json_file_path (Union[str, Path]): The `json_file_path` parameter is the
      file path to the JSON
    file that you want to read and convert into a Python dictionary. This
    function reads the JSON data from the specified file and returns it as a
    dictionary.

    Returns:
      The function `json_to_dict` returns a dictionary object containing the
      data loaded from the JSON
    file located at the specified file path.
    """
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_dict = json.load(file)

    file.close()

    return json_dict


def dict_to_json(json_file_path: Union[str, Path], data: dict):
    """
    The function `dict_to_json` writes a dictionary to a JSON file with
    specified file path and data.

    Args:
      json_file_path (Union[str, Path]): The `json_file_path` parameter is the
      file path where you want
    to save the JSON data. It can be either a string representing the file path
    or a `Path` object from the `pathlib` module.
      data (dict): The `data` parameter in the `dict_to_json` function is a
      dictionary that contains the
    data that you want to write to a JSON file. This dictionary will be
    converted to JSON format and written to the specified file path.
    """
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    file.close()


def read_config(path_directory: Union[str, Path], export=False):
    """
    The `read_config` function reads a YAML file containing experiment configuration data and
    returns the parsed configuration.

    Returns:
      The function `read_config` returns the experiment configuration data loaded from the
      first YAML file found in the specified directory.
    """
    yaml_files = glob.glob(str(Path(path_directory, "*.yaml")))
    yaml_data = open(yaml_files[0], "r", encoding="utf-8")
    yaml_config = yaml.safe_load(yaml_data)
    yaml_data.close()

    todays_date = datetime.today().strftime("%y%m%d")
    if export:
        src_path = Path(yaml_files[0])
        dest_file = f"{src_path.stem}-{todays_date}{src_path.suffix}"
        csv_path_top_dir = Path(path_directory, yaml_config["CSV Export Folder"])
        csv_path_top_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, Path(csv_path_top_dir, dest_file))

    return yaml_config

"""
Code for useful functions needed in other scripts
Created by Yu Luo (yu.8.luo@gsk.com) and Zach Hatzenbeller (zach.a.hatzenbeller@gsk.com)
Created: 2024-04-19
Modified: 2024-04-19
"""
# Standard library imports
from typing import Union
from pathlib import Path

# Third party library imports
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create type hint for the scaler object being passed to SSM class
ScalerType = Union[MinMaxScaler, StandardScaler]

def daily_to_cumulative_feed(model, u_matrix_daily):
    """
    The function `daily_to_cumulative_feed` converts a U matrix's daily feed column to cumulative feed
    for lsim.

    Args:
      model: The "model" parameter is a variable that represents a model object. It is not specified in
    the code snippet provided, so its exact definition and usage would depend on the context in which
    this function is being used.
      u_matrix_daily: The u_matrix_daily parameter is a numpy array representing the U matrix with daily
    feed values.

    Returns:
      the modified U matrix with the daily feed column converted to cumulative feed.
    """
    u_matrix_cumulative = np.copy(u_matrix_daily)
    cumulative_feed_loc = np.where(np.isin(model.inputs, "CUMULATIVE_NORMALIZED_FEED"))[
        0
    ]
    u_matrix_cumulative[:, cumulative_feed_loc] = np.cumsum(
        u_matrix_cumulative[:, cumulative_feed_loc]
    ).reshape([-1, 1])
    return u_matrix_cumulative

def update_json(json_path: Union[str, Path], values_dict: dict):
    """
    The function `update_json` updates a JSON file with values from a dictionary.
    
    Args:
      json_path (Union[str, Path]): The `json_path` parameter in the `update_json` function is the path
    to the JSON file that you want to update. It can be either a string representing the file path or a
    `Path` object from the `pathlib` module.
      values_dict (dict): The `values_dict` parameter in the `update_json` function is a dictionary that
    contains key-value pairs where the key is a string representing the key in the JSON file that you
    want to update, and the value is the new value that you want to set for that key in the JSON file.
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

def scaler_tojson(scaler: MinMaxScaler, save_path: Union[str, Path]):
    """
    The function `scaler_tojson` takes a scaler object and saves its attributes to a JSON file.
    
    Args:
      scaler: The `scaler` parameter is an instance of a scaler object. It could be any scaler object
    from a machine learning library, such as `StandardScaler` from scikit-learn. The scaler object is
    used to scale or normalize data.
      save_path: The `save_path` parameter is the file path where the JSON file will be saved. It should
    include the file name and extension. For example, if you want to save the JSON file as
    "scaler_attributes.json" in the current directory, you can set `save_path` as "sc
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
    The function `json_toscaler` takes a JSON file containing attributes of a scaler and reconstructs
    the scaler object using the loaded attributes.
    
    Args:
      json_file: The path to the JSON file that contains the attributes of the scaler object.
      minmaxscaler: The `minmaxscaler` parameter is a boolean flag that determines whether to use the
    `MinMaxScaler` class for scaling the data. If `minmaxscaler` is set to `True`, the function will use
    `MinMaxScaler` for scaling the data. If it is set to `. Defaults to True
    
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
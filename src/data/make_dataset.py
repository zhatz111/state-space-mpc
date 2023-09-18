"""_summary_

Returns:
    _type_: _description_
"""
# The code is importing various libraries that are used in the code:
import math
import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit


# The `ModelData` class provides methods for data cleaning, preprocessing, and visualization for a
# state space model.
class ModelData:
    """_summary_
    """
    random.seed(10)
    def __init__(
        self,
        raw_data: pd.DataFrame,
        group: str,
        scaler_train: object,
        discard: list,
        states: list,
        inputs: list,
    ) -> None:
        """
        The function initializes an object with various attributes including raw data, a group
        identifier, a scaler object, a list of columns to discard, a list of states, and a list of
        inputs.
        
        Args:
          raw_data (pd.DataFrame): A pandas DataFrame containing the raw data.
          group (str): The "group" parameter is a string that represents the group or category that the
        data belongs to. It is used to identify and differentiate different groups of data within the
        dataset.
          scaler_train (object): The `scaler_train` parameter is an object that is used to scale the
        data. It is likely an instance of a scaler class from a library like scikit-learn, which is used
        to normalize or standardize the data before training a model.
          discard (list): The `discard` parameter is a list that contains the names of columns that
        should be discarded from the `raw_data` DataFrame. These columns will not be used as inputs for
        the model.
          states (list): The "states" parameter is a list that contains the names of the columns in the
        raw_data DataFrame that represent the states or regions. These columns are used to group the
        data and perform calculations or transformations specific to each state or region.
          inputs (list): The `inputs` parameter is a list that contains the names of the input variables
        or features that will be used for training the model. These variables are typically the
        independent variables or predictors that will be used to predict the target variable.
        """
        self.df = raw_data
        self.group = group
        self.scaler_train = scaler_train
        self.discard = discard
        self.states = states
        self.inputs = inputs

    def interpolation(self) -> pd.DataFrame:
        """
        The `interpolation` function performs linear interpolation on a pandas DataFrame, excluding any rows
        that contain specified values in a given column.

        Returns:
          a pandas DataFrame called `df_interpolate`.
        """
        if (len(self.discard) > 0) or (self.discard is None):
            self.df = self.df[
                ~self.df[self.group].str.contains("|".join(self.discard), na=False)
            ]
        grouped = self.df.groupby(self.group, group_keys=False)
        df_interpolate = grouped.apply(
            lambda group: group.interpolate(method="linear", limit_direction="forward")
        )
        return df_interpolate

    def moving_average_smoother(
        self, smoothing_list: list, win_len=2
    ) -> pd.DataFrame:
        """
        The `moving_average_smoother` function takes a list of columns to smooth and a window length,
        and applies a moving average smoothing technique to the specified columns within each group in a
        DataFrame.
        
        Args:
          smoothing_list (list): The `smoothing_list` parameter is a list of column names that you want
        to apply the moving average smoothing to. These columns should be present in the
        `df_interpolated` DataFrame.
          win_len: The `win_len` parameter in the `moving_average_smoother` function represents the
        window length for calculating the rolling mean. It determines the number of consecutive values
        to consider when calculating the average. The default value is 2, which means it will calculate
        the average of each value with its adjacent. Defaults to 2
        
        Returns:
          a pandas DataFrame that contains the smoothed values of the specified columns in the input
        DataFrame.
        """
        df_interpolated = self.interpolation()
        grouped = df_interpolated.groupby("Batch")
        smoothed_df = pd.DataFrame()

        for _, group_data in grouped:
            group_smoothed = group_data.copy()

            for col in smoothing_list:
                # Calculate the rolling mean for the specified column within the group
                group_smoothed[col] = group_data[col].rolling(
                    window=win_len,
                    min_periods=1,
                    center=True
                    ).mean()

                # Keep the first and last values in the column unchanged
                group_smoothed[col].iloc[0] = group_data[col].iloc[0]
                group_smoothed[col].iloc[-1] = group_data[col].iloc[-1]

            smoothed_df = pd.concat([smoothed_df, group_smoothed])
        return smoothed_df

    def train_test_split(
        self,
        smoothing_list: list,
        test_size=0.20,
        n_splits=2,
        random_state=1,
        win_len=2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The function `train_test_split` takes in a list of smoothing values, test size, number of
        splits, random state, and window length, and returns a tuple of two pandas DataFrames.
        
        Args:
          smoothing_list (list): The `smoothing_list` parameter is a list that contains the values used
        for smoothing the data. It is used in the process of splitting the data into training and
        testing sets.
          test_size: The proportion of the dataset that should be allocated for testing. It should be a
        value between 0 and 1, where 0 represents no testing data and 1 represents all data used for
        testing.
          n_splits: The `n_splits` parameter specifies the number of times the dataset will be split
        into train and test sets. In this case, it is set to 2, which means the dataset will be split
        into two sets. Defaults to 2
          random_state: The random_state parameter is used to set the seed for the random number
        generator. This ensures that the same random splits are generated each time the function is
        called with the same random_state value. Defaults to 1
          win_len: The `win_len` parameter represents the length of the sliding window used for
        smoothing the data. It is used in conjunction with the `smoothing_list` parameter, which is a
        list of values used for smoothing the data. The `smoothing_list` is applied to the data using a
        sliding window. Defaults to 2
        """
        df_smoothed = self.moving_average_smoother(smoothing_list, win_len)
        splitter = GroupShuffleSplit(
            test_size=test_size, n_splits=n_splits, random_state=random_state
        )
        split = splitter.split(df_smoothed, groups=df_smoothed[self.group])
        train_index, test_index = next(split)
        return (
            df_smoothed.iloc[list(train_index), :],
            df_smoothed.iloc[list(test_index), :],
        )

    def feature_scaling(self, data: pd.DataFrame, scaler, new_scaler=True):
        """
        The function performs feature scaling on specified columns of a DataFrame using a given scaler
        object.

        Args:
          data (pd.DataFrame): The "data" parameter is a pandas DataFrame that contains the dataset you
        want to perform feature scaling on.
          scaler: The "scaler" parameter is an instance of a scaler class from the scikit-learn library.
        It is used to scale the features in the data. Examples of scaler classes include MinMaxScaler,
        StandardScaler, and RobustScaler.
          new_scaler: The parameter "new_scaler" is a boolean flag that indicates whether a new scaler
        object should be created or an existing one should be used. If "new_scaler" is set to True, a
        new scaler object will be created and fitted on the data. If it is set to False,. Defaults to
        True

        Returns:
          the modified data DataFrame after applying feature scaling.
        """
        # data_set = set(data.columns)
        # exclusion_set = set(scale_exclusion)
        # columns = list(data_set - exclusion_set)
        columns = self.states + self.inputs
        if new_scaler:
            scaler.fit(data.filter(items=columns))
            data[columns] = scaler.transform(data.filter(items=columns))
            return data
        else:
            data[columns] = scaler.transform(data.filter(items=columns))
            return data

    def save_scaler(self, file_name, scaler):
        """
        The function saves a scaler object to a file using joblib.

        Args:
          file_name: The name of the file where the scaler will be saved.
          scaler: The scaler parameter is an instance of a scaler object that is used to scale or
        normalize data. It could be any scaler object such as StandardScaler, MinMaxScaler, etc.

        Returns:
          the result of the joblib.dump() function, which is used to save the scaler object to a file.
        """
        scaler_name = file_name + ".scl"
        return joblib.dump(scaler, scaler_name)

    def clean(
        self,
        column_inclusion,
        smoothing_list,
        test_size=0.10,
        n_splits=2,
        random_state=1,
        win_len=2,
    ):
        """
        The `clean` function takes in several parameters, including `column_inclusion`,
        `smoothing_list`, `test_size`, `n_splits`, `random_state`, and `win_len`, and performs some
        cleaning operations on the data.
        
        Args:
          column_inclusion: A list of columns to include in the cleaning process. Only the columns
        specified in this list will be considered for cleaning.
          smoothing_list: The `smoothing_list` parameter is a list that contains values for smoothing.
        Smoothing is a technique used to reduce noise or fluctuations in data. It involves replacing
        each value in a dataset with an average of its neighboring values. The `smoothing_list`
        parameter allows you to specify the values to
          test_size: The proportion of the dataset that should be used for testing. It should be a value
        between 0 and 1, where 0 represents no testing data and 1 represents all data used for testing.
          n_splits: The number of times the dataset will be split into train and test sets for
        cross-validation. Defaults to 2
          random_state: The random_state parameter is used to set the seed for the random number
        generator. This ensures that the randomization process is reproducible. By setting a specific
        value for random_state, you can obtain the same random splits each time you run the code.
        Defaults to 1
          win_len: The `win_len` parameter represents the window length used for smoothing the data. It
        is used in the `smoothing_list` parameter, which contains a list of values representing the
        smoothing window length for each column in the dataset. Defaults to 2
        """
        train, test = self.train_test_split(
            smoothing_list=smoothing_list,
            test_size=test_size,
            n_splits=n_splits,
            random_state=random_state,
            win_len=win_len,
        )
        train = self.feature_scaling(
            data=train,
            scaler=self.scaler_train,
        )
        test = self.feature_scaling(
            data=test,
            scaler=self.scaler_train,
            new_scaler=False,
        )
        columns = column_inclusion + self.states + self.inputs

        return (
            train[train.columns[train.columns.isin(columns)]],
            test[test.columns[test.columns.isin(columns)]],
        )

    def graph_train_data(self, smoothing_list, test_label, ylim=None):
        """
        The function `graph_train_data` plots the training data grouped by batches in a grid layout.

        Args:
          smoothing_list: The `smoothing_list` parameter is a list that contains the smoothing
        parameters for each batch of train data. It is used as an input to the `moving_average_smoother`
        method.
          test_label: The `test_label` parameter is a string that represents the label or variable that
        you want to plot on the y-axis of the graph. It could be any numerical value that you want to
        visualize, such as "Loss", "Accuracy", "Error", etc.
          ylim: The `ylim` parameter is used to set the y-axis limits for the plots. It allows you to
        specify the minimum and maximum values for the y-axis. If `ylim` is not provided, the y-axis
        limits will be automatically determined based on the data.
        """
        train_data = self.moving_average_smoother(smoothing_list)
        smoothed_grouped = train_data.groupby("Batch")
        cols = 4
        if smoothed_grouped.ngroups > 15:
            rows = math.floor(15 / cols)
        else:
            rows = math.floor(smoothed_grouped.ngroups / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(10, 10), squeeze=False, sharex=True, sharey=True
        )
        smoothed_dict = dict(list(smoothed_grouped))
        dict_keys = list(smoothed_dict.keys())
        for count, ax_test in enumerate(axes.reshape(-1)):
            key = dict_keys[count]
            ax_test.plot(
                smoothed_dict[key]["Day"],
                smoothed_dict[key][test_label],
                "bo-",
                label="Simulated",
                markersize=3.5,
            )
            ax_test.set_title(key)
            if ylim is not None:
                ax_test.set_ylim(0, ylim)
        axes[rows - 1][cols - 1].legend()

        fig.supxlabel("Day")
        fig.supylabel(f"{test_label}")
        fig.tight_layout()
        plt.legend(loc="best")
        plt.show()

    def graph_smoothed_unsmoothed_data(self, smoothing_list, test_label, ylim=None):
        """
        The function `graph_train_data` plots the training data grouped by batches in a grid layout.

        Args:
          smoothing_list: The `smoothing_list` parameter is a list that contains the smoothing
        parameters for each batch of train data. It is used as an input to the `moving_average_smoother`
        method.
          test_label: The `test_label` parameter is a string that represents the label or variable that
        you want to plot on the y-axis of the graph. It could be any numerical value that you want to
        visualize, such as "Loss", "Accuracy", "Error", etc.
          ylim: The `ylim` parameter is used to set the y-axis limits for the plots. It allows you to
        specify the minimum and maximum values for the y-axis. If `ylim` is not provided, the y-axis
        limits will be automatically determined based on the data.
        """
        smoothed_data = self.moving_average_smoother(smoothing_list)
        smoothed_grouped = smoothed_data.groupby("Batch")
        unsmoothed_data = self.interpolation()
        unsmoothed_grouped = unsmoothed_data.groupby("Batch")
        cols = 4
        if smoothed_grouped.ngroups > 15:
            rows = math.floor(15 / cols)
        else:
            rows = math.floor(smoothed_grouped.ngroups / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(10, 10), squeeze=False, sharex=True, sharey=True
        )
        smoothed_dict = dict(list(smoothed_grouped))
        unsmoothed_dict = dict(list(unsmoothed_grouped))
        dict_keys = list(smoothed_dict.keys())
        for count, ax_test in enumerate(axes.reshape(-1)):
            key = dict_keys[count]
            ax_test.plot(
                smoothed_dict[key]["Day"],
                smoothed_dict[key][test_label],
                "bo-",
                label="Smoothed Data",
                markersize=3.5,
            )
            ax_test.plot(
                unsmoothed_dict[key]["Day"],
                unsmoothed_dict[key][test_label],
                "ro-",
                label="Unsmoothed Data",
                markersize=3.5,
            )
            ax_test.set_title(key)
            if ylim is not None:
                ax_test.set_ylim(0, ylim)
        axes[rows - 1][cols - 1].legend()

        fig.supxlabel("Day")
        fig.supylabel(f"{test_label}")
        fig.tight_layout()
        plt.legend(loc="best")
        plt.show()

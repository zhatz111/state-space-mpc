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
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupShuffleSplit


# The `ModelData` class provides methods for data cleaning, preprocessing, and visualization for a
# state space model.
class ModelData:
    """_summary_
    """
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

    def spline_smoothing(
        self, smoothing_list: list, win_len=5, poly_order=2
    ) -> pd.DataFrame:
        """
        The function `spline_smoothing` takes a list of column names, performs interpolation on a
        DataFrame, and then applies Savitzky-Golay smoothing to the specified columns.

        Args:
          smoothing_list (list): A list of column names in the DataFrame that you want to apply the
        smoothing to.
          win_len: The `win_len` parameter specifies the length of the window used for smoothing. It
        determines the number of neighboring data points used to calculate the smoothed value for each
        data point. A larger `win_len` value will result in a smoother curve, but it may also cause the
        smoothed curve to lag behind. Defaults to 5
          poly_order: The `poly_order` parameter specifies the order of the polynomial used in the
        Savitzky-Golay filter. It determines the degree of the polynomial that is used to fit the local
        data points. A higher `poly_order` value will result in a smoother curve, but it may also
        introduce more. Defaults to 2

        Returns:
          a pandas DataFrame object.
        """
        df_interpolated = self.interpolation()
        df_smoothed = df_interpolated.copy()
        if smoothing_list:
            df_smoothed.loc[:, smoothing_list] = df_interpolated.filter(
                items=smoothing_list
            ).apply(
                lambda x: savgol_filter(x, window_length=win_len, polyorder=poly_order)
            )
        return df_smoothed

    def train_test_split(
        self,
        smoothing_list: list,
        test_size=0.20,
        n_splits=2,
        random_state=1,
        win_len=5,
        poly_order=2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        The `train_test_split` function takes a list of data to be smoothed, performs spline smoothing
        on the data, and then splits the smoothed data into training and testing sets using group
        shuffle splitting.

        Args:
          smoothing_list (list): The `smoothing_list` parameter is a list of column names in the
        DataFrame that you want to apply spline smoothing to. These columns will be smoothed using the
        `spline_smoothing` method.
          test_size: The `test_size` parameter determines the proportion of the dataset that will be
        allocated for testing. It is set to 0.20, which means that 20% of the data will be used for
        testing, while the remaining 80% will be used for training.
          n_splits: The `n_splits` parameter specifies the number of times the data will be split into
        train and test sets. In this case, it is set to 2, which means the data will be split into train
        and test sets twice. Defaults to 2
          random_state: The random_state parameter is used to set the seed for the random number
        generator. By setting a specific value for random_state, you can ensure that the train-test
        split is reproducible. If you use the same random_state value in multiple runs of the code, you
        will get the same train-test split. Defaults to 1
          win_len: The `win_len` parameter represents the window length used for spline smoothing. It
        determines the number of data points used to calculate the smoothed value at each point. A
        larger `win_len` value will result in a smoother curve, but it may also introduce more lag in
        the smoothed data. Defaults to 5
          poly_order: The `poly_order` parameter represents the order of the polynomial used for spline
        smoothing. It determines the flexibility of the spline curve. A higher `poly_order` value will
        result in a more flexible curve that can better fit the data but may also be more prone to
        overfitting. Conversely, a. Defaults to 2

        Returns:
          The function `train_test_split` returns a tuple containing two pandas DataFrames. The first
        DataFrame is the training set, and the second DataFrame is the test set.
        """
        df_smoothed = self.spline_smoothing(smoothing_list, win_len, poly_order)
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
        scaler_name = file_name + ".scale"
        return joblib.dump(scaler, scaler_name)

    def clean(
        self,
        column_inclusion,
        smoothing_list,
        test_size=0.10,
        n_splits=2,
        random_state=1,
        win_len=5,
        poly_order=2,
    ):
        """
        The `clean` function takes in various parameters, performs data cleaning and preprocessing
        operations, and returns the cleaned train and test datasets.

        Args:
          column_inclusion: The `column_inclusion` parameter is a list of column names that you want to
        include in the cleaned data. These columns will be retained in the cleaned data, while other
        columns will be excluded.
          smoothing_list: The `smoothing_list` parameter is a list of column names that you want to
        apply smoothing to. It is used in the `train_test_split` method to perform smoothing on the
        specified columns before splitting the data into train and test sets.
          test_size: The proportion of the dataset that should be allocated for testing. It is set to
        0.10, which means 10% of the data will be used for testing.
          n_splits: The `n_splits` parameter is used to specify the number of splits for
        cross-validation. It determines how many times the data will be split into train and test sets
        during cross-validation. In this case, it is set to 2, meaning the data will be split into two
        sets for cross-validation. Defaults to 2
          random_state: The random_state parameter is used to set the seed for random number generation.
        By setting a specific value for random_state, you can ensure that the random processes in your
        code are reproducible. This means that if you run the code multiple times with the same
        random_state value, you will get the same. Defaults to 1
          win_len: The parameter "win_len" represents the window length used for smoothing the data. It
        is used in the train_test_split function to apply a moving average filter to the data before
        splitting it into train and test sets. Defaults to 5
          poly_order: The `poly_order` parameter is used to specify the order of the polynomial features
        to be generated during feature scaling. It is used in the `feature_scaling` method to create
        polynomial features from the input data. The higher the `poly_order`, the more complex the
        polynomial features will be. Defaults to 2

        Returns:
          two dataframes: `train[train.columns[train.columns.isin(columns)]]` and
        `test[test.columns[test.columns.isin(columns)]]`. These dataframes contain the columns specified
        in `column_inclusion` as well as the `states` and `inputs` attributes of the object.
        """
        train, test = self.train_test_split(
            smoothing_list=smoothing_list,
            test_size=test_size,
            n_splits=n_splits,
            random_state=random_state,
            win_len=win_len,
            poly_order=poly_order,
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
        parameters for each batch of train data. It is used as an input to the `spline_smoothing`
        method.
          test_label: The `test_label` parameter is a string that represents the label or variable that
        you want to plot on the y-axis of the graph. It could be any numerical value that you want to
        visualize, such as "Loss", "Accuracy", "Error", etc.
          ylim: The `ylim` parameter is used to set the y-axis limits for the plots. It allows you to
        specify the minimum and maximum values for the y-axis. If `ylim` is not provided, the y-axis
        limits will be automatically determined based on the data.
        """
        train_data = self.spline_smoothing(smoothing_list)
        train_grouped = train_data.groupby("Batch")
        cols = 4
        if train_grouped.ngroups > 15:
            rows = math.floor(15 / cols)
        else:
            rows = math.floor(train_grouped.ngroups / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(10, 10), squeeze=False, sharex=True, sharey=True
        )
        eval_dict = dict(list(train_grouped))
        dict_keys = [k for k in eval_dict.keys()]
        count = 0
        random.seed(10)
        for i in range(rows):
            for j in range(cols):
                key = dict_keys[random.randint(0, len(eval_dict) - 1)]
                axes[i][j].plot(
                    eval_dict[key]["Day"],
                    eval_dict[key][test_label],
                    "bo-",
                    label="Simulated",
                    markersize=3.5,
                )
                axes[i][j].set_title(key)
                if ylim is not None:
                    axes[i][j].set_ylim(0, ylim)
                count += 1
        axes[rows - 1][cols - 1].legend()

        fig.supxlabel("Day")
        fig.supylabel(f"{test_label}")
        plt.legend(loc="best")
        fig.tight_layout()
        plt.show()

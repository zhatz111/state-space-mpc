
# Import Data manipulation and Matrices Libraries
import joblib
import pandas as pd
from scipy.interpolate import splrep, splev
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

class ModelData:
    """_summary_
    """

    def __init__(self, df: pd.DataFrame, group: str, scaler_train, discard, states, inputs):
        self.df = df
        self.group = group
        self.scaler_train = scaler_train
        self.discard = discard
        self.states = states
        self.inputs = inputs

    def interpolation(self):
        if (len(self.discard) > 0) or (self.discard is None):
            self.df = self.df[~self.df[self.group].str.contains("|".join(self.discard))]
        grouped = self.df.groupby(self.group, group_keys=False)
        # grouped.apply(lambda group: group.interpolate(method = 'spline',order=3))
        return grouped.apply(lambda group: group.interpolate(method = 'linear', \
        limit_direction='backward',limit=2))

    def spline_smoothing(self, spline_list: list):
        df_spline = self.interpolation().groupby(self.group)
        smooth_data = []
        splines = set(spline_list)
        for _, group in df_spline:
            for col in splines:
                x = group["Day"].values
                y = group[col].values
                spl = splrep(x,y)
                smooth_y = splev(x, spl)
                group[col] = smooth_y

            smooth_data.append(group)
        smooth_data = pd.concat(smooth_data)
        return smooth_data

    def train_test_split(self, spline_list: list, test_size = 0.10, n_splits = 2, random_state = 1):
        df = self.spline_smoothing(spline_list)
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=random_state)
        split = splitter.split(df, groups=df[self.group])
        train_inds, test_inds = next(split)
        return df.iloc[train_inds], df.iloc[test_inds]

    def feature_scaling(self, data: pd.DataFrame, scaler, new_scaler=True):
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
        scaler_name = file_name + ".scale"
        return joblib.dump(scaler, scaler_name)

    def clean(self, column_inclusion, spline_list, test_size = 0.10, n_splits = 2, random_state = 1):
        train, test = self.train_test_split(
            spline_list=spline_list,
            test_size=test_size,
            n_splits=n_splits,
            random_state=random_state,
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

        return train[train.columns[train.columns.isin(columns)]], \
            test[test.columns[test.columns.isin(columns)]]

# need to split the data first then apply normalization to train set
# and use that scaling data to apply to the test set when predicting
# Add in an input for listing data columns that are contextual
# to the batch and not needed for the analysis


# Import Data manipulation and Matrices Libraries
import joblib
import pandas as pd
from scipy.interpolate import splrep, splev
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

class ModelData:
    """_summary_
    """

    def __init__(self, df: pd.DataFrame, group: str, scaler_train, discard):
        self.df = df
        self.group = group
        self.scaler_train = scaler_train
        self.discard = discard

    def interpolation(self):
        if (len(self.discard) > 0) or (self.discard is None):
            self.df = self.df[~self.df[self.group].str.contains("|".join(self.discard))]
        grouped = self.df.groupby(self.group, group_keys=False)
        grouped.apply(lambda group: group.interpolate(method = 'linear'))
        return grouped.apply(lambda group: group.interpolate(method = 'backfill', \
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

    def feature_scaling(self, data: pd.DataFrame, scale_exclusion: list, scaler, new_scaler=True):
        data_set = set(data.columns)
        exclusion_set = set(scale_exclusion)
        columns = list(data_set - exclusion_set)
        if new_scaler:
            scaler.fit(data[columns])
            data[columns] = scaler.transform(data[columns])
            return data
        else:
            data[columns] = scaler.transform(data[columns])
            return data

    def save_scaler(self, file_name, scaler):
        scaler_name = file_name + ".scale"
        return joblib.dump(scaler, scaler_name)

    def clean(self, scale_exclusion, post_scale_exclusion, spline_list, test_size = 0.10, n_splits = 2, random_state = 1):
        train, test = self.train_test_split(
            spline_list=spline_list,
            test_size=test_size,
            n_splits=n_splits,
            random_state=random_state,
        )
        train = self.feature_scaling(
            data=train,
            scale_exclusion=scale_exclusion,
            scaler=self.scaler_train
        )
        test = self.feature_scaling(
            data=test,
            scale_exclusion=scale_exclusion,
            scaler=self.scaler_train,
            new_scaler=False,
        )

        return train[train.columns[~train.columns.isin(post_scale_exclusion)]], \
            test[test.columns[~test.columns.isin(post_scale_exclusion)]]

# need to split the data first then apply normalization to train set
# and use that scaling data to apply to the test set when predicting
# Add in an input for listing data columns that are contextual
# to the batch and not needed for the analysis

# SPLINES = [
#     "TCC",
#     "VCC",
#     "Lact",
#     "Osmo",
#     "Ammonium",
#     "IGG",
#     "F30 Feed Amount (mL)",
# ]

# DISCARD = [
#     "AR21-048-001",
#     "AR21-048-003",
#     "AR21-048-009",
#     "AR22-001-001",
# ]

# feature_scaling_exclusion = [
#     "Batch",
#     "Day",
#     "Condition",
#     "Sample Time",
#     "Volume",
# ]

# post_scaling_exclusion = [
#     "F30 Feed Amount (mL)",
#     "Glucose Added (mL)",
#     "Daily Feed Fed (mL)",
#     "Daily Glucose Fed (mL)",
#     "CSFR",
#     "Volume",
#     ]

# scaler_train = MinMaxScaler()
# scaler_test = MinMaxScaler()
# data = pd.read_csv(r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\raw\AR22-001-Model-Data.csv")

# dataframe = ModelData(
#     df=data,
#     scaler_train=scaler_train,
#     scaler_test=scaler_test,
#     group="Batch",
#     discard=DISCARD,
# )

# # Class method to clean up all the data
# # this includes interpolation to start, spline smoothing, train and test set splitting
# # and finally feature scaling using the scaler of choice
# train_data, test_data = dataframe.clean(
#     feature_scaling_exclusion,
#     spline_list=SPLINES,
#     post_scale_exclusion=post_scaling_exclusion,
# )

# # Save the Scaler for both the training and test sets to rescale in the future
# dataframe.save_scaler("scaler_train", scaler=scaler_train)
# dataframe.save_scaler("scaler_test", scaler=scaler_test)

# print(train_data)
# print(test_data)

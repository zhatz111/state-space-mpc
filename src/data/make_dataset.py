
# Import Data manipulation and Matrices Libraries
import joblib
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

class ModelData:

    def __init__(self, file_path, scale_dependent_features: list, scaler):
        self.file_path = file_path
        self.scale_dependent_features = scale_dependent_features
        self.scaler = scaler

    def load_data(self):
        return pd.read_csv(self.file_path)

    def interpolation(self):
        df = self.load_data()
        grouped = df.groupby("Batch", group_keys=False)
        grouped.apply(lambda group: group.interpolate(method = 'linear'))
        return grouped.apply(lambda group: group.interpolate(method = 'backfill', \
        limit_direction='backward',limit=2))

    # def spline_smoothing(self, fit_spline_on: str, exclude_features: list):
    #     df_spline = (self.interpolation()).copy()
    #     splines = set(df_spline.columns) - set(exclude_features)
    #     for spline in splines:
    #         spl = UnivariateSpline(df_spline["Day"], df_spline[w])
    #         spl_dict[w] = spl
    #     for w in SPLINES:
    #         df_batch[w] = spl_dict[w](x)

    def train_test_split(self, group: str, test_size = 0.20, n_splits = 2, random_state = 1):
        data = self.interpolation()
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=random_state)
        split = splitter.split(data, groups=data[ group])
        train_inds, test_inds = next(split)
        return data.iloc[train_inds], data.iloc[test_inds]

    def feature_scaling(self, data: pd.DataFrame, scale_exclusion: list, new_scaler = True):
        data = data[set(data.columns) - set(scale_exclusion)]
        if new_scaler:
            self.scaler.fit(data)
            scaled = self.scaler.transform(data)
            features = self.scaler.get_feature_names_out()
            return pd.DataFrame(scaled, columns=features)
        else:
            scaled = self.scaler.transform(data)
            features = self.scaler.get_feature_names_out()
            return pd.DataFrame(scaled, columns=features)

    def get_scaler_data(self, file_name):
        scaler_name = file_name + ".scale"
        return joblib.dump(self.scaler, scaler_name)


# need to split the data first then apply normalization to train set
# and use that scaling data to apply to the test set when predicting
# Add in an input for listing data columns that are contextual
# to the batch and not needed for the analysis



scale_dependent_features = ["F30 Feed Amount (mL)", "Glucose Added (mL)"]
scale_exclusion_list = ["Batch","Day","Condition","Sample Time","Volume"]
scaler = MinMaxScaler()
dataframe = ModelData(
    r"C:\Users\zah48132\OneDrive - GSK\Documents\GitHub\state-space-model\data\raw\AR22-001-Model-Data.csv",
    scale_dependent_features,
    scaler
    )

train, test = dataframe.train_test_split("Batch")
print(train)
print(test)
# interpolated = dataframe.interpolation()
# print(dataframe.feature_scaling(interpolated, scale_exclusion_list))
# dataframe.get_scaler_data("scaled_data")

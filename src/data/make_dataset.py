
# Import Data manipulation and Matrices Libraries
import math
import joblib
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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
            self.df = self.df[~self.df[self.group].str.contains("|".join(self.discard),na=False)]
        grouped = self.df.groupby(self.group, group_keys=False)
        # grouped.apply(lambda group: group.interpolate(method = 'spline',order=3))
        return grouped.apply(lambda group: group.interpolate(method = 'linear', \
        limit_direction='backward',limit=2))

    def spline_smoothing(self, smoothing_list: list, win_len = 5, poly_order = 2):
        df_savgol = self.interpolation().groupby(self.group)
        smooth_data = []
        smooth = set(smoothing_list)
        for _, group in df_savgol:
            for col in smooth:
                # x = group["Day"].values
                y = group[col].values
                sg_filter = savgol_filter(y, window_length=win_len, polyorder=poly_order)
                group[col] = sg_filter

            smooth_data.append(group)
        smooth_data = pd.concat(smooth_data)
        return smooth_data

    def train_test_split(self, smoothing_list: list, test_size = 0.20, n_splits = 2, random_state = 1, win_len = 5, poly_order = 2):
        df = self.spline_smoothing(smoothing_list, win_len, poly_order)
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

    def clean(self, column_inclusion, smoothing_list, test_size = 0.10, n_splits = 2, random_state = 1, win_len = 5, poly_order = 2):
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

        return train[train.columns[train.columns.isin(columns)]], \
            test[test.columns[test.columns.isin(columns)]]
    
    def graph_train_data(self, smoothing_list, test_label, ylim):
        train_data = self.spline_smoothing(smoothing_list)
        train_grouped = train_data.groupby("Batch")
        cols = 4
        if train_grouped.ngroups > 15:
            rows = math.floor(15 / cols)
        else:
            rows = math.floor(train_grouped.ngroups / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10,10),squeeze=False, sharex=True, sharey=True)
        eval_dict = dict(list(train_grouped))
        dict_keys = [k for k in eval_dict.keys()]
        count = 0
        random.seed(10)
        for i in range(rows):
            for j in range(cols):
                key = dict_keys[random.randint(0,len(eval_dict)-1)]
                axes[i][j].plot(eval_dict[key]["Day"], eval_dict[key][test_label],"bo-", label="Simulated",markersize=3.5)
                axes[i][j].set_title(key)
                if ylim != None:
                    axes[i][j].set_ylim(0, ylim)
                count += 1
        axes[rows-1][cols-1].legend()

        fig.supxlabel('Day')
        fig.supylabel(f'{test_label}')
        plt.legend(loc="best")
        fig.tight_layout()

        SMALL_SIZE = 3
        MEDIUM_SIZE = 5
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.show()

# need to split the data first then apply normalization to train set
# and use that scaling data to apply to the test set when predicting
# Add in an input for listing data columns that are contextual
# to the batch and not needed for the analysis

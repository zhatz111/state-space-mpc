
import math
import random
import numpy as np
import pandas as pd
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

class ModelTraining:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, a_matrix, b_matrix, states: list, inputs: list, num_days: int, scaler_dict: dict):

        self.train_data = train_data
        self.test_data = test_data
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.states = states
        self.inputs = inputs
        self.num_days = num_days
        self.scaler_dict = scaler_dict
        self.state_len = len(states)
        self.input_len = len(inputs)
        self.time = np.arange(0, self.num_days, 1)
        self.total = self.state_len + self.input_len
        self.iters = 0

    def first_pass_training(self):

        x = np.array(self.train_data[self.train_data["Day"] < (self.num_days - 1)].filter(items=self.states+self.inputs))
        y = np.array(self.train_data[self.train_data["Day"] > 0].filter(items=self.states))
        regression_matrix = np.zeros([self.state_len, self.total])
        for j in range(self.state_len):
            reg = LinearRegression().fit(x, y[:, j])
            regression_matrix[j, :] = reg.coef_
        return regression_matrix

    def train_model(self, save_path, first_train=True, iterations=10):

        if first_train:
            initial_matrix = self.first_pass_training()
            self.a_matrix = initial_matrix[:self.state_len, :self.state_len]
            self.b_matrix = initial_matrix[:self.state_len, self.state_len:]
            C_Matrix = np.identity(self.state_len)
            D_Matrix = np.zeros([self.state_len, self.input_len])
        else:
            self.a_matrix = self.a_matrix
            self.b_matrix = self.b_matrix
            C_Matrix = np.identity(self.state_len)
            D_Matrix = np.zeros([self.state_len, self.input_len])

        a_sim = self.a_matrix.reshape(-1, 1)
        b_sim = self.b_matrix.reshape(-1, 1)
        combined_mat = np.vstack([a_sim, b_sim])

        def objective_func(mat, info):
            """Objective function to minimize the error of that the A and B matrices
            have when being used in the State Space model.

            Args:
                mat (1D Matrix): This is the u matrix reshaped into a 1D column vector
                info (int): This takes in the minize functions iteration number for keeping
                track of how long that function runs for

            Returns:
                error: This function returns the sum of squared errors
            """
            iter_counter = 0
            train_grouped = self.train_data.groupby("Batch")
            y_sim_all = np.zeros([self.num_days, self.state_len])
            y_actual_all = np.zeros([self.num_days, self.state_len])

            for _, group in train_grouped:

                of_x0 = np.array(group.filter(self.states).iloc[0])
                of_u = np.array(group.filter(self.inputs))
                of_y = np.array(group.filter(self.states))
                a_matrix = mat[:(self.state_len**2)].reshape(self.state_len, self.state_len)
                # # added this in to weight the titer
                # a_matrix[:,self.state_len-1] = a_matrix[:,self.state_len-1]

                b_matrix = mat[(self.state_len**2):].reshape(self.state_len, self.input_len)
                state = signal.StateSpace(a_matrix, b_matrix, C_Matrix, D_Matrix)
                _, of_yout, _ = signal.lsim(state, of_u, self.time, of_x0)

                if iter_counter == 0:
                    y_sim_all = of_yout
                    y_actual_all = of_y
                else:
                    y_sim_all = np.vstack([y_sim_all, of_yout])
                    y_actual_all = np.vstack([y_actual_all, of_y])
                iter_counter += 1
            # if ((y_actual_all - y_sim_all) ** 2).sum() > np.finfo("d").max:
            #     value = np.finfo("d").max
            # else:
            
            value = ((y_actual_all - y_sim_all) ** 2).sum()
            if info["Nfeval"] % 50 == 0:
                print("")
                print("Iteration: ", info["Nfeval"])
                print("Error: ", value)
            info["Nfeval"] += 1
            return value

        
        # minimizer_kwargs = {"method": "SLSQP", "args": ({"Nfeval": 0}, )}
        # res = optimize.basinhopping(
        #     func=objective_func,
        #     x0=combined_mat,
        #     niter=1,
        #     minimizer_kwargs=minimizer_kwargs,
        #     seed=2,
        #     # options={"maxiter": iterations, 'disp': False},
        # )

        res = optimize.minimize(
            fun=objective_func,
            x0=combined_mat,
            method="SLSQP",
            args=({"Nfeval": 0}, ),
            options={"maxiter": iterations, 'disp': False},
        )

        # Assuming you want to search in the range [-5.0, 5.0] for each element in the matrices
        # lw = [-10] * len(combined_mat)
        # up = [10] * len(combined_mat)
        # print(combined_mat)

        # res = optimize.dual_annealing(
        #     func=objective_func,
        #     bounds=list(zip(lw, up)),
        #     args=({"Nfeval": 0},),
        #     # x0=combined_mat,
        #     visit=2,
        #     accept=1,
        #     maxiter=100,
        #     no_local_search=True,
        # )

        # res = optimize.differential_evolution(
        #     func=objective_func,
        #     x0=combined_mat,
        #     bounds=list(zip(lw, up)),
        #     args=({"Nfeval": 0},),
        # )

        opt_matrix = res.x

        # This returns the matrix in the correct shape for use later on in the evaluation
        self.a_matrix = opt_matrix[: (self.state_len**2)].reshape(self.state_len, self.state_len)
        self.b_matrix = opt_matrix[(self.state_len**2) :].reshape(self.state_len, self.input_len)

        pd.DataFrame(self.a_matrix).to_csv(fr"{save_path}\A_Matrix.csv",index=False,header=False)
        pd.DataFrame(self.b_matrix).to_csv(fr"{save_path}\B_Matrix.csv",index=False,header=False)
        # np.savetxt(, self.a_matrix, delimiter=',')
        # np.savetxt(fr"{save_path}\B_Matrix.csv", self.b_matrix, delimiter=',')

    def test_model(self, test_label: str):

        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        # y_out_test = np.zeros([self.num_days, self.state_len])
        eval_dict = {}
        test_model_dict = {}
        test_grouped = self.test_data.groupby("Batch")
        for name, group in test_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix)
            _, test_yout, _ = signal.lsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((test_yout,test_u))
            eval_dict[name] = pd.DataFrame(data=raw_data, columns=columns)
            test_model_dict[name] = group

        scaler_value = self.scaler_dict[test_label][0]
        min_value = self.scaler_dict[test_label][1]
        cols = 2
        rows = math.floor(len(eval_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10,10),squeeze=False)
        dict_keys = [k for k in eval_dict.keys()]
        count = 0
        for i in range(rows):
            for j in range(cols):
                key = dict_keys[count]
                axes[i][j].plot(self.time, (eval_dict[key][test_label]-min_value)/scaler_value,"ro-", label="Simulated",markersize=3.5)
                axes[i][j].plot(self.time, (test_model_dict[key][test_label]-min_value)/scaler_value,"bo-", label="Actual",markersize=3.5)
                axes[i][j].set_title(key)
                count += 1
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

    def evaluate(self, test_label: str, ylim=None):
        df_eval = pd.concat([self.train_data, self.test_data], ignore_index=True)
        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs
        
        eval_dict = {}
        train_test_dict = {}
        eval_grouped = df_eval.groupby("Batch")
        for name, group in eval_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix)
            _, eval_yout, _ = signal.lsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((eval_yout,test_u))
            eval_dict[name] = pd.DataFrame(data=raw_data, columns=columns)
            train_test_dict[name] = group
        scaler_value = self.scaler_dict[test_label][0]
        min_value = self.scaler_dict[test_label][1]
        cols = 4
        if len(eval_dict) > 15:
            rows = math.floor(15 / cols)
        else:
            rows = math.floor(len(eval_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10,10),squeeze=False, sharex=True, sharey=True)
        fig2, axes2 = plt.subplots(rows, cols, figsize=(10,10),squeeze=False, sharex=True, sharey=True)
        dict_keys = [k for k in eval_dict.keys()]
        count = 0
        random.seed(10)
        for i in range(rows):
            for j in range(cols):
                key = dict_keys[random.randint(0,len(eval_dict)-1)]
                axes[i][j].plot(self.time, (eval_dict[key][test_label]-min_value)/scaler_value,"ro-", label="Simulated",markersize=3.5)
                axes[i][j].plot(self.time, (train_test_dict[key][test_label]-min_value)/scaler_value,"bo-", label="Actual",markersize=3.5)
                axes[i][j].set_title(key)
                if ylim != None:
                    axes[i][j].set_ylim(0, ylim)
                count += 1
        axes[rows-1][cols-1].legend()

        fig.supxlabel('Day')
        fig.supylabel(f'{test_label}')
        
        
        plt.legend(loc="best")
        fig.tight_layout()

        random.seed(10)
        for i in range(rows):
            for j in range(cols):
                key = dict_keys[random.randint(0,len(eval_dict)-1)]
                axes2[i][j].plot((train_test_dict[key][test_label]-min_value)/scaler_value, (eval_dict[key][test_label]-min_value)/scaler_value,"bo",markersize=4.5)
                axes2[i][j].set_title(key)
                pt = (0, 0)
                axes2[i][j].axline(pt, slope=1, color='black')
                axes2[i][j].set_ylim(0, ylim)
                axes2[i][j].set_xlim(0, ylim)
                count += 1
        fig2.supxlabel('Measurement')
        fig2.supylabel('Prediction')

        plt.legend(loc="best")
        fig2.tight_layout()

        SMALL_SIZE = 3
        MEDIUM_SIZE = 5
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.show()

    def get_rmse_table(self):
        df_eval = pd.concat([self.train_data, self.test_data], ignore_index=True)
        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        eval_dict = {}
        train_test_dict = {}
        eval_grouped = df_eval.groupby("Batch")
        for name, group in eval_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix)
            _, eval_yout, _ = signal.lsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((eval_yout,test_u))
            eval_dict[name] = pd.DataFrame(data=raw_data, columns=columns)
            train_test_dict[name] = group

        rmse_dict = {}
        for batch, _ in train_test_dict.items():
            state_rmse = []
            for state in self.states:
                scaler_value = self.scaler_dict[state][0]
                min_value = self.scaler_dict[state][1]
                true = np.array((train_test_dict[batch][state]-min_value)/scaler_value)
                pred = np.array((eval_dict[batch][state]-min_value)/scaler_value)
                rmse = np.sqrt((true - pred)**2).sum()/len(train_test_dict[batch][state])
                state_rmse.append(rmse)
            rmse_dict[batch] = state_rmse

        avg_rmse = []
        for _, value in rmse_dict.items():
            new_rmse = (value**2)*self.num_days-1
            avg_rmse.append(new_rmse)
        
        df_rmse = pd.DataFrame.from_dict(rmse_dict, orient="index").reset_index()
        df_rmse.columns = ["Batch"] + self.states
        return df_rmse
    
    def get_r2_table(self):
        df_eval = pd.concat([self.train_data, self.test_data], ignore_index=True)
        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        eval_dict = {}
        train_test_dict = {}
        eval_grouped = df_eval.groupby("Batch")
        for name, group in eval_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix)
            _, eval_yout, _ = signal.lsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((eval_yout,test_u))
            eval_dict[name] = pd.DataFrame(data=raw_data, columns=columns)
            train_test_dict[name] = group

        r2_dict = {}
        for batch, _ in train_test_dict.items():
            state_r2 = []
            for state in self.states:
                scaler_value = self.scaler_dict[state][0]
                min_value = self.scaler_dict[state][1]
                y_pred = np.array((train_test_dict[batch][state]-min_value)/scaler_value)
                y_true = np.array((eval_dict[batch][state]-min_value)/scaler_value)
                r2 = r2_score(y_true,y_pred)
                state_r2.append(r2)

            r2_dict[batch] = state_r2
        df_r2 = pd.DataFrame.from_dict(r2_dict, orient="index").reset_index()
        df_r2.columns = ["Batch"] + self.states
        return df_r2
    
    def get_corrcoef_table(self):
        df_eval = pd.concat([self.train_data, self.test_data], ignore_index=True)
        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        eval_dict = {}
        train_test_dict = {}
        eval_grouped = df_eval.groupby("Batch")
        for name, group in eval_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix)
            _, eval_yout, _ = signal.lsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((eval_yout,test_u))
            eval_dict[name] = pd.DataFrame(data=raw_data, columns=columns)
            train_test_dict[name] = group

        corr_dict = {}
        for batch, _ in train_test_dict.items():
            state_corr = []
            for state in self.states:
                scaler_value = self.scaler_dict[state][0]
                min_value = self.scaler_dict[state][1]
                y_pred = np.array((train_test_dict[batch][state]-min_value)/scaler_value)
                y_true = np.array((eval_dict[batch][state]-min_value)/scaler_value)
                
                corr = (np.corrcoef(np.vstack([y_pred,y_true])))**2
                state_corr.append(corr[0,1])

            corr_dict[batch] = state_corr
        df_corr = pd.DataFrame.from_dict(corr_dict, orient="index").reset_index()
        df_corr.columns = ["Batch"] + self.states
        return df_corr

    def train_test_model(self, save_path, test_label: str, first_train=True, iterations=10):

        self.train_model(
            save_path=save_path,
            first_train=first_train,
            iterations=iterations,
        )
        self.test_model(
            test_label=test_label,
        )
    
    def single_batch_test(self, test_label):

        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        # y_out_test = np.zeros([self.num_days, self.state_len])
        eval_dict = {}
        test_model_dict = {}
        test_grouped = self.test_data.groupby("Batch")
        for name, group in test_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix)
            _, test_yout, _ = signal.lsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((test_yout,test_u))
            eval_dict[name] = pd.DataFrame(data=raw_data, columns=columns)
            test_model_dict[name] = group

        scaler_value = self.scaler_dict[test_label][0]
        min_value = self.scaler_dict[test_label][1]
        cols = 1
        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(10,10),squeeze=False)
        dict_keys = [k for k in eval_dict.keys()]
        key = dict_keys[0]
        axes[0][0].plot(self.time, (eval_dict[key][test_label]-min_value)/scaler_value,"ro-", label="Simulated",markersize=3.5)
        axes[0][0].plot(self.time, (test_model_dict[key][test_label]-min_value)/scaler_value,"bo-", label="Actual",markersize=3.5)
        axes[0][0].set_title(key)
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

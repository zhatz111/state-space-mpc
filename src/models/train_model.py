
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

class ModelTraining:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, a_matrix, b_matrix, states: list, inputs: list, num_days: int):

        self.train_data = train_data
        self.test_data = test_data
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.states = states
        self.inputs = inputs
        self.num_days = num_days
        self.state_len = len(states)
        self.input_len = len(inputs)
        self.time = np.arange(0, self.num_days, 1)
        self.total = self.state_len + self.input_len

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
            print(self.a_matrix)
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
            if info["Nfeval"] % 100 == 0:
                print("")
                print("Iteration: ", info["Nfeval"])
                print("Error: ", ((y_actual_all - y_sim_all) ** 2).sum())
            info["Nfeval"] += 1
            return ((y_actual_all - y_sim_all) ** 2).sum()

        res = optimize.minimize(
            objective_func,
            combined_mat,
            args=({"Nfeval": 0},),
            options={"maxiter": iterations},
        )
        opt_matrix = res.x

        # This returns the matrix in the correct shape for use later on in the evaluation
        self.a_matrix = opt_matrix[: (self.state_len**2)].reshape(self.state_len, self.state_len)
        self.b_matrix = opt_matrix[(self.state_len**2) :].reshape(self.state_len, self.input_len)
        np.savetxt(fr"{save_path}\A_Matrix.csv", self.a_matrix, delimiter=',')
        np.savetxt(fr"{save_path}\B_Matrix.csv", self.b_matrix, delimiter=',')

    def test_model(self, test_label: str):

        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        # y_out_test = np.zeros([self.num_days, self.state_len])
        test_model_dict = {}
        test_grouped = self.test_data.groupby("Batch")
        for name, group in test_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix)
            _, test_yout, _ = signal.lsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((test_yout,test_u))
            test_model_dict[name] = pd.DataFrame(data=raw_data, columns=columns)

            plt.figure(figsize=(10,5))
            plt.plot(self.time, test_model_dict[name][test_label],"ro-", label="Simulated")
            plt.plot(self.time, group[test_label],"bo-", label="Actual")
            plt.legend()
            plt.title(name)
            plt.show()

    def evaluate(self, test_label: str):
        df_eval = pd.concat([self.train_data, self.test_data], ignore_index=True)
        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        eval_dict = {}
        eval_grouped = df_eval.groupby("Batch")
        for name, group in eval_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(self.a_matrix, self.b_matrix, C_Matrix, D_Matrix,dt=1)
            _, eval_yout, _ = signal.dlsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((eval_yout,test_u))
            eval_dict[name] = pd.DataFrame(data=raw_data, columns=columns)

            plt.figure(figsize=(10,5))
            plt.plot(self.time, eval_dict[name][test_label],"ro-", label="Simulated")
            plt.plot(self.time, group[test_label],"bo-", label="Actual")
            plt.legend()
            plt.title(name)
            plt.show()

    def get_matrices(self):
        pass

    def train_test_model(self, save_path, test_label: str, first_train=True, iterations=10):

        self.train_model(
            save_path=save_path,
            first_train=first_train,
            iterations=iterations,
        )
        self.test_model(
            test_label=test_label,
        )

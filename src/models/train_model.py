
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

class ModelTraining:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, states: list, inputs: list, num_days: int):

        self.train_data = train_data
        self.test_data = test_data
        self.states = states
        self.inputs = inputs
        self.num_days = num_days
        self.state_len = len(states)
        self.input_len = len(inputs)
        self.time = np.arange(0, self.num_days, 1)
        self.total = self.state_len + self.input_len

    def first_pass_training(self):

        x = np.array(self.train_data[self.train_data["Day"] < 12].filter(items=self.states))
        y = np.array(self.train_data[self.train_data["Day"] > 0].filter(items=self.states))
        regression_matrix = np.zeros([self.state_len, self.total])
        for j in range(self.state_len):
            reg = LinearRegression().fit(x, y[:, j])
            regression_matrix[j, :] = reg.coef_
        return regression_matrix

    def train_model(self, save_path, a_matrix, b_matrix, first_train=True, iterations=10):

        if first_train:
            initial_matrix = self.first_pass_training()
            A_Matrix = initial_matrix[:self.state_len, :self.state_len]
            B_Matrix = initial_matrix[:self.state_len, self.state_len:]
            C_Matrix = np.identity(self.state_len)
            D_Matrix = np.zeros([self.state_len, self.input_len])
        else:
            A_Matrix = a_matrix
            B_Matrix = b_matrix
            C_Matrix = np.identity(self.state_len)
            D_Matrix = np.zeros([self.state_len, self.input_len])

        a_sim = A_Matrix.reshape(-1, 1)
        b_sim = B_Matrix.reshape(-1, 1)
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
                state = signal.StateSpace(a_matrix, b_matrix, C_Matrix, D_Matrix, dt=1)
                _, of_yout, _ = signal.dlsim(state, of_u, self.time, of_x0)

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
        A_Matrix = opt_matrix[: (self.state_len**2)].reshape(self.state_len, self.state_len)
        B_Matrix = opt_matrix[(self.state_len**2) :].reshape(self.state_len, self.input_len)
        np.savetxt(fr"{save_path}\A_Matrix.csv", A_Matrix, delimiter=',')
        np.savetxt(fr"{save_path}\B_Matrix.csv", B_Matrix, delimiter=',')
        return A_Matrix, B_Matrix

    def test_model(self, a_matrix, b_matrix, test_label: str):

        C_Matrix = np.identity(self.state_len)
        D_Matrix = np.zeros([self.state_len, self.input_len])
        columns = self.states + self.inputs

        # y_out_test = np.zeros([self.num_days, self.state_len])
        test_model_dict = {}
        test_grouped = self.test_data.groupby("Batch")
        for name, group in test_grouped:
            test_x0 = np.array(group.filter(self.states).iloc[0])
            test_u = np.array(group.filter(self.inputs))
            bioreactor = signal.StateSpace(a_matrix, b_matrix, C_Matrix, D_Matrix, dt=1)
            _, test_yout, _ = signal.dlsim(bioreactor, test_u, self.time, test_x0)
            raw_data = np.hstack((test_yout,test_u))
            test_model_dict[name] = pd.DataFrame(data=raw_data, columns=columns)

            plt.figure(figsize=(10,5))
            plt.plot(self.time, test_model_dict[name][test_label],"ro-", label="Simulated")
            plt.plot(self.time, group[test_label],"bo-", label="Actual")
            plt.legend()
            plt.title(name)
            plt.show()

    def data_plots(self):
        pass

    def get_matrices(self):
        pass

    def train_test_model(self, save_path, a_matrix, b_matrix, test_label: str, first_train=True, iterations=10):

        A_Matrix, B_Matrix = self.train_model(
            save_path=save_path,
            a_matrix=a_matrix,
            b_matrix=b_matrix,
            first_train=first_train,
            iterations=iterations,
        )
        self.test_model(
            A_Matrix=A_Matrix,
            B_Matrix=B_Matrix,
            test_label=test_label,
        )

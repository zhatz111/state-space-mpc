import math
import numpy as np
import scipy.stats
import pandas as pd
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt

class ModelOptimizer:

    def __init__(self, target_label: str, a_matrix: np.array, b_matrix: np.array, states: list, inputs: list, scaler, constraint_dict: dict,
    initial_input, initial_condition: np.array, days: int, scaler_dict: dict, max_iters = 1000, volume = 196):
        self.days = days
        self.scaler_dict = scaler_dict
        self.target_label = target_label
        self.scaler = scaler
        self.states = states
        self.inputs = inputs
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.initial_input = initial_input
        self.constraint_dict = constraint_dict
        self.initial_condition = initial_condition
        self.max_iters = max_iters
        self.iterations = 0
        self.volume = volume
        self.bound = True
        self.state_len = len(states)
        self.input_len = len(inputs)
        self.max_iters = max_iters
        self.x_history = []
        self.y_history = []
        self.result = None
        self.glucose = None
        

    def optimizer_function(self, input_array):
        # Change the start of this function according to what your input array will be
        x0 = np.zeros([self.days, self.input_len])
        # Construct feed day matrix column
        for day in [3,5,7,10,12]:
            x0[day,0] = ((input_array[0]/100)*self.scaler_dict["Daily_Feed_Normalized"][0])+self.scaler_dict["Daily_Feed_Normalized"][1]

        # Construct glucose feed column
        x0[:,1] = self.glucose.ravel()
        # construct pH setpoint column
        x0[:,2] = (input_array[1]*self.scaler_dict["pH_Setpoint"][0])+self.scaler_dict["pH_Setpoint"][1]
        x0[:,2] = (7.15*self.scaler_dict["pH_Setpoint"][0])+self.scaler_dict["pH_Setpoint"][1]
        # Construct temperature column
        shift_day = int(input_array[4])
        x0[:shift_day,3] = (input_array[2]*self.scaler_dict["Temperature"][0])+self.scaler_dict["Temperature"][1]
        x0[shift_day:,3] = (input_array[3]*self.scaler_dict["Temperature"][0])+self.scaler_dict["Temperature"][1]

        c_matrix = np.identity(self.state_len)
        d_matrix = np.zeros([self.state_len, self.input_len])
        u_sim = x0
        t_sim = np.arange(0, self.days, 1)
        state = signal.StateSpace(self.a_matrix, self.b_matrix, c_matrix, d_matrix)
        _, y_func, _ = signal.lsim(state, u_sim, t_sim, self.initial_condition)
        return y_func, u_sim
    
    def objective_function(self, input_array):
        y_out, u_sim = self.optimizer_function(input_array)
        data = np.array(self.inverse_scale(y_out, u_sim).filter(like=self.target_label))
        if self.iterations % 2 == 0:
            print("Iteration: ", self.iterations)
            print("Value: ", data[-1].item())
        self.iterations += 1
        self.x_history.append(self.iterations)
        self.y_history.append(data[-1].item())
        return 0 - data[-1].item()
    
    def inverse_scale(self, y_out, u_sim):
        data = np.hstack([y_out, u_sim])
        columns = self.states + self.inputs
        df = pd.DataFrame(data=data, columns=columns)
        for column in df.columns:
            scaler_value = self.scaler_dict[column][0]
            min_value = self.scaler_dict[column][1]
            df[column] = (df[column]-min_value)/scaler_value
        return df

    def optimize(self):
        constraints = [
            # {"type": "ineq", "fun": self.minzero_constraint},
            # {"type": "ineq", "fun": self.vcc_constraint},
            # {"type": "ineq", "fun": self.viability_constraint},
            # {"type": "ineq", "fun": self.ammonium_constraint},
            # {"type": "ineq", "fun": self.lactate_constraint},
            # {"type": "ineq", "fun": self.glucose_constraint},
            {"type": "ineq", "fun": self.titer_constraint},
        ]

        # tuple_list = []
        # for i in range(self.days):
        #     inner_tuple = (0,1)
        #     tuple_list.append(inner_tuple)
        # tuple_of_tuples = tuple(tuple_list)

        bounds = (
            (2,4),
            (6.9, 7.2),
            (36, 37),
            (30, 32),
            (4,6),
            # (2,4.5),
        )

        res = optimize.minimize(
            fun=self.objective_function,
            x0=self.initial_input,
            constraints=constraints,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.max_iters},
        )
        self.result = res.x.ravel()
        print("Optimal Matrix:",self.result.ravel())
        return

    def plot_history(self):
        plt.plot(self.x_history, self.y_history, 'o-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Objective function history')
        plt.show()
    
    def plot_inputs(self):
        y_out, u_sim = self.optimizer_function(self.result.ravel())
        data = self.inverse_scale(y_out, u_sim).filter(items=self.inputs)
        input_dict = {}
        for column in data.columns:
            input_dict[column] = data[column]
        cols = 2
        rows = math.ceil(len(input_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10,10),squeeze=False)
        dict_keys = [k for k in input_dict.keys()]
        count = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    key = dict_keys[count]
                    axes[i][j].plot(np.arange(0,len(input_dict[key]),1),input_dict[key],"ro-",markersize=3.5)
                    axes[i][j].set_title(key)
                    count += 1
                except:
                    pass
        plt.legend(loc="best")
        fig.suptitle(f"Optimal Setpoints: {self.result.ravel()}")
        fig.tight_layout()
        plt.show()
    
    def plot_states(self):
        y_out, u_sim = self.optimizer_function(self.result.ravel())
        data = self.inverse_scale(y_out, u_sim).filter(items=self.states)
        state_dict = {}
        for column in data.columns:
            state_dict[column] = data[column]
        cols = 3
        rows = math.ceil(len(state_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10,5), squeeze=False)
        dict_keys = [k for k in state_dict.keys()]
        count = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    key = dict_keys[count]
                    axes[i][j].plot(np.arange(0,len(state_dict[key]),1),state_dict[key],"ro-",markersize=3.5)
                    axes[i][j].set_title(key)
                    count += 1
                except:
                    pass
        plt.legend(loc="best")
        fig.tight_layout()
        plt.show()

    def mean_confidence_interval(self, confidence=0.95):
        y_out, u_sim = self.optimizer_function(self.result.ravel())
        data = self.inverse_scale(y_out, u_sim).filter(items=self.states)
        confidence_list = []
        for state in self.states:
            a = 1.0 * np.array(data[state])
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            confidence_list.append((m, m-h, m+h))
        return confidence_list

    # Place any constraints to the model here!!!

    def minzero_constraint(self, input_array):  # Nothing can be less than 0
        y_out_opt, _ = self.optimizer_function(input_array)
        return min(y_out_opt.reshape(-1, 1))

    def vcc_constraint(self, input_array):  # VCC cannot be greater than TCC
        y_out, u_sim = self.optimizer_function(input_array)
        # tcc = np.array(self.inverse_scale(y_out, u_sim).filter(like="TCC"))
        vcc = np.array(self.inverse_scale(y_out, u_sim).filter(like="VCC"))
        return self.constraint_dict["VCC"] - max(vcc)

    def viability_constraint(self, input_array):  # Viability Constraint
        y_out, u_sim = self.optimizer_function(input_array)
        tcc = np.array(self.inverse_scale(y_out, u_sim).filter(like="TCC"))
        vcc = np.array(self.inverse_scale(y_out, u_sim).filter(like="VCC"))
        return 100 - max(vcc / tcc)

    def ammonium_constraint(self, input_array):  # Ammonium constraint
        y_out, u_sim = self.optimizer_function(input_array)
        amm = max(np.array(self.inverse_scale(y_out, u_sim).filter(like="Ammonium")))
        return self.constraint_dict["Ammonium"] - amm

    def lactate_constraint(self, input_array):  # Lactate Constraint
        y_out, u_sim = self.optimizer_function(input_array)
        lac = max(np.array(self.inverse_scale(y_out, u_sim).filter(like="Lactate")))
        return self.constraint_dict["Lactate"] - lac
    
    # def glucose_constraint(self, input_array):  # Glucose constraint
    #     y_out, u_sim = self.optimizer_function(input_array)
    #     gluc = np.array(self.inverse_scale(y_out, u_sim).filter(like="Glucose"))
    #     return min(gluc) - self.constraint_dict["Glucose"]
    
    def titer_constraint(self, input_array):  # Titer Constraint
        y_out, u_sim = self.optimizer_function(input_array)
        igg = min(np.array(self.inverse_scale(y_out, u_sim).filter(like="IGG")))
        return  igg


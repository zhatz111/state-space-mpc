import math
import numpy as np
import scipy.stats
import pandas as pd
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt

class ModelOptimizer:
    """_summary_
    """

    def __init__(self, target_label: str, a_matrix: np.ndarray, b_matrix: np.ndarray,
        states: list, inputs: list, scaler, constraint_dict: dict, initial_input,
        initial_condition: np.ndarray, days: int, scaler_dict: dict, volume = 200,
        max_iters = 1000):

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

        # UNCOMMENT FOR UNCONSTRAINED FEEDING STRATEGY
        for count, day in enumerate(range(0,self.days)):
            if day is not self.days:
                x0[day,0] = (input_array[count]-self.scaler_dict["Normalized_Feed_Percent"][0])/self.scaler_dict["Normalized_Feed_Percent"][1]
            else:
                x0[self.days,0] = (0-self.scaler_dict["Normalized_Feed_Percent"][0])/self.scaler_dict["Normalized_Feed_Percent"][1]

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
        if self.iterations % 50 == 0:
            print("Iteration: ", self.iterations)
            print("Value: ", data[-1].item())
        self.iterations += 1
        self.x_history.append(self.iterations)
        self.y_history.append(data[-1].item())
        return 0 - data[-1].item() + 0.5*(np.sum(np.diff(u_sim[:,0])**2))

    def inverse_scale(self, y_out, u_sim):
        data = np.hstack([y_out, u_sim])
        columns = self.states + self.inputs
        df = pd.DataFrame(data=data, columns=columns)
        for column in df.columns:
            scaler_value = self.scaler_dict[column][1]
            min_value = self.scaler_dict[column][0]
            df[column] = (df[column]*scaler_value)+min_value
        return df

    def optimize(self):
        constraints = [
            {"type": "ineq", "fun": self.minzero_constraint},
            # {"type": "ineq", "fun": self.vcc_constraint},
            # {"type": "ineq", "fun": self.ivc_constraint},
            # {"type": "ineq", "fun": self.viability_constraint},
            # {"type": "ineq", "fun": self.ammonium_constraint},
            # {"type": "ineq", "fun": self.lactate_constraint},
            # {"type": "ineq", "fun": self.ILAC_constraint},
            {"type": "ineq", "fun": self.feed_constraint},
            # {"type": "ineq", "fun": self.titer_constraint},
        ]

        num1 = self.days
        feed_bounds = [(0,5)]*num1
        bounds = tuple(feed_bounds)

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

    def volume_calculator(self, feed_arr, glucose_arr):

        volume_arr = []
        feed_scaled = []
        gluc_scaled = []

        for count, _ in enumerate(feed_arr):
            if count == 0:
                volume = self.constraint_dict["Volume"]
                feed = feed_arr[0] * self.constraint_dict["Volume"]
                gluc = glucose_arr[0] * self.constraint_dict["Volume"]
            else:
                volume = volume_arr[count-1] + feed_scaled[count-1] + gluc_scaled[count-1] - self.constraint_dict["Sample_vol"]
                feed = feed_arr[count] * volume
                gluc = glucose_arr[count] * volume

            volume_arr.append(volume)
            feed_scaled.append(feed)
            gluc_scaled.append(gluc)

        return np.array(volume_arr), np.array(feed_scaled), np.array(gluc_scaled)

    def volume_calculator_no_gluc(self, feed_arr):

        volume_arr = []
        feed_scaled = []

        for count, _ in enumerate(feed_arr):
            if count == 0:
                volume = self.constraint_dict["Volume"]
                feed = feed_arr[0] * self.constraint_dict["Volume"]
            else:
                volume = volume_arr[count-1] + feed_scaled[count-1] - self.constraint_dict["Sample_vol"] + 1 # Added 1 to simulate glucose additions in ambrs
                feed = feed_arr[count] * volume

            volume_arr.append(volume)
            feed_scaled.append(feed)

        return np.array(volume_arr), np.array(feed_scaled)

    def plot_history(self):
        plt.plot(self.x_history, self.y_history, 'o-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Objective function history')
        plt.show()

    def plot_inputs(self):
        y_out, u_sim = self.optimizer_function(self.result)
        data = self.inverse_scale(y_out, u_sim).filter(items=self.inputs)
        volume, feed = self.volume_calculator_no_gluc(data[self.inputs[0]])
        input_dict = {}
        for column in data.columns:
            input_dict[column] = data[column]
        input_dict["Volume"] = volume
        input_dict["Scaled_Feed"] = feed
        cols = 2
        rows = math.ceil(len(input_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10,10),squeeze=False)
        dict_keys = list(input_dict)
        print(dict_keys)
        count = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    key = dict_keys[count]
                    axes[i][j].plot(np.arange(0,len(input_dict[key]),1),input_dict[key],"ro-",markersize=3.5)
                    axes[i][j].set_title(key)
                    count += 1
                except KeyError:
                    pass
                except IndexError:
                    pass
        feed_volume = np.sum(feed)/self.constraint_dict["Volume"]
        plt.legend(loc="best")
        if self.result is not None:
            fig.suptitle(
                f"Feed Volume: {feed_volume:.2f}, \
                Final Volume: {volume[-1]:.2f}"
            )
        fig.tight_layout()
        plt.show()

    def plot_states(self):
        y_out, u_sim = self.optimizer_function(self.result)
        data = self.inverse_scale(y_out, u_sim).filter(items=self.states)
        state_dict = {}
        for column in data.columns:
            state_dict[column] = data[column]
        cols = 3
        rows = math.ceil(len(state_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13,5), squeeze=False)
        dict_keys = [k for k in state_dict.keys()]
        count = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    key = dict_keys[count]
                    axes[i][j].plot(np.arange(0,len(state_dict[key]),1),state_dict[key],"ro-",markersize=3.5)
                    axes[i][j].set_title(key)
                    # pd.Series(state_dict[key]).to_clipboard()
                    count += 1
                except KeyError:
                    pass
                except IndexError:
                    pass
        plt.legend(loc="best")
        fig.tight_layout()
        plt.show()

    def mean_confidence_interval(self, confidence=0.95):
        y_out, u_sim = self.optimizer_function(self.result)
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
        y_out, u_sim = self.optimizer_function(input_array)
        data = np.array(self.inverse_scale(y_out, u_sim))
        return min(data.reshape(-1, 1))

    def vcc_constraint(self, input_array):  # EOR VCC cannot be less than a value
        y_out, u_sim = self.optimizer_function(input_array)
        vcc = np.array(self.inverse_scale(y_out, u_sim).filter(like="VCC"))
        return np.mean(vcc[9:]) - self.constraint_dict["VCC"]

    def ivc_constraint(self, input_array):  # EOR IVC needs to be maximized
        y_out, u_sim = self.optimizer_function(input_array)
        ivc = np.array(self.inverse_scale(y_out, u_sim).filter(like="IVC"))
        return self.constraint_dict["IVC"] - ivc[-1]

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
        lac = max(np.array(self.inverse_scale(y_out, u_sim).filter(like="Lactate"))[8:])
        return self.constraint_dict["Lactate"] - lac

    def ILAC_constraint(self, input_array):  # integral of lactate Constraint
        y_out, u_sim = self.optimizer_function(input_array)
        ilac = max(np.array(self.inverse_scale(y_out, u_sim).filter(like="ILAC")))
        return self.constraint_dict["ILAC"] - ilac

    def titer_constraint(self, input_array):  # Titer Constraint
        y_out, u_sim = self.optimizer_function(input_array)
        igg = min(np.array(self.inverse_scale(y_out, u_sim).filter(like="IGG")))
        return igg

    def feed_constraint(self, input_array):  # Feed Constraint
        y_out, u_sim = self.optimizer_function(input_array)
        data = self.inverse_scale(y_out, u_sim).filter(items=self.inputs)
        _, feed = self.volume_calculator_no_gluc(data[self.inputs[0]])
        feed_volume = np.sum(feed)/self.constraint_dict["Volume"]
        return self.constraint_dict["Max_feed_volume"] - feed_volume

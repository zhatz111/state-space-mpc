"""_summary_

Returns:
    _type_: _description_
"""
import math
import numpy as np
import scipy.stats
import pandas as pd
from scipy import signal
from scipy import optimize
import matplotlib.pyplot as plt

# The ModelOptimizer class is used for optimizing machine learning models.
class ModelOptimizer:
    """_summary_"""

    def __init__(
        self,
        target_label: str,
        a_matrix: np.ndarray,
        b_matrix: np.ndarray,
        states: list,
        inputs: list,
        scaler,
        constraint_dict: dict,
        initial_input,
        initial_condition: np.ndarray,
        days: int,
        scaler_dict: dict,
        volume=200,
        max_iters=1000,
    ):
        """
        The function is an initializer for a class that performs a simulation using a given set of
        parameters.
        
        Args:
          target_label (str): The target_label parameter is a string that represents the label of the
        target variable that you want to predict or analyze.
          a_matrix (np.ndarray): The `a_matrix` parameter is a numpy array that represents the
        transition matrix for the hidden Markov model. It defines the probabilities of transitioning
        from one state to another.
          b_matrix (np.ndarray): The `b_matrix` parameter is a numpy array that represents the
        input-output relationship of the system. It is a matrix where each row corresponds to a state
        and each column corresponds to an input. The values in the matrix represent the coefficients of
        the inputs for each state.
          states (list): The `states` parameter is a list that represents the different states or
        variables in your system. Each element in the list represents a state or variable.
          inputs (list): The `inputs` parameter is a list that represents the input variables for the
        system. These input variables can be used to control or influence the behavior of the system.
          scaler: The `scaler` parameter is used to scale the input data. It is an object that
        implements the `fit_transform` method, which is used to fit the scaler to the data and transform
        the data using the fitted scaler.
          constraint_dict (dict): The `constraint_dict` parameter is a dictionary that contains
        constraints for the optimization problem. It specifies the upper and lower bounds for each state
        and input variable. The keys of the dictionary are the variable names (states and inputs), and
        the values are tuples containing the lower and upper bounds for each variable.
          initial_input: The `initial_input` parameter is the initial value of the input variable for
        the system. It represents the starting value of the input variable at the beginning of the
        simulation or optimization process.
          initial_condition (np.ndarray): The initial condition parameter represents the initial state
        of the system. It is a numpy array that contains the initial values for each state variable in
        the system.
          days (int): The number of days for which the model will be run.
          scaler_dict (dict): The `scaler_dict` parameter is a dictionary that contains scaling factors
        for each feature in the dataset. It is used to scale the input data before training the model.
          volume: The volume parameter represents the volume of a system or container. It is used in the
        context of the code you provided, but its specific purpose or meaning may depend on the broader
        context or application of the code. Defaults to 200
          max_iters: The parameter "max_iters" represents the maximum number of iterations that will be
        performed in the algorithm. Defaults to 1000
        """
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
        """
        The function takes an input array, performs some calculations, and returns two arrays as output.
        
        Args:
          input_array: The `input_array` parameter is an array that contains the input values for each
        day. It is used to calculate the initial state values `x0` for the optimization function.
        
        Returns:
          two variables: `y_func` and `u_sim`.
        """
        # Change the start of this function according to what your input array will be
        x0 = np.zeros([self.days, self.input_len])

        # UNCOMMENT FOR UNCONSTRAINED FEEDING STRATEGY
        for count, day in enumerate(range(0, self.days)):
            if day is not self.days:
                x0[day, 0] = (
                    input_array[count] - self.scaler_dict["Normalized_Feed_Percent"][0]
                ) / self.scaler_dict["Normalized_Feed_Percent"][1]
            else:
                x0[self.days, 0] = (
                    0 - self.scaler_dict["Normalized_Feed_Percent"][0]
                ) / self.scaler_dict["Normalized_Feed_Percent"][1]

        c_matrix = np.identity(self.state_len)
        d_matrix = np.zeros([self.state_len, self.input_len])
        u_sim = x0
        t_sim = np.arange(0, self.days, 1)
        state = signal.StateSpace(self.a_matrix, self.b_matrix, c_matrix, d_matrix)
        _, y_func, _ = signal.lsim(state, u_sim, t_sim, self.initial_condition)
        return y_func, u_sim

    def objective_function(self, input_array):
        """
        The objective_function takes an input array, performs optimization, and returns a value that is
        the negative of the last item in the filtered data array plus half the sum of the squared
        differences of the first column of u_sim.
        
        Args:
          input_array: The `input_array` parameter is an array that contains the input values for the
        optimizer function.
        
        Returns:
          the objective function value, which is calculated as 0 minus the last item in the 'data'
        array, plus 0.5 times the sum of the squared differences of the first column of the 'u_sim'
        array.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        data = np.array(self.inverse_scale(y_out, u_sim).filter(like=self.target_label))
        if self.iterations % 50 == 0:
            print("Iteration: ", self.iterations)
            print("Value: ", data[-1].item())
        self.iterations += 1
        self.x_history.append(self.iterations)
        self.y_history.append(data[-1].item())
        return 0 - data[-1].item() + 0.5 * (np.sum(np.diff(u_sim[:, 0]) ** 2))

    def inverse_scale(self, y_out, u_sim):
        """
        The function takes in scaled data and returns the inverse scaled data using a scaler object.
        
        Args:
          y_out: The parameter `y_out` represents the output variable(s) that have been scaled. It is a
        numpy array or pandas DataFrame containing the scaled values of the output variable(s).
          u_sim: The parameter `u_sim` represents the simulated input values.
        
        Returns:
          a DataFrame object named `df_scaled`.
        """
        df_scaled = pd.DataFrame(
                data=self.scaler.inverse_transform(np.hstack((y_out, u_sim))),
                columns=self.scaler.get_feature_names_out(),
        )
        return df_scaled

    def optimize(self):
        """
        The `optimize` function is used to find the optimal solution for a given set of constraints and
        objective function using the SLSQP optimization method.
        
        Returns:
          the optimal matrix, which is stored in the variable `self.result`.
        """
        constraints = [
            {"type": "ineq", "fun": self.minzero_constraint},
            # {"type": "ineq", "fun": self.vcc_constraint},
            # {"type": "ineq", "fun": self.ivc_constraint},
            # {"type": "ineq", "fun": self.viability_constraint},
            # {"type": "ineq", "fun": self.ammonium_constraint},
            # {"type": "ineq", "fun": self.lactate_constraint},
            {"type": "ineq", "fun": self.feed_constraint},
            # {"type": "ineq", "fun": self.titer_constraint},
        ]

        num1 = self.days
        feed_bounds = [(0, 5)] * num1
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
        print("Optimal Matrix:", self.result.ravel())
        return

    def volume_calculator(self, feed_arr, glucose_arr):
        """
        The function calculates the volume, feed, and glucose values based on given arrays and
        constraints.
        
        Args:
          feed_arr: The `feed_arr` parameter is a list or array containing the feed values for each
        iteration or time step. It represents the amount of feed added at each step.
          glucose_arr: The `glucose_arr` parameter is an array containing glucose values.
        
        Returns:
          three arrays: volume_arr, feed_scaled, and gluc_scaled.
        """
        volume_arr = []
        feed_scaled = []
        gluc_scaled = []

        for count, _ in enumerate(feed_arr):
            if count == 0:
                volume = self.constraint_dict["Volume"]
                feed = feed_arr[0] * self.constraint_dict["Volume"]
                gluc = glucose_arr[0] * self.constraint_dict["Volume"]
            else:
                volume = (
                    volume_arr[count - 1]
                    + feed_scaled[count - 1]
                    + gluc_scaled[count - 1]
                    - self.constraint_dict["Sample_vol"]
                )
                feed = feed_arr[count] * volume
                gluc = glucose_arr[count] * volume

            volume_arr.append(volume)
            feed_scaled.append(feed)
            gluc_scaled.append(gluc)

        return np.array(volume_arr), np.array(feed_scaled), np.array(gluc_scaled)

    def volume_calculator_no_gluc(self, feed_arr):
        """
        The function calculates the volume and feed values based on a given feed array and a set of
        constraints.
        
        Args:
          feed_arr: The `feed_arr` parameter is a list that represents the feed values for each
        iteration. Each element in the list corresponds to a specific iteration or count.
        
        Returns:
          two arrays: `volume_arr` and `feed_scaled`.
        """
        volume_arr = []
        feed_scaled = []

        for count, _ in enumerate(feed_arr):
            if count == 0:
                volume = self.constraint_dict["Volume"]
                feed = feed_arr[0] * self.constraint_dict["Volume"]
            else:
                volume = (
                    volume_arr[count - 1]
                    + feed_scaled[count - 1]
                    - self.constraint_dict["Sample_vol"]
                    + 1
                )  # Added 1 to simulate glucose additions in ambrs
                feed = feed_arr[count] * volume

            volume_arr.append(volume)
            feed_scaled.append(feed)

        return np.array(volume_arr), np.array(feed_scaled)

    def plot_history(self):
        """
        The function plots the history of the objective function values.
        """
        plt.plot(self.x_history, self.y_history, "o-")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Objective function history")
        plt.show()

    def plot_inputs(self):
        """
        The function `plot_inputs` plots the input data along with the calculated volume and scaled
        feed.
        """
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
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10), squeeze=False)
        dict_keys = list(input_dict)
        print(dict_keys)
        count = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    key = dict_keys[count]
                    axes[i][j].plot(
                        np.arange(0, len(input_dict[key]), 1),
                        input_dict[key],
                        "ro-",
                        markersize=3.5,
                    )
                    axes[i][j].set_title(key)
                    count += 1
                except KeyError:
                    pass
                except IndexError:
                    pass
        feed_volume = np.sum(feed) / self.constraint_dict["Volume"]
        plt.legend(loc="best")
        if self.result is not None:
            fig.suptitle(
                f"Feed Volume: {feed_volume:.2f}, \
                Final Volume: {volume[-1]:.2f}"
            )
        fig.tight_layout()
        plt.show()

    def plot_states(self):
        """
        The function `plot_states` plots the states of a system using data obtained from an optimizer
        function.
        """
        y_out, u_sim = self.optimizer_function(self.result)
        data = self.inverse_scale(y_out, u_sim).filter(items=self.states)
        state_dict = {}
        for column in data.columns:
            state_dict[column] = data[column]
        cols = 3
        rows = math.ceil(len(state_dict) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13, 5), squeeze=False)
        dict_keys = [k for k in state_dict.keys()]
        count = 0
        for i in range(rows):
            for j in range(cols):
                try:
                    key = dict_keys[count]
                    axes[i][j].plot(
                        np.arange(0, len(state_dict[key]), 1),
                        state_dict[key],
                        "ro-",
                        markersize=3.5,
                    )
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
        """
        The function calculates the mean and confidence interval for a given dataset.
        
        Args:
          confidence: The confidence parameter is a value between 0 and 1 that represents the desired
        level of confidence for the confidence interval. In this case, it is set to 0.95, which
        corresponds to a 95% confidence interval.
        
        Returns:
          a list of tuples, where each tuple contains the mean value of a state variable, as well as the
        lower and upper bounds of the confidence interval for that state variable.
        """
        y_out, u_sim = self.optimizer_function(self.result)
        data = self.inverse_scale(y_out, u_sim).filter(items=self.states)
        confidence_list = []
        for state in self.states:
            a = 1.0 * np.array(data[state])
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
            confidence_list.append((m, m - h, m + h))
        return confidence_list


    # Place any constraints to the model here!!!

    def minzero_constraint(self, input_array):  # Nothing can be less than 0
        """
        The function takes an input array, performs an optimization function on it, and returns the
        minimum value of the resulting data array after inverse scaling.
        
        Args:
          input_array: The input_array is a numpy array that contains the input values for the optimizer
        function.
        
        Returns:
          the minimum value of the data array after it has been reshaped into a column vector.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        data = np.array(self.inverse_scale(y_out, u_sim))
        return min(data.reshape(-1, 1))

    def vcc_constraint(self, input_array):  # EOR VCC cannot be less than a value
        """
        The function calculates the mean value of the VCC (Voltage at the Common Collector) and
        subtracts it from a specified constraint value.
        
        Args:
          input_array: The `input_array` parameter is an array that contains the input values for the
        optimizer function.
        
        Returns:
          the difference between the mean of the VCC values from index 9 onwards and the value specified
        in the "VCC" key of the constraint_dict.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        vcc = np.array(
            self.inverse_scale(y_out, u_sim).filter(like="VCC"), dtype=np.float64
        )
        return np.mean(vcc[9:]) - self.constraint_dict["VCC"]

    def ivc_constraint(self, input_array):  # EOR IVC needs to be maximized
        """
        The function calculates the difference between a constraint value and the last element of an
        array.
        
        Args:
          input_array: The input_array is an array of values that are used as input to the
        optimizer_function.
        
        Returns:
          the difference between the desired IVC constraint value and the last value of the calculated
        IVC.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        ivc = np.array(self.inverse_scale(y_out, u_sim).filter(like="IVC"))
        return self.constraint_dict["IVC"] - ivc[-1]

    def viability_constraint(self, input_array):  # Viability Constraint
        """
        The function calculates the viability constraint by finding the maximum value of the ratio of
        VCC (Variable Cost of Consumption) to TCC (Total Cost of Consumption) and subtracting it from
        100.
        
        Args:
          input_array: The input_array is a parameter that is passed to the viability_constraint
        function. It is an array that contains the input values for the optimizer function.
        
        Returns:
          the viability constraint value, which is calculated as 100 minus the maximum value of the
        ratio of VCC (Variable Cost of Consumption) to TCC (Total Cost of Consumption).
        """
        y_out, u_sim = self.optimizer_function(input_array)
        tcc = np.array(
            self.inverse_scale(y_out, u_sim).filter(like="TCC"), dtype=np.float64
        )
        vcc = np.array(
            self.inverse_scale(y_out, u_sim).filter(like="VCC"), dtype=np.float64
        )
        return 100 - max(vcc / tcc)

    def ammonium_constraint(self, input_array):  # Ammonium constraint
        """
        The function calculates the difference between the maximum value of Ammonium in the output array
        and the Ammonium constraint.
        
        Args:
          input_array: The input_array is a numpy array that contains the input values for the optimizer
        function.
        
        Returns:
          the difference between the value of the "Ammonium" constraint and the maximum value of the
        "Ammonium" variable in the output array.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        amm = max(np.array(self.inverse_scale(y_out, u_sim).filter(like="Ammonium")))
        return self.constraint_dict["Ammonium"] - amm

    def lactate_constraint(self, input_array):  # Lactate Constraint
        """
        The function calculates the difference between a lactate constraint and the maximum lactate
        value obtained from a given input array.
        
        Args:
          input_array: The input_array is a numpy array that contains the input values for the optimizer
        function. It is used as an input to the optimizer function to calculate the output values.
        
        Returns:
          the difference between the desired lactate constraint value and the maximum lactate value
        obtained from the optimizer function.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        lac = max(np.array(self.inverse_scale(y_out, u_sim).filter(like="Lactate"))[8:])
        return self.constraint_dict["Lactate"] - lac

    def titer_constraint(self, input_array):  # Titer Constraint
        """
        The function calculates the minimum value of the "IGG" column in the output of the optimizer
        function.
        
        Args:
          input_array: The input_array is a parameter that is passed to the titer_constraint function.
        It is an array that contains the input values for the optimizer function.
        
        Returns:
          the minimum value of the array `igg`.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        igg = min(np.array(self.inverse_scale(y_out, u_sim).filter(like="IGG")))
        return igg

    def feed_constraint(self, input_array):  # Feed Constraint
        """
        The function calculates the difference between the maximum feed volume and the actual feed
        volume based on the input array.
        
        Args:
          input_array: The input_array is a parameter that represents the input data for the feed
        constraint. It is used as an input to the optimizer_function method.
        
        Returns:
          the difference between the maximum feed volume allowed and the calculated feed volume.
        """
        y_out, u_sim = self.optimizer_function(input_array)
        data = self.inverse_scale(y_out, u_sim).filter(items=self.inputs)
        _, feed = self.volume_calculator_no_gluc(data[self.inputs[0]])
        feed_volume = np.sum(feed) / self.constraint_dict["Volume"]
        return self.constraint_dict["Max_feed_volume"] - feed_volume

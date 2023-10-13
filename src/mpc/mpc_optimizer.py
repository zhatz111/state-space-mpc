"""MPC-related class definitions
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

import numpy as np
import pandas as pd

class Bioreactor:
    """Bioreactor object class for simulation and also processing real off-line data
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-05
        Modified: 2023-10-06
    """

    def __init__(
            self,
            vessel = None,
            process_model = None,
            data = pd.DataFrame,
            state = None,
            duration = 12,
            curr_time = 0):
        self.vessel = vessel # Vessel name for processing multiple bioreactors
        self.process_model = process_model # Model (if provided) for simulating a process
        self.data = data # Data for storing simulation results or real data (if provided)
        self.state = state # Current state of the process
        self.duration = duration # Full culture duration
        self.curr_time = curr_time # Current culture day


    def log_sample(
            self,
            sample_day: int,
            sample_var_names: list[str],
            sample_var_vals: list[float]
            ):
        """Replace the specified day (row) with new data
            Created by Yu Luo (yu.8.luo@gsk.com)
            Created: 2023-10-09
            Modified: 2023-10-09
        """
        self.data.loc[self.data['Day'] == sample_day,sample_var_names] = sample_var_vals

    def update_input(
            self,
            input_days,
            input_var_names,
            input_var_vals
            ):
        """Update column(s) of input
            Created by Yu Luo (yu.8.luo@gsk.com)
            Created: 2023-10-09
            Modified: 2023-10-09
        """
        for i in range(len(input_days)):
            self.data.loc[self.data.Day == input_days[i],input_var_names] = input_var_vals[i,:]


class Controller:
    """Controller object class
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-05
        Modified: 2023-10-06
    """

    def __init__(
            self,
            controller_model,
            bioreactor: Bioreactor,
            ts: np.array, # A 1D, length-T array of time
            pv_sps: np.array, # A T by P array (P process variables)
            pv_names: list[str], # Controlled process variable names
            pv_wts: np.array, # SP tracking weights
            mv_names: list[str], # Manipulated variables
            mv_wts: np.array, # MV cost weights
            pred_horizon: int,
            ctrl_horizon: int,
            constr: np.array, # A 2 by U array (lower and upper limits only)
            curr_time = 0 # Current culture day
    ):
        self.controller_model = controller_model
        self.bioreactor = bioreactor
        self.curr_time = curr_time
        self.ts = ts,
        self.pv_sps = pv_sps
        self.pv_names = pv_names
        self.pv_wts = pv_wts
        self.mv_names = mv_names
        self.mv_wts = mv_wts
        self.pred_horizon = pred_horizon
        self.ctrl_horizon = ctrl_horizon
        self.constr = constr

    def est_state(self):
        """Estimate the current state based on previous measurement"""
        # Gather previous day's state and input
        data_prev = self.data.loc[
            self.data[:,'Day'] == self.curr_time,:]
        x_prev = data_prev[0,self.controller_model.states]
        u_prev = data.prev[0,self.controller_model.states]
        x_curr = self.controller_model.predict(x_prev,u_prev)

        # Replace current missing data with estimated


    def optimize(self):
        """Optimize future inputs"""

        # Retrieve MVs from curr_time to EoR
        data = self.bioreactor.data
        mv_matrix = data.loc[data['Day'] >= self.curr_time,self.mv_names].values

        # Flatten initial mv
        mv_array = mv_matrix.flatten()
        
        # Create constraint matrix
        constr_low_matrix = np.tile(self.constr[0,:],(mv_matrix.shape[0],1))
        constr_low_array = constr_low_matrix.flatten()
        constr_high_matrix = np.tile(self.constr[1,:],(mv_matrix.shape[0],1))
        constr_high_array = constr_high_matrix.flatten()

        # Solve the optimization problem


    def obj_func_wrapper(self,mv_array):
        """Objective function wrapper"""
        
        # Fold mv_array to a 2D array
        mv_matrix = mv_array.reshape([-1,len(self.mv_names)])

        # Retrieve input from curr_time to EoR
        u_matrix = self.bioreactor.data.loc[
            self.bioreactor.data['Day'] >= self.curr_time,
            self.controller_model.inputs].values

        # Replace MVs with mv_matrix
        loc_mv_in_inputs = np.where(np.isin(self.controller_model.inputs,self.mv_names))[0]
        u_matrix[:,loc_mv_in_inputs] = mv_matrix
        print(u_matrix)

        # # Sim
        # xHat = self.controller_model.predict()

        # # Obj
        # return self.obj_func(ts,xHat,u)
        

    def obj_func(
            self,
            ts: np.array,
            x: np.array,
            u: np.array):
        """Return the cost value based on x and u"""

        # Trim to keep only future entries
        x2 = x[ts[:,0] > self.curr_time,:]
        u2 = u[ts[:,0] >= self.curr_time,:]
        pv_sps2 = self.pv_sps[self.pv_sps[:,0] > self.curr_time,1:]

        # Trim to keep the prediction and control horizons
        x3 = x2[0:self.pred_horizon,:]
        pv_sps3 = pv_sps2[0:self.pred_horizon,1]
        u3 = u2[0:self.ctrl_horizon,:]

        # Calculate the cost
        u3_diff = np.diff(u3, axis = 0)
        u3_cost = np.sum(np.multiply(np.sum(np.square(u3_diff),axis = 0),self.mv_wts))
        x3_diff = x3 - pv_sps3
        x3_cost = np.sum(np.multiply(np.sum(np.square(x3_diff),axis = 0),self.pv_wts))
        return u3_cost + x3_cost


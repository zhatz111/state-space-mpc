"""MPC-related class definitions
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

import numpy as np

class Bioreactor:
    """Bioreactor object class for simulation and also processing real off-line data
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-05
        Modified: 2023-10-06
    """

    def __init__(self):
        self.vessel = None # Vessel name for processing multiple bioreactors
        self.model = None # Model (if provided) for simulating a process
        self.data = None # Data for storing simulation results or real data (if provided)
        self.state = None # Current state of the process


class Controller:
    """Controller object class
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-05
        Modified: 2023-10-06
    """

    def __init__(
            self,
            model,
            bioreactor: Bioreactor,
            curr_time: int, # Current culture day
            setpoint: np.array, # A T by 2 array (time and value)
            proc_var: list[str], # Controlled process variable
            proc_var_wt: np.array,
            man_vars: list[str], # Manipulated variables
            man_vars_wt: np.array,
            pred_horizon: int,
            ctrl_horizon: int,
            constr: np.array, # A U by 2 array (lower and upper limits)
    ):
        self.model = model
        self.bioreactor = bioreactor
        self.curr_time = curr_time
        self.setpoint = setpoint
        self.proc_var = proc_var
        self.proc_var_wt = proc_var_wt
        self.man_vars = man_vars
        self.man_vars_wt = man_vars_wt
        self.pred_horizon = pred_horizon
        self.ctrl_horizon = ctrl_horizon
        self.constr = constr

    def obj_func(
            self,
            ts: np.array,
            x: np.array,
            u: np.array):
        """Return the cost value based on x and u"""

        # Trim to keep only future entries
        x2 = x[ts[:,0] > self.curr_time,:]
        u2 = u[ts[:,0] >= self.curr_time,:]
        setpoint2 = self.setpoint[self.setpoint[:,0] > self.curr_time,1:]

        # Trim to keep the prediction and control horizons
        x3 = x2[0:self.pred_horizon,:]
        setpoint3 = setpoint2[0:self.pred_horizon,1]
        u3 = u2[0:self.ctrl_horizon,:]

        # Calculate the cost
        u3_diff = np.diff(u3, axis = 0)
        u3_cost = np.sum(np.multiply(np.sum(np.square(u3_diff),axis = 0),self.man_vars_wt))
        x3_diff = x3 - setpoint3
        x3_cost = np.sum(np.multiply(np.sum(np.square(x3_diff),axis = 0),self.proc_var_wt))
        return u3_cost + x3_cost


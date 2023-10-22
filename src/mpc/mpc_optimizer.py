"""MPC-related class definitions
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import warnings
import math

def daily_to_cumulative_feed(model,u_matrix_daily):
    """Convert a U matrix's daily feed column to cumulative feed for lsim
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-14
        Modified: 2023-10-14
    """
    u_matrix_cumulative = np.copy(u_matrix_daily)
    cumulative_feed_loc = np.where(np.isin(model.inputs,'Cumulative_Normalized_Feed'))[0]
    u_matrix_cumulative[:,cumulative_feed_loc] = np.cumsum(u_matrix_cumulative[:,cumulative_feed_loc]).reshape([-1,1])
    return u_matrix_cumulative

class Bioreactor:
    """Bioreactor object class for simulation and also processing real off-line data
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-05
        Modified: 2023-10-21
    """

    def __init__(
            self,
            vessel = None,
            process_model = None,
            data = pd.DataFrame):
        
        # Update attributes based on user input
        self.vessel = vessel # Vessel name for processing multiple bioreactors
        self.process_model = process_model # Model (if provided) for simulating a process
        # self.data = data # Data for storing simulation results or real data (if provided)
        
        # Check if the data set starts on Day 0
        data = pd.DataFrame.copy(data)
        if data['Day'].values[0] != 0:
            raise Exception('Data set does not start on Day 0!')

        # Initialize other attributes
        self.curr_time = 0
        self.state = data.loc[0,self.process_model.states].values
        self.duration = data['Day'].values[-1]

        # Check if the data set ends on Day duration
        if data.shape[0] - 1 != self.duration:
            raise Exception('Data set has missing or duplicate days!')
        
        # Check if days are consecutive (2023-10-21)
        if any(np.diff(data['Day']) != 1):
            raise Exception('Data set is not in 1-day increments!')
        
        # Convert cumulative feed to daily feed
        if np.isin('Cumulative_Normalized_Feed',data.columns):
            data['Cumulative_Normalized_Feed'] = np.append(np.diff(data.loc[:,'Cumulative_Normalized_Feed']),0)
            self.has_cumulative_feed = True
            warnings.warn('Cumulative feed was converted to daily feed (variable name is unchanged)!')
        else:
            self.has_cumulative_feed = False
        
        # Assign data
        self.data = data
        self.original_data = pd.DataFrame.copy(data)

    def reset(self):
        """Reinitialize the object
            Created by Yu Luo (yu.8.luo@gsk.com)
            Created: 2023-10-14
            Modified: 2023-10-14
        """

        self.data = pd.DataFrame.copy(self.original_data)
        self.curr_time = 0
        self.state = self.data.loc[0,self.process_model.states].values
    
    def show_data(self):
        """Show data with accurate column names
            Created by Yu Luo (yu.8.luo@gsk.com)
            Created: 2023-10-14
            Modified: 2023-10-14
        """
        
        if self.has_cumulative_feed:
            data = pd.DataFrame.copy(self.data)
            data = data.rename(columns={'Cumulative_Normalized_Feed':'Daily_Normalized_Feed'})
        else:
            data = self.data
        
        print(data)

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

    def next_day(self):
        """Advance 24 hours and update the state and current time
            Created by Yu Luo (yu.8.luo@gsk.com)
            Created: 2023-10-14
            Modified: 2023-10-14
        """

        # Get initial state
        initial_state = self.state

        # Get all inputs with daily feed
        u_matrix_daily = self.data.loc[:,self.process_model.inputs].values

        # Convert daily feed to cumulative feed
        u_matrix_cumulative = daily_to_cumulative_feed(self.process_model,u_matrix_daily)
        
        # Filter future inputs
        u_matrix_cumulative = u_matrix_cumulative[self.data['Day'] >= self.curr_time,:]
        
        # Get time array 
        ts = np.arange(u_matrix_cumulative.shape[0])
        
        # Solve
        x_out = self.process_model.ssm_lsim(
            initial_state=initial_state,
            input_matrix=u_matrix_cumulative,
            time=ts
        )

        # Create a DF
        x_out_df = pd.DataFrame(x_out,columns=self.process_model.states)
        x_out_df.insert(0,'Day',ts + self.curr_time)

        # Check if the simulation starts from the current state
        if ~np.all(self.state == x_out[0]):
            raise Exception('Simulation did not start from the current state!')
        
        # Update state and time
        self.state = x_out[1]
        self.curr_time = self.curr_time + 1
        self.data.loc[self.data['Day'] == self.curr_time,self.process_model.states] = self.state

        return x_out_df


class Controller:
    """Controller object class
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-05
        Modified: 2023-10-22
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
        
        # The basics
        self.controller_model = controller_model
        self.bioreactor = bioreactor
        self.curr_time = bioreactor.curr_time
        self.ts = ts
        self.pv_sps = pv_sps
        self.pv_names = pv_names
        self.pv_wts = pv_wts
        self.mv_names = mv_names
        self.mv_wts = mv_wts
        self.pred_horizon = pred_horizon
        self.ctrl_horizon = ctrl_horizon
        self.constr = constr

        # Data snapshots (2023-10-22)
        self.data_before_optim = pd.DataFrame.copy(bioreactor.data)
        self.data_after_optim = pd.DataFrame.copy(bioreactor.data)

        # Plot (2023-10-22)
        cols = 4
        rows = math.ceil(bioreactor.duration / cols)
        figs = []
        fig_axes = []
        for i in range(len(self.pv_names) + len(self.mv_names)):
            fig, axes = plt.subplots(rows, cols, figsize=(9,7), squeeze=False)
            fig.subplots_adjust(top=0.8)
            figs.append(fig)
            fig_axes.append(axes)

        self.figs = figs
        self.fig_axes = fig_axes

    # def est_state(self):
    #     """Estimate the current state based on previous measurement"""
    #     # Gather previous day's state and input
    #     data_prev = self.data.loc[
    #         self.data[:,'Day'] == self.curr_time,:]
    #     x_prev = data_prev[0,self.controller_model.states]
    #     u_prev = data.prev[0,self.controller_model.states]
    #     x_curr = self.controller_model.predict(x_prev,u_prev)

    #     # Replace current missing data with estimated


    def optimize(self,plot=False):
        """Optimize future inputs"""

        # Retrieve MVs from curr_time to EoR
        data = self.bioreactor.data
        self.curr_time = self.bioreactor.curr_time
        is_in_ctrl_horizon = np.logical_and(data['Day'] >= self.curr_time,data['Day'] < (self.curr_time + self.ctrl_horizon))
        mv_matrix = data.loc[is_in_ctrl_horizon,self.mv_names].values

        # Flatten initial mv
        mv_array = mv_matrix.flatten()
        
        # Create constraint matrix
        constr_low_matrix = np.tile(self.constr[:,0],(mv_matrix.shape[0],1))
        constr_low_array = constr_low_matrix.flatten()
        constr_high_matrix = np.tile(self.constr[:,1],(mv_matrix.shape[0],1))
        constr_high_array = constr_high_matrix.flatten()
        bounds = np.vstack((constr_low_array,constr_high_array)).transpose()

        # Simulate before optimization
        _,x_out_before_optim = self.obj_func_wrapper(mv_array)
        data_before_optim = self.data_before_optim
        data_before_optim.loc[data_before_optim['Day'] >= self.curr_time,self.controller_model.states] = x_out_before_optim
        data_before_optim.loc[is_in_ctrl_horizon,self.mv_names] = mv_matrix

        # Solve the optimization problem
        mv_array_star = optimize.minimize(
            fun=lambda x: self.obj_func_wrapper(x)[0],
            x0=mv_array,
            bounds=bounds,
            method="SLSQP",
            options={"disp": False, "maxiter":100}
        )

        # Fold mv to 2D
        mv_matrix_star = mv_array_star.x.reshape([-1,len(self.mv_names)])

        # Simulate after optimization
        _,x_out_after_optim = self.obj_func_wrapper(mv_array_star.x)
        data_after_optim = self.data_after_optim
        data_after_optim.loc[data_after_optim['Day'] >= self.curr_time,self.controller_model.states] = x_out_after_optim
        data_after_optim.loc[is_in_ctrl_horizon,self.mv_names] = mv_matrix_star

        # Plot
        if plot:

            figs = self.figs
            fig_axes = self.fig_axes
            pv_mv_names = np.hstack((self.pv_names,self.mv_names))
            for i in range(len(pv_mv_names)):
                fig = figs[i]
                axes = fig_axes[i]
                ax = axes.reshape(-1)[self.curr_time]
                if i < len(self.pv_names):
                    ax.plot(self.ts,self.pv_sps[:,i],"k--",label="Setpoint")
                    ax.plot(data_before_optim['Day'],data_before_optim[pv_mv_names[i]],"b-",label="Un-optimized")
                    ax.plot(data_after_optim['Day'],data_after_optim[pv_mv_names[i]],"r-",label="Optimized")
                else:
                    ax.step(data_before_optim['Day'],data_before_optim[pv_mv_names[i]],"b-",label="Un-optimized")
                    ax.step(data_after_optim['Day'],data_after_optim[pv_mv_names[i]],"r-",label="Optimized")

                ax.title.set_text("Day " + f"{self.curr_time}")
            
                if (pv_mv_names[i] == 'Cumulative_Normalized_Feed') & self.bioreactor.has_cumulative_feed:
                    fig.suptitle('Daily_Normalized_Feed', size= "x-large", weight= "bold", y=0.98)
                else:
                    fig.suptitle(pv_mv_names[i], size= "x-large", weight= "bold", y=0.98)

                fig.supxlabel("Day", size= "x-large", weight= "bold")
                fig.supylabel("Level", size= "x-large", weight= "bold")
                fig.tight_layout()
                # plt.legend(loc="best")
            

            
            

        # Update the dataset
        data.loc[is_in_ctrl_horizon,self.mv_names] = mv_matrix_star
        self.bioreactor.data = data


    def obj_func_wrapper(self,mv_array):
        """Objective function wrapper"""

        # Rows within the control horizon (2023-10-21)
        ctrl_horizon_where = np.where(np.logical_and(self.bioreactor.data['Day'] >= self.curr_time,self.bioreactor.data['Day'] < (self.curr_time + self.ctrl_horizon)))[0]
        
        # Fold mv_array to a 2D array
        mv_matrix = mv_array.reshape([-1,len(self.mv_names)])

        # Retrieve input from day 0 to EoR
        u_matrix_daily = self.bioreactor.data.loc[:,self.controller_model.inputs].values

        # Replace MVs with mv_matrix
        loc_mv_in_inputs = np.where(np.isin(self.controller_model.inputs,self.mv_names))[0]
        u_matrix_daily_ctrl_horizon = u_matrix_daily[ctrl_horizon_where,:]
        u_matrix_daily_ctrl_horizon[:,loc_mv_in_inputs] = mv_matrix
        u_matrix_daily[ctrl_horizon_where,:] = u_matrix_daily_ctrl_horizon
        u_matrix_cumulative = u_matrix_daily

        # Convert daily feed to cumulative feed
        if self.bioreactor.has_cumulative_feed:
            u_matrix_cumulative = daily_to_cumulative_feed(self.controller_model,u_matrix_daily)
            # cumulative_feed_loc = np.where(np.isin(self.controller_model.inputs,'Cumulative_Normalized_Feed'))[0]
            # u_matrix_cumulative[:,cumulative_feed_loc] = np.cumsum(u_matrix_daily[:,cumulative_feed_loc]).reshape([-1,1])


        # Time array
        ts = np.arange(u_matrix_daily[self.bioreactor.data['Day'] >= self.curr_time,:].shape[0])

        # Sim
        x_out = self.controller_model.ssm_lsim(
            initial_state=self.bioreactor.state,
            input_matrix=u_matrix_cumulative[self.bioreactor.data['Day'] >= self.curr_time,:],
            time=ts
        )

        # Obj
        pv_loc = np.where(np.isin(self.controller_model.states,self.pv_names))[0]
        mv_loc = np.where(np.isin(self.controller_model.inputs,self.mv_names))[0]
        return self.obj_func(ts + self.curr_time,x_out[:,pv_loc],u_matrix_daily[self.bioreactor.data['Day'] >= self.curr_time,:][:,mv_loc]),x_out
        

    def obj_func(
            self,
            ts: np.array,
            x: np.array,
            u: np.array):
        """Return the cost value based on x and u"""

        # Trim to keep only future entries
        x2 = x[ts > self.curr_time,:]
        u2 = u[ts >= self.curr_time,:]
        pv_sps2 = self.pv_sps[self.ts > self.curr_time,:]

        # Trim to keep the prediction and control horizons
        x3 = x2[0:self.pred_horizon,:]
        pv_sps3 = pv_sps2[0:self.pred_horizon,:]
        u3 = u2[0:self.ctrl_horizon,:]

        # Calculate the cost
        u3_diff = np.diff(u3, axis = 0)
        u3_cost = np.sum(np.multiply(np.sum(np.square(u3_diff),axis = 0),self.mv_wts))
        x3_diff = x3 - pv_sps3
        x3_cost = np.sum(np.multiply(np.sum(np.square(x3_diff),axis = 0),self.pv_wts))
        return u3_cost + x3_cost


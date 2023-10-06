"""MPC-related class definitions
    Created by Yu Luo (yu.8.luo@gsk.com)
    Created: 2023-10-05
    Modified: 2023-10-05
"""

class Bioreactor:
    """Bioreactor object class for simulation and also processing real off-line data
        Created by Yu Luo (yu.8.luo@gsk.com)
        Created: 2023-10-05
        Modified: 2023-10-06
    """

    def __init__(self):
        self.vessel = None # Vessel name for processing multiple bioreactors
        self.model = None # Model for simulating a process
        self.data = None # Data for storing simulation results or real data
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
            setpoint,
            pred_horizon,
            ctrl_horizon,
            constr
    ):
        self.model = model
        self.pred_horizon = pred_horizon
        self.ctrl_horizon = ctrl_horizon
        self.obj_func = None
        self.constr = constr

    def add_obj_func(
            self,
            states,
            inputs,
    ):
        """Construct an objective function"""


"""_summary_
"""
import numpy as np

# The StateSpaceModel class represents a mathematical model of a system in state space form.
class StateSpaceModel:
    """_summary_
    """
    def __init__(
            self,
            states: list[str],
            inputs: list[str],
            scaler: object,
            a_matrix: np.ndarray,
            b_matrix: np.ndarray,
            ):
        """
        The function initializes an object with given states, inputs, scaler, a_matrix, and b_matrix
        .
        
        Args:
          states (list[str]): The states parameter is a list of strings representing the possible states of
        a system.
          inputs (list[str]): The `inputs` parameter is a list of strings that represents the possible input
        values for the system. These input values can be used to control or influence the behavior of the
        system.
          scaler (object): The `scaler` parameter is an object that is used to scale the input data. It is
        typically used to normalize or standardize the input values before feeding them into the model. The
        specific implementation of the scaler object will depend on the library or framework being used.
          a_matrix (np.ndarray): The `a_matrix` parameter is a numpy array representing the transition
        probabilities between states in a Markov chain. Each element `a[i][j]` represents the probability of
        transitioning from state `i` to state `j`. The shape of the array should be `(num_states,
        num_states)
          b_matrix (np.ndarray): The `b_matrix` parameter is a numpy array representing the output matrix of
        a system. It defines the relationship between the inputs and the outputs of the system. Each row of
        the matrix corresponds to a state of the system, and each column corresponds to an input.
        """
        self.states = states
        self.inputs = inputs
        self.scaler = scaler
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.c_matrix = np.identity(len(states))
        self.d_matrix = np.zeros([len(states), len(inputs)])

    def predict(self,x,u):
        """Predict a trajectory
          Created by Yu Luo (yu.8.luo@gsk.com)
          Created: 2023-10-11
          Modified: 2023-10-11          
        """

        if u.shape[0] == 1:

          # Transform x and u to row arrays
          x_row = x.reshape(1,-1)
          u_row = u.reshape(1,-1)

          # Concatenate x and u and scale them 
          xu_row = np.hstack((x_row,u_row))
          xu_scaled = self.scaler.transform(xu_row)
          x_scaled = xu_scaled[:,0:x_row.shape[1]].reshape(-1,1)
          u_scaled = xu_scaled[:,x_row.shape[1]:].reshape(-1,1)

          # Predict based on A*x + B*u
          xHat_scaled = np.dot(self.a_matrix,x_scaled) + np.dot(self.b_matrix,u_scaled)
          xuHat_scaled = np.vstack((xHat_scaled,np.zeros((u_row.shape[1],1))))
          xHat = self.scaler.inverse_transform(xuHat_scaled.reshape(1,-1))[:,:x_row.shape[1]]
          
          # Return as a row array
          return xHat.reshape(1,-1)          
        else:

          # Recursively build the prediction results
          xHat = self.predict(x,u[0:1,:])
          xHats = np.vstack((xHat,self.predict(xHat,u[1:,:])))
          return xHats

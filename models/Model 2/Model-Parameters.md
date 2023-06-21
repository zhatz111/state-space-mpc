# AR23-014 Model

## **Description** 

This is a summary of all the model parameters needed to model the data from AR23-014. The raw data used in this model can be found in ***(path to data)***. This experiment was based on a DoE with four factors; pH, feed volume, temperature shift day, and iVCC. The purpose was to determine setpoints that allow us to mimic the Wuxi process in the ambr250 bioreactors. The other goal was to optimize the process if possible for an increase in titer. A state space model was developed to determine the optimal setpoints to achieve this.

# Model Parameters

## **Data Cleaning and Scaling**

### *States*
    1. VCC
    2. Lactate
    3. IGG

### *Inputs*
    1. Daily_Feed_Normalized
    2. Daily_Glucose_Normalized
    3. pH_Setpoint
    4. Temperature

### *Savgol Filtered States*
    1. States: VCC, Lactate, IGG
    2. Window Length: 7
    3. Polyorder: 2

### *Scaling Parameters Table & Equations*

| State/Input              |      scale_ |         min_ |
|:-------------------------|------------:|-------------:|
| VCC                      |  0.0318134  |   0.021012   |
| Lactate                  |  0.46589    |   0.0493622  |
| IGG                      |  0.00020022 |  -0.00538272 |
| Daily_Feed_Normalized    | 28.3235     |   0          |
| Daily_Glucose_Normalized | 92.4286     |   0          |
| pH_Setpoint              |  5          | -34.75       |
| Temperature              |  0.181818   |  -5.63636    |

[AR23-014 Scaler File](https://mygithub.gsk.com/zah48132/state-space-model/blob/main/models/Model%202/scaler_train_AR23-014.scale)

$$\begin{align*}
{\rm Scale} &= \max_{} - \min_{} \\
X_{\rm scale} &= {X}\cdot{\rm Scale} + \min_{}\\
X_{\rm rescale} &= \frac{X - \min_{}}{\rm Scale}\\
\end{align*}$$

### Train, Test, Split Parameters:
- Package: Scipy.model_selection.GroupShuffleSplit
- test_size: 0.15 (Percentage of data used in the test set)
- n_splits: 2 (Number of re-shuffling & splitting iterations)
- random_state: 1 (Controls the randomness of the training and testing indices produced. Pass an int for reproducible output across multiple function calls.)

## **Model Training Parameters**

---

### A-Matrix:

|        VCC |    Lactate |          IGG |
|:----------:|:----------:|:------------:|
| -0.191333  |  0.348639  | -0.0091029   |
| -0.0142399 | -0.575035  |  0.0229528   |
|  0.0397055 | -0.0431091 |  0.000589102 |

### B-Matrix:

|         Daily_Feed_Normalized |         Daily_Glucose_Normalized |           pH_Setpoint |           Temperature |
|:---------:|:---------:|:-----------:|:-----------:|
| -0.1863   |  0.495116 | -0.0304475  | -0.0425543  |
|  0.393914 | -0.395079 |  0.0403181  |  0.385505   |
|  0.124161 |  0.110922 |  0.00110679 | -0.00859193 |

## **Model Optimization Parameters**

---

### Starting Inputs:
    - VCC: 0.0215119
    - Lactate: 0.0246256
    - IGG: 0.00740393
    - Volume: 200

### Starting Setpoints:
    - Daily Feed %: 3.6
    - pH: 7.15
    - Start Temp: 36.5
    - Shift Temp: 31
    - Temp Shift Day: 5

### Setpoint Bounds:

| Bound | Lower Limit | Upper Limit |
|:-----:|:-----------:|:-----------:|
|Daily Feed %| 2 | 4 |
| pH | 6.90 | 7.20 |
|Start Temp| 36 | 37 |
|Final Temp| 30 | 32 |
|Temp Shift Day| 4 | 6 |

### Glucose Input:

|  Day  |        Scaled Daily Glucose |
|---:|---------:|
|  0 | 0        |
|  1 | 0        |
|  2 | 0        |
|  3 | 0.43805  |
|  4 | 0        |
|  5 | 0        |
|  6 | 0.419918 |
|  7 | 0        |
|  8 | 0        |
|  9 | 0.887389 |
| 10 | 0        |
| 11 | 0.549897 |
| 12 | 0.414272 |
| 13 | 0.4046   |
| 14 | 0        |

### Optimizer Function with B Matrix Construction:
```python
    def optimizer_function(self, input_array):
        # Construct the B Matrix based on the input array parameters
        x0 = np.zeros([self.days, self.input_len])

        # Construct feed day matrix column based on daily feed % for the exact feed days
        for day in [3,5,7,10,12]:
            x0[day,0] = ((input_array[0]/100)*self.scaler_dict["Daily_Feed_Normalized"][0])+self.scaler_dict["Daily_Feed_Normalized"][1]

        # Construct glucose feed column based on the input given from batch 004
        x0[:,1] = self.glucose.ravel()

        # Construct pH setpoint column, the pH setpoint was locked at 7.15 for the purposes of the optimization condition
        x0[:,2] = (7.15*self.scaler_dict["pH_Setpoint"][0])+self.scaler_dict["pH_Setpoint"][1]

        # Construct temperature column based on starting temp, final temp and the shift day
        shift_day = int(input_array[4])
        x0[:shift_day,3] = (input_array[2]*self.scaler_dict["Temperature"][0])+self.scaler_dict["Temperature"][1]
        x0[shift_day:,3] = (input_array[3]*self.scaler_dict["Temperature"][0])+self.scaler_dict["Temperature"][1]

        # Use the constructed B matrix and solve the state space equation for the timecourse data
        c_matrix = np.identity(self.state_len)
        d_matrix = np.zeros([self.state_len, self.input_len])
        u_sim = x0
        t_sim = np.arange(0, self.days, 1)
        state = signal.StateSpace(self.a_matrix, self.b_matrix, c_matrix, d_matrix)
        _, y_func, _ = signal.lsim(state, u_sim, t_sim, self.initial_condition)
        return y_func, u_sim
```





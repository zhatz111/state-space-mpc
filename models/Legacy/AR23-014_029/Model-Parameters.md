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
| VCC                      |  0.03153  |   0.029722   |
| Lactate                  |  0.469169    |   0.042672  |
| IGG                      |  0.000183 |  -0.000958 |
| Daily_Feed_Normalized    | 28.32353     |   0          |
| Daily_Glucose_Normalized | 101.199999     |   0          |
| pH_Setpoint              |  5.0          | -34.75       |
| Temperature              |  0.181818   |  -5.636364    |

$$\begin{align*}
{\rm Scale} &= \max_{} - \min_{} \\
X_{\rm scale} &= {X}\cdot{\rm Scale} + \min_{}\\
X_{\rm rescale} &= \frac{X - \min_{}}{\rm Scale}\\
\end{align*}$$

[AR23-014_029 Scaler File](https://mygithub.gsk.com/zah48132/state-space-model/blob/main/models/AR23-014_029/scaler_train_AR23-014_029.scale)

### Train, Test, Split Parameters:
    - Package: Scipy.model_selection.GroupShuffleSplit
    - test_size: 0.20 (Percentage of data used in the test set)
    - n_splits: 2 (Number of re-shuffling & splitting iterations)
    - random_state: 1 (Controls the randomness of the training and testing indices produced. Pass an int for reproducible output across multiple function calls.)

## **Model Training Parameters**

---

### A-Matrix:

|        VCC |    Lactate |          IGG |
|:----------:|:----------:|:------------:|
| -0.243660822  |  0.341488769  | -0.00409376   |
| 0.010977743 | -0.742058541  |  -0.001635321   |
|  0.084184584 | -0.103966116 |  -0.001922327 |

### B-Matrix:

|         Daily_Feed_Normalized |         Daily_Glucose_Normalized |           pH_Setpoint |           Temperature |
|:---------:|:---------:|:-----------:|:-----------:|
| 0.039525947   |  0.320843897 | -0.010934757  | -0.07233171  |
|  0.250424565 | -0.171740803 |  0.050595812  |  0.464156744   |
|  0.091767332 |  0.04303196 |  0.004423241 | 0.027004704 |

## **Model Optimization Parameters**

---

### Starting Inputs:
>   - *VCC*: 0.02496601
>   - Lactate: 0.14186774
>   - IGG: 0.00387
>   - Volume: 200

### Starting Setpoints:
>    - Feed Polynomial: [A: -0.0005418, B: 0.0004637, C: 0.1019456, D: -0.0434798]
>      - Equation: Ax<sup>3</sup> + Bx<sup>2</sup> + Cx + D
>    - pH: 7.05
>    - Start Temp: 36.5
>    - Shift Temp: 31.0
>    - Temp Shift Day: 5

### Setpoint Bounds:

| Bound | Lower Limit | Upper Limit |
|:-----:|:-----------:|:-----------:|
|Feed Polynomial| None | None |
| pH | 6.90 | 7.20 |
|Start Temp| 36 | 37 |
|Final Temp| 30.5 | 31.5 |
|Temp Shift Day| 4 | 6 |

### Glucose Input:

|  Day  |        Scaled Daily Glucose |
|---:|---------:|
|  0 | 0        |
|  1 | 0        |
|  2 | 0.36755172 |
|  3 | 0  |
|  4 | 0.55703941        |
|  5 | 0.71653303        |
|  6 | 0.35160351 |
|  7 | 0        |
|  8 | 0.44276143        |
|  9 | 0 |
| 10 | 0.56491044        |
| 11 | 0 |
| 12 | 0.66025602 |
| 13 | 0.61618868   |
| 14 | 0        |

### Optimizer Function with B Matrix Construction:
```python
    def optimizer_function(self, input_array):
        # Change the start of this function according to what your input array will be
        x0 = np.zeros([self.days, self.input_len])
        # Construct feed day matrix column
        for day in range(0,14):
            x0[day,0] = np.polyval(input_array[0:4],day)
        # if 10 < self.iterations < 20:
        #     print(x0[:,0])
            # x0[day,0] = ((input_array[0]/100)*self.scaler_dict["Daily_Feed_Normalized"][0])+self.scaler_dict["Daily_Feed_Normalized"][1]

        # Construct glucose feed column
        x0[:,1] = self.glucose.ravel()
        # construct pH setpoint column
        x0[:,2] = (input_array[4]*self.scaler_dict["pH_Setpoint"][0])+self.scaler_dict["pH_Setpoint"][1]
        # Construct temperature column
        shift_day = int(input_array[7])
        x0[:shift_day,3] = (input_array[5]*self.scaler_dict["Temperature"][0])+self.scaler_dict["Temperature"][1]
        x0[shift_day:,3] = (input_array[6]*self.scaler_dict["Temperature"][0])+self.scaler_dict["Temperature"][1]
        # if self.iterations == 0:
        #     print(x0)
        c_matrix = np.identity(self.state_len)
        d_matrix = np.zeros([self.state_len, self.input_len])
        u_sim = x0
        t_sim = np.arange(0, self.days, 1)
        state = signal.StateSpace(self.a_matrix, self.b_matrix, c_matrix, d_matrix)
        _, y_func, _ = signal.lsim(state, u_sim, t_sim, self.initial_condition)
        return y_func, u_sim
```





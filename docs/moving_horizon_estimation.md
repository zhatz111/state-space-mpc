# MHE-Based Model Updates for Bioreactor MPC

- Model
$$\begin{align}
\notag \frac{{\rm d}{\mathbf x}}{{\rm d} t} &=  A{\mathbf x} + B{\mathbf u}\\\\
\notag {\mathbf y} &= {\rm diag}({\mathbf p}){\mathbf x}
\end{align}$$

	- State vector: ${\mathbf x}$
	- Output vector: ${\mathbf y}$
	- Model parameters: $A$, $B$
	- Prediction modifier: ${\mathbf p}$

- Model prediction

$$\begin{align}
\notag \hat{{\mathbf x}}(t_k) &= {\mathbf f}({\mathbf x}(t_0),{\mathbf u}(t_0,\ldots,t_{k-1});A,B)\\\\
\notag \hat{{\mathbf y}}(t_k) &= {\rm diag}({\mathbf p})\hat{{\mathbf x}}(t_k)
\end{align}$$

- Initialize on Day 0 and predict the entire $T$-day trajectory
$$\begin{align}
\notag {\mathbf p} &= {\mathbf 1} \\\\
\notag \hat{{\mathbf x}}(t_0) &= {\mathbf y}(t_0) \\\\
\notag \hat{{\mathbf x}}(t_j) &= {\mathbf f}({\mathbf x}(t_0),{\mathbf u}(t_0,\ldots,t_{T-1});A,B)\quad (j = 1,\ldots,T)\\\\
\notag \hat{{\mathbf y}}(t_j) &= \hat{{\mathbf x}}(t_j) \quad (j = 0,\ldots,T)
\end{align}$$

	- Write ${\mathbf p}$ as `STATE_MOD`
	- Write $\hat{{\mathbf x}}(t_{0})$ and $\hat{{\mathbf x}}(t_{1})$ as `STATE_EST`
	- Write $\hat{{\mathbf y}}(t_{0}),\ldots,\hat{{\mathbf y}}(t_{T})$ as `STATE_PRED`

-  Current time: $t_k$
$$J({\mathbf p}) = \sum_{i = k - N +1}^{k}\Big(\Vert \hat{{\mathbf y}}(t_i) - {\mathbf y}(t_i)  \Vert_{W_y}^2\Big) + 
\Vert {\mathbf p} - {\mathbf 1} \Vert^2\cdot w_p
$$ 

	-  Cost function $J$
	-  Horizon length: $N$ or $k$ (whichever is smaller)
	-  Initial condition: $\hat{{\mathbf x}}(t_{k - N})$ or $y_j(t_{k - N})/p_j$ if the $j$th measurement is available
	-  Diagonal weight matrix of measurement deviation: $W_y = {\rm diag}({\mathbf w}_y)$
	-  Scalar weight of prediction modification: $w_p$

- Solve $\min_{\mathbf p} J({\mathbf p})$ to update the prediction modifier ${\mathbf p}$
	- Write ${\mathbf p}$ as `STATE_MOD`
- Estimate next-day's state: $\hat{{\mathbf x}}(t_{k+1})$
	- Write $\hat{{\mathbf x}}(t_{k+1})$ as `STATE_EST`
- Use MPC to update future control actions based on the model and the updated modifier: ${\mathbf u}(t_{k+1},\ldots,t_{T-1})$
- Update the predicted trajectory
$$\begin{align}
\notag \hat{{\mathbf x}}(t_j) &= {\mathbf f}({\mathbf x}(t_{k+1}),{\mathbf u}(t_{k+1},\ldots,t_{j-1});A,B)\quad (j = k+2,\ldots,T)\\\\
\notag \hat{{\mathbf y}}(t_j) &= \hat{{\mathbf x}}(t_j) \quad (j = k+1,\ldots,T)
\end{align}$$

	- Write $\hat{{\mathbf y}}(t_{k+1}),\ldots,\hat{{\mathbf y}}(t_{T})$ as `STATE_PRED`
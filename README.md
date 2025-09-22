# CBF-pythonic
Python implementation for generating text files in the CBF-format for Mixed-Integer Semidefinite Programs (MISDPs) using pythonic operators.

Important note: The CBF-format supports other cones (such as the exponential cone and the quadratic), but they are not implemented here!

The type of problems suitable can be stated as 

$$
\begin{aligned}
    \min \quad & \sum_{j \in J} \langle F_j^{obj}, X_j \rangle + \sum_{\ell \in L} a^{obj}_{\ell} x_{\ell} + b^{obj} \\
    \text{s.t.} \quad & \sum_{j \in J} \langle F_j^i, X_j \rangle + \sum_{\ell \in L} a^i_{\ell} x_{\ell} + b^i \in \mathcal{K}^i, & i = 1, \dots, m_0, \\
    & \sum_{\ell \in L} H_{\ell}^i x_{\ell} + D^i \succeq 0, & i = 1, \dots, m_1, \\
    & X_j \succeq 0, \quad \forall j \in J, \\
    & x_{\ell} \in \mathbb{Z}, \quad \forall \ell \in L_{\mathcal{I}} \subseteq L,
\end{aligned}
$$

where

$$
\begin{aligned}
     \mathcal{K} & = \text{ either "} \geq 0 \text{", "} = 0 \text{", or "} \leq 0 \text{"} \\
     J & = \text{ set of PSD matrix variables} \\
     L & = \text{ set of non-matrix variables} \\
     L_{\mathcal{I}} & = \text{ index set for integer variables} \\
     \\
     m_0 & = \text{ number of linear constraints} \\
     m_1 & = \text{ number of PSD constraints} \\
     \\
     X_j & = \text{ PSD matrix variable of size } n_j \times n_j \\
     F_j & = \text{ symmetric matrix of size } n_j \times n_j \\
     F_j^i & = \text{ symmetric matrix of size } n_j \times n_j \\
     H_{\ell}^i & = \text{ symmetric matrix of size } n_i \times n_i \\
     D^i & = \text{ symmetric matrix of size } n_i \times n_i \\
     a^{obj}, a^i & = \text{ vectors of length } |L| \\
     b^{obj}, b^i & = \text{ real numbers}
\end{aligned}
$$

Problems can be stated in both primal and dual form, i.e., it allows for both PSD-constrained matrix variables as in line 4, and linear PSD-constraints as in line 3. Note that while a problem can contain both PSD-variables and PSD-constraints the (primal) PSD matrix variables cannot be involved in the (dual) PSD constraints.

## The CBF-format
The CBF-format is developed by the [The Conic Benchmark Library](https://cblib.zib.de/). 

This implementation is based on version 2.

## Features

- Generate CBF-format text files compatible with MISDP solvers
- Define semidefinite programs with mixed-integer variables using intuitive Python syntax

## Installation

Simply download the `CBF_pythonic.py` file and place it in your project directory or somewhere in your Python path.

## Usage

Here is a basic example of how to use CBF-pythonic:

```python
# from CBF_pythonic import Model 
M = Model()  
 
# addVars(n, vtype = "C", lb=-np.inf, ub=np.inf)  
x = M.addvars(3, vtype = ["C", "I", "B"], lb = [0,0,0], ub = [2,2,1]) 

# addPSDVar(n): also adds primal PSD-constraint
# To add 'vtype', 'lb', and 'ub' to PSD-variables, see Section ref{sec: addpsdvar} 
X1 = M.addPSDVar(4) 
X2 = M.addPSDVar(2) 

# Linear expressions are one-dimensional 
# LinExpr: sum <F_j, X_j> + sum a_j * x_j + b 
F1 = np.eye(4) # numpy array of size (n,n) 
F2 = np.ones((2,2)) # must be symmetric or lower triangular 
a = np.ones(3) # numpy array of size n 
lin_expr = F1*X1 + F2*X2 + a@x + 7 # NOTE: @ as vector operator 

# addConstraint() 
lin_con = M.addConstraint(lin_expr <= 0) # '<=', '>=', '==' 

# addPSDConstraint(): dual PSD-constraint 
H0 = np.eye(3) # numpy array (symmetric or lower triangular) 
H1 = np.ones((3,3)) # all matrices in one constraint must be same size 
D = np.array([[1,3],[3,2]]) 
mat_expr = D + H0*x[0] + H1*x[1] 
psd_con = M.addPSDConstraint(mat_expr) 

# addObjective(sense, LinExpr) 
lin_expr = np.array([7,3]) @ x[[0,4]] # you can slice variables 
objective = M.addObjective("MIN", lin_expr) # 'MIN' or 'MAX' 

M.writeCBF(file_name) # Write model to file 
```

See the [full documentation](./CBF_documentation.pdf) for detailed usage and examples.

## Examples

This repository includes two example implementation files demonstrating how to use the library:

- `isometry_example.py` — Demonstrates the primal formulation
- `random_example.py` — Demonstrates the dual formulation

## Requirements

- Python 3.7+
- numpy

## Contributing

Contributions are welcome! Please open issues or submit pull requests on GitHub.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

Created by Johanna Skåntorp - [skanto @ kth.se](mailto:skanto@kth.se)
Report any issues via email

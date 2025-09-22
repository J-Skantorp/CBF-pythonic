# CBF-pythonic
Python library for generating text files in the CBF-format for Mixed-Integer Semidefinite Programs (MISDPs) using pythonic operators.

## Features

- Generate CBF-format text files compatible with MISDP solvers
- Define semidefinite programs with mixed-integer variables using intuitive Python syntax

## Installation

Simply download the `CBF_pythonic.py` file and place it in your project directory or somewhere in your Python path.

## Usage

Here is a basic example of how to use pyCBF:

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

For more detailed examples, see the [Documentation](link-to-docs-if-any).

## Requirements

- Python 3.7+
- numpy

## Contributing

Contributions are welcome! Please open issues or submit pull requests on GitHub.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Contact

Created by Johanna Skåntorp - [skanto @ kth.se](mailto:skanto@kth.se)

Report issues at: https://github.com/yourusername/CBF-pythonic/issues

"""
Example usage of CBF_pythonic

Created on Wed Apr 30 10:01:39 2025
@author: johannaskantorp

We create instances of the Isometry-problem, and write it to CBF-file: "temp_cbf.cbf"

  min trace(C, X)
  st  trace(X) = 1
      sum_(j = 1,..,n) z_j <= kappa
      -0.5 * z_j <= X_ij <= 0.5 * z_j,   i = 1,...,n, j = 1,...,n
      X > 0 (PSD)
      X /in R^(n x n),  z /in {0,1}^n,

where C = A^T * A, where A /in R^(m x n) is randomly generated (and m = 10)


"""


from CBF_pythonic import Model
import numpy as np



def _isometry_make_C(n, seed, m=10):
    # Construct n x n matrix C = A^T * A
    np.random.seed(41+seed)
    A = np.random.normal(0, 1, size=(m, n))
    return np.matmul(A.T, A)


def _isometry_make_E(i, j, n):
    # Construct n x n matrix E_ij = 0.5 * (e_i * e_j^T + e_j * e_i^T)
    E = np.zeros((n, n))
    if i == j:
        E[i, i] = 1
    else:
        E[i, j] = 0.5
        E[j, i] = 0.5
    return E


def generate_isometry(n, kappa, file_name="temp_cbf.cbf", seed=0):
    M = Model()
    C = _isometry_make_C(n, seed) # Generate nxn symmetric matrix: C
    X = M.addPSDVar(n) # Add nxn PSD-variable (and constraint): X > 0
    z = M.addVars(n, vtype="B") # Add n variables: z

    obj_expr = C*X # construct linear expression: <C,X>
    M.addObjective("min", obj_expr) # Add objective: min  <C,X>

    I = np.eye(n)
    lin_expr = I*X # construct linear expression: <I,X> = <X>
    M.addConstraint(lin_expr == 1) # Add constraint: <X> == 1

    ones = np.ones(n)
    lin_expr = ones@z # construct linear expression 1^T z
    M.addConstraint(lin_expr <= kappa) # Add constraint sum(z_j) <= kappa

    # Add constraint -0.5 * z_j <= X_ij <= 0.5 * z_j, i = 1,...,n, j = 1,...,n
    for j in range(n):
        z_j = z[j] # SingleVar z_j
        for i in range(n):
            E = _isometry_make_E(i, j, n) # Construct E_ij, s.t. X_ij = <E_ij, X>
            M.addConstraint(E*X <= 0.5*z_j) # Add constraint: X_ij <= 0.5 * z_j
            M.addConstraint(E*X >= -0.5*z_j) # Add constraint: X_ij >= - 0.5 * z_j

    M.writeCBF(file_name)


def main():
    file_name = "temp_cbf.cbf"
    n = 10
    kappa = 3
    generate_isometry(n, kappa, file_name)



if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:01:39 2025

@author: johannaskantorp

We create instances of a random problem in the dual form, and write it to CBF-file: "temp_cbf.cbf"

  min  a^T * y + b^T z
  st   sum H_j * y_j + sum G_j + z_j + D > 0  (PSD-constraint)
       y /in [0,1]^mc
       z /in {0,1}^mb 

where D is chosen s.t. the problem has at least one feasible solution
H_i, G_j and D symmetric matrices of dimension (n x n)

"""


from CBF_pythonic import Model
import numpy as np
import scipy.linalg as scla



def _random_make_matrix(n, seed):
    # Generates random symmetric matrix X of size (n x n)
    # and element c = trace(X)
    np.random.seed(41+seed)
    X = np.zeros((n, n))
    X[np.tril_indices(n)] = np.round(np.random.uniform(
        low=-1.0, high=1.0, size=(1, int(0.5*n*(n+1)))), 3)


    return X + np.transpose(np.tril(X, -1)), np.round(sum(X[np.diag_indices(n)]), 3).item()



def _random_make_parameters(n, mc, mb, seed = 0):
    """
    Constructs randomized parameters a, b, H_j, G_j, and D s.t. the MISDP-instance has a feasible solution.
    
    Args:
        n (int): size (n x n) of PSD-constraint
        mc (int): number of continuous variables [0,1]
        mb (int): number of binary variables {0,1}
        seed (int): optional seed for random generator

    return a, b, H, G, D
        a (ndarray): numpy array of length mc, list elements are scalars
        b (ndarray): numpy array of length mb, list elements are scalars
        H (list): list of length mc, list elements are matrices H_j: (n x n) numpy array
        G (list): list of length mb, list elements are matrices G_j: (n x n) numpy array
        D (ndarray): (n x n) numpy array: matrix D
    """

    H, G, a, b = [], [], [], []


    for i in range(mc):
        # Make H_j matrices, and a for the objective function
        np.random.seed(41+seed)
        H_j, a_j = _random_make_matrix(n, seed)
        a.append(a_j)
        H.append(H_j)
        seed += 13


    for i in range(mb):
        # Make G_j matrices, and b for the objective function
        np.random.seed(41+seed)
        G_j, b_j = _random_make_matrix(n, seed)
        b.append(b_j)
        G.append(G_j)
        seed += 13

    # Create D s.t. at least one solution exists
    # Done by generating one solution y, z
    # Choosing D s.t. sum H_j * y_j + sum G_j + z_j + D > 0,
    # i.e., s.t D > - (sum H_j * y_j + sum G_j + z_j)

    y = np.random.rand(1, mc)[0] #create random cont. y-vector
    z = np.random.randint(2, size=mb) #create random binary z-vector

    D = np.zeros((n, n))
    for j, H_j in enumerate(H):
        D += H_j*y[j]

    for j, G_j in enumerate(G):
        D += G_j*z[j]

    D += -scla.eigh(D, lower=True, eigvals_only=True,
                    subset_by_index=[0, 0])[0]*np.eye(n)
    

    return np.array(a), np.array(b), H, G, D





def main():
    file_name="temp_cbf.cbf" # File name should end in .cbf
    n, mc, mb = 5, 3, 3

    
    M = Model() # Initialize model
    a, b, H, G, D = _random_make_parameters(n, mc, mb) # Make parameters for problem
    
    y = M.addVars(mc, vtype="C", lb=0, ub=1) # Add mc continuous variables: y /in [0,1]^mc
    z = M.addVars(mb, vtype="B") # Add mb binary variables: z /in {0,1}^mb

    M.addObjective("min", a@y + b@z) # Add objective: min  a^T * y + b^T z
    
    mat_expr = sum(H[j]*y[j] for j in range(mc)) + sum(G[j]*z[j] for j in range(mb)) + D # Construct matrix-expression: sum H_j * y_j + sum G_j + z_j + D
    M.addPSDConstraint(mat_expr) # Add PSD-constraint: mat_expr > 0  (dual PSD-constraint)

    M.writeCBF(file_name) # Write to file



if __name__ == "__main__":
    main()

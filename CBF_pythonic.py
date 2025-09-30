#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:08:59 2024
@author: johannaskantorp

This is a script to provide a pythonic way of writing MISDPs in the cbf-format.

Module: CBF_pythonic.py

- Model

- LinExpr 
- MatExpr
- BoundConstraint
- LinConstraint
- PSDConstraint
- MatCoord
- SingleVar
- Variable
- PSDVar
- Objective
"""


"""
# The CBF-format:
# Commented out sections are not in the final file, they are comments
# Often # is used to clarify *type* (such as int/float) of below input
# All indents/added spaces are for readability, and are not in the final file
# Everything after a colon (:) is a comment, and is not in the final file


# File format
VER
2

   - - -
   
# Problem structure
OBJSENSE
# str
  MIN    : or MAX

   - - -
   
# PSD-variables
PSDVAR
# int
   J   : number of PSD matrices
# int
  n_1  : size n of PSD-var 1
  .
  .
  n_j  : size n of PSD-var j
  .
  .
  n_J  : sixe n of PSD-var J

   - - -
   
# Non-matrix variables
VAR
# int   int
  N     K    : number of variables (N), and number of conic domains they are restricted to (K)
# str   int
  c_1   n_1  : name "c" of cone 1 and number of variables n constricted to this cone
  .
  .
  c_k   n_k  : name "c" of cone k and number of variables n constricted to this cone
  .
  .
  c_K   n_K  : name "c" of cone K and number of variables n constricted to this cone
# Note: sum_k n_k = N

# Current implementation adds linear constraints to represent L+, L- and L=, i.e., only use cone F. So this will always look like
# VAR
# N 1 : N vars constricted to 1 cone
# F N : All N vars constricted to F-cone (i.e., R)

   - - -
   
# Integer variables
INT
# int
  N    : number of integer variables (non-matrix)
# int
  j_1  : index j of integer variable 1
  .
  .
  j_n  : index j of integer variable n
  .
  .
  j_N  : index j of integer variable N

   - - -
   
PSDCON, (PSD constraints)
# int
  J    : number of PSD constraints in problem
# int
  n_1  : size n of matrices involved in PSD constraint 1
  .
  .
  n_j  : size n of matrices involved in PSD constraint j
  .
  .
  n_J  : size n of matrices involved in PSD constraint J

   - - -
   
CON, (affine constraints)
# int  int
  N    K      : number of scalar constraints (N) and number of cones they constrict to (K)
# str  int
  c_1  n_1    : name "c" of cone 1 and number of constraints n constricted to this cone
  .
  .
  c_k  n_k    : name "c" of cone k and number of constraints n constricted to this cone
  .
  .
  c_K  n_K    : name "c" of cone K and number of constraints n constricted to this cone
# Note: sum_k n_k = N
# Since all variable bounds are enforced here, if you add e.g., M constraints, L lower bounds, and U upper bounds, then N = M+L+U 

   - - -
   
# Problem data below

   - - -
   
# Objective function:
# sum <OF[j], X[j]> + sum oa[j]*x[j] + ob
#   j   = variable index (either matrix or single variable)
# (k,l) = matrix element index

OBJFCOORD
# int
  N    : number of coordinates to be specified
# int int int float
  j   k   l   OF[j][k,l]  # Note: k >= l
  .
  .
# Total of N lines

OBJACOORD
# int
  N   : number of coordinates to be specified
# int  float
  j    oa[j]
  .
  .
# Total of N lines

OBJBCOORD
# float
  ob

   - - -
   
# Affine constraints:  
# sum <F[i,j],X[j]>  +  sum  a[i,j]*x[j] + b[i] /in cone C, for all i
# the cone C is derives from information in 'CON'
#   i   = constraint index  (number of constraints comes from 'CON')
#   j   = variable index (either matrix or single variable)
# (k,l) = matrix element index


FCOORD
# int
  N   : number of coordinates to be specified
# int int int int float
  i   j   k   l   F[i,j][k,l]  # Note: k >= l
  .
  .
# Total of N lines


ACOORD
# int
  N   : number of coordinates to be specified
# int int  float
  i   j    a[i,j]
  .
  .
# Total of N lines


BCOORD
# int
  N  : number of coordinates to be specified
# int  float
  i    b[i]
  .
  .
# Total of N lines

   - - -

# PSD-constraint:
# sum H[i,j]*x[j] + D[i] /in PSD-cone, for all i
#   i   = constraint index  (number of constraints comes from 'PSDCON')
#   j   = variable index (either matrix or single variable)
# (k,l) = matrix element index


HCOORD
# int
  N   : number of coordinates to be specified
# int int int int  float
  i   j   k   l    H[i,j][k,l]  # Note: k >= l
  .
  .
# Total of N lines


DCOORD
# int
  N   : number of coordinates to be specified
# int int int   float
   i   k   l    D[i][k,l]  # Note: k >= l
   .
   .
# Total of N lines

   - - -
   
CHANGE
# No implementation

   - - -
# END OF FILE


Misc:

relevant cones
 F:    x in Rn
 L+:   x > 0
 L-:   x < 0
 L=:   x = 0


Short note on variable bounds:
    The CBF-format can handle variable bounds separately only if x /in Cone (see list above of relevant cones),
    but we still want to give the user the option to put in variable bounds. 
    I have handled this by not utilizing the conic varible constraint at all -- all variables are in F (i.e., R),
    and adding *all* variable bounds as affine constraints.
    
    For 'readability' the bound constraints are put last in the CON-section,
    and I write out the number of bound constraints as a comment
    
    I thought about fully grouping the constraints based on cone, but settled on writing them out in "input-order"

Note on indices
    Note: index j corresponding to Variable or SingleVar instance is not the same as index j corresponding to PSDVar instance
    In the accompanying PDF I use the (more clear) notation:
        'sum H[i,l]*x[l] + D[i]', and 'sum <F[i,j],X[j]>  +  sum  a[i,l]*x[l] + b[i]'
    But I'm running out of indices :)

"""


import numpy as np


def add(a, b):
    return a + b


class Model:
    """
    A class for constructing MISDP optimization models using Python operators, and writing them to a cbf-file.
    
    Public methods:
        addVar(vtype, lb, ub)
        addVars(n, vtype, lb, ub)
        addPSDvar(n)
        addConstraint(LinConstraint)
        addPSDConstraint(MatExpr)
        addObjective(objectivesense, LinExpr)
        copy()
        writeCBF(file_name)
    
        
    Internal methods (starting with _): Mostly helper functions to write the cbf-file


    Attributes:
        num_of_vars (int): Counter to track the number of variables added to the model.
        PSD_vars (list): List to store positive semidefinite (PSD) matrix variables (elements are of class PSDVar)
        int_list (list): List of indices of the integer variables (elements are ints)
        aff_constraints (list): List of affine (linear equality or inequality) constraints  (elements are of class LinConstraint)
        lb_constraints (list): List of variable bound constraints (e.g., x >= 0) (elements are of class BoundConstraint)
        ub_constraints (list): List of variable bound constraints (e.g., x <= 1) (elements are of class BoundConstraint)
        PSD_constraints (list): List of PSD matrix constraints (e.g., X >= 0 in PSD sense) (elements are of class PSDConstraint)
        objective (LinExpr or False): The objective function of the model (Objective class instance)
    """

    def __init__(self):
        
        self.num_of_vars = 0
        self.PSD_vars = []
        self.int_list = []
        
        self.lb_constraints = []
        self.ub_constraints = []

        self.aff_constraints = []

        self.PSD_constraints = []
        self.objective = False # Currently cannot handle no objective, TODO: fix this?

    @property
    def print(self):
        print(self.output_string())


    def addVar(self, vtype="C", lb=-np.inf, ub=np.inf):
        """
        Add new variable to the model.

        Args:
            vtype (str): Variable type, default is "C" (continuous). Other:
                         "I" for integer, "B" for binary
            lb (float): Lower bound of the variable. Defaults to -infinity.
            ub (float): Upper bound of the variable. Defaults to +infinity.
        Currently: will not check that input is single element
    
        Returns:
            SingleVar: A new variable object representing the added variable.
        """
        # Compute index
        var_idx = self.num_of_vars
        
        # Update the model's total variable count
        self.num_of_vars += 1
        
        # Create new variable
        new_var = SingleVar(var_idx)


        vtype = vtype[0] if isinstance(vtype, (list, np.ndarray)) else vtype
        lb = lb[0] if isinstance(lb, (list, np.ndarray)) else lb
        ub = ub[0] if isinstance(ub, (list, np.ndarray)) else ub


        # Assign bound and handle vtype for new variable
        self._handle_vtype(vtype, var_idx, lb, ub)

        return new_var
        
    def addVars(self, n, vtype="C", lb=-np.inf, ub=np.inf):
        """
        Adds multiple variables to the model.
            Generates new indices, updates num_of_vars
    
        Args:
            n (int): Number of variables to add.
            vtype (str or list): Variable type(s). Can be a single string (applied to all),
                                 or a list/array specifying type for each variable.
                                 Types: "C" (continuous), "I" (integer), "B" (binary)
            lb (float or list): Lower bound(s) for variables. Can be scalar or list of length n.
            ub (float or list): Upper bound(s) for variables. Can be scalar or list of length n.

        Currently: will not check that input is single element or of lengt n (i.e., no error if list is too long)
    
        Returns:
            Variable or SingleVar: If n == 1, returns a SingleVar object. Otherwise, returns a Variable object.
        """

        if n == 1:
            return self.addVar(vtype, lb, ub)
        else:

            var_idx_list = np.array([idx for idx in range(self.num_of_vars, self.num_of_vars + n)])
            self.num_of_vars += n
        
    
            new_var = Variable(n, var_idx_list)
        
            # Handle vtype input if input is string
            if not isinstance(vtype, (list, np.ndarray)):
                vtype = [vtype for _ in range(n)]
        
        
            # Iterate over each variable and handle its type and bounds
            for i, vt in enumerate(vtype):
                var_lb = lb[i] if isinstance(lb, (list, np.ndarray)) else lb
                var_ub = ub[i] if isinstance(ub, (list, np.ndarray)) else ub
                var_idx = var_idx_list[i]
        
                # Add type and bounds info using internal handler
                self._handle_vtype(vt, var_idx, var_lb, var_ub)
        
            # Return the created variable(s)
            return new_var
    
    def addPSDVar(self, n):
        """
        Adds a Positive Semidefinite (PSD) matrix variable of size n x n to the model.
    
        Args:
            n (int): Dimension of the square PSD matrix. Must be greater than 1.
    
        Returns:
            PSDVar: The created PSD matrix variable, or None if dimension is invalid.
        """
        if n > 1:
            index = len(self.PSD_vars)  # Compute index
            PSD_var = PSDVar(n, index)  # Create the PSD variable object
            self.PSD_vars.append(PSD_var)  # Store it in the model
            return PSD_var
        else:
            print("matrix dim should be > 1")

    def addConstraint(self, linear_constraint):
        """
        Adds an affine (linear) constraint to the model.
    
        Args:
            LinConstraint: created by comparing a LinExpression to a scalar or another LinExpr
            The comparison operators ==, <, > will construct a LinConstraint (see LinExpr)
    
        Returns:
            LinConstraint: The added constraint, if valid. Otherwise, prints an error.
        """
        if isinstance(linear_constraint, LinConstraint):
            self.aff_constraints.append(linear_constraint)  # Store in affine constraint list

            linear_constraint._set_index(len(self.aff_constraints)-1)
            return linear_constraint
        else:
            print("input error: not a linear constraint")
    
    def addPSDConstraint(self, mat_expr):
        """
        Adds a PSD constraint to the model.
    
        Args:
            mat_expr (MatExpr): A matrix expression representing the PSD constraint.
    
        Returns:
            PSDConstraint: The created and added PSD constraint.
        """
        if isinstance(mat_expr, MatExpr):
            PSD_con = PSDConstraint(mat_expr)  # Create constraint object
            self.PSD_constraints.append(PSD_con)  # Store it in the model
            PSD_con._set_index(len(self.PSD_constraints) - 1)
            return PSD_con  # Return the constraint
        else: 
            print("input error: not a matrix expression")
    
    def addObjective(self, objsense, lin_expr):
        """
        Sets the objective function of the model.
    
        Args:
            objsense (str): The objective sense, must be one of "min", "MIN", "max", "MAX".
            lin_expr (LinExpr or SingleVar): The linear expression to optimize.
    
        Raises:
            ValueError: If objsense is not a valid string.
        """
        # Validate objective sense
        minsense = {"min", "MIN", "MINIMUM", "minimum"}
        maxsense = {"max", "MAX", "MAXIMUM", "maximum"}

        if objsense not in minsense|maxsense:
            raise ValueError("objsense must be one of: 'min', 'MIN', 'max', 'MAX', 'MINIMUM', 'MAXIMUM', 'minimum', 'maximum'")

        if objsense in minsense:
            objsense = "MIN"
        elif objsense in maxsense:
            objsense = "MAX"

        # Convert SingleVar to LinExpr if needed
        if isinstance(lin_expr, SingleVar):
            lin_expr = lin_expr._convert_to_linear_expression()
    
        # Set the objective
        self.objective = Objective(objsense, lin_expr)

        return self.objective

    def copy(self):
        # Return a copy on the model
        
        m = Model()
        m.num_of_vars = self.num_of_vars
        m.PSD_vars = self.PSD_vars
        m.int_list = self.int_list
        m.aff_constraints = self.aff_constraints
        
        m.lb_constraints = self.lb_constraints
        m.ub_constraints = self.ub_constraints
        
        m.PSD_constraints = self.PSD_constraints
        m.objective = self.objective
        return m

    def _handle_vtype(self, vtype, idx, var_lb, var_ub):
        # Handle integral variables and variable bounds

        # If vtype == "B" - add bounds [0,1] and reassign vtype = "I"
        if vtype == "I" or vtype == "B":
            self.int_list += [idx]
            if vtype == "B":
                var_lb = max(var_lb, 0)
                var_ub = min(var_ub, 1)
                
        # Only add constraints if bound is not +/- infty
        if var_lb != -np.inf: 
            self.lb_constraints.append(BoundConstraint(idx, var_lb, "L+"))
         if var_ub != np.inf:
            self.ub.constraints.append(BoundConstraint(idx, var_ub, "L-"))




    def writeCBF(self, file_name):
        """
        Writes the cbf-file
        Args:
            str: file_name
        """
        file1 = open(file_name, 'w')

        string = self.output_string()

        file1.write(string)

        file1.close()

    def output_string(self):
        string = self._write_format() # add: VER + OBJSENSE
        string += self._write_PSDvars() # add: PSDVAR
        string += self._write_vars() # add: VAR
        string += self._write_ints() # add: INT
        string += self._write_PSDcons_head() # add: PSDCON
        string += self._write_cons_head() # add: CON
        string += self._write_objective() # add COORDS for objective function
        string += self._write_cons() # add COORDS for affine constraints
        string += self._write_PSDcons() # add COORDS for PSD constraints
        return string

        if var_ub != np.inf:
            self.ub_constraints.append(BoundConstraint(idx, var_ub, "L-"))

    # Below are helper functions to write each section of the cbf-file
    # see cbf-format explanation at the top for reference

    def _write_format(self):
        """
        Returns:
            str: VER 2, OBJSENSE 
        """
        # This is version 2
        string = "VER\n2"
    
        # If an objective is set, include the objective sense
        if self.objective:
            string += "\n\nOBJSENSE\n" + self.objective.sense

        return string



    def _write_PSDvars(self):
        """
        Returns:
            str: PSDVAR ... etc
        """        
        if len(self.PSD_vars) > 0:
            string = "\n\nPSDVAR\n"
            string += str(len(self.PSD_vars))
            for psd_var in self.PSD_vars:
                string += "\n" + str(psd_var.n)
        else:
            string = ""
        return string


    def _write_vars(self):
        """
        Returns:
            str: VAR ... etc
        """      
        if self.num_of_vars > 0:
            string = "\n\nVAR\n"
            string += str(self.num_of_vars) + " 1"
            string += "\nF " + str(self.num_of_vars)
        else:
            string = ""
        return string

    def _write_ints(self):
        """
        Returns:
            str: INT ... etc
        """      
        if len(self.int_list) > 0:
            string = "\n\nINT\n"
            string += str(len(self.int_list))
            for idx in self.int_list:
                string += "\n" + str(idx)
        else:
            string = ""
        return string


    def _write_PSDcons_head(self):
        """
        Returns:
            str: COORDS for PSD-constraints
            str: PSDCON ... etc
            
        """      
        string = ""

        if len(self.PSD_constraints) > 0:
            string += "\n\nPSDCON\n"
            string += str(len(self.PSD_constraints))
            for con in self.PSD_constraints:
                string += "\n" + str(con.size)

        return string

    def _write_PSDcons(self):
        """
        Returns:
            str: COORDS for PSD-constraints
            str: PSDCON ... etc
            
        """

        h_string = ""
        d_string = ""
        tot_hcoord = 0
        tot_dcoord = 0
        for idx, constraint in enumerate(self.PSD_constraints):
            if constraint.dcoord is not False:
                tot_dcoord += len(constraint.dcoord.val)
                d_string += constraint.dcoord._stringMat(idx)

            for hcoord in constraint.hcoord:
                tot_hcoord += len(hcoord.val)
                h_string += hcoord._stringMat(idx)

        if len(h_string) > 0:
            h_string = "\n\nHCOORD\n" + str(tot_hcoord) + h_string

        if len(d_string) > 0:
            d_string = "\n\nDCOORD\n" + str(tot_dcoord) + d_string

        return h_string + d_string

    def _write_cons_head(self):
        """
        Returns:
            str: CON ... etc
            
        """  
        domain_list = []
        domain_count = []

        past_domain = False
        num_in_domain = 0
        for con in self.aff_constraints:
            current_domain = con.sense
            if current_domain == past_domain:
                num_in_domain += 1
            else:
                domain_list.append(current_domain)
                domain_count.append(num_in_domain)
                num_in_domain = 1
                past_domain = current_domain

        domain_count.append(num_in_domain)
        domain_count.pop(0)

        domain_list.append("L+")
        domain_count.append(len(self.lb_constraints))

        domain_list.append("L-")
        domain_count.append(len(self.ub_constraints))


        tot_con = len(self.aff_constraints) + len(self.lb_constraints) + len(self.ub_constraints)

        if tot_con > 0:
            string = "\n\nCON\n" + str(tot_con) + " " + str(len(domain_list))
            for idx, domain in enumerate(domain_list):
                string += "\n" + domain + " " + str(domain_count[idx])
            #string += "\n# LB-con: " + str(len(self.lb_constraints)) + ", UB-con: " + str(len(self.ub_constraints)) # print number of bound constraints


        return string


    def _write_cons(self):
        """
        Returns:
            str: COORDS for affine-constraints
            
        """

        tot_fcoord = 0
        tot_acoord = 0
        tot_bcoord = 0
        f_string = ""
        a_string = ""
        b_string = ""

        for idx, constraint in enumerate(self.aff_constraints):
            for fcoord in constraint.lin_expr.fcoord:
                f_string += fcoord._stringMat(idx)
                tot_fcoord += len(fcoord.val)

            a_string += constraint.lin_expr._stringA(idx)
            tot_acoord += len(constraint.lin_expr.acoord)
            if constraint.lin_expr.bcoord:
                b_string +=  constraint.lin_expr._stringB(idx)
                tot_bcoord += 1

        current_idx = len(self.aff_constraints)
        for idx, constraint in enumerate(self.lb_constraints + self.ub_constraints):
            a_string += constraint._stringA(idx + current_idx)
            tot_acoord += 1
            if constraint.bcoord:
                b_string += constraint._stringB(idx + current_idx)
                tot_bcoord += 1



        if tot_fcoord > 0:
            f_string = "\n\nFCOORD\n" + str(tot_fcoord) + f_string
        if tot_acoord > 0:
            a_string = "\n\nACOORD\n" + str(tot_acoord) + a_string
        if tot_bcoord > 0:
            b_string = "\n\nBCOORD\n" + str(tot_bcoord) + b_string

        return f_string + a_string + b_string



    def _write_objective(self):
        """
        (Note that objsense is set in _write_format())
        Returns:
            str: COORDS for objective function
            
        """
        string = ""
        if self.objective:
            if len(self.objective.lin_expr.fcoord) > 0:
                f_string = ""
                tot_fcoord = 0
                for fcoord in self.objective.lin_expr.fcoord:
                    f_string += fcoord._stringMat()
                    tot_fcoord += len(fcoord.val)

                string += "\n\nOBJFCOORD\n" + str(tot_fcoord) + f_string

            if len(self.objective.lin_expr.acoord) > 0:
                string += "\n\nOBJACOORD\n" + \
                    str(len(self.objective.lin_expr.acoord)) + self.objective.lin_expr._stringA()

            if self.objective.lin_expr.bcoord:
                string += "\n\nOBJBCOORD" + self.objective.lin_expr._stringB()

        return string






class LinExpr:
    """
    Class to represent 1-D linear expressions of the form:
        sum <F[i,j],X[j]>  +  sum  a[i,j]*x[j] + b[i]
    Example of input:
        F1*X1 + F2*X2 + a@x + b
        

    index i is assigned when LinExpr is assigned to a cone, i.e., becomes a LinConstraint

    Any expression containing SingleVar, Variable and/or LinExpr will be converted to a linear expression.

    Attributes:
        fcoord (list): list of MatCoord:s (instances in MatCoord class keeps track of the indices of the corr. variables: j /in J)
        acoord (np.array): array of acoord:s
        bcoord (float or bool): bcoord (can be false)
        a_index (np.array): Variable indices corresponding to acoord:s: j /in L

    
    """
    def __init__(self, fcoord=[], acoord=np.array([]), bcoord=0, a_index=np.array([])):
        self.fcoord = fcoord
        self.acoord = acoord
        self.bcoord = bcoord
        self.a_index = a_index

    @property
    def print(self):
        """
        prints:
            str: LinExpression ... etc
            
        """

        print(self._string_all())


    def _string_all(self, constraint_index = False):

        if constraint_index is False:
            prior = ""
        else:
            prior =  "i "

        if len(self.fcoord) > 0:
            f_string = "FCOORD\n" + prior + "j k l val"
            for fcoord in self.fcoord:
                f_string += fcoord._stringMat(constraint_index)
        else:
            f_string = "FCOORD\n" + "# no F-matrices"

        if len(self.acoord) > 0:
            a_string = "\n\nACOORD\n" + prior + "j val" + self._stringA(constraint_index)
        else:
            a_string = "\n\nACOORD\n" + "# no a-elements"

        if self.bcoord:
            b_string = "\n\nBCOORD\n" + prior + "val" + self._stringB(constraint_index)
        else:
            b_string = "\n\nBCOORD\n" + "# no b-element"


        return f_string + a_string + b_string


    def _stringA(self, constraint_index = False):
        """
        prints:
            str: LinExpression ... etc
            
        """
        if constraint_index is False:
            prior = "\n"
        else:
            prior = "\n" + str(constraint_index) + " "

        string = ""

        for jdx, acoord in enumerate(self.acoord):
            j = str(int(self.a_index[jdx])) + " "
            value = str(float(acoord))
            string += prior + j + value

        return string


    def _stringB(self, constraint_index = False):
        """
        prints:
            str: LinExpression ... etc
            
        """
        if constraint_index is False:
            prior = "\n"
        else:
            prior = "\n" + str(constraint_index) + " "

        if self.bcoord:
            string = prior + str(float(self.bcoord))
        else:
            string = ""

        return string



    def __add__(self, other):
        """
        Adds another object to this LinExpr instance.
    
        Supports:
            - Scalar (int or float): Adds to the constant term (bcoord)
            - SingleVar: Converts to LinExpr and adds
            - LinExpr: Adds all components element-wise
    
        Returns:
            LinExpr: A new linear expression representing the sum.
    
        Raises:
            TypeError: If 'other' is not a supported type.
        """
        if isinstance(other, (int, float)): # Scalar input
            return LinExpr(
                fcoord=self.fcoord,
                acoord=self.acoord,
                bcoord=self.bcoord + other,
                a_index=self.a_index
            )
    
        if isinstance(other, SingleVar): # SingleVar input
            other = other._convert_to_linear_expression()
    
        if isinstance(other, LinExpr): # LinExpr input
            return LinExpr(
                fcoord=self.fcoord + other.fcoord,
                acoord=np.concatenate((self.acoord, other.acoord), axis=None),
                bcoord=self.bcoord + other.bcoord,
                a_index=np.concatenate((self.a_index, other.a_index), axis=None)
            )
    
        # Unsupported type
        raise TypeError(f"Unsupported operand type(s) for +: 'LinExpr' and '{type(other).__name__}'")
    
    

    def __radd__(self, other):
        # I honestly can't remember if this is needed, should follow from __add__, but it's been a while
        # TODO: Find out if needed, otherwise clean up
        
        """
        Right-hand addition to handle cases like: scalar + LinExpr, 
        and specifically LinExpr + LinExpr 
    
        Supports:
            - Scalar (int or float)
            - SingleVar: Converts to LinExpr and adds
            - LinExpr: Adds expressions in reverse order (for completeness)
    
        Returns:
            LinExpr: The result of the addition.
    
        Raises:
            TypeError: If 'other' is not a supported type.
        """
        if isinstance(other, (int, float)):
            return LinExpr(
                fcoord=self.fcoord,
                acoord=self.acoord,
                bcoord=self.bcoord + other,
                a_index=self.a_index
            )
    
        if isinstance(other, SingleVar):
            other = other._convert_to_linear_expression()
    
        if isinstance(other, LinExpr):
            # Order reversed from __add__, but addition is commutative? I think I'm trying to keep the intended order of the user?
            return LinExpr(
                fcoord=other.fcoord + self.fcoord,
                acoord=np.concatenate((other.acoord, self.acoord), axis=None),
                bcoord=other.bcoord + self.bcoord,
                a_index=np.concatenate((other.a_index, self.a_index), axis=None)
            )
    
        raise TypeError(f"Unsupported operand type(s) for +: '{type(other).__name__}' and 'LinExpr'")


    def __iadd__(self, other):
        return self + other

    def __neg__(self):
        return LinExpr(fcoord=[-coord for coord in self.fcoord],
                           acoord=-self.acoord, bcoord = -self.bcoord, a_index=self.a_index)


    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __isub__(self, other):
        return self + -other
    

    def __rmul__(self, other): # scalar multiplication
        if isinstance(other, (int, float)):
            return LinExpr(
                fcoord=[other*coord for coord in self.fcoord],
                acoord=other*self.acoord,
                bcoord=other*self.bcoord,
                a_index=self.a_index
            )
                
        raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'LinExpr'")
        
    def __mul__(self, other):
        return other*self
            

    # "conic" operators "<", ">" and "==" transforms LinExpr into LinConstraint
    # First it moves the rhs to the lhs s.t. rhs = 0
    # This is the only way to construct a LinConstraint
    def __le__(self, other):
        lin_expr = self - other
        new_con = LinConstraint(lin_expr, "L-")
        return new_con

    def __eq__(self, other):
        lin_expr = self - other
        new_con = LinConstraint(lin_expr, "L=")
        return new_con

    def __ge__(self, other):
        lin_expr = self - other
        new_con = LinConstraint(lin_expr, "L+")
        return new_con

    __array_priority__ = 10000 # Ensures NumPy respects LinExpr's operator overloads







class MatExpr:
    """
    Helper class to represent matrix expressions on the form:
        sum H[i,j]*x[j] + D[i]
    Example of input: 
        H1*x1 + H2*x2 + D
        
    index i is assigned when MatExpr is assigned to a cone, i.e., becomes a PSDConstraint

    Any matrix expression containing SingleVar, Variable and/or MatrExpr will be converted to a  matrix expression.

    Attributes:
        size (int): size of matrices in expression
        hcoord (list): list of MatCoord:s (instances in MatCoord class keeps track of the indices of the corr. variables)
        dcoord (np.array): matrix D or 
    """
    def __init__(self, hcoord, dcoord=False):
        self.size = hcoord[0].size
        self.hcoord = hcoord
        if dcoord is False:
            self.dcoord = np.zeros((self.size, self.size))
        else:
            self.dcoord = dcoord

    @property
    def print(self):
        """
        Prints
            str: COORDS for MatExpression            
        """
        COORD_string = ""

        h_string = ""
        d_string = ""

        if self.dcoord.any():
            d_string = MatCoord(self.dcoord)._stringMat()

        for hcoord in self.hcoord:
            h_string += hcoord._stringMat()

        if len(h_string) > 0:
            h_string = "HCOORD\n" + "j k l val" + h_string
        else:
            h_string = "HCOORD\n" + "# no H-matrices"

        COORD_string += h_string

        if len(d_string) > 0:
            d_string = "\n\nDCOORD\n" + "k l val" + d_string
        else:
            d_string = "\n\nDCOORD\n" + "# no D-matrix"
        COORD_string += d_string

        print(COORD_string)



    def __add__(self, other):
        """
        Defines: self + other

        Supports:
        - MatExpr (if sizes match) (H-matrix)
        - NumPy array (if dimensions match self.size) (D-matrix)
        - Scalar 0 (returns self) (To be able to handle 'sum')

        Returns:
            MatExpr: New matrix expression

        Raises:
            ValueError: On dimension mismatch or unsupported type
        """
        if isinstance(other, MatExpr):
            if self.size != other.size:
                raise ValueError("Dimension mismatch between MatExpr instances.")
            return MatExpr(
                hcoord=self.hcoord + other.hcoord,
                dcoord=self.dcoord + other.dcoord
            )

        elif isinstance(other, np.ndarray):
            if self.size != len(other):
                raise ValueError("Dimension mismatch between MatExpr and ndarray.")
            return MatExpr(
                hcoord=self.hcoord,
                dcoord=self.dcoord + other
            )
        

        elif isinstance(other, int):
            if other == 0:
                return self # This is to handle sum(), which works by always adding a zero first. Maybe there is a better way to handle/override this
            else:
                raise TypeError(f"Unsupported operand type(s) for +: 'MatExpr' and '{type(other).__name__}'")
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'MatExpr' and '{type(other).__name__}'")
    
    
    def __radd__(self, other):
        return self.__add__(other)
            

    def __iadd__(self, other):
        return self + other

    def __rmul__(self, other): # Scalar multiplication
        if isinstance(other, (int, float)):
            return MatExpr(
                hcoord=[other*coord for coord in self.hcoord],
                dcoord=other*self.dcoord,
            )
                
        raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'LinExpr'")
        
    def __mul__(self, other):
        return other*self


    def __neg__(self):
        new_expr = MatExpr(hcoord=[-x for x in self.hcoord])
        if self.dcoord:
            new_expr.dcoord = -self.dcoord
        return new_expr

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __isub__(self, other):
        return self + -other

    __array_priority__ = 10000


class BoundConstraint:
    """
    Helper class to handle variable bounds. Could as easily be handled by LinConstraint, but I was lazy.
    Also that would mean we have to create SingleVar instances of all bounded Variables
    Cannot be used by Matrix-variables
    """
    def __init__(self, index, bound, sense):
        self.bcoord = -bound
        self.a_index = int(index)
        self.sense = sense


    def _stringA(self, constraint_index):
        return "\n" + str(constraint_index) + " " + str(self.a_index) + " 1"


    def _stringB(self, constraint_index = False):
        return "\n" + str(constraint_index) + " " + str(float(self.bcoord))




class LinConstraint:
    """
    Class to represent 1-D linear constraints of the form:
        sum <F[i,j],X[j]>  +  sum  a[i,j]*x[j] + b[i] /in Cone
    Example of input:
        F1*X1 + F2*X2 + a@x + b > 0
    
    All LinConstraints are added to aff_constraints-list when created. 
    Index i is simply it's corresponding index in the aff_constraints-list.


    Attributes:
        fcoord (list): list of MatCoord:s (instances in MatCoord class keeps track of the indices of the corr. variables: j /in J)
        acoord (np.array): array of acoord:s
        bcoord (float or bool): bcoord (can be false)
        a_index (np.array): Variable indices corresponding to acoord:s: j /in L
        sense (string): "L-", "L+", or "L="
    """
    def __init__(self, lin_expr, sense):
        self.lin_expr = LinExpr(lin_expr.fcoord, lin_expr.acoord[np.where(lin_expr.acoord != 0)], lin_expr.bcoord, lin_expr.a_index[np.where(lin_expr.acoord != 0)])
        self.sense = sense
        self._index = False

    @property
    def index(self):
        return self._index

    @property
    def print(self):
        """
        Prints the linear constraint
        """
        print("SENSE\n" + str(self.sense) + "\n\n" + self.lin_expr._string_all(self._index))

    def _set_index(self, index):
        self._index = index


class PSDConstraint:
    """
    Class to represent matrix consraints on the form:
        sum H[i,j]*x[j] + D[i] /in PSD-cone
    Example of input:
        H1*x1 + H2*x2 + D
        
    All PSDConstraints are added to PSD_constraints-list when created. 
    Index i is simply it's corresponding index in the PSD_constraints-list.

    Any expression containing SingleVar, Variable and/or MatrExpr will be converted to a  matrix expression.

    Attributes:
        size (int): size of matrices in expression
        hcoord (list): list of MatCoord:s (instances in MatCoord class keeps track of the indices of the corr. variables)
        dcoord (MatCoord): D is treated as a numpy array in the MatExpr-class. Here it is converted to a MatCoord-instance
    """
    def __init__(self, mat_expr):
        self.size = mat_expr.size
        self.hcoord = mat_expr.hcoord
        

        if mat_expr.dcoord.any(): # save only if there are any non-zero elements
            self.dcoord = MatCoord(mat_expr.dcoord)
        else:
            self.dcoord = False

        self._index = False

    @property
    def index(self):
        return self._index

    def _set_index(self, index):
        self._index = index


    @property
    def print(self):
        """
        Prints PSD constraint           
        """
        if self._index is not False:
            COORD_string = "SIZE\n" + str(self.size)

    
            h_string = ""
            d_string = ""

            if self.dcoord is not False:
                d_string = self.dcoord._stringMat(self._index)

            for hcoord in self.hcoord:
                h_string += hcoord._stringMat(self._index)

            if len(h_string) > 0:
                h_string = "\n\nHCOORD\n" + "i j k l val" + h_string
            else:
                h_string = "\n\nHCOORD\n" + "# no H-matrices"

            COORD_string += h_string
            if len(d_string) > 0:
                d_string = "\n\nDCOORD\n" + "i k l val" + d_string
            else:
                d_string = "\n\nDCOORD\n" + "# no D-matrix"
            COORD_string += d_string
    
            print(COORD_string)
        else:
            print("Not a constraint")




class MatCoord:
    """
    Helper class to handle matrices, i.e., both F, H and D in expressions such as:
        'H[i,j]*x[j] + D[i]', and 'sum <F[i,j],X[j]>  +  sum  a[i,j]*x[j] + b[i]'
    

    The i-index is handled/assigned when the MatCoord becomes part of either LinConstraint or PSDConstraint
    
    Keeps track of values corresponding to following lines (apart from i):
    i   j   k   l   F[i,j][k,l]  # Note: k >= l,
    i   j   k   l   H[i,j][k,l]  # Note: k >= l, and 
    i   k   l   D[i][k,l]  # Note: k >= l    
    
    (Will use the name F for matrices)
    
    Args:
        F (np.array) : Symmetric or lower triangular matrix
        var_index (int): False if matrix is D-type
        
    
    Attributes:
        size (int): size of matrix
        k_index (int): k
        l_index (int): l
        j_index (int): j, False for matrix-type D
        val (float): F[i,j][k,l]
        F (np.array): copy of original matrix (used for multiplication with scalars)
    """
    def __init__(self, F, var_index=False):

        self.F = F

        # save index and and val if k >= l
        self.size = F.shape[0]
        
        self._validate_matrix(F)
        
        lower_tri_indices = np.tril_indices(self.size, k=0)
        non_zero_mask = F[lower_tri_indices] != 0 # Only interested in non-zero elements
        
        
        self.k_index = lower_tri_indices[0][non_zero_mask]
        self.l_index = lower_tri_indices[1][non_zero_mask]
        
        self.j_index = var_index
        self.val = F[self.k_index, self.l_index].astype(float)

    def _stringMat(self, constraint_index = False):
        """
        Args:
            constraint_index (int): Default: False
        returns: 
            print_string (str):  string to print COORD for one matrix, if constraint_index is given, this is also printed on each row
        """
        print_string = ""

        if constraint_index is False:
            prior = "\n"
        else:
            prior = "\n" + str(constraint_index) + " "

        if self.j_index is False:
            j = ""
        else:
            j = str(self.j_index) + " "

        for jdx in range(len(self.val)):
            k = str(self.k_index[jdx]) + " "
            l = str(self.l_index[jdx]) + " "
            value = str(self.val[jdx])
            print_string += prior + j + k + l + value

        return print_string

    def _validate_matrix(self, F):
        # Check symmetry: F == F.T
        is_symmetric = np.allclose(F, F.T)
        
        # Check upper triangular elements are zero (excluding diagonal)
        upper_elements = F[np.triu_indices(self.size, k=1)]
        is_upper_zero = np.all(upper_elements == 0)
        
        if not (is_symmetric or is_upper_zero):
            raise ValueError("Matrix must be symmetric or have all upper triangular elements zero.")
        return True


    def __neg__(self):
        return -1*self

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return MatCoord(other*self.F, self.j_index)
        raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'LinExpr'")
        
    def __mul__(self, other):
        return other*self




class SingleVar:
    """
    Class to handle single variables
    SingleVar instances of each individual variable of the Variable-class will also be created

    Here we handle the cases, when 'x' is a single variable, 'a' is a scalar and 'H' is a symmetric or lower-triangular matrix:
        x*a (__mul__)
        a*x (__rmul__)
        H*x (__rmul__)


    Attributes:
        n (int): Size is of course 1 (Why to I keep track of this? TODO: Find out)
        var_index (int): assigned at creation (not by user)
    
    """
    def __init__(self, index):
        self.n = 1
        self.var_index = index

    @property
    def size(self):
        return self.n

    @property
    def index(self):
        return self.var_index

    def _validate_matrix(self, F):
        # Check symmetry: F == F.T
        is_symmetric = np.allclose(F, F.T)
        
        # Check upper triangular elements are zero (excluding diagonal)
        upper_elements = F[np.triu_indices(F.shape[0], k=1)]
        is_upper_zero = np.all(upper_elements == 0)
        
        if not (is_symmetric or is_upper_zero):
            raise ValueError("Matrix must be symmetric or have all upper triangular elements zero.")
        return True


    def _convert_to_linear_expression(self):
        """
        Helper function. 
        LinExpr is designed to handle e.g., 1*x + 2*y - 3 so if user input is e.g., x + 2, this turns it into 1*x + 2
        """
        return LinExpr(acoord=np.ones(1), bcoord=0, a_index=np.array([self.var_index]))

    def __add__(self, other):
        return self._convert_to_linear_expression() + other

    def __radd__(self, other):
        return other + self._convert_to_linear_expression()

    def __iadd__(self, other):
        return self + other

    def __neg__(self):
        return - self._convert_to_linear_expression()

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __isub__(self, other):
        return self + -other

    def __rmul__(self, other):
        # This operator defines both H*x and a*x

        if isinstance(other, (int, float)): # a * x
            return other*self._convert_to_linear_expression()

        if isinstance(other, np.ndarray): # H * x
            self._validate_matrix(other)

            hcoord = MatCoord(other, var_index=self.var_index)
            return MatExpr(hcoord=[hcoord], dcoord=False)

        raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'SingleVar'")

    def __mul__(self, other):
        # x*a is OK, but x*H is not
        if isinstance(other, (int, float)):
            return other*self._convert_to_linear_expression()

        raise TypeError(f"Unsupported operand type(s) for *: 'SingleVar' and '{type(other).__name__}'")


    # I don't think these are used. self - other should turn into LinExpr and use the operators of LinExpr. Keeping them just in case :)
    def __le__(self, other):
        return LinConstraint(self - other, "L-")

    def __eq__(self, other):
        return LinConstraint(self - other, "L=")

    def __ge__(self, other):
        return LinConstraint(self - other, "L+")

    __array_priority__ = 10000


class Variable:
    """
    Class to handle multible variables
    Basically just a list, that you can slice

    Here we handle the cases, when 'x' is an n-variable and 'a' is an n-vector
        a@x (__matmul__)
    It only accepts a@x, where a is a numpy array of shape (n,)


    I'm not sure '@' is the most intuitive notation/choice of operator. Might swap to '*'


    Attributes:
        n (int): Size is of course 1 (Why to I keep track of this? TODO: Find out)
        var_index (int): assigned at creation (not by user)
    
    """
    def __init__(self, n, index):
        self.n = n
        self.var_index = index

    @property
    def size(self):
        return self.n

    @property
    def index(self):
        return self.var_index


    def _validate_vector(self, other):
        if not isinstance(other, np.ndarray):
            raise TypeError(f"Input must be a numpy.ndarray, but got {type(other)}")

        if other.shape != (self.n,):
            raise ValueError(f"Input array must have shape ({self.n},), but got {other.shape}")
        return True

    def __matmul__(self, other):
        # Currently accepting x@a, where a is numpy array of size (n,)
        self._validate_vector(other)
        acoord = other[np.where(other != 0)]
        a_index = self.var_index[np.where(other != 0)]
        return LinExpr(acoord=acoord, a_index=a_index)


    def __rmatmul__(self, other):
        """
        Handles the operation a@x
        Args:
            a (numpy-array): of size (n,)
        """
        self._validate_vector(other)
        acoord = other[np.where(other != 0)]
        a_index = self.var_index[np.where(other != 0)]
        return LinExpr(acoord=acoord, a_index=a_index)

    def __getitem__(self, key):
        """
        Slicing. Should work the same as numpy slicing
        
        return self[index]    
        """
        if isinstance(key, int):
            return SingleVar(self.var_index[key])
        else:
            var_index = self.var_index[key]
            n = len(var_index)
            return Variable(n, var_index)

    __array_priority__ = 10000


class PSDVar:
    """
    Class to handle PSD-variables
    Here we define <F,X>, i.e., X*F or F*X

    Attributes:
        n (int): size of nxn variable
        index (int): Gets assigned at creation (not by user)
    Note: 
    """
    def __init__(self, n, var_index):
        self.n = n
        self.var_index = var_index


    @property
    def size(self):
        return (self.n, self.n)

    @property
    def index(self):
        return self.var_index

    @property
    def print(self):
        print("Matrix variable of size " + str(self.size) + ", constrained to the PSD-cone. Index = " + str(self.index))


    def _validate_matrix(self, F):
        if not isinstance(F, np.ndarray):
            raise TypeError(f"Input must be a numpy.ndarray, but got {type(F)}")
    
        if F.shape != (self.n, self.n):
            raise ValueError(f"Input array must have shape ({self.n}, {self.n}), but got {F.shape}")
    
        # Check symmetry: F == F.T
        is_symmetric = np.allclose(F, F.T)
    
        # Check upper triangular elements are zero (excluding diagonal)
        upper_elements = F[np.triu_indices(self.n, k=1)]
        is_upper_zero = np.all(upper_elements == 0)
    
        if not (is_symmetric or is_upper_zero):
            raise ValueError("Matrix must be symmetric or have all upper triangular elements zero.")
    
        return True

    def __mul__(self, other): # Trace
        return other*self

    def __rmul__(self, other): # Trace
        self._validate_matrix(other)
        fcoord = MatCoord(other, var_index=self.var_index)
        return LinExpr(fcoord=[fcoord])

    __array_priority__ = 10000



class Objective:
    """
    Class to maintain the linear expression of the objective function

    Attributes:
        lin_exp (LinExpr):
        sense (str):

    """
    def __init__(self, objsense, lin_expr):

        self.lin_expr = LinExpr(lin_expr.fcoord, lin_expr.acoord[np.where(lin_expr.acoord != 0)], lin_expr.bcoord, lin_expr.a_index[np.where(lin_expr.acoord != 0)])
        self.sense = objsense


    @property
    def print(self):

        print("SENSE\n" + self.sense + "\n\n" + self.lin_expr._string_all())









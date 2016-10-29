----------------------
APC 524 Assignment 3: Root finding and automated testing
Betul Uralcan, Submitted on: 10/28/2016
----------------------
This project offers an implementation of the Newton's method in one and higher dimensions with the functionalities described below. The project includes the following python codes:

1. newton.py: implements the simple root finder Newton's method in one and higher dimensions.

2. functions.py: contains one and higher dimensional functions and functionalities to compute the analytical and approximate jacobians

3. testNewton.py: tests the convergence, accuracy of the Newton method, and a single step of the Newton method. Also contains tests for cases of divergence when the equation does not have a real root or the initial guess is outside of a radius r. 

4. testFunctions.py: tests the accuracy of approximate and analytic Jacobians for all functions in functions.py as well as additional functions (functions that don't have roots, etc.) 

----------------------
newton.py
----------------------
The user can input the following variables:

tol:    		    Tolerance for iteration (default: 1.e-6)

maxiter: 	      Maximum number of iterations to perform (default: 20)

dx:     		    Step size for computing approximate Jacobian (default: 1.e-6)

Df:     		    Analytical jacobian (default: None (uses approximate Jacobian if Df not specified))

norm:    	      treshold for distance between initial guess and approximated root (default: None)

----------------------
functions.py
----------------------
Polynomial:        1D polynomial with a function to compute the analytical jacobian
                   
		   Example usage: to construct the polynomial p(x) = x^2 + 2x + 3, and evaluate p(5):
		   p = Polynomial([1, 2, 3]) 

Linear3D:          Set of three equations with a function to compute the analytical jacobian
	           
		   Example usage: to construct the set of equations p1(x1, x2, x3)=5x1-3, p2(x1, x2, x3)=4x2-1, p3(x1, x2, x3)=x3-1:
		   p = Polynomial([5,4,1])

WrongLinear3D:     Linear set of three equations with a function to compute a wrong analytical jacobian

PolynomialMD:      Set of three equations containing trigonometric functions with the analytical jacobian

WrongPolynomialMD: Set of three equations containing sine/cosine terms with the wrong analytical jacobian

# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import functions as F
import sys

class Newton(object):
    def __init__(self, f, Df=None, Dim=None, norm=None, tol=1.e-6, maxiter=20, dx=1.e-6 ):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian"""
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        #Df=None
        self._Df = Df
        self._Dimension = Dim
        self._norm = norm

    def solve(self, x0):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
        for i in xrange(self._maxiter):
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            else:
                x = self.step(x, fx)
            if self._norm is not None and N.linalg.norm(x-x0) > self._norm:
                #raise Exception()
                raise Exception ("Approximated x is not within an acceptable radius of initial guess x0")
        return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        if self._Df is None:
            Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        else:
            Df_x = self._Df(x)
        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        return x - h

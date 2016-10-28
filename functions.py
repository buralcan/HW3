import math
import numpy as N

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx) / dx
    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""
    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans

    def __call__(self, x):
        return self.f(x)

    def AnalyticalJacobian(self,x):
        df = 0
        for n in range(1,len(self._coeffs)):
            df += n*self._coeffs[n] * x ** (n-1)
        return N.matrix(df)


class PolynomialMD(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def f(self,x):
        ans=N.matrix(N.zeros((len(x),1)))
        ans[0,0]=5*x[1]
        ans[1,0]=4*x[0]**2-2*math.sin(x[1]*x[2])
        ans[2,0]=x[1]*x[2]
        return ans

    def __call__(self, x):
        return self.f(x)

    def AnalyticalJacobian(self,x):
        df = N.matrix(N.zeros((len(x),len(x))))
        df[0,0]=0
        df[0,1]=5
        df[0,2]=0
        df[1,0]=8*x[0]
        df[1,1]=-2*x[2]*math.cos(x[1]*x[2])
        df[1,2]=-2*x[1]*math.cos(x[1]*x[2])
        df[2,0]=0
        df[2,1]=x[2]
        df[2,2]=x[1]
        return df

class WrongPolynomialMD(object):

    def f(self,x):
        ans=N.matrix(N.zeros((len(x),1)))
        ans[0,0]=5*x[1]
        ans[1,0]=4*x[0]**2-2*math.sin(x[1]*x[2])
        ans[2,0]=x[1]*x[2]
        return ans

    def __call__(self, x):
        return self.f(x)

    def AnalyticalJacobian(self,x):
        df = N.matrix(N.zeros((len(x),len(x))))
        df[0,0]=0
        df[0,1]=5
        df[0,2]=0
        df[1,0]=8*x[2]
        df[1,1]=-2*x[2]*math.cos(x[1]*x[2])
        df[1,2]=-2*x[0]*math.cos(x[1]*x[2])
        df[2,0]=9
        df[2,1]=x[2]
        df[2,2]=x[1]
        return df

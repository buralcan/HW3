#!/usr/bin/env python
import functions as F
import newton
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    # test function given in the homework
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    # compare the approximated roots of the polynomial
    # with the real roots using both the approximated and analytical jacobian
    def testPolynomial(self):
        # p(x) = x^2 + 3x + 2
        p = F.Polynomial([1, 3, 2])
        solver = newton.Newton(p.f, tol=1.e-17, maxiter=125)
        x1Appr = solver.solve(0.5)
        self.assertAlmostEqual(x1Appr, -1)
        x2Appr = solver.solve(-3.5)
        self.assertAlmostEqual(x2Appr, -2)
        solver = newton.Newton(p.f, Df=p.AnalyticalJacobian, tol=1.e-17, maxiter=125)
        x2Analy = solver.solve(-3.5)
        self.assertAlmostEqual(x2Analy, -2)

    def testPolynomial2(self):
        f = lambda x : 3*x**3 - 3
        solver = newton.Newton(f, tol=1.e-9, maxiter=25)
        x = solver.solve(2.0)
        self.assertAlmostEqual(x, 1.0)

    # test the 3 dimensional problem using the approximate and analytical jacobian
    # compare the results with the real roots and each other    
    def testLinear3D(self):
        x=N.matrix("1;0.5;-3")
        xReal = N.matrix("0.6;0.25;-3.0")
        solver = newton.Newton(F.Linear3D().f, Df=F.Linear3D().AnalyticalJacobian)
        solver2 = newton.Newton(F.Linear3D().f)
        # roots match, so no exception will be raised
        N.testing.assert_array_almost_equal(solver2.solve(x), solver.solve(x))
        N.testing.assert_array_almost_equal(xReal, solver.solve(x))

    # the analytical jacobian of the 3 dimensional problem is wrong,
    # therefore the computed roots will be wrong, raises and exception.
    def testWrongJacobianPolynomialMD(self):
        # p(x) = x^2 + 3x + 2
        x=N.matrix("2; 3; 5")
        xReal=N.matrix("7; 1; 1")
        solver = newton.Newton(F.WrongPolynomialMD().f, Df=F.WrongPolynomialMD().AnalyticalJacobian)
        try:  self.assertAlmostEqual(xReal, solver.solve(x))
        except Exception:
            print "Computed roots don't match the real ones, wrong analytical Jacobian"
        else:
            pass

    # the equation 2^(x-1)+3=0 does not have a root 
    # xk will diverge and the iteration loop will raise and exception
    def testNorm(self):
        f = lambda x : 2**(x-1)+3
        x=3.0
        try: x=newton.Newton(f, tol=1.e-15).solve(x)
        except Exception:
            print "Roots cant be found, Approximated x is too far from initial guess x0"
 
    # the approximated root must lie within a radius r of the initial guess x0,
    # or the iteration loop raises an exception
    def testRadius(self):
        p = F.Polynomial([1, 3, 2])
        # with the approximate jacobian
        solver = newton.Newton(p.f, tol=1.e-15, norm=1 )
        try: x = solver.solve(35)
        except Exception:
            print "Approximated x is not within an acceptable radius of initial guess x0"
        # with the analytical jacobian
        solver = newton.Newton(p.f, Df=p.AnalyticalJacobian, tol=1.e-15, maxiter=25, norm=1 )
        try: x = solver.solve(35)
        except Exception:
            print "x is not within an acceptable radius of initial guess x0"

    # checking a single step of the newton method
    def testOneStep(self):
        x = 3
        # f = x^2 + 3x + 2  
        p = F.Polynomial([1, 3, 2])
        solver=newton.Newton(p.f)
        x1 = solver.step(x)
        x2 = solver.step(x, fx=p.f(x))
        self.assertEqual(x1, x2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
import functions as F
import newton
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

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

    def testWrongJacobianPolynomialMD(self):
        # p(x) = x^2 + 3x + 2
        x=N.matrix("2; 3; 5")
        xReal=N.matrix("7; 1; 1")
        solver = newton.Newton(F.WrongPolynomialMD().f, Df=F.WrongPolynomialMD().AnalyticalJacobian)
        try: xCalculated = solver.solve(x)
        except Exception:
            print "Wrong Jacobian"
        else:
            pass

    def testPolynomial2(self):
        f = lambda x : 3*x**3 - 3
        solver = newton.Newton(f, tol=1.e-9, maxiter=25)
        x = solver.solve(2.0)
        self.assertAlmostEqual(x, N.matrix(1.0))

    def testNorm(self):
        f = lambda x : 2**(x-1)+3
        x=3.0
        try: x=newton.Newton(f, tol=1.e-15, maxiter=1).solve(x)
        except Exception:
            print "Approximated x is not within an acceptable radius of initial guess x0"

    def testRadius(self):
        p = F.Polynomial([1, 3, 2])
        solver = newton.Newton(p.f, tol=1.e-15, norm=1 )
        try: x = solver.solve(35)
        except Exception:
            print "Approximated x is not within an acceptable radius of initial guess x0"
        solver = newton.Newton(p.f, Df=p.AnalyticalJacobian, tol=1.e-15, maxiter=25.0, norm=1 )
        try: x = solver.solve(35)
        except Exception:
            print "Approximated x is not within an acceptable radius of initial guess x0"

    def testOneStep(self):
        x = 3
        p = F.Polynomial([1, 3, 2])
        solver=newton.Newton(p.f)
        x1 = solver.step(x)
        x2 = solver.step(x, fx=p.f(x))
        self.assertEqual(x1, x2)


if __name__ == "__main__":
    unittest.main()

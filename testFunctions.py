#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    # test function as given in the bugged files
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    # compare the approximate to analytical jacobian for polynomial function
    def testJacobian1(self):
        x=2
        dx = 1.e-6
        # p(x) = 5x^3+7*x^2+11*x+15
        ApproximateDf_x=F.ApproximateJacobian(F.Polynomial([5, 7, 11, 15]).f,x,dx)
        AnalyticalDf_x=F.Polynomial([15, 11, 7, 5]).AnalyticalJacobian(x) # call analytical J
        self.assertEqual(ApproximateDf_x.shape,  AnalyticalDf_x.shape)
        self.assertAlmostEqual(ApproximateDf_x,AnalyticalDf_x, places=3)

    # 2D system, compare the approximate jacobian to real
    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)
    
    # compare the analytically computed jacobian of the 3D nonlinear system
    # to the approximate jacobian
    def testJacobianMD(self):
        x=N.matrix("2; 3; 5")
        dx = 1.e-6
        ApproximateDf_x=F.ApproximateJacobian(F.PolynomialMD().f,x,dx)
        AnalyticalDf_x=F.PolynomialMD().AnalyticalJacobian(x)
        self.assertEqual(ApproximateDf_x.shape,  AnalyticalDf_x.shape)
        N.testing.assert_array_almost_equal(ApproximateDf_x,  AnalyticalDf_x, decimal=3)

    # test the polynomial function for multiple x to make sure it works as expected in 
    # the given x-range
    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    # compare the wrong analytical jacobian with the approximatre and real jacobian 
    # for the 3D linear system
    def testWrongLinear3D(self):
        # p1(x)=5x-3, p2(x)=4x-1, p3(x)=x-1
        p = F.WrongLinear3D([5, 4, 1])
        x=N.matrix("1;0.5;-3")
        dx = 1.e-6
        Df_xReal = N.matrix("5.;4.;1.")
        ApproximateDf_x=F.ApproximateJacobian(p.f,x,dx)
        AnalyticalDf_x=p.AnalyticalJacobian(x)
        self.assertEqual(ApproximateDf_x.shape,  AnalyticalDf_x.shape)
        #self.assertAlmostEqual(ApproximateDf_x,AnalyticalDf_x, places=3)
        try: N.testing.assert_array_almost_equal(ApproximateDf_x,  AnalyticalDf_x, decimal=3)
        except Exception:
            print "Approximate DF_x and Analytical DF_x dont match"
        else:
            pass
        try: N.testing.assert_array_almost_equal(Df_xReal,  AnalyticalDf_x, decimal=3)
        except Exception:
            print "Wrong anlaytically computed Jacobian: Real and Analytical DF_x dont match"
        else:
            pass

     
    # test the analytical jacobian of the 3D nonlinear system
    # compare to the approximate jacobian, the analytical J is wrong
    def testWrongPolynomialMD(self):
        x=N.matrix("1;0.5;-3")
        dx = 1.e-6
        ApproximateDf_x=F.ApproximateJacobian(F.WrongPolynomialMD().f,x,dx)
        AnalyticalDf_x=F.WrongPolynomialMD().AnalyticalJacobian(x)
        self.assertEqual(ApproximateDf_x.shape,  AnalyticalDf_x.shape)
        try: N.testing.assert_array_almost_equal(ApproximateDf_x,  AnalyticalDf_x, decimal=3)
        except Exception:
            print "Wrong analytical Jacobian"
        else:
            pass

if __name__ == '__main__':
    unittest.main()

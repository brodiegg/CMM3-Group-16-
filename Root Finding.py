#Root finding 


"""
Root Finding Methods for Suspension Optimization
CMM3 Group Project - Group 16

Goal: Find optimal damping coefficient c* that minimizes passenger discomfort
Method: Find where df/dc = 0 (minimum of comfort function)
"""

import numpy as np
import matplotlib.pyplot as plt

# Root finding methods

def bisection(f, a, b, N):
    """
    Bisection method for finding roots
    
    Parameters:
    -----------
    f : function
        Function for which we are searching for a root f(x)=0
    a, b : float
        Initial interval [a, b] that brackets the root
    N : int
        Maximum number of iterations
    
    Returns:
    --------
    float : Approximate root
    """
    # Check if a and b bound a root
    if f(a)*f(b) >= 0:
        print("a and b do not bound a root")
        return None 
    
    a_n = a
    b_n = b
    
    for n in range(1, N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    
    return (a_n + b_n)/2


def newton(f, Df, x0, epsilon, max_iter):
    """
    Newton-Raphson method for finding roots
    
    Parameters:
    -----------
    f : function
        Function for which we are searching for a solution f(x)=0
    Df : function
        Derivative of f(x)
    x0 : float
        Initial guess for a solution f(x)=0
    epsilon : float
        Stopping criteria is abs(f(x)) < epsilon
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    float : Approximate solution
    """
    xn = x0
    
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after', n, 'iterations.')
            return xn
        
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        
        xn = xn - fxn/Dfxn
    
    print('Exceeded maximum iterations. No solution found.')
    return None
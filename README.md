This project simulates a quarter-car suspension system and finds the optimal damping coefficient for comfort. The main document Final.py generates road profiles, interpolates them into continuous functions, solves the suspension dynamics using RK4, applies ISO frequency-weighting, and finally estimates the optimal damping coefficient using Bisection and Newton Raphson methods. 


The project is implemented in standard Python and requires Python 3.x (e.g.3.9 or later).

Before running, make sure the required packages are installed:

pip install numpy scipy matplotlib

Then, navigate to the folder and run Final.py.

 You may also run it directly inside VS Code by selecting a Python environment with the required packages and clicking “Run Python File”.


ODE Solver (Line 356 - 458)
We convert the coupled second-order suspension equations into a first-order state system and solve the time-domain dynamics using a fixed-step RK4 method.

Interpolation (Line 299 - 353) and Regression (Line 179 - 183)
Road height samples are turned into a smooth continuous function using cubic spline interpolation so the ODE can read road input at any position. ISO frequency-weighting data is fitted with polynomial regression to create a continuous comfort weighting curve.

Root Finding (Line 541 - 614)
The optimal damping coefficient c is estimated by minimizing the comfort function using both Newton–Raphson and Bisection.

# Quarter-Car Suspension Optimization Project

This project simulates a quarter-car suspension system and finds the optimal damping coefficient for comfort.  

The main script, `Final.py`, performs the following tasks:

- Generates road profiles
- Interpolates them into continuous functions 
- Solves the suspension dynamics using RK4 and Forward Euler
- Applies ISO frequency-weighting
- Estimates the optimal damping coefficient using Bisection and Newton-Raphson methods

---

__Requirements__

- Python 3.x (e.g., 3.9 or later)
- One package: pip install numpy scipy matplotlib

## Key Features

<details>
  <summary>1. ODEs (Lines 385 - 482)</summary>

- Converts the coupled second-order suspension equations into a first-order state system
- Solves the time-domain dynamics using a fixed-step RK4 and Forward Euler

</details>

<details>
  <summary>2. Interpolation (Lines 281 - 379) & Regression (Lines 176 - 180)</summary>

- Road height samples are turned into a smooth continuous function using cubic spline interpolation
- ISO frequency-weighting data is fitted with polynomial regression to create a continuous comfort weighting curve

</details>

<details>
  <summary>3. Root Finding (Lines 565 - 638)</summary>

- The optimal damping coefficient `c` is estimated by minimizing the comfort function
- Uses both Newton–Raphson and Bisection methods

</details>

## Visuals & Outputs

<details>
  <summary>What does our code do?</summary>

- Generates road profiles
- Tracks suspension displacement and velocity
- Graphs Comfort Function curves
- Calculates convergence of the optimal damping coefficient

*(Plots are generated automatically when running `Final.py`)*

</details>

---

## Final Message

Thank you for checking out our project! The programme should take around 20 minutes in total, wth visuals popping up throughout. 
We hope you enjoy what we've made :)

~ From Brodie, Tom, Libby, Yang and Lucas



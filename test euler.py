"""
Quarter-Car Suspension System Optimization for Electric Vehicles
CMM3 Group Project - Group 16
(Code for testing the RK45 solver output)
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# System Parameters for different EV types (Copy from original code)
VEHICLE_CONFIGS = {
    'Tesla_Model_X': {
        'M1': 625,     # Sprung mass (kg)
        'M2': 50,      # Unsprung mass (kg)
        'k': 35000,    # Suspension spring stiffness (N/m)
        'kt': 250000,  # Tire stiffness (N/m)
        'c_range': (500, 5000),
    },
    # ... other configs omitted for brevity
    'Standard_EV': {
        'M1': 500,
        'M2': 40,
        'k': 25000,
        'kt': 200000,
        'c_range': (300, 4000),
    }
}

class RoadProfile:
    """Generate and interpolate road profiles for simulation (Simplified for testing)"""
    
    def __init__(self, profile_type='smooth', distance=100, dx=0.1):
        self.profile_type = profile_type
        self.distance = distance
        self.dx = dx
        self.x = np.arange(0, distance, dx)
        self.profile = self._generate_profile()
        self.interpolator = None
        self._create_interpolator()
    
    def _generate_profile(self):
        # A simple, single-bump profile for clear visual results
        profile = np.zeros_like(self.x)
        bump_start = int(self.distance / 4 / self.dx)
        bump_end = int(self.distance / 2 / self.dx)
        bump_indices = np.arange(bump_start, bump_end)
        
        # Gaussian bump shape
        x_bump = self.x[bump_indices]
        amplitude = 0.05 # 5 cm bump
        center = (self.x[bump_start] + self.x[bump_end]) / 2
        sigma = 1.0
        profile[bump_indices] = amplitude * np.exp(-((x_bump - center)**2) / (2 * sigma**2))
        return profile
    
    def _create_interpolator(self):
        """Create cubic spline interpolator for road profile"""
        self.interpolator = interpolate.CubicSpline(self.x, self.profile, bc_type='natural')
        self.velocity_interpolator = self.interpolator.derivative()
    
    def get_height(self, position):
        return self.interpolator(position)
    
    def get_velocity(self, position, vehicle_speed):
        return self.velocity_interpolator(position) * vehicle_speed

class QuarterCarModel:
    """Quarter-car suspension system dynamics model (The main code snippet)"""
    
    def __init__(self, config_name='Standard_EV'):
        self.config = VEHICLE_CONFIGS[config_name]
        self.M1 = self.config['M1']
        self.M2 = self.config['M2']
        self.k = self.config['k']
        self.kt = self.config['kt']
        
    def system_dynamics(self, t, y, c, road_func, vehicle_speed, current_position):
        """
        System ODEs for quarter-car model
        State vector: y = [x1, x1_dot, x2, x2_dot]
        """
        x1, x1_dot, x2, x2_dot = y
        
        # Get road input at current position
        position = current_position + vehicle_speed * t
        xr = road_func(position)
        xr_dot = road_func.derivative()(position) * vehicle_speed
        
        # Calculate accelerations
        x1_ddot = -(self.k * (x1 - x2) + c * (x1_dot - x2_dot)) / self.M1
        x2_ddot = (self.k * (x1 - x2) + c * (x1_dot - x2_dot) - 
                    self.kt * (x2 - xr) - 0.01 * self.kt * (x2_dot - xr_dot)) / self.M2
        
        return [x1_dot, x1_ddot, x2_dot, x2_ddot]
    
    def simulate(self, c, road_profile, vehicle_speed, sim_time):
        """Simulate system response with given damping coefficient"""
        # Initial conditions (at rest)
        y0 = [0, 0, 0, 0]
        
        # Time span
        t_span = [0, sim_time]
        # 100 Hz sampling for a 5-second sim -> 501 points (0 to 5.0)
        t_eval = np.linspace(0, sim_time, int(sim_time * 100) + 1)
        
        # Solve ODEs using RK45 (Runge-Kutta 4th/5th order)
        sol = solve_ivp(
            lambda t, y: self.system_dynamics(t, y, c, road_profile.interpolator, vehicle_speed, 0),
            t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6
        )

        # Print the outputs for verification
        print("\n==============================================")
        print("         RK45 Solution Verification")
        print("==============================================")
        
        # sol.t verification
        print(f"\n✅ sol.t (Time Vector) verification:")
        print(f"   Length: {len(sol.t)}")
        print(f"   Expected Length: {len(t_eval)}")
        print(f"   Start Time: {sol.t[0]:.2f} s, End Time: {sol.t[-1]:.2f} s")
        print(f"   First 5 values: {sol.t[:5]}")
        
        # sol.y verification
        print(f"\n✅ sol.y (State Matrix) verification:")
        print(f"   Shape: {sol.y.shape} (Variables x Time Steps)")
        print(f"   Expected Shape: (4, {len(sol.t)})")
        print(f"   Initial State (y0): {sol.y[:, 0]}")
        print(f"   Final State (y_end) for x1 (car body pos): {sol.y[0, -1]:.5f} m")
        
        if len(sol.t) == len(t_eval) and sol.y.shape == (4, len(sol.t)):
            print("\n**SUCCESS: sol.t and sol.y have the correct dimensions and boundary values.**")
        
        return sol.t, sol.y

# --- Execution Block ---

# 1. Instantiate the Model and Road
model = QuarterCarModel(config_name='Tesla_Model_X')
road = RoadProfile(profile_type='custom_bump', distance=50, dx=0.01) # Use a 50m road for simplicity

# 2. Set Test Parameters
TEST_DAMPING_C = 2500.0  # Ns/m (A value within the Tesla range)
TEST_SPEED = 20.0        # m/s (approx 72 km/h)
TEST_TIME = 2.5          # seconds (to cover 50m distance)

# 3. Run the Simulation
t, y = model.simulate(TEST_DAMPING_C, road, TEST_SPEED, TEST_TIME)
    
    
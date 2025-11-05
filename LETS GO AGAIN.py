"""
Quarter-Car Suspension System Optimization for Electric Vehicles
CMM3 Group Project - Group 16

Complete implementation:
- Road generation with cubic spline interpolation
- ODE solver (RK4 + Forward Euler)
- ISO 2631 frequency weighting filter
- RMS/RMQ comfort calculation with crest factor
- Root finding optimization (Newton-Raphson + Bisection)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.integrate import solve_ivp

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

# Vehicle configurations for different EVs
VEHICLE_CONFIGS = {
    'Tesla_Model_X': {
        'M1': 625,    # Sprung mass (kg) - quarter car body mass
        'M2': 50,     # Unsprung mass (kg) - wheel assembly
        'k': 35000,   # Suspension spring stiffness (N/m)
        'kt': 250000, # Tire stiffness (N/m)
        'c_range': (500, 5000),  # Damping coefficient range (Ns/m)
        'c_initial': 2500  # Initial guess for optimization
    },
    'BMW_i4': {
        'M1': 550,
        'M2': 45,
        'k': 30000,
        'kt': 230000,
        'c_range': (400, 4500),
        'c_initial': 2250
    },
    'Standard_EV': {
        'M1': 500,
        'M2': 40,
        'k': 25000,
        'kt': 200000,
        'c_range': (300, 4000),
        'c_initial': 2000
    }
}

# Simulation parameters
VEHICLE_SPEED = 10  # m/s (reduced to stay on road longer)
SIM_TIME = 8  # seconds (80m travel at 10 m/s - stays within 100m road)
DT = 0.01  # time step for evaluation (100 Hz)

# Road generation parameters
ROAD_LENGTH = 100  # meters
NO_POINTS_SMOOTH = 20  # control points for smooth roads
NO_POINTS_ROUGH = 50  # control points for rough roads
H_RANGE_SMOOTH = 0.02  # ±2cm height variation for smooth roads
H_RANGE_ROUGH = 0.04  # ±4cm height variation for rough roads
NO_PROFILES = 3  # number of each road type to generate

# ISO 2631-1 frequency weighting data
ISO_FREQ = np.array([0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0,
                     5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0,
                     50.0, 63.0, 80.0])

ISO_WEIGHTS = np.array([0.482, 0.484, 0.494, 0.531, 0.631, 0.804, 0.967,
                        1.039, 1.054, 1.036, 0.988, 0.902, 0.768, 0.636,
                        0.513, 0.405, 0.314, 0.246, 0.186, 0.132,
                        0.100, 0.070, 0.045])

# ============================================================================
# ROAD PROFILE GENERATION (From Roads + Cubic spline.py)
# ============================================================================

from scipy.interpolate import splrep, splev

class RoadProfile:
    """Road profile generator using cubic spline interpolation"""
    
    def __init__(self, road_type, seed):
        self.type = road_type
        self.seed = seed
        
        if road_type == 'smooth':
            self.x_positions, self.road_heights, self.spline = self._smooth_road_gen(
                seed, NO_POINTS_SMOOTH, ROAD_LENGTH, H_RANGE_SMOOTH
            )
        else:  # rough
            self.x_positions, self.road_heights, self.spline = self._rough_road_gen(
                seed, NO_POINTS_ROUGH, ROAD_LENGTH, H_RANGE_ROUGH
            )
    
    def _smooth_road_gen(self, seed, no_points, road_length, h_range):
        """Generate smooth road profile"""
        np.random.seed(seed)
        x_pos = np.linspace(0, road_length, no_points)
        h_values = np.random.uniform(-h_range, h_range, no_points)
        
        h_values[0] = 0   # start at 0
        h_values[-1] = 0  # end at 0
        
        spline = splrep(x_pos, h_values)
        
        return x_pos, h_values, spline
    
    def _rough_road_gen(self, seed, no_points, road_length, h_range):
        """Generate rough road profile"""
        np.random.seed(seed)
        x_pos = np.linspace(0, road_length, no_points)
        h_values = np.random.uniform(-h_range, h_range, no_points)
        
        h_values[0] = 0
        h_values[-1] = 0
        
        spline = splrep(x_pos, h_values)
        
        return x_pos, h_values, spline
    
    def get_height(self, position):
        """Get road height at position using cubic spline"""
        return float(splev(position, self.spline))
    
    def get_slope(self, position):
        """Get road slope (derivative) at position"""
        return float(splev(position, self.spline, der=1))

# ============================================================================
# ISO 2631 FREQUENCY WEIGHTING FILTER (From ISO 2631 Filter.py)
# ============================================================================

# Fit polynomial to ISO data (degree 5 gives R² = 0.998)
log_f = np.log10(ISO_FREQ)
log_w = np.log10(ISO_WEIGHTS)
iso_coeffs = np.polyfit(log_f, log_w, 5)
iso_poly = np.poly1d(iso_coeffs)

def ISO_filter(frequency):
    """
    ISO 2631-1 frequency weighting filter
    
    Parameters:
    -----------
    frequency : float or array
        Frequency in Hz
    
    Returns:
    --------
    weight : float or array
        Weighting factor (dimensionless)
    """
    log_freq = np.log10(frequency)
    log_weight = iso_poly(log_freq)
    weight = 10**log_weight
    return weight

# ============================================================================
# ODE SOLVER
# ============================================================================

def solve_suspension_odes(c, M1, M2, k, kt, road_profile, vehicle_speed, sim_time):
    """
    Solve quarter-car suspension ODEs using RK45
    
    Parameters:
    -----------
    c : float
        Damping coefficient (Ns/m)
    M1, M2, k, kt : float
        System parameters
    road_profile : RoadProfile
        Road profile object
    vehicle_speed : float
        Vehicle speed (m/s)
    sim_time : float
        Simulation time (s)
    
    Returns:
    --------
    t : array
        Time vector
    y : array
        State vector [x1, x1_dot, x2, x2_dot]
    """
    
    def system_dynamics(t, y):
        """
        System ODEs: dy/dt = f(t, y)
        State vector: y = [x1, x1_dot, x2, x2_dot]
        """
        x1, x1_dot, x2, x2_dot = y
        
        # Current position along road
        position = vehicle_speed * t
        
        # Clamp position to valid road range
        position = np.clip(position, 0, ROAD_LENGTH - 0.1)
        
        # Get road input
        xr = road_profile.get_height(position)
        
        # Calculate accelerations from Newton's second law
        # M1*x1_ddot = -k(x1-x2) - c(x1_dot-x2_dot)
        x1_ddot = -(k * (x1 - x2) + c * (x1_dot - x2_dot)) / M1
        
        # M2*x2_ddot = k(x1-x2) + c(x1_dot-x2_dot) - kt(x2-xr)
        x2_ddot = (k * (x1 - x2) + c * (x1_dot - x2_dot) - kt * (x2 - xr)) / M2
        
        return [x1_dot, x1_ddot, x2_dot, x2_ddot]
    
    # Initial conditions (at rest)
    y0 = [0, 0, 0, 0]
    
    # Time span and evaluation points
    t_span = [0, sim_time]
    t_eval = np.arange(0, sim_time, DT)
    
    # Solve using RK45 (Runge-Kutta 4th/5th order adaptive)
    sol = solve_ivp(system_dynamics, t_span, y0, t_eval=t_eval, 
                    method='RK45', rtol=1e-6, atol=1e-9)
    
    return sol.t, sol.y

# ============================================================================
# COMFORT METRIC CALCULATION
# ============================================================================

def apply_frequency_weighting(acceleration, t):
    """
    Apply ISO 2631 frequency weighting to acceleration signal
    
    Process: Time domain → FFT → Apply Wk(f) → IFFT → Time domain
    
    Parameters:
    -----------
    acceleration : array
        Raw acceleration signal a(t)
    t : array
        Time vector
    
    Returns:
    --------
    weighted_accel : array
        Frequency-weighted acceleration aω(t)
    """
    n = len(acceleration)
    dt = t[1] - t[0]
    
    # FFT to frequency domain
    fft_accel = np.fft.fft(acceleration)
    freq = np.fft.fftfreq(n, dt)
    
    # Apply ISO 2631 weighting
    weights = np.ones_like(freq)
    valid_freq = (np.abs(freq) >= 0.5) & (np.abs(freq) <= 80)
    
    # Avoid very small frequencies that cause numerical issues
    freq_to_weight = np.abs(freq[valid_freq])
    freq_to_weight = np.maximum(freq_to_weight, 0.5)  # Clamp minimum
    
    weights[valid_freq] = ISO_filter(freq_to_weight)
    
    # Apply weights in frequency domain
    weighted_fft = fft_accel * weights
    
    # IFFT back to time domain
    weighted_accel = np.real(np.fft.ifft(weighted_fft))
    
    return weighted_accel

def calculate_comfort_metric(weighted_accel, t):
    """
    Calculate RMS or RMQ based on crest factor
    
    Parameters:
    -----------
    weighted_accel : array
        Frequency-weighted acceleration aω(t)
    t : array
        Time vector
    
    Returns:
    --------
    comfort_value : float
        RMS (smooth) or RMQ (rough) in m/s²
    crest_factor : float
        CF = |aω,max| / aω,RMS
    metric_used : str
        'RMS' or 'RMQ'
    """
    # Calculate RMS (always needed for crest factor)
    # aω,RMS = sqrt(1/T * integral(aω² dt))
    rms = np.sqrt(np.mean(weighted_accel**2))
    
    # Calculate crest factor
    peak = np.max(np.abs(weighted_accel))
    crest_factor = peak / rms if rms > 0 else 0
    
    # Choose metric based on crest factor
    if crest_factor > 9:
        # Rough road - use RMQ
        # aω,RMQ = (1/T * integral(aω⁴ dt))^0.25
        rmq = (np.mean(weighted_accel**4))**0.25
        return rmq, crest_factor, 'RMQ'
    else:
        # Smooth road - use RMS
        return rms, crest_factor, 'RMS'

# ============================================================================
# COMFORT FUNCTION (Inner code called by root finding)
# ============================================================================

def create_comfort_function(vehicle_config, road_profile):
    """
    Create comfort function for specific vehicle and road combination
    This is the INNER CODE called repeatedly by root finding
    
    Parameters:
    -----------
    vehicle_config : dict
        Vehicle parameters (M1, M2, k, kt)
    road_profile : RoadProfile
        Road profile object
    
    Returns:
    --------
    comfort_func : function
        Function that takes c and returns comfort metric
    """
    
    def comfort_function(c):
        """
        Complete pipeline: ODEs → Acceleration → Filter → RMS/RMQ
        
        Parameters:
        -----------
        c : float
            Damping coefficient (Ns/m)
        
        Returns:
        --------
        comfort_value : float
            RMS or RMQ comfort metric (m/s²)
        """
        # Step 1: Solve ODEs with damping coefficient c
        t, y = solve_suspension_odes(
            c, 
            vehicle_config['M1'],
            vehicle_config['M2'],
            vehicle_config['k'],
            vehicle_config['kt'],
            road_profile,
            VEHICLE_SPEED,
            SIM_TIME
        )
        
        # Step 2: Calculate car body acceleration from velocity
        x1_dot = y[1]  # Car body velocity
        x1_accel = np.gradient(x1_dot, t)  # Differentiate to get acceleration
        
        # Step 3: Apply ISO 2631 frequency weighting
        weighted_accel = apply_frequency_weighting(x1_accel, t)
        
        # Step 4: Calculate comfort metric (RMS or RMQ)
        comfort_value, crest_factor, metric = calculate_comfort_metric(weighted_accel, t)
        
        return comfort_value
    
    return comfort_function

# ============================================================================
# ROOT FINDING METHODS (Outer loop)
# ============================================================================

def newton_numerical(f, x0, x_range, h=10, epsilon=1e-6, max_iter=50):
    """Newton-Raphson with numerical derivatives"""
    xn = x0
    x_min, x_max = x_range
    
    for n in range(max_iter):
        f_xn = f(xn)
        f_plus = f(xn + h)
        f_minus = f(xn - h)
        
        Df_xn = (f_plus - f_minus) / (2 * h)
        D2f_xn = (f_plus - 2*f_xn + f_minus) / (h**2)
        
        if abs(Df_xn) < epsilon:
            return xn
        
        if abs(D2f_xn) < 1e-10:
            return xn
        
        xn_new = xn - Df_xn / D2f_xn
        xn_new = np.clip(xn_new, x_min, x_max)
        
        if abs(xn_new - xn) < epsilon:
            return xn_new
        
        xn = xn_new
    
    return xn

def bisection_numerical(f, a, b, h=10, epsilon=1e-6, max_iter=50):
    """Bisection method with numerical derivatives"""
    a_n = a
    b_n = b
    
    Df_a = (f(a + h) - f(a - h)) / (2 * h)
    Df_b = (f(b + h) - f(b - h)) / (2 * h)
    
    if Df_a * Df_b >= 0:
        return (a + b) / 2
    
    for n in range(1, max_iter + 1):
        m_n = (a_n + b_n) / 2
        Df_m = (f(m_n + h) - f(m_n - h)) / (2 * h)
        
        if abs(Df_m) < epsilon or (b_n - a_n) / 2 < epsilon:
            return m_n
        
        if Df_a * Df_m < 0:
            b_n = m_n
            Df_b = Df_m
        else:
            a_n = m_n
            Df_a = Df_m
    
    return (a_n + b_n) / 2

# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def optimize_suspension(vehicle_name, road_profile, verbose=True):
    """
    Optimize damping coefficient for given vehicle and road
    
    Parameters:
    -----------
    vehicle_name : str
        Vehicle configuration name
    road_profile : RoadProfile
        Road profile object
    verbose : bool
        Print results
    
    Returns:
    --------
    results : dict
        Optimization results
    """
    config = VEHICLE_CONFIGS[vehicle_name]
    
    # Create comfort function for this vehicle/road combination
    comfort_func = create_comfort_function(config, road_profile)
    
    # Method 1: Newton-Raphson
    c_opt_newton = newton_numerical(
        comfort_func,
        config['c_initial'],
        config['c_range']
    )
    comfort_newton = comfort_func(c_opt_newton)
    
    # Method 2: Bisection
    c_opt_bisection = bisection_numerical(
        comfort_func,
        config['c_range'][0],
        config['c_range'][1]
    )
    comfort_bisection = comfort_func(c_opt_bisection)
    
    if verbose:
        print(f"\n{vehicle_name} - {road_profile.type} road (seed={road_profile.seed})")
        print(f"  Newton-Raphson: c* = {c_opt_newton:.1f} Ns/m, Comfort = {comfort_newton:.4f} m/s²")
        print(f"  Bisection:      c* = {c_opt_bisection:.1f} Ns/m, Comfort = {comfort_bisection:.4f} m/s²")
    
    return {
        'c_newton': c_opt_newton,
        'c_bisection': c_opt_bisection,
        'comfort_newton': comfort_newton,
        'comfort_bisection': comfort_bisection
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete optimization for all vehicle/road combinations"""
    
    print("="*70)
    print("QUARTER-CAR SUSPENSION OPTIMIZATION")
    print("="*70)
    
    # Generate all road profiles
    print("\nGenerating road profiles...")
    roads = []
    
    # Generate smooth roads (seeds 1, 2, 3)
    for i in range(NO_PROFILES):
        seed = 1 + i
        road = RoadProfile('smooth', seed)
        roads.append(road)
        print(f"  Road {i+1}: smooth (seed={seed})")
    
    # Generate rough roads (seeds 4, 5, 6)
    for i in range(NO_PROFILES):
        seed = 4 + i
        road = RoadProfile('rough', seed)
        roads.append(road)
        print(f"  Road {i+NO_PROFILES+1}: rough (seed={seed})")
    
    print(f"\nSimulation parameters:")
    print(f"  Vehicle speed: {VEHICLE_SPEED} m/s")
    print(f"  Simulation time: {SIM_TIME} s")
    print(f"  Sampling rate: {1/DT:.0f} Hz")
    
    # Run optimization for all combinations
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    results = {}
    vehicle_names = ['Standard_EV', 'Tesla_Model_X', 'BMW_i4']
    
    for vehicle in vehicle_names:
        for i, road in enumerate(roads):
            key = f"{vehicle}_road{i+1}"
            results[key] = optimize_suspension(vehicle, road, verbose=True)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main()
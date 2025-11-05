"""
Quarter-Car Suspension System Optimization for Electric Vehicles
CMM3 Group Project - Group 16

This code optimizes the damping coefficient for a passive suspension system
to minimize passenger discomfort according to ISO 2631 standards.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# System Parameters for different EV types
VEHICLE_CONFIGS = {
    'Tesla_Model_X': {
        'M1': 625,    # Sprung mass (kg) - quarter car body mass
        'M2': 50,     # Unsprung mass (kg) - wheel assembly
        'k': 35000,   # Suspension spring stiffness (N/m)
        'kt': 250000, # Tire stiffness (N/m)
        'c_range': (500, 5000),  # Damping coefficient range (Ns/m)
    },
    'BMW_i4': {
        'M1': 550,
        'M2': 45,
        'k': 30000,
        'kt': 230000,
        'c_range': (400, 4500),
    },
    'Standard_EV': {
        'M1': 500,
        'M2': 40,
        'k': 25000,
        'kt': 200000,
        'c_range': (300, 4000),
    }
}

# ISO 2631-1 Frequency weighting parameters
ISO_2631_WEIGHTS = {
    'frequencies': [0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 
                   5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0],
    'weights': [0.418, 0.459, 0.477, 0.482, 0.484, 0.494, 0.531, 0.631, 0.804, 0.967,
               1.039, 1.054, 1.036, 0.988, 0.902, 0.768, 0.636, 0.513, 0.405, 0.314, 0.246, 0.186, 0.132]
}

class RoadProfile:
    """Generate and interpolate road profiles for simulation"""
    
    def __init__(self, profile_type='smooth', distance=100, dx=0.1):
        self.profile_type = profile_type
        self.distance = distance
        self.dx = dx
        self.x = np.arange(0, distance, dx)
        self.profile = self._generate_profile()
        self.interpolator = None
        self._create_interpolator()
    
    def _generate_profile(self):
        """Generate random road profile based on type"""
        np.random.seed(42)  # For reproducibility
        
        if self.profile_type == 'smooth':
            # Smooth road with long wavelength irregularities
            frequencies = [0.1, 0.2, 0.3, 0.5]
            amplitudes = [0.02, 0.015, 0.01, 0.005]  # meters
        else:  # rough
            # Rough road with shorter wavelength irregularities
            frequencies = [0.3, 0.5, 0.8, 1.2, 2.0]
            amplitudes = [0.03, 0.025, 0.02, 0.015, 0.01]
        
        profile = np.zeros_like(self.x)
        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.random() * 2 * np.pi
            profile += amp * np.sin(2 * np.pi * freq * self.x + phase)
        
        # Add random noise
        profile += np.random.normal(0, 0.002, len(self.x))
        
        # Add occasional bumps for rough road
        if self.profile_type == 'rough':
            n_bumps = int(self.distance / 20)
            for _ in range(n_bumps):
                bump_pos = np.random.randint(10, len(self.x) - 10)
                bump_width = np.random.randint(5, 15)
                bump_height = np.random.uniform(0.02, 0.05)
                x_bump = self.x[bump_pos:bump_pos + bump_width]
                profile[bump_pos:bump_pos + bump_width] += bump_height * np.exp(-((x_bump - x_bump.mean())**2) / (2 * (bump_width/4)**2))
        
        return profile
    
    def _create_interpolator(self):
        """Create cubic spline interpolator for road profile"""
        # Use scipy's cubic spline interpolation
        self.interpolator = interpolate.CubicSpline(self.x, self.profile, bc_type='natural')
        
        # Also create derivative for road velocity input
        self.velocity_interpolator = self.interpolator.derivative()
    
    def get_height(self, position):
        """Get road height at given position"""
        return self.interpolator(position)
    
    def get_velocity(self, position, vehicle_speed):
        """Get road input velocity (dxr/dt) at given position"""
        return self.velocity_interpolator(position) * vehicle_speed

class QuarterCarModel:
    """Quarter-car suspension system dynamics model"""
    
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
        x1: car body position
        x2: wheel position
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
        # Generate time vector at 100 Hz sampling
        sample_rate = 100
        t_eval = np.linspace(0, sim_time, int(sim_time * sample_rate))

        # Define system dynamics function once (avoids recreating lambda each call)
        def dynamics(t, y):
            return self.system_dynamics(t, y, c, road_profile.interpolator, vehicle_speed, 0)

        # Solve ODEs using adaptive RK45 solver
        sol = solve_ivp(
            dynamics, t_span, y0, t_eval=t_eval,
            method='RK45', rtol=1e-6, atol=1e-9
        )
        
        # Solve ODEs using Euler method
        def euler(system_dynamics_func, y0, t_eval, h, c, road_interpolator, vehicle_speed):
            y_sol = np.zeros((len(t_eval), len(y0)))
            y_current = np.array(y0)
            y_sol[0, :]=y_current
        
            for i in range(len(t_eval)-1):
                t_current = t_eval[i]
                
                # Calculate the derivative
                f_ty = system_dynamics_func(t_current, y_current, c, road_interpolator, vehicle_speed, 0)
                
                # Apply the Euler step
                y_next = y_current + h * f_ty
                
                # Store the result and update the current y value
                y_sol[i+1, :] = y_next
                y_current = y_next

            return t_eval, y_sol.T
        
        t_euler, y_euler = euler(self.system_dynamics, y0, t_eval, h, c, road_profile.interpolator, vehicle_speed)

        # Compares the result from the Runge-Kutta method and Euler method
        if np.isclose(sol.y, y_sol.T, 0.05, 0):
            return sol,y, sol.t
        else:
            return sol.y, y_euler
    
class ISO2631Evaluator:
    """Evaluate comfort metrics according to ISO 2631-1"""
    
    def __init__(self):
        # Create frequency weighting filter using regression
        self._create_weighting_filter()
    
    def _create_weighting_filter(self):
        """Create frequency weighting filter using least-squares regression"""
        # Fit rational function to ISO 2631-1 Wk weighting
        frequencies = np.array(ISO_2631_WEIGHTS['frequencies'])
        weights = np.array(ISO_2631_WEIGHTS['weights'])
        
        # Use polynomial regression for simplicity
        # In practice, would use rational function fitting
        log_freq = np.log10(frequencies)
        log_weights = np.log10(weights)
        
        # Fit 4th order polynomial
        self.weight_coeffs = np.polyfit(log_freq, log_weights, 4)
        
    def apply_frequency_weighting(self, acceleration, dt):
        """Apply ISO 2631-1 frequency weighting to acceleration signal"""
        # FFT to frequency domain
        n = len(acceleration)
        freq = np.fft.fftfreq(n, dt)[:n//2]
        fft_accel = np.fft.fft(acceleration)[:n//2]
        
        # Apply weighting
        weights = np.zeros_like(freq)
        valid_freq = (freq > 0.1) & (freq < 100)
        log_freq = np.log10(freq[valid_freq])
        log_weights = np.polyval(self.weight_coeffs, log_freq)
        weights[valid_freq] = 10**log_weights
        
        # Apply weights and inverse FFT
        weighted_fft = fft_accel * weights
        full_fft = np.concatenate([weighted_fft, np.conj(weighted_fft[::-1])])
        weighted_accel = np.real(np.fft.ifft(full_fft[:n]))
        
        return weighted_accel
    
    def calculate_comfort_metrics(self, time, acceleration):
        """Calculate comfort metrics from acceleration data"""
        dt = time[1] - time[0]
        
        # Apply frequency weighting
        weighted_accel = self.apply_frequency_weighting(acceleration, dt)
        
        # Calculate crest factor
        rms = np.sqrt(np.mean(weighted_accel**2))
        peak = np.max(np.abs(weighted_accel))
        crest_factor = peak / rms if rms > 0 else 0
        
        # Choose appropriate metric
        if crest_factor > 9:
            # Use RMQ for rough roads
            rmq = (np.mean(weighted_accel**4))**0.25
            comfort_value = rmq
        else:
            # Use RMS for smooth roads
            comfort_value = rms
        
        return comfort_value, crest_factor

class SuspensionOptimizer:
    """Optimize damping coefficient for ride comfort"""
    
    def __init__(self, model, road_profile, vehicle_speed=20):
        self.model = model
        self.road_profile = road_profile
        self.vehicle_speed = vehicle_speed  # m/s
        self.evaluator = ISO2631Evaluator()
        self.sim_time = 10  # seconds
        
        # Constraints
        self.max_suspension_travel = 0.2  # meters
        self.min_tire_force_variance = 1000  # N
    
    def objective_function(self, c):
        """Objective function to minimize (comfort + penalties)"""
        # Simulate system
        t, y = self.model.simulate(c, self.road_profile, self.vehicle_speed, self.sim_time)
        
        # Extract accelerations
        x1_accel = np.gradient(y[1], t)  # Car body acceleration
        
        # Calculate comfort metric
        comfort_value, _ = self.evaluator.calculate_comfort_metrics(t, x1_accel)
        
        # Calculate constraints
        suspension_travel = np.max(np.abs(y[0] - y[2]))
        
        # Calculate tire force variance (simplified)
        positions = self.vehicle_speed * t
        road_heights = np.array([self.road_profile.get_height(p) for p in positions])
        tire_force = self.model.kt * (y[2] - road_heights)
        tire_force_var = np.var(tire_force)
        
        # Apply penalties for constraint violations
        penalty = 0
        if suspension_travel > self.max_suspension_travel:
            penalty += 100 * (suspension_travel - self.max_suspension_travel)**2
        if tire_force_var < self.min_tire_force_variance:
            penalty += 0.01 * (self.min_tire_force_variance - tire_force_var)
        
        return comfort_value + penalty
    
    def newton_raphson(self, c_initial, tol=1e-6, max_iter=50):
        """Newton-Raphson method to find optimal damping coefficient"""
        c = c_initial
        h = 10  # Step size for numerical differentiation
        
        for i in range(max_iter):
            # Calculate first derivative (gradient)
            f_plus = self.objective_function(c + h)
            f_minus = self.objective_function(c - h)
            f_prime = (f_plus - f_minus) / (2 * h)
            
            # Calculate second derivative (Hessian)
            f_center = self.objective_function(c)
            f_double_prime = (f_plus - 2 * f_center + f_minus) / h**2
            
            # Avoid division by zero
            if abs(f_double_prime) < 1e-10:
                print(f"Warning: Second derivative near zero at iteration {i}")
                break
            
            # Newton-Raphson update
            c_new = c - f_prime / f_double_prime
            
            # Ensure c stays within bounds
            c_new = np.clip(c_new, self.model.config['c_range'][0], 
                          self.model.config['c_range'][1])
            
            # Check convergence
            if abs(c_new - c) < tol:
                print(f"Newton-Raphson converged in {i+1} iterations")
                return c_new
            
            c = c_new
        
        print(f"Newton-Raphson reached max iterations")
        return c
    
    def bisection(self, tol=1e-6, max_iter=50):
        """Bisection method for finding optimal damping coefficient"""
        a, b = self.model.config['c_range']
        h = 10
        
        # Check that derivative has opposite signs at boundaries
        f_prime_a = (self.objective_function(a + h) - self.objective_function(a - h)) / (2 * h)
        f_prime_b = (self.objective_function(b + h) - self.objective_function(b - h)) / (2 * h)
        
        if f_prime_a * f_prime_b > 0:
            print("Warning: Derivative has same sign at boundaries")
            # Use middle point as fallback
            return (a + b) / 2
        
        for i in range(max_iter):
            c = (a + b) / 2
            f_prime_c = (self.objective_function(c + h) - self.objective_function(c - h)) / (2 * h)
            
            if abs(f_prime_c) < tol or (b - a) / 2 < tol:
                print(f"Bisection converged in {i+1} iterations")
                return c
            
            if f_prime_a * f_prime_c < 0:
                b = c
            else:
                a = c
                f_prime_a = f_prime_c
        
        print(f"Bisection reached max iterations")
        return (a + b) / 2

def main():
    """Main execution function"""
    print("=" * 60)
    print("Quarter-Car Suspension Optimization System")
    print("=" * 60)
    
    # Test different vehicle configurations and road types
    results = {}
    
    for vehicle in ['Standard_EV', 'Tesla_Model_X', 'BMW_i4']:
        for road_type in ['smooth', 'rough']:
            print(f"\nAnalyzing {vehicle} on {road_type} road...")
            print("-" * 40)
            
            # Create models
            model = QuarterCarModel(vehicle)
            road = RoadProfile(road_type, distance=200, dx=0.1)
            optimizer = SuspensionOptimizer(model, road, vehicle_speed=20)
            
            # Initial guess (middle of range)
            c_initial = np.mean(model.config['c_range'])
            
            # Optimize using Newton-Raphson
            c_optimal_nr = optimizer.newton_raphson(c_initial)
            comfort_nr = optimizer.objective_function(c_optimal_nr)
            
            # Verify with Bisection method
            c_optimal_bis = optimizer.bisection()
            comfort_bis = optimizer.objective_function(c_optimal_bis)
            
            # Store results
            key = f"{vehicle}_{road_type}"
            results[key] = {
                'c_newton': c_optimal_nr,
                'c_bisection': c_optimal_bis,
                'comfort_newton': comfort_nr,
                'comfort_bisection': comfort_bis
            }
            
            print(f"Newton-Raphson: c* = {c_optimal_nr:.1f} Ns/m, Comfort = {comfort_nr:.4f}")
            print(f"Bisection:      c* = {c_optimal_bis:.1f} Ns/m, Comfort = {comfort_bis:.4f}")
    
    # Visualization
    plot_results(results)
    
    # Detailed analysis for one configuration
    print("\n" + "=" * 60)
    print("Detailed Analysis: Standard EV on Smooth Road")
    print("=" * 60)
    
    model = QuarterCarModel('Standard_EV')
    road = RoadProfile('smooth', distance=100, dx=0.1)
    optimizer = SuspensionOptimizer(model, road, vehicle_speed=20)
    
    # Get optimal damping
    c_optimal = optimizer.newton_raphson(np.mean(model.config['c_range']))
    
    # Simulate with optimal damping
    t, y = model.simulate(c_optimal, road, 20, 5)
    
    # Create detailed plots
    detailed_analysis_plots(t, y, road, c_optimal, model)
    
    print("\nOptimization Complete!")
    return results

def plot_results(results):
    """Create visualization of optimization results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Suspension Optimization Results', fontsize=16, fontweight='bold')
    
    vehicles = ['Standard_EV', 'Tesla_Model_X', 'BMW_i4']
    road_types = ['smooth', 'rough']
    
    for i, vehicle in enumerate(vehicles):
        # Damping coefficients
        c_smooth = results[f"{vehicle}_smooth"]['c_newton']
        c_rough = results[f"{vehicle}_rough"]['c_newton']
        
        axes[0, i].bar(['Smooth', 'Rough'], [c_smooth, c_rough], 
                      color=['skyblue', 'coral'])
        axes[0, i].set_title(f'{vehicle}\nOptimal Damping')
        axes[0, i].set_ylabel('Damping Coefficient (Ns/m)')
        axes[0, i].grid(axis='y', alpha=0.3)
        
        # Comfort values
        comfort_smooth = results[f"{vehicle}_smooth"]['comfort_newton']
        comfort_rough = results[f"{vehicle}_rough"]['comfort_newton']
        
        axes[1, i].bar(['Smooth', 'Rough'], [comfort_smooth, comfort_rough],
                      color=['lightgreen', 'salmon'])
        axes[1, i].set_title('Comfort Metric')
        axes[1, i].set_ylabel('Comfort Value (lower is better)')
        axes[1, i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def detailed_analysis_plots(t, y, road, c_optimal, model):
    """Create detailed analysis plots for a specific configuration"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Detailed System Analysis - Optimal Damping: {c_optimal:.1f} Ns/m', 
                 fontsize=16, fontweight='bold')
    
    # Road profile
    x_road = np.linspace(0, 100, 1000)
    axes[0, 0].plot(x_road, road.get_height(x_road), 'b-', linewidth=0.5)
    axes[0, 0].set_title('Road Profile')
    axes[0, 0].set_xlabel('Distance (m)')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Body and wheel displacement
    axes[0, 1].plot(t, y[0], 'b-', label='Car Body', linewidth=2)
    axes[0, 1].plot(t, y[2], 'r--', label='Wheel', linewidth=1.5)
    axes[0, 1].set_title('Vertical Displacement')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Displacement (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Suspension deflection
    suspension_deflection = y[0] - y[2]
    axes[0, 2].plot(t, suspension_deflection, 'g-', linewidth=2)
    axes[0, 2].axhline(y=model.config['M1'] * 9.81 / model.k, 
                       color='k', linestyle='--', label='Static deflection')
    axes[0, 2].set_title('Suspension Deflection')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Deflection (m)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Body acceleration
    accel = np.gradient(y[1], t)
    axes[1, 0].plot(t, accel, 'b-', linewidth=1)
    axes[1, 0].set_title('Car Body Acceleration')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Acceleration (m/s²)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frequency spectrum
    freq = np.fft.fftfreq(len(accel), t[1] - t[0])[:len(accel)//2]
    fft_accel = np.abs(np.fft.fft(accel))[:len(accel)//2]
    
    axes[1, 1].semilogy(freq[1:100], fft_accel[1:100], 'b-', linewidth=2)
    axes[1, 1].set_title('Acceleration Frequency Spectrum')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 30])
    
    # Tire force
    positions = 20 * t  # vehicle_speed * time
    road_heights = np.array([road.get_height(p) for p in positions])
    tire_force = model.kt * (y[2] - road_heights)
    
    axes[1, 2].plot(t, tire_force/1000, 'r-', linewidth=1.5)
    axes[1, 2].set_title('Tire Contact Force')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Force (kN)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = main()
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Method':<15} {'Damping (Ns/m)':<15} {'Comfort':<10}")
    print("-" * 65)
    
    for key, value in results.items():
        vehicle, road = key.split('_')
        print(f"{key:<25} {'Newton-Raphson':<15} {value['c_newton']:<15.1f} {value['comfort_newton']:<10.4f}")
        print(f"{'':<25} {'Bisection':<15} {value['c_bisection']:<15.1f} {value['comfort_bisection']:<10.4f}")
        print("-" * 65)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import math

# ============================================================================
# ISO 2631 FILTER (ORIGINAL POLYNOMIAL METHOD)
# ============================================================================

# ISO 2631 frequency weighting data
ISO_freq = np.array([0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0,
    5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0,
    50.0, 63.0, 80.0])

ISO_weights = np.array([0.482, 0.484, 0.494, 0.531, 0.631, 0.804, 0.967,
    1.039, 1.054, 1.036, 0.988, 0.902, 0.768, 0.636,
    0.513, 0.405, 0.314, 0.246, 0.186, 0.132,
    0.100, 0.070, 0.045])

# Transform to logs and create polynomial fit (degree 5 as before)
log_f = np.log10(ISO_freq)
log_w = np.log10(ISO_weights)
coeffs = np.polyfit(log_f, log_w, 5)
p = np.poly1d(coeffs)

def ISO_filter(frequency):
    frequency = np.array(frequency) #converts input frequency to a numpy array
    boolean_f = (frequency >= 0.5) & (frequency <= 80) #creates a boolean array with frequency values that fall between 0.5 and 80
    weights = np.ones_like(frequency) * 0.045 #creates an array like frequency with every value at 0.045
    
    if np.any(boolean_f):
        log_freq = np.log10(frequency[boolean_f]) #selects only the elements in the array that are true
        log_weight = p(log_freq)
        weights[boolean_f] = 10**log_weight #calculates the weights and added into the weights array
    
    return weights

def apply_iso_filter_simple(signal, time, dt):
    n = len(signal) #calculates the number of data points in the input signal
    frequencies = np.fft.rfftfreq(n, dt) #calculates the discrete frequencies
    fft_signal = np.fft.rfft(signal) #fourier transform
    iso_weights = ISO_filter(frequencies)
    weighted_fft = fft_signal * iso_weights
    filtered_signal = np.fft.irfft(weighted_fft, n) #converts back into time domain
    
    return filtered_signal

# ============================================================================
# RMS/RMQ CALCULATOR WITH CREST FACTOR
# ============================================================================

def calculate_comfort_metrics(weighted_accel, time):
    """
    Calculate RMS or RMQ based on crest factor as per ISO 2631
    """
    rms_value = np.sqrt(np.mean(weighted_accel**2))
    peak = np.max(np.abs(weighted_accel))
    crest_factor = peak / rms_value if rms_value > 0 else 0
    
    if crest_factor > 9:
        rmq_value = (np.mean(weighted_accel**4))**0.25
        return rmq_value, crest_factor, 'RMQ', rms_value
    else:
        return rms_value, crest_factor, 'RMS', rms_value

# ============================================================================
# SEPARATED PENALTY SYSTEMS - COMPLETELY ISOLATED
# ============================================================================

def calculate_suspension_deflection_penalty_smooth(y_rk, max_suspension_travel=0.15):
    """
    Suspension deflection penalty for SMOOTH roads only
    """
    x1 = y_rk[0, :]
    x2 = y_rk[2, :]
    suspension_deflection = np.abs(x1 - x2)
    max_deflection = np.max(suspension_deflection)
    
    # SMOOTH ROAD PENALTIES ONLY
    soft_limit = 0.01  # 1cm - start penalizing
    hard_limit = max_suspension_travel  # 15cm - maximum
    
    if max_deflection <= soft_limit:
        penalty = 0.0
    elif max_deflection <= hard_limit:
        ratio = (max_deflection - soft_limit) / (hard_limit - soft_limit)
        penalty = 5 * (ratio ** 2)  # 0 to 0.5
    else:
        excess_ratio = (max_deflection - hard_limit) / hard_limit
        penalty = 0.5 + 5.0 * excess_ratio
    
    return penalty, max_deflection

def calculate_suspension_deflection_penalty_rough(y_rk, max_suspension_travel=0.4):
    """
    Suspension deflection penalty for ROUGH roads only
    """
    x1 = y_rk[0, :]
    x2 = y_rk[2, :]
    suspension_deflection = np.abs(x1 - x2)
    max_deflection = np.max(suspension_deflection)
    
    # ROUGH ROAD PENALTIES ONLY - more forgiving
    soft_limit = 0.3  # 12cm - start penalizing
    hard_limit = max_suspension_travel
    
    if max_deflection <= soft_limit:
        penalty = 0.0
    elif max_deflection <= hard_limit:
        ratio = (max_deflection - soft_limit) / (hard_limit - soft_limit)
        penalty =  5 * (ratio ** 2)  # 0 to 0.3
    else:
        excess_ratio = (max_deflection - hard_limit) / hard_limit
        penalty = 0.3 + 3.0 * excess_ratio
    
    return penalty, max_deflection

def calculate_suspension_force_penalty_smooth(y_rk, k, c):
    """
    Suspension force penalty for SMOOTH roads only
    """
    x1 = y_rk[0, :]
    x2 = y_rk[2, :]
    x1_dot = y_rk[1, :]
    x2_dot = y_rk[3, :]
    
    suspension_force = k * (x1 - x2) + c * (x1_dot - x2_dot)
    force_rms = np.sqrt(np.mean(suspension_force**2))
    
    # SMOOTH ROAD FORCE TARGETS
    ideal_min = 270.0
    ideal_max = 1000.0
    
    if ideal_min <= force_rms <= ideal_max:
        penalty = 0.0
    elif force_rms < ideal_min:
        deficit_ratio = (ideal_min - force_rms) / ideal_min
        penalty = 0.5 * (deficit_ratio ** 2)
    else:  # force_rms > ideal_max
        excess_ratio = (force_rms - ideal_max) / ideal_max
        penalty = 1 * (excess_ratio ** 2)
    
    # Cap maximum penalty for smooth roads
    penalty = min(penalty, 1.5)
    
    return penalty, force_rms, suspension_force

def calculate_suspension_force_penalty_rough(y_rk, k, c):
    """
    Suspension force penalty for ROUGH roads only
    """
    x1 = y_rk[0, :]
    x2 = y_rk[2, :]
    x1_dot = y_rk[1, :]
    x2_dot = y_rk[3, :]
    
    suspension_force = k * (x1 - x2) + c * (x1_dot - x2_dot)
    force_rms = np.sqrt(np.mean(suspension_force**2))
    
    # ROUGH ROAD FORCE TARGETS - higher thresholds
    ideal_min = 2000.0
    ideal_max = 5000.0
    
    if ideal_min <= force_rms <= ideal_max:
        penalty = 0.0
    elif force_rms < ideal_min:
        deficit_ratio = (ideal_min - force_rms) / ideal_min
        penalty = 0.1 * (deficit_ratio ** 2)
    else:  # force_rms > ideal_max
        excess_ratio = (force_rms - ideal_max) / ideal_max
        penalty = 5 * (excess_ratio ** 2)
    
    # Cap maximum penalty for rough roads
    penalty = min(penalty, 2.0)
    
    return penalty, force_rms, suspension_force

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

VEHICLE_CONFIGS = {
    'Tesla_Model_X': {
        'M1': 625, 'M2': 50, 'k': 35000, 'kt': 250000, 
        'c_range': (500, 5000), 'c_initial': 2500
    },
    'BMW_i4': {
        'M1': 550, 'M2': 45, 'k': 30000, 'kt': 230000,
        'c_range': (400, 4500), 'c_initial': 2250
    },
    'Nissan_Leaf': {
        'M1': 500, 'M2': 40, 'k': 25000, 'kt': 200000,
        'c_range': (300, 4000), 'c_initial': 2000
    }
}
VEHICLE_SPEED = 10
SIM_TIME = 8
DT = 0.01

# ============================================================================
# ROAD PROFILE GENERATION
# ============================================================================

road_length = 100
No_points_smooth = 20 
No_points_rough = 2000
h_range_smooth = 0.02
h_range_rough = 0.5
No_Road_Profiles = 1

def smooth_road_gen(seed, No_points_smooth, road_length, h_range_smooth):
    np.random.seed(seed)
    x_pos_smooth = np.linspace(0, road_length, No_points_smooth)
    h_values_smooth = np.random.uniform(-h_range_smooth, h_range_smooth, No_points_smooth)
    
    h_values_smooth[0] = 0
    h_values_smooth[-1] = 0

    return splrep(x_pos_smooth, h_values_smooth)

def rough_road_gen(seed, No_points_rough, road_length, h_range_rough):
    """
    Create natural rough road base + sharp impulses for CF > 9
    """
    np.random.seed(seed)
    x_pos_rough = np.linspace(0, road_length, No_points_rough)
    
    # LAYER 1: Natural rough road base
    base_amplitude = 0.05
    h_values_rough = np.random.uniform(-base_amplitude, base_amplitude, No_points_rough)
    window = 5
    h_values_rough = np.convolve(h_values_rough, np.ones(window)/window, mode='same')
    
    # LAYER 2: Medium-scale features
    n_medium_features = np.random.randint(5, 8)
    for _ in range(n_medium_features):
        pos = np.random.uniform(10, road_length - 10)
        idx = np.argmin(np.abs(x_pos_rough - pos))
        width = np.random.randint(20, 40)
        
        for j in range(-width, width):
            if 0 <= idx + j < No_points_rough:
                amplitude = np.random.uniform(0.045, 0.065)
                sign = np.random.choice([-1, 1])
                h_values_rough[idx + j] += sign * amplitude * np.exp(-(j**2)/(2*(width/3)**2))
    
    # LAYER 3: Sharp impulses for high crest factor
    n_impulses = np.random.randint(3, 5)
    impulse_positions = np.linspace(20, road_length - 20, n_impulses)
    
    for pos in impulse_positions:
        idx = np.argmin(np.abs(x_pos_rough - pos))
        impulse_amplitude = np.random.uniform(0.3, 0.6)
        impulse_sign = np.random.choice([-1, 1])
        impulse_width = 5
        
        for j in range(-impulse_width, impulse_width + 1):
            if 0 <= idx + j < No_points_rough:
                decay = 1.0 - abs(j) / (impulse_width + 1)
                h_values_rough[idx + j] += impulse_sign * impulse_amplitude * decay
    
    h_values_rough[0] = 0
    h_values_rough[-1] = 0
    
    return splrep(x_pos_rough, h_values_rough)

# ============================================================================
# ODE SOLVERS
# ============================================================================

def solve_suspension_odes_rk4(c, M1, M2, k, kt, road_spline, vehicle_speed, sim_time, h=0.001):
    def model(y, t):
        x1, x1_dot, x2, x2_dot = y
        position = vehicle_speed * t
        xr = splev(position, road_spline)
        x1_ddot = -(k * (x1 - x2) + c * (x1_dot - x2_dot)) / M1
        x2_ddot = (k * (x1 - x2) + c * (x1_dot - x2_dot) - kt * (x2 - xr)) / M2
        return [x1_dot, x1_ddot, x2_dot, x2_ddot]

    n_step = math.ceil(sim_time/h)
    y_rk = np.zeros((4, n_step+1))
    t_rk = np.zeros(n_step+1)
    y_rk[:, 0] = [0, 0, 0, 0]
    t_rk[0] = 0
    
    for i in range(n_step):
        t_rk[i+1] = t_rk[i] + h
    
    for i in range(n_step):
        y_current = y_rk[:, i]
        t_current = t_rk[i]
        k1 = model(y_current, t_current)
        k2 = model(y_current + np.array(k1) * h/2, t_current + h/2)
        k3 = model(y_current + np.array(k2) * h/2, t_current + h/2)
        k4 = model(y_current + np.array(k3) * h, t_current + h)
        slope = (np.array(k1) + 2*np.array(k2) + 2*np.array(k3) + np.array(k4)) / 6
        y_rk[:, i+1] = y_current + h * slope
    
    return t_rk, y_rk

# ============================================================================
# COMFORT FUNCTIONS - WITH SEPARATED PENALTIES
# ============================================================================

def create_comfort_function_smooth(vehicle_config, road_spline, vehicle_speed, sim_time):
    """
    Comfort function for SMOOTH roads with SMOOTH-only penalties
    """
    M1 = vehicle_config['M1']
    M2 = vehicle_config['M2'] 
    k = vehicle_config['k']
    kt = vehicle_config['kt']
    
    def comfort_function(c):
        t_rk, y_rk = solve_suspension_odes_rk4(c, M1, M2, k, kt, road_spline, 
                                              vehicle_speed, sim_time, h=0.001)
        accel_rk = y_rk[1, :]
        
        # Apply ISO 2631 frequency weighting
        weighted_accel = apply_iso_filter_simple(accel_rk, t_rk, t_rk[1]-t_rk[0])

        # Calculate comfort metrics (RMS/RMQ)
        comfort_value, crest_factor, metric_used, rms_value = calculate_comfort_metrics(weighted_accel, t_rk)
        
        # Use SMOOTH-ONLY penalty functions
        suspension_penalty, max_deflection = calculate_suspension_deflection_penalty_smooth(y_rk)
        force_penalty, force_rms, suspension_force = calculate_suspension_force_penalty_smooth(y_rk, k, c)
        
        total_comfort = comfort_value + suspension_penalty + force_penalty
        
        return total_comfort
    
    return comfort_function

def create_comfort_function_rough(vehicle_config, road_spline, vehicle_speed, sim_time):
    """
    Comfort function for ROUGH roads with ROUGH-only penalties
    """
    M1 = vehicle_config['M1']
    M2 = vehicle_config['M2'] 
    k = vehicle_config['k']
    kt = vehicle_config['kt']
    
    def comfort_function(c):
        t_rk, y_rk = solve_suspension_odes_rk4(c, M1, M2, k, kt, road_spline, 
                                              vehicle_speed, sim_time, h=0.001)
        accel_rk = y_rk[1, :]
        
        # Apply ISO 2631 frequency weighting
        weighted_accel = apply_iso_filter_simple(accel_rk, t_rk, t_rk[1]-t_rk[0])

        # Calculate comfort metrics (RMS/RMQ)
        comfort_value, crest_factor, metric_used, rms_value = calculate_comfort_metrics(weighted_accel, t_rk)

        # Use ROUGH-ONLY penalty functions
        suspension_penalty, max_deflection = calculate_suspension_deflection_penalty_rough(y_rk)
        force_penalty, force_rms, suspension_force = calculate_suspension_force_penalty_rough(y_rk, k, c)
        
        total_comfort = comfort_value + suspension_penalty + force_penalty
        
        return total_comfort
    
    return comfort_function

# ============================================================================
# ROOT FINDING METHODS
# ============================================================================

def bisection(f, a, b, N):
    """Bisection method for finding minimum"""
    print(f"Bisection method: searching in [{a}, {b}] with {N} iterations")
    
    best_c = a
    best_value = f(a)
    
    for n in range(1, N+1):
        mid = (a + b) / 2
        f_mid = f(mid)
        left_mid = (a + mid) / 2
        right_mid = (mid + b) / 2
        f_left = f(left_mid)
        f_right = f(right_mid)
        
        print(f"Bisection iter {n}: c={mid:.1f}, f(c)={f_mid:.6f}")
        
        if f_mid < best_value:
            best_value = f_mid
            best_c = mid
        
        if f_left < f_mid and f_left < f_right:
            b = mid
        elif f_right < f_mid and f_right < f_left:
            a = mid
        else:
            a = left_mid
            b = right_mid
    
    print(f"Bisection final: c={best_c:.1f}, f(c)={best_value:.6f}")
    return best_c, best_value

def newton_numerical(f, x0, x_range, h=10, epsilon=1e-6, max_iter=20):
    """Newton-Raphson method with numerical derivatives"""
    xn = x0
    x_min, x_max = x_range
    
    print(f"Newton-Raphson: starting from c={x0}, range [{x_min}, {x_max}]")
    
    for n in range(max_iter):
        f_xn = f(xn)
        f_plus = f(xn + h)
        f_minus = f(xn - h)
        Df_xn = (f_plus - f_minus) / (2 * h)
        D2f_xn = (f_plus - 2*f_xn + f_minus) / (h**2)
        
        print(f"Newton iter {n+1}: c={xn:.1f}, f(c)={f_xn:.6f}, f'(c)={Df_xn:.6f}")
        
        if abs(Df_xn) < epsilon:
            print(f"Newton converged: gradient is near zero")
            return xn, f_xn
        
        if abs(D2f_xn) < 1e-10:
            print(f"Newton: small second derivative, using gradient descent")
            xn_new = xn - 0.1 * Df_xn
        else:
            xn_new = xn - Df_xn / D2f_xn
        
        xn_new = np.clip(xn_new, x_min, x_max)
        
        if abs(xn_new - xn) < epsilon:
            print(f"Newton converged: small parameter change")
            return xn_new, f(xn_new)
        
        xn = xn_new
    
    print(f"Newton reached maximum iterations")
    return xn, f(xn)

# ============================================================================
# MAIN OPTIMIZATION - COMPLETE OUTPUT FOR ALL CONFIGURATIONS
# ============================================================================

def analyze_vehicle_road_combination(vehicle_name, road_spline, road_type, road_id):
    """
    Analyze suspension performance for a specific vehicle and road combination
    """
    config = VEHICLE_CONFIGS[vehicle_name]
    
    # Choose the appropriate comfort function based on road type
    if road_type == "smooth":
        comfort_func = create_comfort_function_smooth(config, road_spline, VEHICLE_SPEED, SIM_TIME)
        road_name = f"Smooth_Road_{road_id}"
    else:
        comfort_func = create_comfort_function_rough(config, road_spline, VEHICLE_SPEED, SIM_TIME)
        road_name = f"Rough_Road_{road_id}"
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {vehicle_name} on {road_name}")
    print(f"{'='*70}")
    
    # Use BOTH root finding methods
    c_newton, comfort_newton = newton_numerical(
        comfort_func, 
        x0=config['c_initial'],
        x_range=config['c_range'],
        h=50,
        epsilon=1e-6,
        max_iter=15
    )
    
    c_bisection, comfort_bisection = bisection(
        comfort_func, 
        config['c_range'][0], 
        config['c_range'][1], 
        12
    )
    
    # Determine which is better
    if comfort_newton <= comfort_bisection:
        c_optimal = c_newton
        comfort_optimal = comfort_newton
        best_method = "Newton-Raphson"
    else:
        c_optimal = c_bisection  
        comfort_optimal = comfort_bisection
        best_method = "Bisection"
    
    # Get detailed results for optimal point
    t_rk, y_rk = solve_suspension_odes_rk4(c_optimal, config['M1'], config['M2'], 
                                          config['k'], config['kt'], road_spline, 
                                          VEHICLE_SPEED, SIM_TIME, h=0.001)
    accel_rk = y_rk[1, :]
    
    # Apply complete pipeline for optimal case
    weighted_accel_optimal = apply_iso_filter_simple(accel_rk, t_rk, t_rk[1]-t_rk[0])
    comfort_value_opt, crest_factor_opt, metric_used_opt, rms_value_opt = calculate_comfort_metrics(weighted_accel_optimal, t_rk)
    
    # Use appropriate penalty functions based on road type
    if road_type == "smooth":
        suspension_penalty_opt, max_deflection_opt = calculate_suspension_deflection_penalty_smooth(y_rk)
        force_penalty_opt, force_rms_opt, _ = calculate_suspension_force_penalty_smooth(y_rk, config['k'], c_optimal)
    else:
        suspension_penalty_opt, max_deflection_opt = calculate_suspension_deflection_penalty_rough(y_rk)
        force_penalty_opt, force_rms_opt, _ = calculate_suspension_force_penalty_rough(y_rk, config['k'], c_optimal)
    
    # Print detailed breakdown for optimal solution
    print(f"\n{'='*70}")
    print(f"OPTIMAL SOLUTION BREAKDOWN:")
    print(f"  Base Comfort:        {comfort_value_opt:.6f} m/s²")
    print(f"  Suspension Penalty:  {suspension_penalty_opt:.6f}")
    print(f"  Force Penalty:       {force_penalty_opt:.6f}")
    print(f"  TOTAL:              {comfort_optimal:.6f}")
    print(f"  Max Deflection:      {max_deflection_opt*100:.2f} cm")
    print(f"  Force RMS:           {force_rms_opt:.1f} N")
    print(f"  Crest Factor:        {crest_factor_opt:.2f} ({metric_used_opt})")
    print(f"{'='*70}")
    
    # Generate data for plotting
    c_values_plot = np.linspace(config['c_range'][0], config['c_range'][1], 50)
    comfort_values_plot = [comfort_func(c) for c in c_values_plot]
    
    # Generate deflection and force data for plotting
    max_deflections_plot = []
    force_rms_plot = []
    
    for c_val in c_values_plot:
        t_temp, y_temp = solve_suspension_odes_rk4(c_val, config['M1'], config['M2'], 
                                                  config['k'], config['kt'], road_spline, 
                                                  VEHICLE_SPEED, 2.0, h=0.001)
        x1 = y_temp[0, :]
        x2 = y_temp[2, :]
        x1_dot = y_temp[1, :]
        x2_dot = y_temp[3, :]
        
        max_deflections_plot.append(np.max(np.abs(x1 - x2)))
        
        suspension_force = config['k'] * (x1 - x2) + c_val * (x1_dot - x2_dot)
        force_rms_plot.append(np.sqrt(np.mean(suspension_force**2)))
    
    # Store all data for later plotting
    plot_data = {
        'vehicle_name': vehicle_name,
        'road_name': road_name,
        'road_type': road_type,
        'road_spline': road_spline,
        'c_values_plot': c_values_plot,
        'comfort_values_plot': comfort_values_plot,
        'max_deflections_plot': max_deflections_plot,
        'force_rms_plot': force_rms_plot,
        't_rk': t_rk,
        'weighted_accel_optimal': weighted_accel_optimal,
        'c_newton': c_newton,
        'comfort_newton': comfort_newton,
        'c_bisection': c_bisection,
        'comfort_bisection': comfort_bisection,
        'c_optimal': c_optimal,
        'comfort_optimal': comfort_optimal,
        'best_method': best_method,
        'crest_factor_opt': crest_factor_opt,
        'metric_used_opt': metric_used_opt,
        'max_deflection_opt': max_deflection_opt,
        'force_rms_opt': force_rms_opt
    }
    
    return {
        'vehicle': vehicle_name,
        'road_name': road_name,
        'road_type': road_type,
        'c_newton': c_newton,
        'comfort_newton': comfort_newton,
        'c_bisection': c_bisection,
        'comfort_bisection': comfort_bisection,
        'c_optimal': c_optimal,
        'comfort_optimal': comfort_optimal,
        'best_method': best_method,
        'crest_factor': crest_factor_opt,
        'max_deflection': max_deflection_opt,
        'force_rms': force_rms_opt,
        'base_comfort': comfort_value_opt,
        'suspension_penalty': suspension_penalty_opt,
        'force_penalty': force_penalty_opt,
        'plot_data': plot_data
    }

def generate_all_plots(all_results):
    """
    Generate all 5-graph figures for each vehicle-road combination
    """
    print(f"\n{'='*80}")
    print("GENERATING ALL GRAPHS")
    print(f"{'='*80}")
    
    for i, result in enumerate(all_results):
        plot_data = result['plot_data']
        
        print(f"Generating plot {i+1}/{len(all_results)}: {result['vehicle']} on {result['road_name']}")
        
        fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(22, 11))
        _.remove()
        
        # GRAPH 1: Road Profile
        road_positions = np.linspace(0, road_length, 1000)
        road_heights = [splev(pos, plot_data['road_spline']) for pos in road_positions]
        ax1.plot(road_positions, road_heights, 'b-', linewidth=2)
        ax1.set_xlabel('Position (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Road Height (m)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Road Profile: {plot_data["road_name"]}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, road_length])
        
        # GRAPH 2: ISO Filtered Acceleration
        ax2.plot(plot_data['t_rk'], plot_data['weighted_accel_optimal'], 'r-', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Weighted Acceleration (m/s²)', fontsize=12, fontweight='bold')
        ax2.set_title(f'ISO 2631 Filtered Acceleration\n(CF: {plot_data["crest_factor_opt"]:.2f}, {plot_data["metric_used_opt"]})', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, SIM_TIME])
        
        # GRAPH 3: Comfort vs Damping
        ax3.plot(plot_data['c_values_plot'], plot_data['comfort_values_plot'], 'g-', linewidth=3, label='Total Comfort', alpha=0.7)
        ax3.plot(plot_data['c_newton'], plot_data['comfort_newton'], 'bo', markersize=12, markeredgecolor='black', 
                 label=f'Newton: {plot_data["c_newton"]:.0f} Ns/m')
        ax3.plot(plot_data['c_bisection'], plot_data['comfort_bisection'], 'rs', markersize=12, markeredgecolor='black',
                 label=f'Bisection: {plot_data["c_bisection"]:.0f} Ns/m')
        ax3.set_xlabel('Damping Coefficient c (Ns/m)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Comfort Value (m/s²)', fontsize=12, fontweight='bold')
        ax3.set_title('Comfort vs Damping - Root Finding Results', 
                      fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10, loc='upper right')
        
        # GRAPH 4: Suspension Deflection
        ax4.plot(plot_data['c_values_plot'], plot_data['max_deflections_plot'], 'orange', linewidth=3, label='Max Suspension Deflection')
        ax4.axvline(x=plot_data['c_newton'], color='blue', linestyle='--', linewidth=2, label=f'Newton: {plot_data["c_newton"]:.0f}')
        ax4.axvline(x=plot_data['c_bisection'], color='red', linestyle='--', linewidth=2, label=f'Bisection: {plot_data["c_bisection"]:.0f}')
        ax4.axhline(y=0.15, color='black', linestyle='--', linewidth=1.5, 
                    label='Max Limit (0.15 m)', alpha=0.7)
        ax4.set_xlabel('Damping Coefficient c (Ns/m)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Max Suspension Deflection (m)', fontsize=12, fontweight='bold')
        ax4.set_title('Suspension Deflection vs Damping', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        # GRAPH 5: Suspension Force RMS
        ax5.plot(plot_data['c_values_plot'], plot_data['force_rms_plot'], 'purple', linewidth=3, label='Suspension Force RMS')
        ax5.axvline(x=plot_data['c_newton'], color='blue', linestyle='--', linewidth=2, label=f'Newton: {plot_data["c_newton"]:.0f}')
        ax5.axvline(x=plot_data['c_bisection'], color='red', linestyle='--', linewidth=2, label=f'Bisection: {plot_data["c_bisection"]:.0f}')
        
        # Set appropriate force thresholds based on road type
        if plot_data['road_type'] == "smooth":
            ax5.axhline(y=200, color='orange', linestyle='--', linewidth=1.5, 
                        label='Min Target (200 N)', alpha=0.7)
            ax5.axhline(y=2000, color='darkred', linestyle='--', linewidth=1.5, 
                        label='Max Target (2000 N)', alpha=0.7)
        else:
            ax5.axhline(y=500, color='orange', linestyle='--', linewidth=1.5, 
                        label='Min Target (500 N)', alpha=0.7)
            ax5.axhline(y=4000, color='darkred', linestyle='--', linewidth=1.5, 
                        label='Max Target (4000 N)', alpha=0.7)
        
        ax5.set_xlabel('Damping Coefficient c (Ns/m)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Suspension Force RMS (N)', fontsize=12, fontweight='bold')
        ax5.set_title('Suspension Force RMS vs Damping', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9)
        
        # Final formatting
        plt.suptitle(f'{plot_data["vehicle_name"]} - {plot_data["road_name"]}\n'
                     f'Newton: {plot_data["c_newton"]:.0f} Ns/m ({plot_data["comfort_newton"]:.6f}) | '
                     f'Bisection: {plot_data["c_bisection"]:.0f} Ns/m ({plot_data["comfort_bisection"]:.6f}) | '
                     f'Best: {plot_data["best_method"]}', 
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()
        
        print(f"  ✓ Plot generated: {result['vehicle']} on {result['road_name']}")

# ============================================================================
# COMPREHENSIVE ANALYSIS - ALL VEHICLES AND ROADS
# ============================================================================

def run_comprehensive_analysis():
    """
    Run analysis for all vehicle and road combinations
    """
    print("COMPREHENSIVE SUSPENSION ANALYSIS")
    print("=" * 80)
    print("3 VEHICLES × 6 ROADS = 18 COMBINATIONS")
    print("=" * 80)
    
    # Generate road profiles
    road_profiles = {}
    
    # 3 Smooth roads
    for i in range(No_Road_Profiles):
        road_profiles[f"smooth_{i+1}"] = smooth_road_gen(1 + i, No_points_smooth, road_length, h_range_smooth)
    
    # 3 Rough roads  
    for i in range(No_Road_Profiles):
        road_profiles[f"rough_{i+1}"] = rough_road_gen(4 + i, No_points_rough, road_length, h_range_rough)
    
    all_results = []
    
    # Analyze all combinations (without plotting)
    print("\n🚀 RUNNING ALL SIMULATIONS...")
    print("   (Graphs will be generated after all simulations complete)")
    print("-" * 80)
    
    for vehicle_name in VEHICLE_CONFIGS.keys():
        for road_key, road_spline in road_profiles.items():
            road_type, road_id = road_key.split('_')
            result = analyze_vehicle_road_combination(vehicle_name, road_spline, road_type, road_id)
            all_results.append(result)
            
            # Print quick summary for this combination
            print(f"  ✓ {vehicle_name} on {result['road_name']}: "
                  f"c_opt={result['c_optimal']:.0f} Ns/m, "
                  f"comfort={result['comfort_optimal']:.6f}")
    
    return all_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("COMPREHENSIVE VEHICLE-ROAD SUSPENSION ANALYSIS")
    print("=" * 80)
    print("3 Vehicles: Tesla Model X, BMW i4, Nissan Leaf")
    print("6 Roads: 3 Smooth + 3 Rough")
    print("=" * 80)
    
    # Run comprehensive analysis (simulations only)
    all_results = run_comprehensive_analysis()

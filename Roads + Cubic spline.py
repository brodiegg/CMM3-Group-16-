import numpy as np 
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

# Road Parameters

road_length = 100 #100 meters
No_points_smooth = 20 
No_points_rough = 50 

h_range_smooth = 0.02   #0.02 meters, +- 2cm 
h_range_rough = 0.04    #0.04 meters, +- 4cm 

No_profiles = 3         #Number of roads to generater per type

#Dictionaries to store profiles
smooth_roads = {}
rough_roads = {}
 
def smooth_road_gen(seed,No_points_smooth, road_length, h_range_smooth):

    np.random.seed(seed)
    x_pos_smooth = np.linspace(0,road_length,No_points_smooth)
    h_values_smooth = np.random.uniform(-h_range_smooth,h_range_smooth,No_points_smooth)

    h_values_smooth[0]= 0       #start value = 0
    h_values_smooth[-1]= 0      #end value = 0 

    spline_smooth = splrep(x_pos_smooth,h_values_smooth)

    return x_pos_smooth, h_values_smooth, spline_smooth

def rough_road_gen(seed, No_points_rough, road_length, h_range_rough):
    
    np.random.seed(seed)
    x_pos_rough = np.linspace(0, road_length, No_points_rough)
    h_values_rough = np.random.uniform(-h_range_rough, h_range_rough, No_points_rough)

    h_values_rough[0] = 0
    h_values_rough[-1] = 0 

    spline_rough = splrep(x_pos_rough, h_values_rough) # creation of cubic spline 
    
    return x_pos_rough, h_values_rough, spline_rough

for i in range(No_profiles):
    smooth_seed = 1 + i
    rough_seed = 4 + i

    smooth_roads[i] = smooth_road_gen(smooth_seed,No_points_smooth, road_length, h_range_smooth)
    rough_roads[i] = rough_road_gen(rough_seed, No_points_rough, road_length, h_range_rough)

# Create high-resolution points for smooth interpolation
x_high_res = np.linspace(0, road_length, 200)  # 200 high-resolution points

# Plot Smooth Roads
for i, (x_smooth, h_smooth, spline_smooth) in smooth_roads.items():
    h_smooth_high_res = splev(x_high_res, spline_smooth)

    plt.figure()
    plt.scatter(x_smooth, h_smooth, color='red', s=50, label='Control Points')
    plt.plot(x_high_res, h_smooth_high_res, '-b', linewidth=2, label='Smooth Road Profile')
    plt.ylim(-1, 1)
    plt.xlabel('Distance along road (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Smooth Road {i+1} (Seed: {1+i})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Plot Rough Roads
for i, (x_rough, h_rough, spline_rough) in rough_roads.items():
    h_rough_high_res = splev(x_high_res, spline_rough)

    plt.figure()
    plt.scatter(x_rough, h_rough, color='red', s=50, label='Control Points')
    plt.plot(x_high_res, h_rough_high_res, '-b', linewidth=2, label='Rough Road Profile')
    plt.ylim(-1, 1)
    plt.xlabel('Distance along road (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Rough Road {i+1} (Seed: {4+i})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
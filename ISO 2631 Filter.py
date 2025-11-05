# ISO 2631 Filter

import numpy as np 
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt


ISO_freq = np.array([ 0.5, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0,
    5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0,
    50.0, 63.0, 80.0])

ISO_weights = np.array([0.482, 0.484, 0.494, 0.531, 0.631, 0.804, 0.967,
    1.039, 1.054, 1.036, 0.988, 0.902, 0.768, 0.636,
    0.513, 0.405, 0.314, 0.246, 0.186, 0.132,
    0.100, 0.070, 0.045])

# Transform to logs, linearize the exponential decay 

log_f = np.log10(ISO_freq)      #transform x-axis; frequencey to log
log_w = np.log10(ISO_weights)   #transform y-axis; weights to log

# Test different polynomial degrees to determin best fit 
degrees = [1,2,3,4,5,6,7,8,9,10]   # testing degrees 1 - 10
results = {}

for deg in degrees:
    coeffs = np.polyfit(log_f,log_w, deg)
    log_w_prediction = np.polyval(coeffs,log_f)

    difference_w = log_w - log_w_prediction
    
    sum_squares_difference_w = np.sum(difference_w**2)
    sum_squares_total = np.sum((log_w - np.mean(log_w))**2)

    r_squared = 1 -(sum_squares_difference_w/sum_squares_total)

    results[deg]= {'R2':r_squared}

for deg in sorted(results.keys()):
    print(f'Degree {deg}: R2 = {results[deg]['R2']:.5f}')

"""
We tested polynomial degrees 1-10 and found degree 5 provides the best balance with 
R² = 0.998 while avoiding overfitting. Higher degrees showed marginal improvement (<0.001 R²) 
with increased complexity.
"""
# Polynomial Regression 
coeffs = np.polyfit(log_f, log_w, 5)
p = np.poly1d(coeffs)
w_fit = p(log_f)

#Create Filter Function
"""
ISO 2631-1 frequency weighting filter
    
Parameters:
    frequency : array 
        Frequency in Hz
Returns:
    weight : array
        Weighting factor (dimensionless)
 """

def ISO_filter(frequency):

    # Transform to log space
    log_freq = np.log10(frequency)
    
    # Evaluate polynomial in log space
    log_weight = p(log_freq)
    
    # Transform back to linear space
    weight = 10**log_weight
    
    return weight

# Create fine frequency array for smooth line
f_fine = np.linspace(0.5, 80, 1000)
w_fine = ISO_filter(f_fine)

# Simple, clear plot
plt.figure(figsize=(10, 6))

# Plot fitted line
plt.plot(f_fine, w_fine, 'b-', linewidth=2.5, label='Regression Fit (degree 5)')

# Plot ISO data points
plt.scatter(ISO_freq, ISO_weights, c='red', s=100, zorder=5, 
           label='ISO 2631-1 Data Points', edgecolors='darkred', linewidth=2)

# Highlight peak sensitivity region
plt.axvspan(4, 12.5, alpha=0.15, color='yellow', label='Peak Sensitivity (4-12.5 Hz)')

plt.xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
plt.ylabel('Weighting Factor Wk(f)', fontsize=12, fontweight='bold')
plt.title('ISO 2631-1 Frequency Weighting Filter', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim([0, 85])
plt.ylim([0, 1.15])

# Add R² text box
textstr = f'R² = {r_squared:.4f}'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()
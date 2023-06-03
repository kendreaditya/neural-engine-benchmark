import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Set macOS-inspired theme
plt.style.use('ggplot')

# Given points
x_t5 = np.array([0, 60, 220])
y_t5 = np.array([0, 9.23, 22.23])
z_t5 = np.array([0, 0.231, 0.851])

# Additional GPT-2 data
x_gpt2 = np.array([124, 355])
y_gpt2 = np.array([11.73, 31.06])
z_gpt2 = np.array([0.670, 1.6])

# Combined data
x = np.concatenate((x_t5, x_gpt2))
y = np.concatenate((y_t5, y_gpt2))
z = np.concatenate((z_t5, z_gpt2))

# Power-law regression
def power_law_func(x, a, b):
    return a * np.power(x, b)

# Curve fitting for all data points
power_law_coeffs, _ = curve_fit(power_law_func, x, y)

# Generating data for curve
x_curve = np.linspace(0, 500, 100)
y_power_law_curve = power_law_func(x_curve, *power_law_coeffs)

# Plotting the points and curve with transparency
plt.plot(x_curve, y_power_law_curve, alpha=0.5, label='Power-law Regression', linestyle='dashed')
plt.scatter(x_t5, y_t5, color='blue', alpha=0.5, label='T5 Data Points')
plt.scatter(x_gpt2, y_gpt2, color='green', alpha=0.5, label='GPT-2 Data Points')

# Annotating data points with labels
for i, (xi, yi) in enumerate(zip(x_t5, y_t5)):
    if xi == 0:
        continue
    plt.annotate(f'T5 {xi} million parms', (xi, yi), textcoords="offset points", xytext=(10,5), ha='center', fontsize=8, color='blue')
for i, (xi, yi) in enumerate(zip(x_gpt2, y_gpt2)):
    plt.annotate(f'GPT-2 {xi} million parms', (xi, yi), textcoords="offset points", xytext=(10,5), ha='center', fontsize=8, color='green')

# Extrapolation for 11 billion parameters
x_extrapolate = np.array([500])
y_power_law_extrapolate = power_law_func(x_extrapolate, *power_law_coeffs)

# Plotting the extrapolation point with transparency
# plt.scatter(x_extrapolate, y_power_law_extrapolate, color='blue', alpha=0.5, label='Extrapolation')

# Title and subtitle
plt.title('Prediction Time of T5 and GPT-2 Transformer Models', color='white')
plt.suptitle('Number of Parameters vs. Prediction Time', color='white')
plt.xlabel('Number of Parameters (in millions)', color='white')
plt.ylabel('Prediction Time (in ms)', color='white')

# Labeling axes and adding legend
# plt.legend()

# Setting figure background to transparent
plt.gcf().set_facecolor('none')

# Predicting parameter size based on z value for T5 and GPT-2 models
z_predict_t5 = 2
x_predict_t5 = (z_predict_t5 / power_law_coeffs[0]) ** (1 / power_law_coeffs[1])

z_predict_gpt2 = 2
x_predict_gpt2 = (z_predict_gpt2 / power_law_coeffs[0]) ** (1 / power_law_coeffs[1])

# Printing predicted parameter size
print(f"Predicted parameter size for T5 with z = {z_predict_t5}: {x_predict_t5} million params")
print(f"Predicted parameter size for GPT-2 with z = {z_predict_gpt2}: {x_predict_gpt2} million params")

# Displaying the plot
plt.show()

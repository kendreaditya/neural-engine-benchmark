import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Set macOS-inspired theme
plt.style.use('ggplot')

fig, ax1 = plt.subplots()

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

# Power-law regression for x vs y
def power_law_func(x, a, b):
    return a * np.power(x, b)

# Fit the power law regression
params, _ = curve_fit(power_law_func, z, x)

# Extract the fitted parameters
a, b = params

# Predict the values for the input data
z_limit = 2
predicted_z = power_law_func([z_limit], a, b)[0]

# Print the fitted parameters
print("a:", a)
print("b:", b)

# Print the predicted values
print("Predicted values:", predicted_z)

power_law_coeffs, _ = curve_fit(power_law_func, x, y)

# Generating data for regression curve
x_curve = np.linspace(0, predicted_z, 100)
y_power_law_curve = power_law_func(x_curve, *power_law_coeffs)

z_limit_time = power_law_func([predicted_z], *power_law_coeffs)[0]

# Creating the first subplot (x vs y)
ax1.scatter(x_t5, y_t5, color='blue', alpha=0.5, label='T5 Data Points')
ax1.scatter(x_gpt2, y_gpt2, color='green', alpha=0.5, label='GPT-2 Data Points')
ax1.plot(x_curve, y_power_law_curve, 'r-', alpha=0.3, label='Power-law Regression', linestyle='dashed')

ax1.scatter(predicted_z, z_limit_time, color='black', alpha=0.5, label=f"Predicted Neural Engine\nLimit: {round(predicted_z)} million")
ax1.annotate(f"Predicted Neural Engine\nLimit {round(predicted_z)} million @ {z_limit}GB", (predicted_z, z_limit_time), textcoords="offset points", xytext=(10, 5), ha='center', fontsize=8, color='orange')

# Annotating data points with labels for x vs y
for xi, yi in zip(x_t5, y_t5):
    if xi == 0:
        continue
    ax1.annotate(f'T5 ({xi} million params)', (xi, yi), textcoords="offset points", xytext=(10, 5), ha='center', fontsize=8, color='blue')
for xi, yi in zip(x_gpt2, y_gpt2):
    ax1.annotate(f'GPT-2 ({xi} million params)', (xi, yi), textcoords="offset points", xytext=(10, 5), ha='center', fontsize=8, color='green')

ax1.set_xlabel('Number of Parameters (in millions)')
ax1.set_ylabel('Prediction Time (in ms)')

# Set the text, axis numbers, and ticks color to white
ax1.xaxis.label.set_color('white')
ax1.yaxis.label.set_color('white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')

# Adjusting figure size and DPI
fig = plt.gcf()
fig.set_size_inches(7, 5)
fig.set_dpi(150)

# Setting figure background to transparent
fig.patch.set_alpha(0)

# Title and subtitle
plt.suptitle('Prediction Time and Model Parameters/Size\n', color='white')
plt.tight_layout()

# Displaying the plot

ax2 = ax1.secondary_xaxis("top", functions=(lambda x: 2*(x/predicted_z), lambda x: predicted_z * (x/2)))
ax2.set_xlabel('Model Size (in GB)')
ax2.xaxis.label.set_color('white')
ax2.tick_params(axis='x', colors='white')

# Set the figure facecolor to transparent
fig.patch.set_alpha(0)

plt.show()

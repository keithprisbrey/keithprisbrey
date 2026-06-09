import numpy as np
import matplotlib.pyplot as plt

# Generate x values from 0 to 2π (two complete cycles)
x = np.linspace(0, 2 * np.pi, 1000)

# Calculate the sine wave values
y = np.sin(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='blue', label='sin(x)')

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Labels and title
plt.xlabel('x (radians)', fontsize=12)
plt.ylabel('sin(x)', fontsize=12)
plt.title('Sine Wave Plot', fontsize=14, fontweight='bold')

# Add legend
plt.legend(fontsize=11)

# Set x-axis ticks at multiples of π
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', 'π/2', 'π', '3π/2', '2π'])

# Display the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the points and their corresponding labels
points = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
labels = np.array([-1, 1, 1, -1])

# Plot the points with labels
plt.scatter(points[labels == 1][:, 0], points[labels == 1][:, 1], label='+1', marker='o', c='blue')
plt.scatter(points[labels == -1][:, 0], points[labels == -1][:, 1], label='-1', marker='x', c='red')

# Plot lines above and below points with label -1
x_vals = np.linspace(-2, 2, 400)

# Line above points with label -1
plt.plot(x_vals, 1 * x_vals + 0.5, color='green', linestyle='dashed', label='Decision Boundary')

# Line below points with label -1
plt.plot(x_vals, 1 * x_vals - 0.5, color='green', linestyle='dashed')

# Set plot attributes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('f(x) graph for D == 2')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Set legend outside the plot in the upper-right corner
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

# Save the plot in the "Image/q3" directory with the name "decision_boundary_plot.png"
plt.savefig("Image/q3/decision_boundary_plot.png")

# Show the plot
plt.show()

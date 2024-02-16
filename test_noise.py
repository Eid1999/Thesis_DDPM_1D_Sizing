import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define the size of the matrix
matrix_size = (16661, 91)

# Generate Gaussian noise matrix
gaussian_noise = np.random.normal(size=matrix_size)

# Reshape the matrix to the desired size
gaussian_noise = gaussian_noise.reshape(matrix_size)

# Create a heatmap using Seaborn
plt.imshow(gaussian_noise, cmap='viridis',aspect='auto')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Gaussian Noise Heatmap')

# Show the plot
plt.show()
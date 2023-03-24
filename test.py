import numpy as np

#use seed
np.random.seed(0)

# Create two 20x20 matrices
matrix1 = np.random.rand(20, 20)
matrix2 = np.random.rand(20, 20)

# Flatten the matrices into 1D arrays
vector1 = matrix1.flatten()
vector2 = matrix2.flatten()


# Calculate the correlation coefficient between the two vectors
print("Using np.corrcoef")
correlation_coefficient = np.corrcoef(vector1, vector2)[0,1]
print(correlation_coefficient)



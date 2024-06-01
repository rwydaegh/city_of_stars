import numpy as np
import matplotlib.pyplot as plt
from aztec_code_generator import AztecCode

# Generate Aztec code for the given data
data = "https://imgur.com/gallery/just-cat-XM3rdEH"
aztec_code = AztecCode(data)

# Get the matrix representing the Aztec code
aztec_matrix = np.array(aztec_code.matrix)
print(aztec_matrix)

# Now make a larger matrix with random 1 and 0 values that is 10x larger than data
large_matrix = np.random.randint(0, 2, (aztec_matrix.shape[0] * 10, aztec_matrix.shape[1] * 10))

# Create a copy of the larger matrix to avoid overwriting
matrix = large_matrix.copy()

# Add the Aztec code to the center of the matrix
matrix[matrix.shape[0]//2 - aztec_code.size//2 - 1:matrix.shape[0]//2 + aztec_code.size//2, matrix.shape[1]//2 - aztec_code.size//2 - 1:matrix.shape[1]//2 + aztec_code.size//2] = aztec_matrix

# Around the data, add a border of 1s to make it more visible
#matrix[matrix.shape[0]//2 - aztec_code.size//2 - 1:matrix.shape[0]//2 + aztec_code.size//2 + 1, matrix.shape[1]//2 - aztec_code.size//2 - 1:matrix.shape[1]//2 + aztec_code.size//2 + 1] = 1

# Display the Aztec code
plt.imshow(matrix, cmap='gray')
plt.show()
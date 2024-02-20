# Import the PyTorch library
import torch
import torch.nn as nn 
import torch.nn.functional as F

# Define the input size and the number of filters
n = 28 # The size of the input image
f = 32 # The number of filters

# Create a random input image of size n x n
x = torch.randn(n, n)

# Create a convolutional layer with f filters of size 3 x 3, stride 1 and padding 1
conv1 = nn.Conv2d(1, f, 3, padding=1)

# Reshape the input image to a 4D tensor of shape (batch_size, channels, height, width)
x = x.view(1, 1, n, n)

# Apply the convolutional layer to the input image
x = conv1(x)

# Apply the ReLU activation function to the output
x = F.relu(x)

# Print the shape of the output
print(x.shape) # It should be (1, f, n, n)

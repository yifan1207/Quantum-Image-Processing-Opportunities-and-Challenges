# Import the PyTorch library
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Define the class for the fully connected layer 1
class FC1(nn.Module):
    def __init__(self, n):
        # Initialize the parent class
        super(FC1, self).__init__()
        # Define the linear transformation parameters
        self.weight = nn.Parameter(torch.randn(256, 64 * n * n // 16))
        self.bias = nn.Parameter(torch.randn(256))

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 64 * n * n // 16)
        # Apply the linear transformation
        x = torch.matmul(x, self.weight.t()) + self.bias
        # Apply the ReLU activation function
        x = F.relu(x)
        # Return the output tensor
        return x

# Define the input size
n = 28 # Assuming the input is a 28x28 image
# Create an instance of the fully connected layer 1
fc1 = FC1(n)
# Create a random input tensor of size 64x28x28
x = torch.randn(64, n, n)
# Feed the input tensor to the fully connected layer 1
y = fc1(x)
# Print the shape of the output tensor
print(y.shape) # Expected output: torch.Size([64, 256])

# Import the PyTorch library
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Define the class for the fully connected layer 2
class FC2(nn.Module):
    def __init__(self):
        # Initialize the parent class
        super(FC2, self).__init__()
        # Define the linear transformation parameters
        self.weight = nn.Parameter(torch.randn(2, 256))
        self.bias = nn.Parameter(torch.randn(2))

    def forward(self, x):
        # Apply the linear transformation
        x = torch.matmul(x, self.weight.t()) + self.bias
        # Apply the softmax activation function
        x = F.softmax(x, dim=1)
        # Return the output tensor
        return x

# Create an instance of the fully connected layer 2
fc2 = FC2()
# Create a random input tensor of size 64x256
x = torch.randn(64, 256)
# Feed the input tensor to the fully connected layer 2
y = fc2(x)
# Print the shape of the output tensor
print(y.shape) # Expected output: torch.Size([64, 2])

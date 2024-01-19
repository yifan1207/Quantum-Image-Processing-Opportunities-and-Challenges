import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume the input image is a 4D tensor of shape (batch_size, channels, height, width)
# For example, a batch of 10 grayscale images of size 28x28 pixels
image = torch.randn(10, 1, 28, 28)

# Define the pooling layer 1 as a nn.MaxPool2d object with kernel_size=2 and stride=2
pooling_layer_1 = nn.MaxPool2d(2, 2)

# Apply the pooling layer 1 to the input image and get the output feature maps
output = pooling_layer_1(image)

# The output shape should be (batch_size, channels, height/2, width/2)
# For example, (10, 1, 14, 14)
print(output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming the input is a 4D tensor of shape (batch_size, channels, height, width)
# and the output of the convolutional layer 2 is stored in a variable x
x = F.relu(self.conv2(x)) # apply convolutional layer 2

# Define the pooling layer 2 as an instance of nn.MaxPool2d class
# with kernel_size=2 and stride=2
pool2 = nn.MaxPool2d(2, 2)

# Apply the pooling layer 2 to the output of the convolutional layer 2
x = pool2(x) # apply pooling layer 2

# The output of the pooling layer 2 is a 4D tensor of shape
# (batch_size, channels, height/4, width/4)

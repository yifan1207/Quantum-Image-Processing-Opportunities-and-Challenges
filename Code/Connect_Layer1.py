# Assume that x is the output of the second pooling layer, and n is the number of pixels in one dimension of the image
x = x.view(-1, 64 * n * n // 16) # flatten the output
fc1 = nn.Linear(64 * n * n // 16, 256) # define the linear transformation
x = F.relu(fc1(x)) # apply the ReLU activation function

import torch
import torch.nn as nn[^1^][1]
import torch.nn.functional as F[^2^][2]

class QuantumImageCNN(nn.Module):[^3^][3]
    def __init__(self, n):[^4^][4]
        super(QuantumImageCNN, self).__init__()[^5^][5]
        self.n = n
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)[^6^][6][^7^][7]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (n // 4) * (n // 4), 256)[^8^][8]
        self.fc2 = nn.Linear(256, 2)[^9^][9]

    def forward(self, x):[^10^][10]
        x = F.relu(self.conv1(x))[^11^][11][^13^][13]
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))[^12^][12][^11^][11]
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * (self.n // 4) * (self.n // 4))[^14^][14]
        x = F.relu(self.fc1(x))[^13^][13][^11^][11]
        x = self.fc2(x)[^15^][15]
        return F.log_softmax(x, dim=1)

# Example usage:
# Assuming 'n' is the dimension of the image (n x n)
# and 'quantum_image' is a torch.Tensor of shape (batch_size, 1, n, n)
n = 28  # Example image dimension
model = QuantumImageCNN(n)
quantum_image = torch.randn(10, 1, n, n)  # Example batch of quantum images
output = model(quantum_image)

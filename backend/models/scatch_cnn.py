import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ScatchCNN(nn.Module):
    def __init__(self):
        super(ScatchCNN, self).__init__()
        
        # Define convolutional and pooling layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 3)

        self._initialize_flattened_size()

    def _initialize_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.view(-1).size(0)
        
        # Update the first fully connected layer with the correct flattened size
        self.fc1 = nn.Linear(self.flattened_size, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
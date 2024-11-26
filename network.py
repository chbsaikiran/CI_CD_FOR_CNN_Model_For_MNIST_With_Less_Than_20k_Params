import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) #input -? OUtput? RF
        self.bn1 = nn.BatchNorm2d(16)
        self.do1 = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.do2 = nn.Dropout2d(0.1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.transition1 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.do3 = nn.Dropout2d(0.1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.do4 = nn.Dropout2d(0.1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.transition2 = nn.Conv2d(16, 8, kernel_size=1)
        self.conv5 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(8)
        self.do5 = nn.Dropout2d(0.1)
        self.conv6 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(8)
        self.do6 = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear(8 * 7 * 7, 21)  # Assuming MNIST input size of 28x28
        self.fc2 = nn.Linear(21, 10)

    def forward(self, x):
        x = self.pool1(self.do2(self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x))))))))
        x = self.transition1(x)
        x = self.pool2(self.do4(self.bn4(F.relu(self.conv4(self.bn3(F.relu(self.conv3(x))))))))
        x = self.transition2(x)
        x = self.do6(self.bn6(F.relu(self.conv6(self.bn5(F.relu(self.conv5(x)))))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
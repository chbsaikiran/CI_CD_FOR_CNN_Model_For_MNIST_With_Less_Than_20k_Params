import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model architecture (needed to load the state dict)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
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
        self.fc1 = nn.Linear(8 * 7 * 7, 21)
        self.fc2 = nn.Linear(21, 10)

    def forward(self, x):
        x = self.pool1(self.do2(self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x))))))))
        x = self.transition1(x)
        x = self.pool2(self.do4(self.bn4(F.relu(self.conv4(self.bn3(F.relu(self.conv3(x))))))))
        x = self.transition2(x)
        x = self.do6(self.bn6(F.relu(self.conv6(self.bn5(F.relu(self.conv5(x)))))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model():
    model = Net()
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    print("\nTest 1: Checking if total parameters are less than 20000...")
    model = load_model()
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    assert total_params < 20000, "Model has too many parameters"
    print("PASS: Parameter count test successful!")

def test_batch_norm():
    print("\nTest 2: Checking if Batch Normalization is used...")
    model = load_model()
    has_bn = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert has_bn, "Model doesn't use Batch Normalization"
    print("PASS: Batch Normalization test successful!")

def test_dropout():
    print("\nTest 3: Checking if Dropout is used...")
    model = load_model()
    has_dropout = any(isinstance(module, nn.Dropout2d) for module in model.modules())
    assert has_dropout, "Model doesn't use Dropout"
    print("PASS: Dropout test successful!")

def test_fc_layer():
    print("\nTest 4: Checking if Fully Connected Layer is used...")
    model = load_model()
    has_fc = any(isinstance(module, nn.Linear) for module in model.modules())
    assert has_fc, "Model doesn't use Fully Connected Layer"
    print("PASS: Fully Connected Layer test successful!")

if __name__ == "__main__":
    try:
        test_parameter_count()
        test_batch_norm()
        test_dropout()
        test_fc_layer()
        print("\nAll tests passed successfully! ✅")
    except AssertionError as e:
        print(f"\nTest failed: {str(e)} ❌")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: {str(e)} ❌")
        sys.exit(1) 
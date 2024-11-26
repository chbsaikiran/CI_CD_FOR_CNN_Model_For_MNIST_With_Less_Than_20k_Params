import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from network import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    print("\nTest 1: Checking if total parameters are less than 20000...")
    model = Net()
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    assert total_params < 20000, "Model has too many parameters"
    print("PASS: Parameter count test successful!")

def test_batch_norm():
    print("\nTest 2: Checking if Batch Normalization is used...")
    model = Net()
    has_bn = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert has_bn, "Model doesn't use Batch Normalization"
    print("PASS: Batch Normalization test successful!")

def test_dropout():
    print("\nTest 3: Checking if Dropout is used...")
    model = Net()
    has_dropout = any(isinstance(module, nn.Dropout2d) for module in model.modules())
    assert has_dropout, "Model doesn't use Dropout"
    print("PASS: Dropout test successful!")

def test_fc_layer():
    print("\nTest 4: Checking if Fully Connected Layer is used...")
    model = Net()
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
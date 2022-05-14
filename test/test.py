import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2,1)
        self.relu = nn.ReLU()
        
    def execute(self, x):
        x = self.fc1(x)
        return self.relu(x)
    
net = Net()
x = nn.ones((10,2))
y = net(x)

    
import pdb;pdb.set_trace()
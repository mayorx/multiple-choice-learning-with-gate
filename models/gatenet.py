import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, model_nums):
        super(GateNet, self).__init__()
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, model_nums)

    def forward(self, x):
        # x = x.view(-1, 3*32*32)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = GateNet(1)
    # input = torch.autograd.Variable(torch.randn(784))
    # y = net(input)
    # print(net)
    # print(y.size())
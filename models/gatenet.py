import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, model_nums, sm=1):
        super(GateNet, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 20)
        self.fc2 = nn.Linear(20, model_nums)
        self.sm = sm

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.fc1(x)
        x = self.fc2(x)
        if self.sm:
            return F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    net = GateNet()
    input = torch.autograd.Variable(torch.randn(784))
    y = net(input)
    print(net)
    print(y.size())

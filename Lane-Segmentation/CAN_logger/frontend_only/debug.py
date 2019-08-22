import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 1, verbose = True, min_lr = 0.000001)

print(optimizer.step())
print(optimizer.param_groups[0]['lr'])
scheduler.step(100)
print(optimizer.step())
print(optimizer.param_groups[0]['lr'])
scheduler.step(10)
optimizer.step()
print(optimizer.param_groups[0]['lr'])
scheduler.step(11)
optimizer.step()
print(optimizer.param_groups[0]['lr'])
scheduler.step(12)
optimizer.step()
print(optimizer.param_groups[0]['lr'])
scheduler.step(13)
scheduler.step(14)
scheduler.step(15)

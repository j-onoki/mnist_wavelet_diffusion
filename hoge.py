import torch

a = torch.rand((100, 128))
a = a.unsqueeze(2)
a = a.unsqueeze(3)
print(a.size())
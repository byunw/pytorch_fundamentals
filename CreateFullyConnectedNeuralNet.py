import torch
import torch.nn as nn

layer = nn.Linear(4,2)
with torch.no_grad():
    layer.weight.fill_(1.0)
    layer.bias.fill_(2.0)

inputs = torch.tensor([1.0,2.0,3.0,4.0],dtype=torch.float32)
out = layer(inputs)

#test code for output for the fully-connected neural net 
assert out[0] == 12
assert out[1] == 12

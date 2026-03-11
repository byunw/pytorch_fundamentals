import torch

x = torch.tensor([
    [2,3],
    [5,6]
],dtype=torch.float32)
y = torch.tensor([
    [1,1],
    [1,1]
],dtype=torch.float32)

#method 1
z = x @ y
assert z[0][0]==5.0
assert z[0][1]==5.0
assert z[1][0]==11.0
assert z[1][1]==11.0

#method 2
a = torch.tensor([
    [1,1],
    [1,1]
], dtype=torch.float32)

b = torch.tensor([
    [2,2],
    [2,2]
],dtype=torch.float32)
c = torch.matmul(a,b)
assert c[0][0]==4.0
assert c[0][1]==4.0
assert c[1][0]==4.0
assert c[1][1]==4.0

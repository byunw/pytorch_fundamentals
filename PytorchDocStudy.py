import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data,dtype=torch.float32)

assert data[0][0] == 1.0
assert data[0][1] == 2.0
assert data[1][0] == 3.0
assert data[1][1] == 4.0

#creates a tensor with randomized values
tensor = torch.rand(2,2)

#creates a tensor with 1s
tensor2 = torch.ones(2,2)
assert tensor2[0][0]==1.0
assert tensor2[0][1]==1.0
assert tensor2[1][0]==1.0
assert tensor2[1][1]==1.0

#creates a tensor with 0s
tensor3 = torch.zeros(2,2)
assert tensor3[0][0]==0.0
assert tensor3[0][1]==0.0
assert tensor3[1][0]==0.0
assert tensor3[1][1]==0.0

# understand this code
tensor1 = torch.ones(2,2)

if torch.accelerator.is_available:
  tensor2 = tensor1.to(torch.accelerator.current_accelerator())

tensor3 = torch.ones(2,2).cuda()
matrix_multiplication_result = torch.matmul(tensor2,tensor3)

assert matrix_multiplication_result[0][0]==2.0
assert matrix_multiplication_result[0][1]==2.0
assert matrix_multiplication_result[1][0]==2.0
assert matrix_multiplication_result[1][1]==2.0

# element-wise multiplication
tensor4 = torch.zeros(1,1).cuda() #making a tensor in VRAM
tensor5 = torch.zeros(1,1).cuda() #making a tensor in VRAM
element_wise_multiplication_result = tensor4*tensor5
assert element_wise_multiplication_result[0][0]==0.0

tensor6 = torch.ones(2,2).cuda() #making a tensor in VRAM
tensor7 = tensor6.add(1)

assert tensor6[0][0]==1.0
assert tensor6[0][1]==1.0
assert tensor6[1][0]==1.0
assert tensor6[1][1]==1.0

assert tensor7[0][0]==2.0
assert tensor7[0][1]==2.0
assert tensor7[1][0]==2.0
assert tensor7[1][1]==2.0

import math
import numpy as np
import atomgrad.cpu.init as atom_init
import torch
from torch.nn import init
from torch.nn.init import kaiming_uniform_

# def torch_kaiming_initialization(input_size, output_size):
#     gen_w_matrix = torch.empty(size=(input_size, output_size))
#     gen_b_matrix = torch.empty(size=(output_size,))
#     weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
#     fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
#     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#     bias = init.uniform_(gen_b_matrix, -bound, bound)

#     return [torch.tensor(weights, requires_grad=True), torch.tensor(bias, requires_grad=True)]

gen_x = np.random.randn(10, 5)
gen_b = np.random.randn(5)

def test_kaiming_uniform():
    a_weight = atom_init.kaiming_uniform(gen_x, a=math.sqrt(5))
    t_weight = kaiming_uniform_(torch.tensor(gen_x), a=math.sqrt(5), nonlinearity='relu')
    
    print(gen_x[0])
    print(a_weight[0])
    print(t_weight[0])

    satisfy = np.allclose(a_weight, t_weight.numpy())

    # print(satisfy)

test_kaiming_uniform()

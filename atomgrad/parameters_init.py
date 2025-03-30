
import math
import torch
import atomgrad.atom as atom
from torch.nn import init
from torch.nn.init import kaiming_uniform_

'''Collection of parameters initializations'''

def kaiming_initialization(input_size, output_size):
    gen_w_matrix = torch.empty(size=(input_size, output_size))
    gen_b_matrix = torch.empty(size=(output_size,))
    weights = kaiming_uniform_(gen_w_matrix, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = init.uniform_(gen_b_matrix, -bound, bound)

    return [atom.tensor(weights, requires_grad=True), atom.tensor(bias, requires_grad=True)]

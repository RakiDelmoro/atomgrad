import math
import numpy as np
import atomgrad.cpu.init as atom_init
from atomgrad.cpu.atom import tensor

'''Collection of parameters initializations'''

def atom_kaiming_init(input_size, output_size):
    gen_w = np.empty(shape=(output_size, input_size))
    weights = atom_init.kaiming_uniform(gen_w, a=math.sqrt(5))
    fan_in, _ = atom_init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = np.random.uniform(-bound, bound, size=(output_size,))

    return [tensor(weights, requires_grad=True), tensor(bias, requires_grad=True)]

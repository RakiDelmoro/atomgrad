import math
import cupy as cp
import atomgrad.cuda.init as atom_init
from atomgrad.cuda.atom import cuda_tensor
# import init as atom_init
# from atom import cuda_tensor

'''Collection of parameters initializations'''

def atom_kaiming_init(input_size, output_size):
    gen_w = cp.empty(shape=(output_size, input_size))
    weights = atom_init.kaiming_uniform(gen_w, a=math.sqrt(5))
    fan_in, _ = atom_init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = cp.random.uniform(-bound, bound, size=(output_size,))

    return [cuda_tensor(weights, requires_grad=True), cuda_tensor(bias, requires_grad=True)]

def atom_embedding_weight(vocab_size, embedding_dim):
    gen_w = cp.empty(shape=(vocab_size, embedding_dim))
    weight = atom_init.kaiming_uniform(gen_w, a=math.sqrt(5))

    return cuda_tensor(weight, requires_grad=True)

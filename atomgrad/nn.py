import math
import cupy as cp
import numpy as np
from tensor import atom

''' Parameters Initialization'''
def calculate_gain(nonlinearity, param=None):
    if nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float)):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    
    else: raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def _calculate_fan_in_and_fan_out(atom_tensor):
    num_dim = atom_tensor.ndim
    if num_dim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = atom_tensor.shape[1]
    num_output_fmaps = atom_tensor.shape[0]
    receptive_field_size = 1
    if atom_tensor.ndim > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in atom_tensor.shape[2:]: receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(atom_tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(atom_tensor)
    return fan_in if mode == "fan_in" else fan_out

def kaiming_uniform(atom_tensor, a=0, mode='fan_in', nonlinearity='relu'):
    fan = _calculate_correct_fan(atom_tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    return atom.uniform(-bound, bound, size=atom_tensor.shape, device=atom_tensor.device)

def kaiming_params_init(input_size, output_size, device='cpu', bias=True):
    generate_weights = atom.empty((input_size, output_size), device)
    # Applied weight initialization
    weights = kaiming_uniform(generate_weights, a=math.sqrt(5))
    fan_in, _ = _calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    bias = atom.uniform(-bound, bound, size=(output_size,), device=device)

    if bias: return [weights, bias]
    else: return [weights]

'''Neural Network operations'''
def linear(input_size, output_size, device='cpu', bias=True, parameters=None):
    learnable_params = kaiming_params_init(input_size, output_size, device, bias)
    
    if parameters is not None: learnable_params = parameters

    def forward(data):
        weights = learnable_params[0]
        bias = learnable_params[1]

        weight_proj = atom.matmul(data, weights)
        result = weight_proj + bias
        
        return result

    return forward, learnable_params

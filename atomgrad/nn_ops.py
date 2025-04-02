'''Collection of Neural Network operations'''
import atomgrad.atom as atom
import math
import torch
import atomgrad.parameters_init as init

'''NN ops contains the operations for Neural Network this include the forward and backward'''

def linear_layer(input_size, output_size, bias=True):
    #https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py#L50
    weight_param = torch.empty((output_size, input_size))

    torch.nn.init.kaiming_uniform_(weight_param, a=math.sqrt(5))
    if bias:
        bias_param = torch.empty(output_size)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight_param)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(bias_param, -bound, bound)
    else: bias_param = None

    # Convert to ATOM tensor
    weight_param = atom.tensor(weight_param, requires_grad=True)
    bias_param = atom.tensor(bias_param, requires_grad=True)

    def forward(data):
        result = atom.matmul(data, weight_param)
        if bias: result = atom.add(result, bias_param)

        return result

    return forward, weight_param, bias_param

def recurrent_layer():
    pass

def conv_layer():
    pass

def attention_layer():
    pass

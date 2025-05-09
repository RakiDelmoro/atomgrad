'''Collection of Neural Network operations'''
import math
import torch
import atomgrad.cpu.atom as cpu_atom
import atomgrad.cpu.params_init as init

'''NN ops contains the operations for Neural Network this include the forward and backward'''

def linear_layer(input_size, output_size, bias=True):
    parameters = init.atom_kaiming_init(input_size, output_size)

    def forward(data):
        result = cpu_atom.matmul(data, parameters[0])
        if bias: result = cpu_atom.add(result, parameters[1])

        return result

    return forward, parameters

def recurrent_layer():
    pass

def conv_layer():
    pass

def attention_layer():
    pass

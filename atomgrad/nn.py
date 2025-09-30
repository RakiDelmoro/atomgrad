import math
import cupy as cp
import numpy as np
from tensor import atom

class Parameter:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.moment = None
        self.velocity = None

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

    if bias: return [atom(weights.data, device, requires_grad=True), atom(bias.data, device, requires_grad=True)]
    else: return [atom(weights.data, device, requires_grad=True)]

def embeddinigs_params_init(vocab_size, embedding_dim, device='cpu'):
    if device == 'cpu':
        weight = np.random.normal(loc=0.0, scale=1.0, size=(vocab_size, embedding_dim))
    else:
        weight = cp.random.normal(loc=0.0, scale=1.0, size=(vocab_size, embedding_dim))

    return atom(weight, device)

'''Neural Network operations'''
def linear(input_size, output_size, device='cpu', bias=True, parameters=None):
    learnable_params = kaiming_params_init(input_size, output_size, device, bias)
    
    if parameters is not None:
        parameters[0].requires_grad = True
        parameters[1].requires_grad = True
        learnable_params = parameters

    def forward(data):
        weights = learnable_params[0]
        bias = learnable_params[1]

        weight_proj = atom.matmul(data, weights)
        result = weight_proj + bias

        return result

    return forward, learnable_params

def embeddings(num_embeddings, embedding_dim, device='cpu', parameters=None):
    learnable_params = embeddinigs_params_init(num_embeddings, embedding_dim, device)

    if parameters is not None:
        parameters.requires_grad = True
        learnable_params = parameters

    def forward(indices):
        return indices.embeddings_(learnable_params)
    
    return forward, learnable_params

def layer_norm(normalized_shape, eps=1e-5, device='cpu', parameters=None):
    learnable_parameters = []

    if parameters is None:
        weight = atom.ones(shape=(normalized_shape,), device=device, requires_grad=True)
        bias = atom.zeros(shape=(normalized_shape,), device=device, requires_grad=True)
    else:
        weight = parameters[0]
        bias = parameters[1]

    learnable_parameters.extend([weight])
    learnable_parameters.extend([bias])

    def forward(atom_tensor):
        x_normalized = atom_tensor.layer_norm_(eps)
        mul_result = weight * x_normalized

        return mul_result + bias
    
    return forward, learnable_parameters

def dropout(p, train=True, mask=None):
    def forward(atom_tensor):
        return atom_tensor.dropout_(p, train, mask=mask)
    
    return forward

'''Activation Function'''
def relu():

    def forward(atom_tensor):
        return atom_tensor.relu()
    
    return forward

def softmax():
    
    def forward(atom_tensor):
        return atom_tensor.softmax(dim=-1)
    
    return forward

''' LOSS Function'''
def cross_entropy():

    def forward(model_output, expected_output):
        assert model_output.data.dtype in [cp.float32, np.float32], f'model output should have float32 dtype, got {model_output.data.dtype}'
        assert expected_output.data.dtype in [cp.float32, np.float32], f'model output should have float32 dtype, got {expected_output.data.dtype}'

        if expected_output.ndim == 1:
            zeros_arr = np.zeros(model_output.shape) if model_output.device == 'cpu' else cp.zeros(model_output.shape)
            arrange = np.arange(len(model_output.data)) if model_output.device == 'cpu' else cp.arange(len(model_output.data))
            data_type = np.longlong if model_output.device == 'cpu' else cp.longlong
            zeros_arr[arrange, expected_output.data.astype(data_type)] = 1
            expected_output.data = zeros_arr
            expected_output.shape = zeros_arr.shape
        else:
            expected_output = expected_output

        if model_output.device == 'cpu':
            avg_loss = -np.mean(np.sum((expected_output * model_output.log_softmax(dim=-1)).data, axis=-1))
        else:
            avg_loss = -cp.mean(cp.sum((expected_output * model_output.log_softmax(dim=-1)).data, axis=-1))

        def grad_fn(grad):
            model_output.grad = grad
            expected_output.grad = grad

        scalar_loss = atom(avg_loss, model_output.device, model_output.requires_grad, 'cross_entropy', [model_output, expected_output], grad_fn)
        scalar_loss.is_leaf = False

        return scalar_loss
    
    return forward
import cupy as cp
import numpy as np 

# This function makes it if we need to trach the compution for gradient calculation
def atom(data, requires_grad=False, device='cpu'):
    """Properties atom tensor"""

    data = cp.array(data) if device != 'cpu' else np.array(data)

    if requires_grad:
        if device == 'cpu': grad = np.zeros_like(data)
        else: grad = cp.zeros_like(data)
    else: grad = None

    return {'data': data, 'shape': data.shape, 'device': device, 'requires_grad': requires_grad, 'grad': grad, 'grad_fn': None, 'depends_on': []}

# This function should not be care if will include to the computional graph
def tensor(x, requires_grad=False, device='cpu'):
    if device not in ['cpu', 'cuda']: raise ValueError('Device must be cpu or cuda')

    data = np.array(x) if device == 'cpu' else cp.array(x)

    return atom(data, requires_grad, device)

def randn(size, requires_grad=False, device='cpu'):
    if device not in ['cpu', 'cuda']: raise ValueError('Device must be cpu or cuda')

    data = np.random.randn(*size) if device == 'cpu' else cp.random.randn(*size)

    return atom(data, requires_grad, device)

def rand(size, requires_grad=False, device='cpu'):
    if device not in ['cpu', 'cuda']: raise ValueError('Device must be cpu or cuda')

    data = np.random.rand(*size) if device == 'cpu' else cp.random.rand(*size)

    return atom(data, requires_grad, device)

def randint(size, low, high, requires_grad=False, device='cpu'):
    if device not in ['cpu', 'cuda']: raise ValueError('Device must be cpu or cuda')

    data = np.random.randint(low=low, high=high, size=size) if device == 'cpu' else cp.random.randint(low=low, high=high, size=size)

    return atom(data, requires_grad, device)

def zeros(size, requires_grad=False, device='cpu'):
    if device not in ['cpu', 'cuda']: raise ValueError('Device must be cpu or cuda')

    data = np.zeros(*size) if device == 'cpu' else cp.zeros(*size)

    return atom(data, requires_grad, device)

def ones(size, requires_grad=False, device='cpu'):
    if device not in ['cpu', 'cuda']: raise ValueError('Device must be cpu or cuda')

    data = np.ones(*size) if device == 'cpu' else cp.ones(*size)

    return atom(data, requires_grad, device)

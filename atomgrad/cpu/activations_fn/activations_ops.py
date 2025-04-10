import numpy as np
import atomgrad.cpu.atom as cpu_atom

'''Activation ops consist of forward pass and backward pass calculation'''

def softmax_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    shifted_data = atom_tensor['data'] - np.max(atom_tensor['data'], axis=-1, keepdims=True)
    exp_data = np.exp(shifted_data)
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    softmax_data = cpu_atom.cpu_tensor(exp_data / sum_exp_data, requires_grad=requires_grad)

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                sum_term = (softmax_data['data'] * grad).sum(axis=-1, keepdims=True)
                atom_tensor['grad'] = softmax_data['data'] * (grad - sum_term)
            else:
                grad = np.sum(grad, axis=-1)
                sum_term = (softmax_data['data'] * grad).sum(axis=-1, keepdims=True)
                atom_tensor['grad'] = softmax_data['data'] * (grad - sum_term)

    softmax_data['grad_fn'] = backward

    return softmax_data

def log_softmax(atom_tensor):
    shifted = atom_tensor['data'] - np.max(atom_tensor['data'], axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

def relu_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    relu_data = cpu_atom.cpu_tensor(np.maximum(0, atom_tensor['data']), requires_grad=requires_grad)
    relu_data['depends_on'] = [atom_tensor]

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                # Derivative of ReLU applied with chain rule
                atom_tensor['grad'] = np.where(relu_data['data'] > 0, 1, 0) * grad
            else:
                atom_tensor['grad'] = np.where(relu_data['data'] > 0, 1, 0) * np.sum(grad, axis=-1)

    relu_data['grad_fn'] = backward

    return relu_data

def leaky_relu_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    leaky_data = cpu_atom.cpu_tensor(np.maximum(atom_tensor['data'] * 0.05, atom_tensor['data']), requires_grad=requires_grad)
    leaky_data['depends_on'] = [atom_tensor]

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                # Derivative of ReLU applied with chain rule
                atom_tensor['grad'] = np.where(leaky_data['data'] > 0, 1, 0.05 * leaky_data['data']) * grad
            else:
                atom_tensor['grad'] = np.where(leaky_data['data'] > 0, 1, 0.05 * leaky_data['data']) * np.sum(grad, axis=-1)

    leaky_data['grad_fn'] = backward

    return leaky_data

def tanh_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    tanh_data = cpu_atom.cpu_tensor(np.exp(atom_tensor['data']) - np.exp(-atom_tensor['data']))/(np.exp(atom_tensor['data']) + np.exp(-atom_tensor['data']))
    tanh_data['depends_on'] = [atom_tensor]

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                # Derivative of Tanh
                deriv = (np.exp(tanh_data['data']) - np.exp(-tanh_data['data']))/(np.exp(tanh_data['data']) + np.exp(-tanh_data['data']))
                # apply chain rule
                atom_tensor['grad'] = deriv * grad
            else:
                atom_tensor['grad'] = deriv * np.sum(grad, axis=-1)

    tanh_data['grad_fn'] = backward

    return tanh_data

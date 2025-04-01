import numpy as np
import atomgrad.atom as atom

'''Activation ops consist of forward pass and backward pass calculation'''

def softmax_ops(data):
    requires_grad = data['requires_grad']

    shifted_data = data['data'] - np.max(data['data'], axis=-1, keepdims=True)
    exp_data = np.exp(shifted_data)
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    softmax_data = atom.tensor(exp_data / sum_exp_data, requires_grad=True)
        
    # TODO: Figure out the backward fn of softmax
    def backward(grad):
        pass

def relu_ops(data):
    requires_grad = data['requires_grad']

    relu_data = atom.tensor(np.maximum(0, data['data']), requires_grad=True)
    relu_data['depends_on'] = [data]

    def backward(grad):
        if requires_grad:
            if data['grad'].ndim == grad.ndim:
                # Derivative of ReLU applied with chain rule
                data['grad'] = np.where(relu_data['data'] > 0, 1, 0) * grad
            else:
                data['grad'] = np.where(relu_data['data'] > 0, 1, 0) * np.sum(grad, axis=-1)

    relu_data['grad_fn'] = backward

    return relu_data

def leaky_relu_ops(data: dict):
    pass

def tanh_ops(data: dict):
    pass

def sigmoid_ops(data: dict):
    pass

import cupy as cp
# import atomgrad.cuda.atom as atom
import atom as atom

'''Activation ops consist of forward pass and backward pass calculation'''

def softmax_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    shifted_data = atom_tensor['data'] - cp.max(atom_tensor['data'], axis=-1, keepdims=True)
    exp_data = cp.exp(shifted_data)
    sum_exp_data = cp.sum(exp_data, axis=-1, keepdims=True)

    softmax_data = atom.cuda_tensor(exp_data / sum_exp_data, requires_grad=requires_grad)

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                sum_term = (softmax_data['data'] * grad).sum(axis=-1, keepdims=True)
                atom_tensor['grad'] = softmax_data['data'] * (grad - sum_term)
            else:
                grad = cp.sum(grad, axis=-1)
                sum_term = (softmax_data['data'] * grad).sum(axis=-1, keepdims=True)
                atom_tensor['grad'] = softmax_data['data'] * (grad - sum_term)

    softmax_data['grad_fn'] = backward

    return softmax_data

def log_softmax(atom_tensor):
    shifted = atom_tensor['data'] - cp.max(atom_tensor['data'], axis=-1, keepdims=True)
    return shifted - cp.log(cp.sum(cp.exp(shifted), axis=-1, keepdims=True))

def relu_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    relu_data = atom.cuda_tensor(cp.maximum(0, atom_tensor['data']), requires_grad=requires_grad)
    relu_data['depends_on'] = [atom_tensor]

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                # Derivative of ReLU applied with chain rule
                atom_tensor['grad'] = cp.where(relu_data['data'] > 0, 1, 0) * grad
            else:
                atom_tensor['grad'] = cp.where(relu_data['data'] > 0, 1, 0) * cp.sum(grad, axis=-1)

    relu_data['grad_fn'] = backward

    return relu_data

def leaky_relu_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    leaky_data = atom.cuda_tensor(cp.maximum(atom_tensor['data'] * 0.05, atom_tensor['data']), requires_grad=requires_grad)
    leaky_data['depends_on'] = [atom_tensor]

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                # Derivative of ReLU applied with chain rule
                atom_tensor['grad'] = cp.where(leaky_data['data'] > 0, 1, 0.05 * leaky_data['data']) * grad
            else:
                atom_tensor['grad'] = cp.where(leaky_data['data'] > 0, 1, 0.05 * leaky_data['data']) * cp.sum(grad, axis=-1)

    leaky_data['grad_fn'] = backward

    return leaky_data

def tanh_ops(atom_tensor):
    requires_grad = atom_tensor['requires_grad']

    tanh_data = atom.cuda_tensor(cp.exp(atom_tensor['data']) - cp.exp(-atom_tensor['data']))/(cp.exp(atom_tensor['data']) + cp.exp(-atom_tensor['data']))
    tanh_data['depends_on'] = [atom_tensor]

    def backward(grad):
        if requires_grad:
            if atom_tensor['grad'].ndim == grad.ndim:
                # Derivative of Tanh
                deriv = (cp.exp(tanh_data['data']) - cp.exp(-tanh_data['data']))/(cp.exp(tanh_data['data']) + cp.exp(-tanh_data['data']))
                # apply chain rule
                atom_tensor['grad'] = deriv * grad
            else:
                atom_tensor['grad'] = deriv * cp.sum(grad, axis=-1)

    tanh_data['grad_fn'] = backward

    return tanh_data

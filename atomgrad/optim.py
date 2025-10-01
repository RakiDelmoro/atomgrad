import numpy as np
import cupy as cp
from atomgrad.tensor import atom

def SGD(parameters, lr=0.001):
    def update(scale):
        for param in parameters:
            grad = param.grad / scale
            param.data -= lr * grad.data

    def zero_grad():
        for param in parameters:
            param.grad = atom.zeros_like(param, param.device)

    return update, zero_grad

def Adam(parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

    def update():
        # Initialize timestep counter for bias correction
        if not hasattr(update, 't'): update.t = 0

        update.t += 1

        for param in parameters:
            if not hasattr(param, 'moment'):
                param.moment = atom.zeros_like(param, param.device)
            if not hasattr(param, 'velocity'):
                param.velocity = atom.zeros_like(param, param.device)

            param.moment.data = beta1 * param.moment.data + (1 - beta1) * param.grad.data
            param.velocity.data = beta2 * param.velocity.data + (1 - beta2) * (param.grad.data**2)

            m_hat = param.moment.data / (1 - beta1**update.t)
            v_hat = param.velocity.data / (1 - beta2**update.t)
            v_hat = np.sqrt(v_hat) if param.device == 'cpu' else cp.sqrt(v_hat)

            param.data -= lr * (m_hat / (v_hat + epsilon))

    def zero_grad():
        for param in parameters:
            param.grad = atom.zeros_like(param, param.device)

    return update, zero_grad

def AdamW(parameters, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, epsilon=1e-8):

    def update(scale):
        # Initialize timestep counter for bias correction
        if not hasattr(update, 't'): update.t = 0

        update.t += 1

        for param in parameters:
            grad = param.grad / scale
            grad += weight_decay * param

            if not hasattr(param, 'moment'):
                param.moment = atom.zeros_like(param, param.device)
            if not hasattr(param, 'velocity'):
                param.velocity = atom.zeros_like(param, param.device)

            param.moment.data = beta1 * param.moment.data + (1 - beta1) * grad.data
            param.velocity.data = beta2 * param.velocity.data + (1 - beta2) * (grad.data**2)

            m_hat = param.moment.data / (1 - beta1**update.t)
            v_hat = param.velocity.data / (1 - beta2**update.t)
            v_hat = np.sqrt(v_hat) if param.device == 'cpu' else cp.sqrt(v_hat)

            param.data -= lr * (m_hat / (v_hat + epsilon))
            param -= lr * (m_hat / v_hat**2 + epsilon)
    
    def zero_grad():
        for param in parameters:
            param.grad = atom.zeros_like(param, param.device)

    return update, zero_grad
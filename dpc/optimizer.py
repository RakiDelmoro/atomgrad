import numpy as np

def step(parameters: dict, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Initialize timestep counter for bias correction
    if not hasattr(step, 't'):
        step.t = 0
    step.t += 1

    # Collect all trainable parameters from the parameters dictionary
    trainable_params = []
    # Add Vk parameters (transition matrices)
    trainable_params.extend(parameters['vk'])
    # Add hypernetwork parameters (each layer has weight and bias)
    for layer in parameters['hyper_network']:
        trainable_params.extend(layer)
    # Add higher-level RNN parameters (weights and biases for ih and hh)
    for part in parameters['higher_rnn']:
        trainable_params.extend(part)
    # Add digit classifier parameters (weight and bias)
    trainable_params.extend(parameters['digit_classifier'])

    # Update each parameter using Adam optimization rules
    for idx, param in enumerate(trainable_params):
        # Skip if gradient computation is not required
        # if not param.requires_grad:
            # continue

        # Initialize first and second moment estimates if not present
        if not hasattr(param, 'm'):
            param['m'] = np.zeros_like(param['data'])
        if not hasattr(param, 'v'):
            param['v'] = np.zeros_like(param['data'])

        # Retrieve the gradient
        # grad = param['grad'] / 128

        # if idx == len(trainable_params)-2:
            # grad = param['grad'] / 128
        # else:
        grad = param['grad'] / (128*5)

        # Update moments with current gradients
        param['m'] = beta1 * param['m'] + (1 - beta1) * grad
        param['v'] = beta2 * param['v'] + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected moments
        m_hat = param['m'] / (1 - beta1 ** step.t)
        v_hat = param['v'] / (1 - beta2 ** step.t)

        # Apply parameter update
        param['data'] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

import numpy as np

def step(batch_size, parameters: dict, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
    for param in trainable_params:
        # Skip if gradient computation is not required
        if not param['requires_grad']: continue

        grad = (param['grad'] / (batch_size * 5))

        # Initialize first and second moment estimates if not present
        if 'm' not in param:
            param['m'] = np.zeros_like(param['data'])
        if 'v' not in param:
            param['v'] = np.zeros_like(param['data'])

        # Update moments with current gradients
        param['m'] = beta1 * param['m'] + (1 - beta1) * grad
        param['v'] = beta2 * param['v'] + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected moments
        m_hat = param['m'] / (1 - beta1 ** step.t)
        v_hat = param['v'] / (1 - beta2 ** step.t)

        param['data'] -= lr * (m_hat / (np.sqrt(v_hat) + epsilon) + 0.01 * param['data']) 

        # Fix the optimizer it causes the loss to be Nan when using Adam optimizer but when SGD is not getting Nan

def zero_grad(parameters):
    # Collect all trainable parameters from the parameters dictionary
    trainable_params = []
    # Add Vk parameters (transition matrices)

    trainable_params.extend([parameters['lower_network']])
    trainable_params.extend(parameters['vk'])
    # Add hypernetwork parameters (each layer has weight and bias)
    for layer in parameters['hyper_network']:
        trainable_params.extend(layer)
    # Add higher-level RNN parameters (weights and biases for ih and hh)
    for part in parameters['higher_rnn']:
        trainable_params.extend(part)
    # Add digit classifier parameters (weight and bias)
    trainable_params.extend(parameters['digit_classifier'])

    for param in trainable_params:
        if param['grad'] is None: continue
        param['grad'] = np.zeros_like(param['grad'], dtype=np.float32)

import numpy as np

def step(batch_size, parameters: list, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6):
    # Initialize timestep counter for bias correction
    if not hasattr(step, 't'):
        step.t = 0
    
    step.t += 1

    # Update each parameter using Adam optimization rules
    for param in parameters:
        # Skip if gradient computation is not required
        if not param['requires_grad']: continue

        grad = param['grad'] / batch_size

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

def zero_grad(parameters: list):
    for param in parameters:
        if param['grad'] is None: continue
        param['grad'] = np.zeros_like(param['grad'], dtype=np.float32)

import numpy as np

def tensor(data, requires_grad=False):
    """Create a tensor with data and gradient tracking."""
    return {'data': np.array(data, dtype=np.float32), 'shape': np.array(data).shape, 'grad': np.zeros_like(data) if requires_grad else None, 'requires_grad': requires_grad, 'grad_fn': None, 'depends_on': []}

'''TENSOR OPS'''
def add(x1, x2):
    if isinstance(x1, np.ndarray): x1 = tensor(x1)
    if isinstance(x2, np.ndarray): x2 = tensor(x2)

    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = tensor(x1['data'] + x2['data'], requires_grad)
    result['depends_on'] = [x1, x2]
    
    def grad_fn(grad):
        if x1['requires_grad']:
            if x1['grad'].ndim == grad.ndim:
                x1['grad'] += grad
            else:
                x1['grad'] += np.sum(grad, axis=0)
        if x2['requires_grad']:
            if x2['grad'].ndim == grad.ndim:
                x2['grad'] += grad
            else:
                x2['grad'] += np.sum(grad, axis=0)

    result['grad_fn'] = grad_fn

    return result

def relu(x, requires_grad=False):
    if isinstance(x, np.ndarray): x = tensor(x)

    result = tensor(np.maximum(0, x['data']), requires_grad)
    result['depends_on'] = [x]

    def grad_fn(grad):
        if requires_grad:
            # x['grad'] = np.zeros_like(x['data'])
            if x['grad'].ndim == grad.ndim:
                x['grad'] = (np.where(result['data'] > 0, 1, 0) * grad)
            else:
                grad = np.sum(grad, axis=-1)
                x['grad'] = (np.where(result['data'] > 0, 1, 0) * grad)
    result['grad_fn'] = grad_fn

    return result

def softmax(x):
    # Subtract max value for numerical stability
    shifted_data = x['data']- np.max(x['data'], axis=-1, keepdims=True)
    # Calculate exp
    exp_data = np.exp(shifted_data)
    # Sum along axis=1 and keep dimensions for broadcasting
    sum_exp_data = np.sum(exp_data, axis=-1, keepdims=True)

    return exp_data / sum_exp_data

def sub(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    return tensor(x1['data'] - x2['data'], requires_grad)

def broadcasted_mul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    broadcasting_results = []
    for i in range(len(x2)):
        w = x1['data'][:, i].reshape(-1, 1, 1)
        result = w * x2[i]['data']
        broadcasting_results.append(result)

    result = tensor(broadcasting_results, requires_grad=requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        grad = grad[0]
        # for each in grad:
        if x1['requires_grad']:
            for i in range(len(x2)):
                # Multiply by vk_params[i]['data'], then sum over extra dimensions
                x1['grad'][:, i] += np.sum(grad * x2[i]['data'][np.newaxis, :, :], axis=(1, 2))

        if x2[0]['requires_grad']:
            for i, each in enumerate(x2):
                each['grad'] += np.sum(grad * x1['data'][:, i].reshape(-1, 1, 1), axis=0)
    result['grad_fn'] = grad_fn

    return result

def mul(x1, x2, weights, vk_params):
    if isinstance(x1, np.ndarray): x1 = tensor(x1)
    if isinstance(x2, np.ndarray): x2 = tensor(x2)

    requires_grad = x1['requires_grad'] or x2['requires_grad']
    result = tensor(x1['data'] * x2['data'], requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        """Backward function for multiplication."""
        if x1['requires_grad']:
            x1['grad'] = np.zeros((x1['data'].shape[0], len(vk_params)))
            # Accumulate gradients for each vk_param
            for i in range(len(vk_params)):
                # Multiply by vk_params[i]['data'], then sum over extra dimensions
                x1['grad'][:, i] += np.sum(grad * vk_params[i]['data'][np.newaxis, :, :], axis=(1, 2))

        if x2['requires_grad']:
            x2['grad'] += np.sum(grad * x1['data'], axis=0)
    result['grad_fn'] = grad_fn

    return result

def matmul(x1, x2):
    if isinstance(x1, np.ndarray): x1 = tensor(x1)
    if isinstance(x2, np.ndarray): x2 = tensor(x2)

    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = tensor(np.matmul(x1['data'], x2['data'].T), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        if x1['requires_grad']:
            x1['grad'] += (grad @ x2['data'])
        if x2['requires_grad']:
            x2['grad'] += (grad.T @ x1['data']) 
    result['grad_fn'] = grad_fn

    return result

def matmul_3d(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']
    
    x1_data = x1['data']
    x2_data = x2['data']
    
    x1_shape = x1_data.shape
    x2_shape = x2_data.shape

    # Check dimensions
    is_x1_3d = len(x1_shape) == 3
    is_x2_3d = len(x2_shape) == 3
    
    # x1 @ x2 (2d shape size)
    if not is_x1_3d and not is_x2_3d:
        result_data = np.matmul(x1_data, x2_data.T)

    # x1(3d shape size) @ x2(2d shape size) 
    elif is_x1_3d and not is_x2_3d:
        if len(x2_shape) == 1: result_data = np.matmul(x1_data, x2_data)
        elif x2_shape[0] == x1_shape[0]:
            result_data = np.zeros((x1_shape[0], x1_shape[1]))
            for i in range(x1_shape[0]): result_data[i] = np.matmul(x1_data[i], x2_data[i])
        else:
            result_data = np.zeros((x1_shape[0], x1_shape[1], x2_shape[0]))
            for i in range(x1_shape[0]): result_data[i] = np.matmul(x1_data[i], x2_data.T)

    # x1(2d shape size) @ x2(3d shape size)
    elif not is_x1_3d and is_x2_3d:
        if len(x1_shape) == 1:
            result_data = np.zeros((x2_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1_data, x2_data[i])
        elif x1_shape[0] == x2_shape[0]:
            result_data = np.zeros((x2_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1_data[i], x2_data[i])
        else:
            result_data = np.zeros((x2_shape[0], x1_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1_data, x2_data[i])
    
    # x1 @ x2 (3d shape size)
    else:
        # Both 3D case - batch dimensions should match
        if x1_shape[0] != x2_shape[0]:
            raise ValueError("Batch dimensions must match for 3D-3D matmul")
        
        # Case: (batch, m, n) @ (batch, n, p) -> (batch, m, p)
        result_data = np.zeros((x1_shape[0], x1_shape[1], x2_shape[2]))
        for i in range(x1_shape[0]):
            result_data[i] = np.matmul(x1_data[i], x2_data[i])
    
    result = tensor(result_data, requires_grad)
    result['depends_on'] = [x1, x2]
    
    def grad_fn(grad):
        # Zero grad
        # x1['grad'] = np.zeros_like(x1['data'])
        # x2['grad'] = np.zeros_like(x1['data'])

        if is_x1_3d and not is_x2_3d and len(x2_shape) == 2 and x2_shape[0] == x1_shape[0]:
            if x1['requires_grad']:
                for i in range(x1_shape[0]): x1['grad'][i] += np.outer(grad[i], x2_data[i])
            if x2['requires_grad']:
                for i in range(x2_shape[0]): x2['grad'][i] += np.matmul(grad[i], x1_data[i])
        else:
            if x1['requires_grad']: 
                if not is_x1_3d: x1['grad'] += grad @ x2_data
                else:
                    for i in range(x1_shape[0]): x1['grad'][i] += grad[i][:, np.newaxis] @ x2_data[i][np.newaxis, :]
            if x2['requires_grad']:
                if not is_x2_3d: x2['grad'] += grad.T @ x1_data
                else:
                    for i in range(x2_shape[0]): x2['grad'][i] += x1_data[i].T @ grad[i]

    result['grad_fn'] = grad_fn
    return result

def sum_tensors(x):
    result = tensor(sum([t for t in x['data']]), requires_grad=True)
    result['depends_on'] = [x]

    def grad_fn(grad):
        for i in range(len(x['data'])):
            x['grad'][i] += grad

    result['grad_fn'] = grad_fn
    return result

def build_topo(nodes):
    """Build topological order starting from the given node."""
    visited = set()
    topo = []

    def visit(node):
        node_identity = id(node)
        if node_identity not in visited:
            visited.add(node_identity)
            if type(node) == list:
                for each in node:
                    for depends_on in each['depends_on']:
                        visit(depends_on)
            else:
                for depends_on in node['depends_on']:
                    visit(depends_on)
            topo.append(node)
    visit(nodes)
    return topo

def backward(atom_tensor, grad=None):
    """Compute gradients via reverse-mode autodiff."""

    if not atom_tensor['requires_grad']: return

    topo = build_topo(atom_tensor)
    if grad is None: atom_tensor['grad'] = np.ones_like(atom_tensor['data']) if atom_tensor['grad'] is None else atom_tensor['grad']
    else: atom_tensor['grad'] = grad
    
    for node in reversed(topo):
        if type(node) == list:
            for each in node:
                if each['grad_fn'] is not None:
                    each['grad_fn'](each['grad'])

                # Throw it away after calculating/propagate the gradient
                each['depends_on'] = []
                each['grad_fn'] = None

        else:
            if node['grad_fn'] is not None:
                node['grad_fn'](node['grad'])

            # Throw it away after calculating/propagate the gradient
            node['depends_on'] = []
            node['grad_fn'] = None

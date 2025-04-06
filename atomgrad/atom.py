import numpy as np
import atomgrad.ops as ops

def tensor(data, requires_grad=False):
    """Create a tensor with data and gradient tracking."""
    return {'data': np.array(data, dtype=np.float32), 'shape': np.array(data).shape, 'grad': np.zeros_like(data) if requires_grad else None, 'requires_grad': requires_grad, 'grad_fn': None, 'depends_on': []}

'''TENSOR OPS'''

def add(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = tensor(ops._add(x1['data'], x2['data']), requires_grad)
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

def sub(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    return tensor(ops._sub(x1['data'], x2['data']), requires_grad)

# TODO: Transfer the operation in ops.py file
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

def broadcasted_mul(x1, x2):
    pass

def mul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']
    result = tensor(ops._mul(x1['data'], x2['data']), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        """Backward function for multiplication."""
        if x1['requires_grad']:
            x1['grad'] += grad * x2['data']
        if x2['requires_grad']:
            x2['grad'] += grad * x1['data']
    result['grad_fn'] = grad_fn

    return result

def matmul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = tensor(ops._matmul(x1['data'], x2['data'].T), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        x1_shape = x1['shape']
        x2_shape = x2['shape']

        x1_is_3d = x1['data'].ndim == 3
        x2_is_3d = x2['data'].ndim == 3

        if not x1_is_3d and not x2_is_3d:
            if x1['requires_grad']: x1['grad'] += ((grad @ x2['data'])) 
            if x2['requires_grad']: x2['grad'] += (grad.T @ x1['data']) 
        else:
            if x1_is_3d and not x2_is_3d and len(x2_shape) == 2 and x2_shape[0] == x1_shape[0]:
                if x1['requires_grad']:
                    for i in range(x1_shape[0]): x1['grad'][i] += np.outer(grad[i], x2['data'][i])
                if x2['requires_grad']:
                    for i in range(x2_shape[0]): x2['grad'][i] += np.matmul(grad[i], x1['data'][i])
            else:
                if x1['requires_grad']: 
                    if not x1_is_3d: x1['grad'] += grad @ x2['data']
                    else:
                        for i in range(x1_shape[0]): x1['grad'][i] += grad[i][:, np.newaxis] @ x2['data'][i][np.newaxis, :]

                if x2['requires_grad']:
                    if not x2_is_3d: x2['grad'] += grad.T @ x1['data']
                    else:
                        for i in range(x2_shape[0]): x2['grad'][i] += x1['data'][i].T @ grad[i]

    result['grad_fn'] = grad_fn

    return result

def sum_tensors(x: list | dict, axis=0):
    if type(x) == list: list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    else: list_atom_data = [atom_tensor for atom_tensor in x['data']]

    result = tensor(ops._sum_arrays(list_atom_data, axis), requires_grad=True)
    result['depends_on'] = [x]

    def grad_fn(grad):
        for i in range(len(x['data'])):
            x['grad'][i] += grad

    result['grad_fn'] = grad_fn
    return result

def mean_tensor(x: list | dict, axis=0):
    if type(x) == list: list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    else: list_atom_data = [atom_tensor for atom_tensor in x['data']]

    list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    result = tensor(ops._mean_arrays(list_atom_data, axis), requires_grad=True)
    result['depends_on'] = [i for i in x]

    def grad_fn(grad):
        for each_act in x:
            each_act['grad'] += grad

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

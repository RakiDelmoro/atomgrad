import numpy as np

def tensor(data, requires_grad=False):
    """Create a tensor with data and gradient tracking."""
    return {
        'data': np.array(data, dtype=np.float32),
        'shape': np.array(data).shape,
        'grad': np.zeros_like(data) if requires_grad else None,
        'requires_grad': requires_grad,
        'grad_fn': None,
        'depends_on': []
        }


'''TENSOR OPS'''
def add(x1, x2):
    if type(x1) != dict: x1 = tensor(x1)
    if type(x2) != dict: x2 = tensor(x2)

    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = tensor(x1['data'] + x2['data'], requires_grad)
    result['depends_on'] = [x1, x2]
    
    def grad_fn(grad):
        if x1['requires_grad']: x1['grad'] += grad
        if x2['requires_grad']: x2['grad'] += np.sum(grad, axis=0)
    result['grad_fn'] = grad_fn

    return result

def div():
    pass

def pow():
    pass

def relu(x):
    if type(x) != dict:
        x = tensor(x)

    requires_grad = x['requires_grad']

    result = tensor(np.maximum(0, x['data']), requires_grad)
    result['depends_on'] = [x]

    def grad_fn(grad):
        if requires_grad:
            x['grad'] = np.where(result['data'] > 0, 1, 0) * grad
    result['grad_fn'] = grad_fn

    return result

def sub(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    return tensor(x1['data'] - x2['data'], requires_grad)

def mul(x1, x2):
    if type(x1) != dict: x1 = tensor(x1)
    if type(x2) != dict: x2 = tensor(x2)

    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = tensor(x1['data'] * x2['data'], requires_grad)
    result['depends_on'] = [x1, x2]
    
    def grad_fn(grad):
        """Backward function for multiplication."""
        if x1['requires_grad']: x1['grad'] += grad * x2['data']
        if x2['requires_grad']: x2['grad'] += grad * x1['data']
    result['grad_fn'] = grad_fn

    return result

def matmul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = tensor(np.matmul(x1['data'], x2['data'].T), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        if x1['requires_grad']: x1['grad'] += grad @ x2['data']
        if x2['requires_grad']: x2['grad'] += grad.T @ x1['data']
    result['grad_fn'] = grad_fn

    return result

def build_topo(node):
    """Build topological order starting from the given node."""
    visited = set()
    topo = []
    
    def visit(node):
        node_identity = id(node)
        if node_identity not in visited:
            visited.add(node_identity)
            for depends_on in node['depends_on']:
                visit(depends_on)
            topo.append(node)
    visit(node)
    return topo

def backward(atom_tensor, grad=None):
    """Compute gradients via reverse-mode autodiff."""
    if not atom_tensor['requires_grad']:
        return
    
    topo = build_topo(atom_tensor)
    if grad is None:
        atom_tensor['grad'] = np.ones_like(atom_tensor['data'])
    else:
        atom_tensor['grad'] = grad
    
    for node in reversed(topo):
        if node['grad_fn'] is not None:
            node['grad_fn'](node['grad'])

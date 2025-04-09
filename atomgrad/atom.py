import numpy as np
import cupy as cp
import atomgrad.cpu.ops as cpu_ops
import atomgrad.cuda.ops as cuda_ops

def tensor(data, requires_grad=False, device='cpu'):
    """Create a tensor with data and gradient tracking."""

    data = np.array(data, dtype=np.float32) if device == 'cpu' else cp.array(data, dtype=cp.float32)
    if requires_grad:
        if device == 'cpu':
            grad = np.zeros_like(data, dtype=np.float32)
        else:
            grad = cp.zeros_like(data, dtype=np.float32)
    else:
        grad = None

    return {'data': data, 'shape': data.shape, 'device': device, 'grad': grad, 'requires_grad': requires_grad, 'grad_fn': None, 'depends_on': []}

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

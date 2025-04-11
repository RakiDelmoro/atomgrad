import atom
import torch
import cupy as cp
import nn_ops as ops
import random
import numpy as np
import activations_fn.activations as act
import loss_fn.loss_fn_nn as loss_nn

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'
UNDERLINE = "\033[4m"

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

def deriv_softmax():
    # Init
    logits = torch.randn(3, 3)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    grad = torch.randn_like(probs)

    # TORCH deriv softmax
    t_logits = logits.clone().detach().requires_grad_(True)
    t_probs = torch.nn.functional.softmax(t_logits, dim=-1)
    t_loss_scalar = (t_probs * grad).sum()
    t_loss_scalar.backward()
    t_calculated_grad = t_logits.grad

    # ATOM deriv softmax
    a_logits = atom.cuda_tensor(logits.numpy(), requires_grad=True)
    a_probs = act.softmax()(a_logits)
    a_grad = atom.cuda_tensor(grad.numpy())['data']
    # call backward in atom
    a_probs['grad_fn'](a_grad)
    a_calculated_grad = a_logits['grad']

    # for double checking
    print(t_probs)
    print(a_probs['data'])

    satisfied = torch.allclose(torch.tensor(a_calculated_grad), t_calculated_grad)

    if satisfied:
        print(f"softmax derivative --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"softmax derivative --->>> {RED}FAILED{RESET}")

def test_layer_norm():
    # Init
    t_tensor_x = torch.randn(2, 5, requires_grad=True)
    a_tensor_x = atom.cuda_tensor(t_tensor_x.detach().numpy(), requires_grad=True)
    
    t_tensor_y = torch.zeros(2, 5)
    t_tensor_y[torch.arange(len(t_tensor_y)), [random.randint(0, 4) for _ in range(len(t_tensor_x))]] = 1

    a_tensor_y = t_tensor_y.detach().numpy()

    # Loss fn
    t_loss_fn = torch.nn.CrossEntropyLoss()
    a_loss_fn = loss_nn.cross_entropy_loss()

    a_layer_norm, params = ops.layer_norm(5)
    t_layer_norm = torch.nn.LayerNorm(5)

    a_res = a_layer_norm(a_tensor_x)
    t_res = t_layer_norm(t_tensor_x)

    t_res.retain_grad()
    t_tensor_x.retain_grad()

    # Backprop
    t_loss = t_loss_fn(t_res, t_tensor_y)
    t_loss.backward()

    a_avg_loss, a_grad = a_loss_fn(a_res, a_tensor_y)
    backward(a_res, a_grad)

    a_tensor_x_grad = a_tensor_x['grad'] / len(t_tensor_x)

    backward_satisfied = torch.allclose(torch.tensor(cp.asnumpy(a_tensor_x_grad)), t_tensor_x.grad)
    # backward_satisfied = torch.tensor(a_tensor_x_grad) == t_tensor_x.grad
    
    # Double check
    print(t_tensor_x.grad)
    print(a_tensor_x['grad'] / len(t_tensor_x))

    if backward_satisfied:
        print(f"layer norm backward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"layer norm backward --->>> {RED}FAILED{RESET}")

    forward_satisfied = torch.allclose(torch.tensor(cp.asnumpy(a_res['data'])), t_res)

    if forward_satisfied:
        print(f"layer norm forward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"layer norm forward --->>> {RED}FAILED{RESET}")

def test_dropout():
    # Init
    pass

# deriv_softmax()
# test_dropout()
test_layer_norm()

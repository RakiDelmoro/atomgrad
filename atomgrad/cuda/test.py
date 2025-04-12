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
    # print(t_probs)
    # print(a_probs['data'])

    satisfied = torch.allclose(torch.tensor(a_calculated_grad), t_calculated_grad, atol=1e-5)

    if satisfied:
        print(f"softmax derivative --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"softmax derivative --->>> {RED}FAILED{RESET}")

def test_layer_norm():
    # Init
    t_tensor_x = torch.randn(1, 5, requires_grad=True)
    a_tensor_x = atom.cuda_tensor(t_tensor_x.detach().numpy(), requires_grad=True)
    
    t_tensor_y = torch.zeros(1, 5)
    t_tensor_y[torch.arange(len(t_tensor_y)), [random.randint(0, 4) for _ in range(len(t_tensor_x))]] = 1

    a_tensor_y = cp.array(t_tensor_y.detach().numpy())

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

    mean_atom_grad = cp.mean(a_tensor_x_grad, axis=0)
    mean_torch_grad = torch.mean(t_tensor_x.grad, dim=0)

    forward_satisfied = torch.allclose(torch.tensor(cp.asnumpy(a_res['data'])), t_res, atol=1e-5)
    backward_satisfied = torch.allclose(mean_torch_grad, torch.tensor(mean_atom_grad), atol=1e-5)

    # NOTE: Sometimes the torch.allclose is not Happy/Satisfied because of the precision of floating point just uncomment the printing for double checking
    # print(mean_atom_grad)
    # print(mean_torch_grad)

    if forward_satisfied:
        print(f"layer norm forward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"layer norm forward --->>> {RED}FAILED{RESET}")

    if backward_satisfied:
        print(f"layer norm backward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"layer norm backward --->>> {RED}FAILED{RESET}")

def test_embedding():
    # Init
    # batch, seq_len
    BATCH = 2
    SEQ_LEN = 3
    VOCAB_SIZE = 5

    indices = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH, SEQ_LEN))
    expected = torch.zeros(BATCH*SEQ_LEN, VOCAB_SIZE)
    expected[torch.arange(len(expected)), indices.view(BATCH*SEQ_LEN)] = 1
    pos_emb = torch.arange(SEQ_LEN)

    torch_loss_fn = torch.nn.CrossEntropyLoss()
    atom_loss_fn = loss_nn.cross_entropy_loss()

    torch_char_embedding = torch.nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
    torch_pos_embedding = torch.nn.Embedding(SEQ_LEN, VOCAB_SIZE)
    # add parameters to use #NOTE: for testing and comparing to TORCH
    atom_char_embedding, char_parameter = ops.embeddings(VOCAB_SIZE, VOCAB_SIZE, torch_char_embedding.weight.data)
    atom_pos_embedding, pos_parameter = ops.embeddings(BATCH, VOCAB_SIZE, torch_pos_embedding.weight.data)

    # TORCH and ATOM forward pass
    torch_emb_char = torch_char_embedding.forward(indices)
    torch_emb_pos = torch_pos_embedding.forward(pos_emb)
    torch_output = torch_emb_char + torch_emb_pos
    torch_emb_char.retain_grad()
    torch_emb_pos.retain_grad()

    atom_emb_char = atom_char_embedding(atom.cuda_tensor(indices.numpy(), requires_grad=True))
    atom_emb_pos = atom_pos_embedding(atom.cuda_tensor(pos_emb.numpy(), requires_grad=True))
    atom_output = atom.add(atom_emb_char, atom_emb_pos)

    char_emb_satisfied = torch.allclose(torch.tensor(cp.asnumpy(atom_emb_char['data'])), torch_emb_char, atol=1e-5)
    pos_emb_satisfied = torch.allclose(torch.tensor(cp.asnumpy(atom_emb_pos['data'])), torch_emb_pos, atol=1e-5)
    forward_satisfied = torch.allclose(torch.tensor(cp.asnumpy(atom_output['data'])), torch_output, atol=1e-5)
    
    if forward_satisfied:
        print(f"Embedding forward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"Embedding forward --->>> {RED}FAILED{RESET}")

    if char_emb_satisfied:
        print(f"Char Embedding forward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"Char Embedding forward --->>> {RED}FAILED{RESET}")

    if pos_emb_satisfied:
        print(f"Pos Embedding forward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"Pos Embedding forward --->>> {RED}FAILED{RESET}")

    torch_loss = torch_loss_fn(torch_output.view(BATCH*SEQ_LEN, VOCAB_SIZE), expected)
    torch_loss.backward()

    atom_output_data = atom_output['data'].reshape(BATCH*SEQ_LEN, VOCAB_SIZE)
    atom_output['data'] = atom_output_data    
    atom_expected = atom.cuda_tensor(expected.numpy())['data']
    atom_loss, atom_loss_grad = atom_loss_fn(atom_output, atom_expected)
    backward(atom_output, atom_loss_grad.reshape(BATCH, SEQ_LEN, VOCAB_SIZE))

    atom_char_grad = char_parameter['grad'] / (BATCH*SEQ_LEN)
    atom_pos_grad = pos_parameter['grad'] / (BATCH*SEQ_LEN)

    char_grad_satisfied = torch.allclose(torch.tensor(cp.asnumpy(atom_char_grad)), torch_char_embedding.weight.grad)
    pos_grad_satisfied = torch.allclose(torch.tensor(cp.asnumpy(atom_pos_grad)), torch_pos_embedding.weight.grad)

    if char_grad_satisfied:
        print(f"Char Embedding backward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"Char Embedding backward --->>> {RED}FAILED{RESET}")

    if pos_grad_satisfied:
        print(f"Pos Embedding backward --->>> {GREEN}PASSED{RESET}")
    else:
        print(f"Pos Embedding backward --->>> {RED}FAILED{RESET}")

def test_pos_embedding():
    pass

# deriv_softmax()
# test_layer_norm()
test_embedding()

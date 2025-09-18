import torch
import numpy as np
from tensor import atom
import nn 

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def test_zeros():
    try:
        atom.zeros((2, 3), device='cuda', requires_grad=True)
        atom.zeros((2, 3), device='cpu', requires_grad=False)
        print(f'Test generate array of zeros --> {GREEN}Pass!{RESET}')
    except:
        print(f'Test generate array of zeros --> {RED}Failed!{RESET}')

def test_empty():
    try:
        atom.empty((2, 3), device='cpu')
        atom.empty((2, 3), device='cuda')
        print(f'Test generate random value in array --> {GREEN}Pass!{RESET}')
    except:
        print(f'Test generate random value in array --> {RED}Failed!{RESET}')

def test_add_ops():
    x1_cuda = atom.zeros((2, 3), device='cuda')
    x2_cuda = atom.ones((2, 3), device='cuda')

    x1_cpu = atom.zeros((2, 3), device='cpu')
    x2_cpu = atom.ones((2, 3), device='cpu')

    try:
        x1_cuda + x2_cuda
        x2_cuda + x1_cuda
        x1_cpu + x2_cpu
        x2_cpu + x1_cpu
        print(f'Test adding two arrays --> {GREEN}Pass!{RESET}')
    except:
        print(f'Test adding two arrays --> {RED}Failed!{RESET}')

def test_matmul_for_2d():
    x1_atom = atom.randn((2, 3), device='cpu')
    x2_atom = atom.randn((3, 2), device='cpu')

    # Check if it has error
    try:
        atom.matmul(x1_atom, x2_atom)
        print(f'Test if matmul works --> {GREEN}Pass!{RESET}')
    except:
        print(f'Test if matmtul works --> {RED}Failed!{RESET}')

    # Comparing Torch and Atom
    x1_torch = torch.tensor(x1_atom.data, dtype=torch.float32)
    x2_torch = torch.tensor(x2_atom.data, dtype=torch.float32)

    y_atom = atom.matmul(x1_atom, x2_atom)
    y_torch = torch.matmul(x1_torch, x2_torch)

    satisfied = np.allclose(y_atom.data, y_torch.numpy())

    if satisfied:
        print(f'Comparing matmul ops of Torch and Atom --> {GREEN}Pass!{RESET}')
    else:
        print(f'Comparing matmul ops of Torch and Atom --> {RED}Failed!{RESET}')

def test_linear_ops():
    x_test = torch.randn(2, 10)
    y_test = torch.nn.functional.one_hot(torch.randint(0, 5, size=(2,)), num_classes=5).float()

    torch_ln = torch.nn.Linear(10, 5)
    torch_loss_fn = torch.nn.CrossEntropyLoss()

    atom_x = atom(x_test, device='cuda', requires_grad=True)
    atom_w = atom(torch_ln.weight.detach().numpy().T, device='cuda')
    atom_b = atom(torch_ln.bias.detach().numpy().T, device='cuda')
    atom_ln, atom_params = nn.linear(input_size=10, output_size=5, device='cuda', parameters=[atom_w,atom_b])
    
    y_atom = atom_ln(atom_x)
    y_torch = torch_ln(x_test)

    torch_loss = torch_loss_fn(y_torch, y_test)
    y_torch.retain_grad()
    
    # Torch automatic gradient calculation
    torch_loss.backward()
    # Manual gradient calculation
    grad = (y_torch.softmax(dim=-1) - y_test) / y_torch.shape[0]

    print(y_torch.grad)
    print()
    print(grad)


    # TODO: Add compare the Torch and Atom for both forward pass and backward pass to make sure Atom works correctly

test_zeros()
test_empty()
test_add_ops()
test_matmul_for_2d()
test_linear_ops()

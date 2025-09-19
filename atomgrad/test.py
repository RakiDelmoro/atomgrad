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

    print(torch_loss)

    # print(y_torch.grad)
    # print()
    # print(grad)

    # TODO: Add compare the Torch and Atom for both forward pass and backward pass to make sure Atom works correctly

def test_relu():
    x_atom = atom.randn((2, 5))
    x_torch = torch.tensor(x_atom.data)

    y_atom = x_atom.relu() 
    y_torch = x_torch.relu()

    satified = np.allclose(y_atom.data, y_torch.numpy())

    if satified:
        print(f'ReLu test --> {GREEN}Pass!{RESET}')
    else:
        print(f'ReLu test --> {RED}Failed!{RESET}')

def test_softmax():
    x_atom = atom.randn((2, 5))
    x_torch = torch.tensor(x_atom.data)

    y_atom = x_atom.softmax(dim=-1) 
    y_torch = x_torch.softmax(dim=-1)

    satified = np.allclose(y_atom.data, y_torch.numpy())

    if satified:
        print(f'Softmax test --> {GREEN}Pass!{RESET}')
    else:
        print(f'Softmax test --> {RED}Failed!{RESET}')

def test_log_softmax():
    x_atom = atom.randn((2, 5))
    x_torch = torch.tensor(x_atom.data)

    y_atom = x_atom.log_softmax(dim=-1) 
    y_torch = x_torch.log_softmax(dim=-1)

    satified = np.allclose(y_atom.data, y_torch.numpy())

    if satified:
        print(f'Log Softmax test --> {GREEN}Pass!{RESET}')
    else:
        print(f'Log Softmax test --> {RED}Failed!{RESET}')

def test_one_hot():
    x_atom = atom.randint(0, 5, size=(2,))
    x_torch = torch.tensor(x_atom.data, dtype=torch.int64) # torch one hot funtion expect tensor should have dtype int64

    y_atom = x_atom.one_hot(num_classes=5)
    y_torch = torch.nn.functional.one_hot(x_torch, num_classes=5)

    satified = np.allclose(y_atom.data, y_torch.numpy())

    if satified:
        print(f'One hot test --> {GREEN}Pass!{RESET}')
    else:
        print(f'One hot test --> {RED}Failed!{RESET}')

def test_cross_entropy():
    x_atom = atom.randn((2, 5), requires_grad=True)
    y = atom.randint(0, 5, size=(2,), requires_grad=True)
    y_atom_one_hot = y.one_hot(num_classes=5)

    x_torch = torch.tensor(x_atom.data, dtype=torch.float32, requires_grad=True)
    y_torch_one_hot = torch.tensor(y_atom_one_hot.data, dtype=torch.float32, requires_grad=True)

    atom_loss_fn = nn.cross_entropy()
    torch_loss_fn = torch.nn.CrossEntropyLoss()
    x_torch.retain_grad()
    y_torch_one_hot.retain_grad()

    atom_loss = atom_loss_fn(x_atom, y_atom_one_hot)
    torch_loss = torch_loss_fn(x_torch, y_torch_one_hot)

    torch_loss.retain_grad()

    torch_loss.backward()
    atom_loss.backward()

    print(x_torch.grad)
    print(x_atom.grad.data / 2)
    print()
    print(torch_loss.grad)
    print(atom_loss.grad)

test_zeros()
test_empty()
test_add_ops()
test_matmul_for_2d()
# test_linear_ops()
test_softmax()
test_relu()
test_log_softmax()
test_one_hot()
test_cross_entropy()
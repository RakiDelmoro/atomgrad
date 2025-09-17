import torch
from tensor import atom

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def test_zeros():
    try:
        atom.zeros((2, 3), device='cuda', requires_grad=True)
        atom.zeros((2, 3), device='cpu', requires_grad=False)
        print(f'{GREEN}Pass!{RESET}')
    except:
        print(f'{RED}Failed!{RESET}')

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
        print(f'{GREEN}Pass!{RESET}')
    except:
        print(f'{RED}Failed!{RESET}')

def test_empty():
    try:
        atom.empty((2, 3), device='cpu')
        atom.empty((2, 3), device='cuda')
        print(f'{GREEN}Pass!{RESET}')
    except:
        print(f'{RED}Failed!{RESET}')

def test_backprop():
    x1 = atom.randn((2, 3), device='cuda', requires_grad=True)
    x2 = atom.randn((2, 3), device='cuda', requires_grad=True)

    test_grad = atom.randn((2, 3), device='cuda')

    y = x1 + x2

    y.backward(test_grad)

    print(x1.grad)
    print(x2.grad)
    print(y.grad)

test_zeros()
test_add_ops()
test_empty()
test_backprop()

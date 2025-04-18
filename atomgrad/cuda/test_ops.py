import atom
import cupy as cp

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def test_matmul():
    # Test case x1 (2, 3, 5) x2 (2, 3, 5)
    # y (2, 3, 3)
    x1 = atom.cuda_tensor(cp.random.randn(2, 3, 5))
    x2 = atom.cuda_tensor(cp.random.randn(2, 3, 5))
    try:
        x1_shape = x1['shape']
        x2_shape = x2['shape']
        y = atom.matmul(x1, x2)
        print(f'Matrix Multiply test.... array1 shape: {x1_shape} array2 shape: {x2_shape} {GREEN}PASSED!{RESET}')
    except ValueError:
        print(f'Matrix Multiply test.... array1 shape: {x1_shape} array2 shape: {x2_shape} {RED}FAIL!{RESET}')

    # Test case x1 (2, 3, 5) x2 (10, 5)
    # y (2, 3, 10)
    x1 = atom.cuda_tensor(cp.random.randn(2, 3, 5))
    x2 = atom.cuda_tensor(cp.random.randn(10, 5))
    y = atom.matmul(x1, x2)
    try:
        x1_shape = x1['shape']
        x2_shape = x2['shape']
        y = atom.matmul(x1, x2)
        print(f'Matrix Multiply test.... array1 shape: {x1_shape} array2 shape: {x2_shape} {GREEN}PASSED!{RESET}')
    except ValueError:
        print(f'Matrix Multiply test.... array1 shape: {x1_shape} array2 shape: {x2_shape} {RED}FAIL!{RESET}')

    # Test case x1 (2, 3, 3) x2 (2, 3, 5)
    # y (2, 3, 5)
    x1 = atom.cuda_tensor(cp.random.randn(2, 3, 3))
    x2 = atom.cuda_tensor(cp.random.randn(2, 3, 5))
    y = atom.matmul(x1, x2)
    try:
        x1_shape = x1['shape']
        x2_shape = x2['shape']
        y = atom.matmul(x1, x2)
        print(f'Matrix Multiply test.... array1 shape: {x1_shape} array2 shape: {x2_shape} {GREEN}PASSED!{RESET}')
    except ValueError:
        print(f'Matrix Multiply test.... array1 shape: {x1_shape} array2 shape: {x2_shape} {RED}FAIL!{RESET}')

test_matmul()

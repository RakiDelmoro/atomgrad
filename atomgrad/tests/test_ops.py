import ops
import torch
import numpy as np

RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'

def test_matmul_ops():
    # 2d test case
    x1 = np.random.randn(1, 10)
    x2 = np.random.randn(10, 5)

    # 3d test case
    x1_3d = np.random.randn(2, 256)
    x2_3d = np.random.randn(2, 256, 256)

    # 2d calculation
    atom_res = ops.matmul(x1, x2)
    torch_res = torch.matmul(torch.tensor(x1), torch.tensor(x2))
    satisfy_2d = np.allclose(atom_res, torch_res.numpy())

    #LOOK OUT for this! 👀
    # 3d calculation
    atom_res_3d = ops.matmul(x2_3d, x1_3d.T)
    torch_res_3d = torch.matmul(torch.tensor(x1_3d), torch.tensor(x2_3d))
    print(torch_res_3d.shape)
    print(atom_res_3d.shape)
    # satisfy_3d = np.allclose(atom_res_3d, torch_res_3d.numpy())
    satisfy_3d = False

    if satisfy_2d: print(f'test matmul ops >>>> {GREEN}{satisfy_2d}{RESET}')
    else: print(f'test 2D matmul ops >>>> {RED}{satisfy_2d}{RESET}')

    if satisfy_3d: print(f'matmul ops >>>> {GREEN}{satisfy_3d}{RESET}')
    else: print(f'test 3D matmul ops >>>> {RED}{satisfy_3d}{RESET}')

def test_add_ops():
    pass

def test_mean_ops():
    pass

def test_sub_ops():
    pass

def test_mul_ops():
    pass

test_matmul_ops()

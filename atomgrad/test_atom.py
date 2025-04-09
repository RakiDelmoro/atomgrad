import tensor as atom
import atomgrad.cpu.atom as atom_ops

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'
UNDERLINE = "\033[4m"

def init_tensor_test():
    print(f'{UNDERLINE}INIT TENSOR TEST{RESET}')

    try:
        atom.tensor([1, 2, 3], requires_grad=True, device='cuda')
        print(f"tensor test --->>> {GREEN}PASSED{RESET}")
    except: print(f'randn test --->>> {RED}FAILED{RESET}')

    try:
        atom.randn((3, 3), requires_grad=True, device='cuda')
        print(f"randn test --->>> {GREEN}PASSED{RESET}")
    except: print(f'randn test --->>> {GREEN}FAILED{RESET}')

    try:
        atom.rand((3, 3), requires_grad=True, device='cuda')
        print(f"rand test --->>> {GREEN}PASSED{RESET}")
    except: print(f'rand test --->>> {RED}FAILED{RESET}')

    try:
        atom.randint(low=0, high=5, size=(3, 3), requires_grad=True, device='cuda')
        print(f"randint test --->>> {GREEN}PASSED{RESET}")
    except: print(f'randint test --->>> {RED}FAILED{RESET}')

def ops_tensor_test():
    print(f'{UNDERLINE}TENSOR CALCULATIONS{RESET}')

    generate_x = atom.tensor([1, 2, 3], device='cuda', requires_grad=True)['data']
    generate_y = atom.tensor([4], device='cuda', requires_grad=True)['data']

    matmul_x = atom.randn(size=(2, 4), requires_grad=True, device='cuda')['data']
    matmul_y = atom.randn(size=(3, 4), requires_grad=True, device='cuda')['data']

    try:
        add_tensor = atom_ops.add(generate_x, generate_y)
        print(f"adding tensor test --->>> {GREEN}PASSED{RESET}")
    except: print(f'adding tensor test --->>> {RED}FAILED{RESET}')

    try:
        sub_tensor = atom_ops.sub(generate_x, generate_y)
        print(f"subracting tensor test --->>> {GREEN}PASSED{RESET}")
    except: print(f'subracting tensor test --->>> {RED}FAILED{RESET}')

    try:
        mul_tensor = atom_ops.mul(generate_x, generate_y)
        print(f"multiply tensor test --->>> {GREEN}PASSED{RESET}")
    except: print(f'multiply tensor test --->>> {RED}FAILED{RESET}')

    try:
        matmul_tensor = atom_ops.matmul(matmul_x, matmul_y)
        print(f"matmul tensor test --->>> {GREEN}PASSED{RESET}")
    except: print(f'matmul tensor test --->>> {RED}FAILED{RESET}')

    # print(add_tensor['data'])
    # print(sub_tensor['data'])
    # print(mul_tensor['data'])

init_tensor_test()
ops_tensor_test()

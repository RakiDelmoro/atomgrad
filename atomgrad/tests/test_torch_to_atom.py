import torch
import atom as atom
from features import GREEN, RED, RESET

def torch_to_atom_tensor():
    try:
        torch_tensor = torch.randn(10, 1)
        atom_tensor = atom.tensor(torch_tensor)
        print(f'{GREEN}PASS! conversion from torch tensor to atom tensor{RESET}')
    except:
        print(f'{RED}FAIL! conversion from torch tensor to atom tensor{RESET}')

def runner():
    torch_to_atom_tensor()


runner()


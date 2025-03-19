import atom
import torch
import numpy as np

def nn_forward_test():
    gen_x = np.random.randn(2, 10)
    gen_w = np.random.randn(5, 10)
    gen_b = np.random.randn(5)
    gen_y = np.random.randn(2, 5)

    atom_x = atom.tensor(gen_x, requires_grad=True)
    atom_y = atom.tensor(gen_y, requires_grad=True)

    atom_w = atom.tensor(gen_w, requires_grad=True)
    atom_b = atom.tensor(gen_b, requires_grad=True)
    pre_act = atom.matmul(atom_x, atom_w)
    a_act = atom.add(pre_act, atom_b)
    grad = (a_act['data'] - atom_y['data']) / atom_y['data'].shape[-1] # MSELoss
    atom.backward(a_act, grad)

    torch_x = torch.tensor(gen_x, requires_grad=True, dtype=torch.float32)
    torch_y = torch.tensor(gen_y, requires_grad=True, dtype=torch.float32)

    torch_w = torch.tensor(gen_w, requires_grad=True, dtype=torch.float32)
    torch_b = torch.tensor(gen_b, requires_grad=True, dtype=torch.float32)
    t_act = torch.matmul(torch_x, torch_w.T) + torch_b
    loss = torch.nn.MSELoss().forward(t_act, torch_y)
    loss.backward()

    # forward pass satisfaction
    print(a_act['data'].tolist())
    print(t_act.data.tolist())

    # backward pass satisfaction
    print(atom_w['grad'])
    print(torch_w.grad)

nn_forward_test()

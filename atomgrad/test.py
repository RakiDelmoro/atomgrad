import atom
import torch
import numpy as np

def nn_forward_test():
    # Generate input x, expected y and parameters w, b
    gen_x = np.random.randn(2, 10)
    gen_y = np.random.randn(2, 3)

    gen_w_1 = np.random.randn(5, 10)
    gen_b_1 = np.random.randn(5)
    gen_w_2 = np.random.randn(3, 5)
    gen_b_2 = np.random.randn(3)

    '''ATOM in action'''
    atom_x = atom.tensor(gen_x, requires_grad=True)
    atom_y = atom.tensor(gen_y, requires_grad=True)
    # First weight and bias parameters
    atom_w_1 = atom.tensor(gen_w_1, requires_grad=True)
    atom_b_1 = atom.tensor(gen_b_1, requires_grad=True)
    # Second weight and bias parameters
    atom_w_2 = atom.tensor(gen_w_2, requires_grad=True)
    atom_b_2 = atom.tensor(gen_b_2, requires_grad=True)
    
    atom_linear_1 = atom.relu(atom.add(atom.matmul(atom_x, atom_w_1), atom_b_1))
    atom_out = atom.add(atom.matmul(atom_linear_1, atom_w_2), atom_b_2)

    grad = (atom_out['data'] - atom_y['data']) / atom_y['data'].shape[-1] # MSELoss
    atom.backward(atom_out, grad)

    '''TORCH in action'''
    torch_x = torch.tensor(gen_x, requires_grad=True, dtype=torch.float32)
    torch_y = torch.tensor(gen_y, requires_grad=True, dtype=torch.float32)
    # First weight and bias parameters
    torch_w_1 = torch.tensor(gen_w_1, requires_grad=True, dtype=torch.float32)
    torch_b_1 = torch.tensor(gen_b_1, requires_grad=True, dtype=torch.float32)
    # Second weight and bias parameters
    torch_w_2 = torch.tensor(gen_w_2, requires_grad=True, dtype=torch.float32)
    torch_b_2 = torch.tensor(gen_b_2, requires_grad=True, dtype=torch.float32)

    # Apply Relu activation
    torch_linear_1 = (torch.matmul(torch_x, torch_w_1.T) + torch_b_1).relu()
    torch_out = torch.matmul(torch_linear_1, torch_w_2.T) + torch_b_2

    loss = torch.nn.MSELoss().forward(torch_out, torch_y)
    loss.backward()

    # forward pass satisfaction
    print(atom_out['data'])
    print(torch_out.data)

    # backward pass satisfaction
    print(atom_w_1['grad'])
    print(torch_w_1.grad)
    print()
    print(atom_w_2['grad'])
    print(torch_w_2.grad)

def test_with_3d_shape():
    gen_a = np.random.randn(2, 10)
    gen_b = np.random.randn(10)
    gen_w = np.random.randn(2, 10, 10)
    gen_y = np.random.randn(2, 10)

    # ATOM in action
    atom_a = atom.tensor(gen_a, requires_grad=True)
    atom_b = atom.tensor(gen_b, requires_grad=True)
    atom_w = atom.tensor(gen_w, requires_grad=True)
    atom_y = atom.tensor(gen_y, requires_grad=True)

    a_x = atom.sum_tensor([atom_a, atom_b])    
    a_result = atom.matmul_3d(atom_w, a_x)
    grad = (a_result['data'] - atom_y['data']) / atom_y['data'].shape[-1]
    a_loss = np.mean((a_result['data'] - atom_y['data'])**2)
    atom.backward(a_result, grad)

    # TORCH in action
    torch_a = torch.tensor(gen_a, requires_grad=True, dtype=torch.float32)
    torch_b = torch.tensor(gen_b, requires_grad=True, dtype=torch.float32)
    torch_w = torch.tensor(gen_w, requires_grad=True, dtype=torch.float32)
    torch_y = torch.tensor(gen_y, requires_grad=True, dtype=torch.float32)

    t_x = sum([torch_a, torch_b])
    t_result = torch.matmul(torch_w, t_x.unsqueeze(-1)).squeeze(-1)
    t_loss = torch.nn.MSELoss().forward(t_result, torch_y)
    t_x.retain_grad()
    t_result.retain_grad()
    t_loss.backward()

    # Scalar loss satisfaction
    print(a_loss, t_loss)
    # Activation satisfaction
    print(a_result)
    print(t_result)

def classifier_test():
    gen_x = np.random.randn(2, 10)
    gen_w = np.random.randn(5, 10)
    gen_b = np.random.randn(5)
    gen_y = np.zeros((2, 5))
    gen_y[np.arange(len(gen_x)), [np.random.randint(0, 2) for _ in range(len(gen_x))]] = 1

    '''ATOM'''
    atom_x = atom.relu(gen_x, requires_grad=True)
    atom_w = atom.tensor(gen_w, requires_grad=True)
    atom_b = atom.tensor(gen_b, requires_grad=True)
    atom_y = atom.tensor(gen_y)
    # forward pass
    a_act = atom.add(atom.matmul(atom_x, atom_w), atom_b)
    a_grad = (atom.softmax(a_act) - gen_y)
    atom.backward(a_act, a_grad)

    '''TORCH'''
    torch_x = torch.tensor(gen_x, requires_grad=True, dtype=torch.float32).relu()
    torch_w = torch.tensor(gen_w, requires_grad=True, dtype=torch.float32)
    torch_b = torch.tensor(gen_b, requires_grad=True, dtype=torch.float32)
    torch_y = torch.tensor(gen_y, requires_grad=True, dtype=torch.float32)
    # forward pass
    t_act = torch.matmul(torch_x, torch_w.T) + torch_b
    loss = torch.nn.CrossEntropyLoss().forward(t_act, torch_y)
    torch_x.retain_grad()
    t_act.retain_grad()
    loss.backward()

    print()

def test_combined_transition_weight():
    gen_w = np.random.randn(2, 5)
    gen_vk_params = [np.random.randn(4, 4) for _ in range(5)]
    gen_rt = np.random.randn(2, 4)
    gen_y = np.zeros((2, 4))
    gen_y[np.arange(len(gen_y)), [np.random.randint(0, 2) for _ in range(len(gen_y))]] = 1

    # ATOM
    a_rt = atom.tensor(gen_rt, requires_grad=True)
    a_vk_params = [atom.tensor(vk_params, requires_grad=True) for vk_params in gen_vk_params]
    a_w = atom.tensor(gen_w, requires_grad=True)

    a_tensors_to_sum = []
    for i in range(5):
        w = atom.tensor(a_w['data'][:, i].reshape(-1, 1, 1), requires_grad=True)
        tensor = atom.mul(w, a_vk_params[i], a_w, a_vk_params)
        a_tensors_to_sum.append(tensor)

    a_summed_tensor = atom.sum_tensor(a_tensors_to_sum)

    a_act = atom.matmul_3d(a_summed_tensor, a_rt)
    grad = atom.softmax(a_act) - gen_y
    atom.backward(a_act, grad)

    # TORCH
    t_rt = torch.tensor(gen_rt, dtype=torch.float32, requires_grad=True)
    t_vk_params = [torch.tensor(vk_params, dtype=torch.float32, requires_grad=True) for vk_params in gen_vk_params]
    t_w = torch.tensor(gen_w, dtype=torch.float32, requires_grad=True)

    t_summed_tensor = sum(t_w[:, idx].unsqueeze(-1).unsqueeze(-1) * t_vk_params[idx] for idx in range(5))
    t_act = torch.einsum('bij,bj->bi', t_summed_tensor, t_rt)
    loss = torch.nn.CrossEntropyLoss().forward(t_act, torch.tensor(gen_y, dtype=torch.float32, requires_grad=True))

    t_act.retain_grad()
    t_summed_tensor.retain_grad()
    loss.backward()

    print()

test_combined_transition_weight()

# classifier_test()
# test_with_3d_shape()
# nn_forward_test()

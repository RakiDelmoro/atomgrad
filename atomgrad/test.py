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
    torch_out.retain_grad()

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
    hyper_nn_w = np.random.randn(5, 10)
    hyper_nn_b = np.random.randn(5)
    
    gen_hyper_act = np.random.randn(2, 10)
    gen_vk_params = [np.random.randn(4, 4) for _ in range(5)]
    gen_rt = np.random.randn(2, 4)
    gen_y = np.zeros((2, 10))
    gen_y[np.arange(len(gen_y)), [np.random.randint(0, 2) for _ in range(len(gen_y))]] = 1
    gen_noise = np.random.randn(2, 4)

    gen_class_w = np.random.randn(10, 4)
    gen_class_b = np.random.randn(10)

    # ATOM
    a_rt = atom.tensor(gen_rt, requires_grad=True)
    a_vk_params = [atom.tensor(vk_params, requires_grad=True) for vk_params in gen_vk_params]
    a_noise = atom.tensor(gen_noise)

    a_hyper_w = atom.tensor(hyper_nn_w, requires_grad=True)
    a_hyper_b = atom.tensor(hyper_nn_b, requires_grad=True)

    a_class_w = atom.tensor(gen_class_w, requires_grad=True)
    a_class_b = atom.tensor(gen_class_b, requires_grad=True)

    # hyper network
    a_hyper_act = atom.tensor(gen_hyper_act, requires_grad=True)
    a_w = atom.add(atom.matmul(a_hyper_act, a_hyper_w), a_hyper_b)

    # combine transition weight
    # a_tensors_to_sum = []
    # for i in range(5):
    #     w = atom.tensor(a_w['data'][:, i].reshape(-1, 1, 1), requires_grad=True)
    #     tensor = atom.mul(w, a_vk_params[i], a_w, a_vk_params)
    #     a_tensors_to_sum.append(tensor)

    a_tensors_to_sum = atom.broadcasting_mul(a_w, a_vk_params)

    # Lower net update
    a_summed_tensor = atom.sum_tensor(a_tensors_to_sum)
    a_act_3d = atom.matmul_3d(a_summed_tensor, a_rt)
    a_act_with_noise = atom.add(a_act_3d, a_noise)
    relu_act = atom.relu(a_act_with_noise, requires_grad=True)

    # Classifier
    a_act = atom.add(atom.matmul(relu_act, a_class_w), a_class_b)

    grad = atom.softmax(a_act) - gen_y
    atom.backward(a_act, grad)

    '''TORCH'''
    t_rt = torch.tensor(gen_rt, dtype=torch.float32, requires_grad=True)
    t_vk_params = [torch.tensor(vk_params, dtype=torch.float32, requires_grad=True) for vk_params in gen_vk_params]
    t_noise = torch.tensor(gen_noise, dtype=torch.float32)

    t_hyper_w = torch.tensor(hyper_nn_w, requires_grad=True, dtype=torch.float32)
    t_hyper_b = torch.tensor(hyper_nn_b, requires_grad=True, dtype=torch.float32)

    t_class_w = torch.tensor(gen_class_w, dtype=torch.float32, requires_grad=True)
    t_class_b = torch.tensor(gen_class_b, dtype=torch.float32, requires_grad=True)

    # Hyper network
    t_hyper_act = torch.tensor(gen_hyper_act, dtype=torch.float32, requires_grad=True)
    t_w = torch.matmul(t_hyper_act, t_hyper_w.T) + t_hyper_b

    # Lower net update
    t_tensor_to_sum = [t_w[:, idx].unsqueeze(-1).unsqueeze(-1) * t_vk_params[idx] for idx in range(5)]
    t_summed_tensor = sum(t_tensor_to_sum)
    t_act_3d = torch.einsum('bij,bj->bi', t_summed_tensor, t_rt)
    t_act_with_noise = t_act_3d + t_noise
    t_relu_act = t_act_with_noise.relu()

    # Classifier
    t_act = torch.matmul(t_relu_act, t_class_w.T) + t_class_b
    loss = torch.nn.CrossEntropyLoss().forward(t_act, torch.tensor(gen_y, dtype=torch.float32, requires_grad=True))


    t_tensor_to_sum[0].retain_grad()
    t_tensor_to_sum[1].retain_grad()
    t_tensor_to_sum[2].retain_grad()
    t_tensor_to_sum[3].retain_grad()
    t_tensor_to_sum[4].retain_grad()

    t_hyper_act.retain_grad()
    t_w.retain_grad()
    t_summed_tensor.retain_grad()
    t_act_3d.retain_grad()
    t_act_with_noise.retain_grad()
    t_relu_act.retain_grad()
    t_act.retain_grad()
    t_summed_tensor.retain_grad()

    loss.backward()

    print()

import torch
import torch.nn as nn

def detailed_gradient_analysis():
    # Network configuration
    lower_dim = 256
    input_dim = 784
    
    # Create a linear layer
    lower_level_network = nn.Linear(lower_dim, input_dim, bias=False)
    
    # Initialize rt as zeros
    rt = torch.zeros(32, lower_dim, requires_grad=True)
    
    # Create a non-zero target input
    input_frame = torch.randn(32, input_dim)
    
    # Detailed step-by-step analysis
    print("Step 1: Forward Pass with Zero Input")
    predicted_frame = lower_level_network(rt)
    print("Predicted Frame Shape:", predicted_frame.shape)
    print("First few values of Predicted Frame:", predicted_frame[0][:10])
    
    print("\nStep 2: Loss Computation")
    loss = torch.nn.functional.mse_loss(predicted_frame, input_frame)
    print("Loss Value:", loss.item())
    
    print("\nStep 3: Backward Pass")
    loss.backward()
    
    print("\nWeight Gradient Details:")
    print("Weight Gradient Shape:", lower_level_network.weight.grad.shape)
    print("First few values of Weight Gradient:")
    print(lower_level_network.weight.grad[0][:10])
    
    # Explicit gradient computation
    print("\nStep 4: Explicit Gradient Computation")
    error = predicted_frame - input_frame
    print("Error First Few Values:", error[0][:10])
    
    # Demonstrate gradient computation mechanism
    manual_grad = 2 * torch.matmul(error.unsqueeze(1), rt.unsqueeze(0)).mean(dim=0)
    print("\nManual Gradient Computation:")
    print("First few values:", manual_grad[0][:10])

# detailed_gradient_analysis()

test_combined_transition_weight()
# classifier_test()
# test_with_3d_shape()
# nn_forward_test()



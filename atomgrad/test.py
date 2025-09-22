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
    # DATASET
    BATCH = 2
    OUTPUT_DIM = 5
    x_atom = atom.randn((BATCH, OUTPUT_DIM), requires_grad=True) # Example model output
    y = atom.randint(BATCH, OUTPUT_DIM, size=(BATCH,), requires_grad=True) 
    y_atom_one_hot = y.one_hot(num_classes=OUTPUT_DIM) # Example expected output

    # Torch version
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

    model_output_grad_satisfied = np.allclose((x_atom.grad / BATCH).data, x_torch.grad.detach().numpy())
    loss_fn_satisfied = np.allclose(atom_loss.data, torch_loss.detach().numpy())

    if model_output_grad_satisfied and loss_fn_satisfied:
        print(f'CrossEntropy Loss test --> {GREEN}Pass!{RESET}')
    else:
        print(f'CrossEntropy Loss test --> {RED}Failed!{RESET}')

def test_1_layer_linear_ops():
    # Dataset
    BATCH_SIZE = 2
    INPUT_DIM = 10
    OUTPUT_DIM = 5
    x_test = torch.randn(2, INPUT_DIM, requires_grad=True)
    y_test = torch.nn.functional.one_hot(torch.randint(0, OUTPUT_DIM, size=(BATCH_SIZE,)), num_classes=OUTPUT_DIM).float()
    atom_x = atom(x_test.detach().numpy(), requires_grad=True)
    atom_y = atom(y_test)

    # Torch Configs
    torch_ln = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM)
    torch_loss_fn = torch.nn.CrossEntropyLoss()

    # Atom Configs
    atom_w = atom(torch_ln.weight.detach().numpy().T, requires_grad=True)
    atom_b = atom(torch_ln.bias.detach().numpy(), requires_grad=True)
    atom_ln, _ = nn.linear(input_size=INPUT_DIM, output_size=OUTPUT_DIM, parameters=[atom_w,atom_b])
    atom_loss_fn = nn.cross_entropy()

    atom_out = atom_ln(atom_x) # Atom forward pass
    torch_out = torch_ln(x_test) # Torch forward pass

    # Torch loss calculation
    torch_loss = torch_loss_fn(torch_out, y_test)
    x_test.retain_grad()
    torch_out.retain_grad()
    # Atom loss calculation
    atom_loss = atom_loss_fn(atom_out, atom_y)

    # Torch automatic gradient calculation
    torch_loss.backward()
    # Atom automatic gradient calculation
    atom_grad = atom_out.softmax(dim=-1) - atom_y
    atom_loss.backward(atom_grad)

    weight_grad_satisfied = np.allclose((atom_w.grad / BATCH_SIZE).data, torch_ln.weight.grad.T.numpy())
    bias_grad_satisfied = np.allclose((atom_b.grad / BATCH_SIZE).data,  torch_ln.bias.grad.numpy())

    if weight_grad_satisfied and bias_grad_satisfied:
        print(f'One linear layer test --> {GREEN}Pass!{RESET}')
    else:
        print(f'One linear layer test --> {RED}Failed!{RESET}')

def test_2_layer_linear_ops():
    # Dataset
    BATCH_SIZE = 2
    INPUT_DIM = 10
    HIDDEN_DIM = 5
    OUTPUT_DIM = 5
    x_test = torch.randn(2, INPUT_DIM)
    y_test = torch.nn.functional.one_hot(torch.randint(0, OUTPUT_DIM, size=(BATCH_SIZE,)), num_classes=OUTPUT_DIM).float()
    atom_x = atom(x_test)
    atom_y = atom(y_test)

    # Torch Configs
    torch_ln_1 = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM)
    torch_act = torch.nn.ReLU()
    torch_ln_2 = torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
    torch_loss_fn = torch.nn.CrossEntropyLoss()

    # Atom Configs
    atom_w_ln_1 = atom(torch_ln_1.weight.detach().numpy().T, requires_grad=True)
    atom_b_ln_1 = atom(torch_ln_1.bias.detach().numpy(), requires_grad=True)
    atom_ln_1, _ = nn.linear(input_size=INPUT_DIM, output_size=HIDDEN_DIM, parameters=[atom_w_ln_1,atom_b_ln_1])
    atom_w_ln_2 = atom(torch_ln_2.weight.detach().numpy().T, requires_grad=True)
    atom_b_ln_2 = atom(torch_ln_2.bias.detach().numpy(), requires_grad=True)
    atom_ln_2, _ = nn.linear(input_size=HIDDEN_DIM, output_size=OUTPUT_DIM, parameters=[atom_w_ln_2, atom_b_ln_2])
    atom_loss_fn = nn.cross_entropy()

    # Forward passes
    atom_lin_1_out = atom_ln_1(atom_x)
    atom_lin_2_out = atom_ln_2(atom_lin_1_out)

    torch_lin_1_out = torch_ln_1(x_test)
    torch_lin_2_out = torch_ln_2(torch_lin_1_out)

    # Torch loss calculation
    torch_loss = torch_loss_fn(torch_lin_2_out, y_test)
    torch_lin_1_out.retain_grad()
    torch_lin_2_out.retain_grad()
    # Atom loss calculation
    atom_loss = atom_loss_fn(atom_lin_2_out, atom_y)
    torch_loss.retain_grad()
    # Torch automatic gradient calculation
    torch_loss.backward()
    # Atom automatic gradient calculation
    atom_loss.backward()


    weight_grad_satisfied = np.allclose((atom_w_ln_1.grad / BATCH_SIZE).data, torch_ln_1.weight.grad.T.numpy())
    bias_grad_satisfied = np.allclose((atom_b_ln_1.grad / BATCH_SIZE).data,  torch_ln_1.bias.grad.numpy())

    if weight_grad_satisfied and bias_grad_satisfied:
        print(f'Two linear layer test --> {GREEN}Pass!{RESET}')
    else:
        print(f'Two linear layer test --> {RED}Failed!{RESET}')

test_zeros()
test_empty()
test_add_ops()
test_matmul_for_2d()
test_softmax()
test_relu()
test_log_softmax()
test_one_hot()
test_cross_entropy()
test_1_layer_linear_ops()
test_2_layer_linear_ops()

import torch
import numpy as np
import cupy as cp
from tensor import atom
import nn 

def test_matmul_for_2d():
    x1_atom = atom.randn((2, 3), device='cpu')
    x2_atom = atom.randn((3, 2), device='cpu')

    # Comparing Torch and Atom
    x1_torch = torch.tensor(x1_atom.data, dtype=torch.float32)
    x2_torch = torch.tensor(x2_atom.data, dtype=torch.float32)

    y_atom = atom.matmul(x1_atom, x2_atom)
    y_torch = torch.matmul(x1_torch, x2_torch)

    assert np.allclose(y_atom.data, y_torch.numpy())

def test_matmul_for_3d():
    x1_atom = atom.randn((2, 3, 10))
    x2_atom = atom.randn((10, 5))

    x1_torch = torch.tensor(x1_atom.data, dtype=torch.float32)
    x2_torch = torch.tensor(x2_atom.data, dtype=torch.float32)

    y_atom = atom.matmul(x1_atom, x2_atom)
    y_torch = torch.matmul(x1_torch, x2_torch)
    
    assert np.allclose(y_atom.data, y_torch.numpy())

def test_relu():
    x_atom = atom.randn((2, 5))
    x_torch = torch.tensor(x_atom.data)

    y_atom = x_atom.relu() 
    y_torch = x_torch.relu()

    satified = np.allclose(y_atom.data, y_torch.numpy())

    assert satified

def test_softmax():
    x_atom = atom.randn((2, 5))
    x_torch = torch.tensor(x_atom.data)

    y_atom = x_atom.softmax(dim=-1) 
    y_torch = x_torch.softmax(dim=-1)

    satified = np.allclose(y_atom.data, y_torch.numpy())

    assert satified

def test_log_softmax():
    x_atom = atom.randn((2, 5))
    x_torch = torch.tensor(x_atom.data)

    y_atom = x_atom.log_softmax(dim=-1) 
    y_torch = x_torch.log_softmax(dim=-1)

    assert np.allclose(y_atom.data, y_torch.numpy())

def test_one_hot():
    x_atom = atom.randint(0, 5, size=(2,))
    x_torch = torch.tensor(x_atom.data, dtype=torch.int64) # torch one hot funtion expect tensor should have dtype int64

    y_atom = x_atom.one_hot(num_classes=5)
    y_torch = torch.nn.functional.one_hot(x_torch, num_classes=5)

    assert np.allclose(y_atom.data, y_torch.numpy())

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

    assert model_output_grad_satisfied and loss_fn_satisfied

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

    assert weight_grad_satisfied and bias_grad_satisfied

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

    assert weight_grad_satisfied and bias_grad_satisfied

def test_2_layer_with_act_linear_ops():
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
    
    atom_act = nn.relu()

    atom_w_ln_2 = atom(torch_ln_2.weight.detach().numpy().T, requires_grad=True)
    atom_b_ln_2 = atom(torch_ln_2.bias.detach().numpy(), requires_grad=True)
    atom_ln_2, _ = nn.linear(input_size=HIDDEN_DIM, output_size=OUTPUT_DIM, parameters=[atom_w_ln_2, atom_b_ln_2])
    
    atom_loss_fn = nn.cross_entropy()

    # Forward passes
    atom_lin_1_out = atom_ln_1(atom_x)
    atom_act_out = atom_act(atom_lin_1_out)
    atom_lin_2_out = atom_ln_2(atom_act_out)

    torch_lin_1_out = torch_ln_1(x_test)
    torch_act_out = torch_act(torch_lin_1_out)
    torch_lin_2_out = torch_ln_2(torch_act_out)

    # Torch loss calculation
    torch_loss = torch_loss_fn(torch_lin_2_out, y_test)
    torch_lin_1_out.retain_grad()
    torch_act_out.retain_grad()
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

    assert weight_grad_satisfied and bias_grad_satisfied

def test_self_attention():
    BATCH = 2
    SEQ_LEN = 5
    NUM_HEADS = 1
    EMBEDDING_DIM = 16
    HEAD_DIM = EMBEDDING_DIM // 1

    torch_x_emb = torch.randn(BATCH, SEQ_LEN, EMBEDDING_DIM)
    atom_x_emb = atom(torch_x_emb.numpy(), requires_grad=True)

    torch_y = torch.randint(low=0, high=EMBEDDING_DIM, size=(BATCH*SEQ_LEN,))
    atom_y = atom(torch_y.numpy())

    torch_tril = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN)))
    atom_tril = atom.tril(atom.ones((SEQ_LEN, SEQ_LEN)))

    torch_loss_fn = torch.nn.CrossEntropyLoss()
    atom_loss_fn = nn.cross_entropy()

    torch_q_proj = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
    torch_k_proj = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
    torch_v_proj = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
    torch_ln_attn_out = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    atom_q_proj, q_params = nn.linear(EMBEDDING_DIM, EMBEDDING_DIM, parameters=[atom(torch_q_proj.weight.T.detach().numpy(), requires_grad=True), atom(torch_q_proj.bias.detach().numpy(), requires_grad=True)])
    atom_k_proj, k_params = nn.linear(EMBEDDING_DIM, EMBEDDING_DIM, parameters=[atom(torch_k_proj.weight.T.detach().numpy(), requires_grad=True), atom(torch_k_proj.bias.detach().numpy(), requires_grad=True)])
    atom_v_proj, v_params = nn.linear(EMBEDDING_DIM, EMBEDDING_DIM, parameters=[atom(torch_v_proj.weight.T.detach().numpy(), requires_grad=True), atom(torch_v_proj.bias.detach().numpy(), requires_grad=True)])
    atom_ln_attn_out, atom_ln_params = nn.linear(EMBEDDING_DIM, EMBEDDING_DIM, parameters=[atom(torch_ln_attn_out.weight.T.detach().numpy(), requires_grad=True), atom(torch_ln_attn_out.bias.detach().numpy(), requires_grad=True)])

    # TORCH forward pass
    torch_query = torch_q_proj(torch_x_emb).reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    torch_key = torch_k_proj(torch_x_emb).reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    torch_value = torch_v_proj(torch_x_emb).reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    
    torch_key_transposed = torch_key.transpose(-2, -1)
    torch_qk_matmul = torch.matmul(torch_query, torch_key_transposed)
    torch_scale = torch_qk_matmul * EMBEDDING_DIM**-0.5
    torch_mask = torch_scale.masked_fill(torch_tril[:SEQ_LEN, :SEQ_LEN] == 0, float('-inf'))
    torch_softmax = torch.nn.functional.softmax(torch_mask, dim=-1)
    torch_softmax_v_matmul = torch.matmul(torch_softmax, torch_value)
    torch_attention_out = torch_ln_attn_out(torch_softmax_v_matmul)

    torch_loss = torch_loss_fn(torch_attention_out.view(BATCH*SEQ_LEN, EMBEDDING_DIM), torch_y)

    # Tell pytorch to retain gradient on each forward pass calculcation
    torch_query.retain_grad()
    torch_key.retain_grad()
    torch_value.retain_grad()
    torch_key_transposed.retain_grad()
    torch_qk_matmul.retain_grad()
    torch_scale.retain_grad()
    torch_key.retain_grad()
    torch_softmax.retain_grad()
    torch_softmax_v_matmul.retain_grad()
    torch_attention_out.retain_grad()

    # ATOM forward pass
    atom_query = atom_q_proj(atom_x_emb).reshape((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
    atom_key = atom_k_proj(atom_x_emb).reshape((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
    atom_value = atom_v_proj(atom_x_emb).reshape((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))

    atom_key_transposed = atom_key.transpose((0, 1, 3, 2))
    atom_qk_matmul = atom.matmul(atom_query, atom_key_transposed)
    atom_scale = atom_qk_matmul * atom(EMBEDDING_DIM**-0.5)
    mask = atom_tril.data[:SEQ_LEN, :SEQ_LEN] == 0 # atom_mask
    atom_scale.data[:, :, (mask == 1)] = -np.inf if atom_scale.device == 'cpu' else -cp.inf
    atom_softmax = atom_scale.softmax(dim=-1)
    atom_softmax_v_matmul = atom.matmul(atom_softmax, atom_value)
    atom_attention_out = atom_ln_attn_out(atom_softmax_v_matmul)

    atom_attention_out.data = atom_attention_out.data.reshape(BATCH*SEQ_LEN, EMBEDDING_DIM)
    atom_attention_out.shape = atom_attention_out.data.shape
    
    atom_loss = atom_loss_fn(atom_attention_out, atom_y)

    torch_loss.backward()
    atom_loss.backward()

    SCALE = BATCH * SEQ_LEN

    print(torch_key.grad.detach().numpy()[0])
    print()
    print((atom_key.grad / SCALE).data[0])

    # Check the gradient for each forward pass calculation
    assert np.allclose(torch_softmax_v_matmul.grad.detach().numpy(), (atom_softmax_v_matmul.grad / SCALE).data)
    assert np.allclose(torch_softmax.grad.detach().numpy(), (atom_softmax.grad / SCALE).data)
    assert np.allclose(torch_scale.grad.detach().numpy(), (atom_scale.grad / SCALE).data)
    assert np.allclose(torch_qk_matmul.grad.detach().numpy(), (atom_qk_matmul.grad / SCALE).data)
    assert np.allclose(torch_value.grad.detach().numpy(), (atom_value.grad / SCALE).data)
    assert np.allclose(torch_key.grad.detach().numpy(), (atom_key.grad / SCALE).data)
    assert np.allclose(torch_query.grad.detach().numpy(), (atom_query.grad / SCALE).data)

    # Check the gradient for each parameters
    # Query projection parameters
    assert np.allclose(torch_q_proj.weight.grad.detach().numpy(), (q_params[0].grad / SCALE).data)
    assert np.allclose(torch_q_proj.bias.grad.detach().numpy(), (q_params[1].grad / SCALE).data)
    # Key projection parameters
    assert np.allclose(torch_k_proj.weight.grad.detach().numpy(), (k_params[0].grad / SCALE).data)
    assert np.allclose(torch_k_proj.bias.grad.detach().numpy(), (k_params[1].grad / SCALE).data)
    # Value projection parameters
    assert np.allclose(torch_v_proj.weight.grad.detach().numpy(), (v_params[0].grad / SCALE).data)
    assert np.allclose(torch_v_proj.bias.grad.detach().numpy(), (v_params[1].grad / SCALE).data)
    # Attention linear layer parameters
    assert np.allclose(torch_ln_attn_out.weight.grad.detach().numpy(), (atom_ln_params[0].grad / SCALE).data)
    assert np.allclose(torch_ln_attn_out.bias.grad.detach().numpy(), (atom_ln_params[1].grad / SCALE).data)

def test_multi_head_attention():
    BATCH = 2
    SEQ_LEN = 5
    NUM_HEADS = 4 
    EMBEDDING_DIM = 16
    HEAD_DIM = EMBEDDING_DIM // NUM_HEADS

    torch_x_emb = torch.randn(BATCH, SEQ_LEN, EMBEDDING_DIM)
    atom_x_emb = atom(torch_x_emb.numpy(), requires_grad=True)

    torch_y = torch.randint(low=0, high=EMBEDDING_DIM, size=(BATCH*SEQ_LEN,))
    atom_y = atom(torch_y.numpy())

    torch_tril = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN)))
    atom_tril = atom.tril(atom.ones((SEQ_LEN, SEQ_LEN)))

    torch_loss_fn = torch.nn.CrossEntropyLoss()
    atom_loss_fn = nn.cross_entropy()

    torch_q_proj = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
    torch_k_proj = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
    torch_v_proj = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
    torch_ln_out = torch.nn.Linear(NUM_HEADS*HEAD_DIM, EMBEDDING_DIM)

    atom_q_proj, q_params = nn.linear(EMBEDDING_DIM, EMBEDDING_DIM, parameters=[atom(torch_q_proj.weight.T.detach().numpy(), requires_grad=True), atom(torch_q_proj.bias.detach().numpy(), requires_grad=True)])
    atom_k_proj, k_params = nn.linear(EMBEDDING_DIM, EMBEDDING_DIM, parameters=[atom(torch_k_proj.weight.T.detach().numpy(), requires_grad=True), atom(torch_k_proj.bias.detach().numpy(), requires_grad=True)])
    atom_v_proj, v_params = nn.linear(EMBEDDING_DIM, EMBEDDING_DIM, parameters=[atom(torch_v_proj.weight.T.detach().numpy(), requires_grad=True), atom(torch_v_proj.bias.detach().numpy(), requires_grad=True)])
    atom_ln_out, atom_ln_params = nn.linear(NUM_HEADS*HEAD_DIM, EMBEDDING_DIM, parameters=[atom(torch_ln_out.weight.T.detach().numpy(), requires_grad=True), atom(torch_ln_out.bias.detach().numpy(), requires_grad=True)])

    # TORCH forward pass
    torch_query = torch_q_proj(torch_x_emb).reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    torch_key = torch_k_proj(torch_x_emb).reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    torch_value = torch_v_proj(torch_x_emb).reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)

    torch_key_transposed = torch_key.transpose(-2, -1)
    torch_qk_matmul = torch.matmul(torch_query, torch_key_transposed)
    torch_scale = torch_qk_matmul * HEAD_DIM**-0.5
    torch_mask = torch_scale.masked_fill(torch_tril[:SEQ_LEN, :SEQ_LEN] == 0, float('-inf'))
    torch_softmax = torch.nn.functional.softmax(torch_mask, dim=-1)
    torch_softmax_v_matmul = torch.matmul(torch_softmax, torch_value).reshape(BATCH, SEQ_LEN, NUM_HEADS*HEAD_DIM)
    torch_attn_out = torch_ln_out(torch_softmax_v_matmul)

    torch_query.retain_grad()
    torch_key.retain_grad()
    torch_value.retain_grad()
    torch_key_transposed.retain_grad()
    torch_qk_matmul.retain_grad()
    torch_scale.retain_grad()
    torch_key.retain_grad()
    torch_softmax.retain_grad()
    torch_softmax_v_matmul.retain_grad()
    torch_attn_out.retain_grad()

    # ATOM forward pass
    atom_query = atom_q_proj(atom_x_emb).reshape((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
    atom_key = atom_k_proj(atom_x_emb).reshape((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))
    atom_value = atom_v_proj(atom_x_emb).reshape((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM))

    atom_key_transposed = atom_key.transpose((0, 1, 3, 2))
    atom_qk_matmul = atom.matmul(atom_query, atom_key_transposed)
    atom_scale = atom_qk_matmul * HEAD_DIM**-0.5
    #atom_mask
    mask = atom_tril.data[:SEQ_LEN, :SEQ_LEN] == 0
    atom_scale.data[:, :, (mask == 1)] = -np.inf if atom_scale.device == 'cpu' else -cp.inf

    atom_softmax = atom_scale.softmax(dim=-1)
    atom_softmax_v_matmul = atom.matmul(atom_softmax, atom_value).reshape((BATCH, SEQ_LEN, NUM_HEADS*HEAD_DIM))
    atom_attn_out = atom_ln_out(atom_softmax_v_matmul)

    torch_loss = torch_loss_fn(torch_attn_out.view(BATCH*SEQ_LEN, EMBEDDING_DIM), torch_y)
    atom_loss = atom_loss_fn(atom_attn_out.reshape((BATCH*SEQ_LEN, EMBEDDING_DIM)), atom_y)

    torch_loss.backward()
    atom_loss.backward()

    SCALE = (BATCH * SEQ_LEN)

    # Check the gradient for each parameters
    # Query projection parameters
    assert np.allclose(torch_q_proj.weight.grad.detach().numpy(), (q_params[0].grad / SCALE).data)
    assert np.allclose(torch_q_proj.bias.grad.detach().numpy(), (q_params[1].grad / SCALE).data)
    # Key projection parameters
    assert np.allclose(torch_k_proj.weight.grad.detach().numpy(), (k_params[0].grad / SCALE).data)
    assert np.allclose(torch_k_proj.bias.grad.detach().numpy(), (k_params[1].grad / SCALE).data)
    # Value projection parameters
    assert np.allclose(torch_v_proj.weight.grad.detach().numpy(), (v_params[0].grad / SCALE).data)
    assert np.allclose(torch_v_proj.bias.grad.detach().numpy(), (v_params[1].grad / SCALE).data)
    # Attention linear layer parameters
    assert np.allclose(torch_ln_out.weight.grad.detach().numpy(), (atom_ln_params[0].grad / SCALE).data)
    assert np.allclose(torch_ln_out.bias.grad.detach().numpy(), (atom_ln_params[1].grad / SCALE).data)

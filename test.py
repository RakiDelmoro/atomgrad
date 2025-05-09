import torch
import cupy
import numpy as np
import atomgrad.atom as atom
import atomgrad.cuda.atom as cuda_atom
import atomgrad.cuda.loss_fn.loss_fn_nn as loss_nn
import atomgrad.torch_examples.transformer.neural_network as torch_transformer
import atomgrad.examples.transformer.neural_network as atom_transformer

import atomgrad.cuda.nn_ops as nn_ops

BATCH = 2
VOCAB_SIZE = 10
SEQ_LEN = 3
NUM_HEAD = 2

# Colors
RED = '\033[31m'
RESET = '\033[0m'
GREEN = '\033[32m'
UNDERLINE = "\033[4m"

def build_topo(nodes):
    """Build topological order starting from the given node."""
    visited = set()
    topo = []

    def visit(node):
        node_identity = id(node)
        if node_identity not in visited:
            visited.add(node_identity)
            if type(node) == list:
                for each in node:
                    for depends_on in each['depends_on']:
                        visit(depends_on)
            else:
                for depends_on in node['depends_on']:
                    visit(depends_on)
            topo.append(node)
    visit(nodes)
    return topo

def backward(atom_tensor, grad=None):
    """Compute gradients via reverse-mode autodiff."""

    if not atom_tensor['requires_grad']: return

    topo = build_topo(atom_tensor)
    if grad is None: atom_tensor['grad'] = cupy.ones_like(atom_tensor['data']) if atom_tensor['grad'] is None else atom_tensor['grad']
    else: atom_tensor['grad'] = grad
    
    for node in reversed(topo):
        if type(node) == list:
            for each in node:
                if each['grad_fn'] is not None:
                    each['grad_fn'](each['grad'])

                # Throw it away after calculating/propagate the gradient
                each['depends_on'] = []
                each['grad_fn'] = None

        else:
            if node['grad_fn'] is not None:
                node['grad_fn'](node['grad'])

            # Throw it away after calculating/propagate the gradient
            node['depends_on'] = []
            node['grad_fn'] = None

def compare_embeddings(atom_grad=None, torch_grad=None):
    if atom_grad is None and torch_grad is None:
        torch_grad = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
        atom_grad = cupy.array(torch_grad.cpu().numpy())

    torch_x = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH, SEQ_LEN), device='cuda')
    atom_x = cuda_atom.cuda_tensor(torch_x.detach().cpu().numpy())

    # TORCH
    torch_model = torch_transformer.Embeddings()
    torch_char_params = torch_model.char_embedding.weight.detach().cpu()
    torch_pos_params = torch_model.pos_embedding.weight.detach().cpu()

    # ATOM
    atom_model, atom_params = atom_transformer.char_embeddings(torch_char_params, torch_pos_params)

    # Forward passes
    torch_out = torch_model(torch_x)
    atom_out = atom_model(atom_x)
    torch_out.retain_grad()

    # Backward passes
    backward(atom_out, atom_grad)
    torch_out.backward(torch_grad)

    embeddings_satisfied = torch.allclose(torch.tensor(cupy.asnumpy(atom_params[0]['grad'])), torch_model.char_embedding.weight.grad.cpu())

    print(f"{UNDERLINE}Character Embeddings satisfied{RESET}: {GREEN}{embeddings_satisfied}{RESET}")

def compare_attention(atom_grad=None, torch_grad=None):
    if atom_grad is None and torch_grad is None:
        torch_grad = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
        atom_grad = cupy.array(torch_grad.cpu().numpy())

    torch_x = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
    atom_x = cuda_atom.cuda_tensor(torch_x.detach().cpu().numpy())

    # TORCH
    torch_model = torch_transformer.AttentionLayer(VOCAB_SIZE // 1)
    
    query_weight = torch_model.query.weight.detach().cpu()    
    key_weight = torch_model.key.weight.detach().cpu()
    value_weight = torch_model.value.weight.detach().cpu()

    # ATOM
    atom_model, atom_params = atom_transformer.attention_layer(VOCAB_SIZE // 1, VOCAB_SIZE, [query_weight], [key_weight], [value_weight])

    # Forward passes
    torch_out, torch_qk, torch_q, torch_k, torch_v = torch_model(torch_x)
    atom_out, atom_qk, atom_q, atom_k, atom_v = atom_model(atom_x)

    torch_out.retain_grad()
    torch_q.retain_grad()
    torch_k.retain_grad()
    torch_v.retain_grad()
    torch_qk.retain_grad()

    # Backward passes
    backward(atom_out, atom_grad)
    torch_out.backward(torch_grad)

    atom_query_params_grad = cupy.asnumpy(atom_params[0]['grad'])
    atom_key_params_grad = cupy.asnumpy(atom_params[1]['grad'])
    atom_value_params_grad = cupy.asnumpy(atom_params[2]['grad'])

    query_satisfied = torch.allclose(torch.tensor(atom_query_params_grad), torch_model.query.weight.grad.cpu(), atol=1e-5)
    key_satisfied = torch.allclose(torch.tensor(atom_key_params_grad), torch_model.key.weight.grad.cpu(), atol=1e-5)
    value_satisfied = torch.allclose(torch.tensor(atom_value_params_grad), torch_model.value.weight.grad.cpu(), atol=1e-5)

    print(f'{UNDERLINE}Attention layer Query satisfied{RESET}: {query_satisfied}')
    print(f'{UNDERLINE}Attention layer Key satisfied{RESET}: {key_satisfied}')
    print(f'{UNDERLINE}Attention layer Value satisfied{RESET}: {value_satisfied}')

def compare_multi_head_attn(atom_grad=None, torch_grad=None):
    attn_head_size = VOCAB_SIZE // NUM_HEAD

    if atom_grad is None and torch_grad is None:
        torch_grad = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
        atom_grad = cupy.array(torch_grad.cpu().numpy())

    torch_x = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
    atom_x = cuda_atom.cuda_tensor(torch_x.detach().cpu().numpy())

    torch_model = torch_transformer.MultiHeadAttention(NUM_HEAD, attn_head_size)
    linear_proj_params = [torch_model.proj.weight.detach().cpu(), torch_model.proj.bias.detach().cpu()]

    atom_model, atom_params = atom_transformer.multi_head_attention_layer(NUM_HEAD, attn_head_size, VOCAB_SIZE, torch_model.heads, linear_proj_params)
    
    # Forward passes
    torch_out = torch_model(torch_x)
    torch_out.retain_grad()
    atom_out = atom_model(atom_x)
    
    # Backward passes
    backward(atom_out, atom_grad)
    torch_out.backward(torch_grad)

    linear_proj_w_grad = torch.tensor(cupy.asnumpy(atom_params[-2]['grad']))
    linear_proj_b_grad = torch.tensor(cupy.asnumpy(atom_params[-1]['grad']))

    linear_proj_satisfied = torch.allclose(linear_proj_w_grad, torch_model.proj.weight.grad.cpu(), atol=1e-5) and torch.allclose(linear_proj_b_grad, torch_model.proj.bias.grad.cpu(), atol=1e-5)

    attn_head_1_query_grad = torch.tensor(cupy.asnumpy(atom_params[0]['grad']))
    attn_head_1_key_grad = torch.tensor(cupy.asnumpy(atom_params[1]['grad']))
    attn_head_1_value_grad = torch.tensor(cupy.asnumpy(atom_params[2]['grad']))

    attn_head_2_query_grad = torch.tensor(cupy.asnumpy(atom_params[3]['grad']))
    attn_head_2_key_grad = torch.tensor(cupy.asnumpy(atom_params[4]['grad']))
    attn_head_2_value_grad = torch.tensor(cupy.asnumpy(atom_params[5]['grad']))

    head_1_query_satisfied = torch.allclose(attn_head_1_query_grad, torch_model.heads[0].query.weight.grad.cpu(), atol=1e-5)
    head_1_key_satisfied = torch.allclose(attn_head_1_key_grad, torch_model.heads[0].key.weight.grad.cpu(), atol=1e-5)
    head_1_value_satisfied = torch.allclose(attn_head_1_value_grad, torch_model.heads[0].value.weight.grad.cpu(), atol=1e-5)

    head_2_query_satisfied = torch.allclose(attn_head_2_query_grad, torch_model.heads[1].query.weight.grad.cpu(), atol=1e-5)
    head_2_key_satisfied = torch.allclose(attn_head_2_key_grad, torch_model.heads[1].key.weight.grad.cpu(), atol=1e-5)
    head_2_value_satisfied = torch.allclose(attn_head_2_value_grad, torch_model.heads[1].value.weight.grad.cpu(), atol=1e-5)

    print(linear_proj_satisfied)

    print(f'{UNDERLINE}Attn head 1 query satisfied{RESET}: {GREEN}{head_1_query_satisfied}{RESET}')
    print(f'{UNDERLINE}Attn head 1 key satisfied{RESET}: {GREEN}{head_1_key_satisfied}{RESET}')
    print(f'{UNDERLINE}Attn head 1 value satisfied{RESET}: {GREEN}{head_1_value_satisfied}{RESET}')

    print(f'{UNDERLINE}Attn head 2 query satisfied{RESET}: {GREEN}{head_2_query_satisfied}{RESET}')
    print(f'{UNDERLINE}Attn head 2 key satisfied{RESET}: {GREEN}{head_2_key_satisfied}{RESET}')
    print(f'{UNDERLINE}Attn head 2 value satisfied{RESET}: {GREEN}{head_2_value_satisfied}{RESET}')

def compare_mlp(atom_grad=None, torch_grad=None):
    if atom_grad is None and torch_grad is None:
        torch_grad = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
        atom_grad = cupy.array(torch_grad.cpu().numpy())

    torch_x = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
    atom_x = cuda_atom.cuda_tensor(torch_x.detach().cpu().numpy(), requires_grad=True)

    torch_model = torch_transformer.FeedForward(VOCAB_SIZE)
    linear_1_w = torch_model.ff_linear_1.weight.detach().cpu()   
    linear_1_b = torch_model.ff_linear_1.bias.detach().cpu()

    linear_2_w = torch_model.ff_linear_2.weight.detach().cpu()
    linear_2_b = torch_model.ff_linear_2.bias.detach().cpu()

    layer_norm_w = torch_model.ff_layer_norm.weight.detach().cpu()
    layer_norm_b = torch_model.ff_layer_norm.bias.detach().cpu()

    atom_model, atom_params = atom_transformer.feedforward(VOCAB_SIZE, [linear_1_w, linear_1_b], [linear_2_w, linear_2_b], [layer_norm_w, layer_norm_b])

    # Forward passes
    torch_out = torch_model(torch_x)
    torch_out.retain_grad()
    atom_out = atom_model(atom_x)

    # Backward passes
    torch_out.backward(torch_grad)
    backward(atom_out, atom_grad)
    
    laye_norm_w_grad = cupy.asnumpy(atom_params[0]['grad'])
    laye_norm_b_grad = cupy.asnumpy(atom_params[1]['grad'])
    
    linear_1_w_grad = cupy.asnumpy(atom_params[2]['grad'])
    linear_1_b_grad = cupy.asnumpy(atom_params[3]['grad'])

    linear_2_w_grad = cupy.asnumpy(atom_params[4]['grad'])
    linear_2_b_grad = cupy.asnumpy(atom_params[5]['grad'])

    layer_norm_satisfied = torch.allclose(torch.tensor(laye_norm_w_grad), torch_model.ff_layer_norm.weight.grad.cpu(), atol=1e-5) and torch.allclose(torch.tensor(laye_norm_b_grad), torch_model.ff_layer_norm.bias.grad.cpu(), atol=1e-5)
    linear_1_satisfied = torch.allclose(torch.tensor(linear_1_w_grad), torch_model.ff_linear_1.weight.grad.cpu(), atol=1e-5) and torch.allclose(torch.tensor(linear_1_b_grad), torch_model.ff_linear_1.bias.grad.cpu(), atol=1e-5)
    linear_2_satisfied = torch.allclose(torch.tensor(linear_2_w_grad), torch_model.ff_linear_2.weight.grad.cpu(), atol=1e-5) and torch.allclose(torch.tensor(linear_2_b_grad), torch_model.ff_linear_2.bias.grad.cpu(), atol=1e-5)

    print(f'{UNDERLINE}FeedForward layer norm{RESET}: {GREEN}{layer_norm_satisfied}{RESET}')
    print(f'{UNDERLINE}FeedForward linear layer 1{RESET}: {GREEN}{linear_1_satisfied}{RESET}')
    print(f'{UNDERLINE}FeedForward linear layer 2{RESET}: {GREEN}{linear_2_satisfied}{RESET}')

    # print(torch_model.ff_linear_1.weight.grad)
    # print(atom_params[2]['grad'][0])

def compare_transformer_block(atom_grad=None, torch_grad=None):
    '''Test with 2 attention heads with 2 transformer blocks'''

    if atom_grad is None and torch_grad is None:
        torch_grad = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
        atom_grad = cupy.array(torch_grad.cpu().numpy())

    torch_x = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, device='cuda')
    atom_x = cuda_atom.cuda_tensor(torch_x.detach().cpu().numpy(), requires_grad=True)

    torch_y = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH, SEQ_LEN), device='cuda')
    atom_y = cuda_atom.cuda_tensor(cupy.array(torch_y.detach().cpu().numpy()))

    torch_model = torch_transformer.TransformerBlocks(VOCAB_SIZE, NUM_HEAD)

    '''Transformer Block 1'''
    block_1_input_layer_norm = [torch_model.blocks[0].ln1.weight.detach().cpu(), torch_model.blocks[0].ln1.bias.detach().cpu()]
    # Attention heads
    block_1_attn_heads = torch_model.blocks[0].attention_layer.heads
    # Attn proj out params
    block_1_attn_proj_out_params = [torch_model.blocks[0].attention_layer.proj.weight.detach().cpu(), torch_model.blocks[0].attention_layer.proj.bias.detach().cpu()]
    # FeedForward params
    block_1_ff_layer_norm = [torch_model.blocks[0].feedforward.ff_layer_norm.weight.detach().cpu(), torch_model.blocks[0].feedforward.ff_layer_norm.bias.detach().cpu()]
    block_1_ff_lin_1_params = [torch_model.blocks[0].feedforward.ff_linear_1.weight.detach().cpu(), torch_model.blocks[0].feedforward.ff_linear_1.bias.detach().cpu()]
    block_1_ff_lin_2_params = [torch_model.blocks[0].feedforward.ff_linear_2.weight.detach().cpu(), torch_model.blocks[0].feedforward.ff_linear_2.bias.detach().cpu()]

    '''Transformer Block 2'''
    block_2_input_layer_norm = [torch_model.blocks[1].ln1.weight.detach().cpu(), torch_model.blocks[1].ln1.bias.detach().cpu()]
    # Attention heads
    block_2_attn_heads = torch_model.blocks[1].attention_layer.heads
    # Attn proj out params
    block_2_attn_proj_out_params = [torch_model.blocks[1].attention_layer.proj.weight.detach().cpu(), torch_model.blocks[1].attention_layer.proj.bias.detach().cpu()]
    # FeedForward params
    block_2_ff_layer_norm = [torch_model.blocks[1].feedforward.ff_layer_norm.weight.detach().cpu(), torch_model.blocks[1].feedforward.ff_layer_norm.bias.detach().cpu()]
    block_2_ff_lin_1_params = [torch_model.blocks[1].feedforward.ff_linear_1.weight.detach().cpu(), torch_model.blocks[1].feedforward.ff_linear_1.bias.detach().cpu()]
    block_2_ff_lin_2_params = [torch_model.blocks[1].feedforward.ff_linear_2.weight.detach().cpu(), torch_model.blocks[1].feedforward.ff_linear_2.bias.detach().cpu()]

    atom_model, atom_params = atom_transformer.transformer_blocks(num_transformer_blocks=2, num_attn_heads=2, embedding_dim=VOCAB_SIZE, input_layer_norm_params=[block_1_input_layer_norm, block_2_input_layer_norm], attn_heads_params=[block_1_attn_heads, block_2_attn_heads], attn_out_params=[block_1_attn_proj_out_params, block_2_attn_proj_out_params], feed_forward_params=[[block_1_ff_lin_1_params, block_1_ff_lin_2_params], [block_2_ff_lin_1_params, block_2_ff_lin_2_params]], feed_forward_layer_norm_params=[block_1_ff_layer_norm, block_2_ff_layer_norm])

    # Forward passes
    atom_out = atom_model(atom_x)
    torch_out = torch_model(torch_x)

    # Loss functions
    torch_loss_fn = torch.nn.CrossEntropyLoss()
    atom_loss_fn = loss_nn.cross_entropy_loss()

    # Backward passes
    torch_loss = torch_loss_fn(torch_out.view(BATCH*SEQ_LEN, VOCAB_SIZE), torch_y.view(BATCH*SEQ_LEN))
    torch_loss.backward()
    
    reshaped_atom_out = atom_out['data'].reshape(BATCH*SEQ_LEN, VOCAB_SIZE)
    reshaped_atom_y = atom_y['data'].reshape(BATCH*SEQ_LEN)
    atom_out['data'] = reshaped_atom_out
    atom_y['data'] = reshaped_atom_y
    atom_loss, atom_loss_grad = atom_loss_fn(atom_out, atom_y)
    backward(atom_out, atom_loss_grad.reshape(BATCH, SEQ_LEN, VOCAB_SIZE))

    # Check each parameters gradient
    for i, params in enumerate(torch_model.parameters()):
        # For 2d grad / Batch_size
        # for 3d Batch*seq_len
        atom_param_grad = cupy.asnumpy(atom_params[i]['grad'] / (BATCH*SEQ_LEN))
        torch_param_grad = params.grad.cpu().numpy()
        print(atom_param_grad.shape, torch_param_grad.shape, np.allclose(torch_param_grad, atom_param_grad, atol=1e-5))
        # print(atom_param_grad.shape, torch_param_grad.shape)
        # print(torch_param_grad)
        # print(atom_param_grad)

# compare_transformer_block()

compare_embeddings()
compare_attention()

# NOTE: This test is disable the dropout since it will make a random dropout for the activation
# I already have a test case about dropout and works perfectly fine

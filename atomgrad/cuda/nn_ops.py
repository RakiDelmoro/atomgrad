'''Collection of Neural Network operations'''
import math
import torch
import cupy as cp
import atomgrad.cuda.atom as cuda_atom
import atomgrad.cuda.params_init as init
import atomgrad.cuda.activations_fn.activations as atom_act

# import atom as cuda_atom
# import params_init as init 
# import activations_fn.activations as atom_act

'''NN ops contains the operations for Neural Network this include the forward and backward'''
def recurrent_layer():
    pass

def conv_layer():
    pass

def dropout(p, train=True):
    '''Dropout does not have learnable parameters it's just a regulation that randomly zeros the activation'''

    def forward(atom_tensor):
        return cuda_atom.dropout_(atom_tensor, p, train)

    return forward

def layer_norm(normalized_shape, eps=1e-5, params=None):
    learnable_parameters = []
    
    if params is None:
        weight = cuda_atom.cuda_tensor(cp.ones(shape=normalized_shape, dtype=cp.float32), requires_grad=True)
        bias = cuda_atom.cuda_tensor(cp.zeros(shape=normalized_shape, dtype=cp.float32), requires_grad=True)
    else:
        weight = cuda_atom.cuda_tensor(params[0].numpy(), requires_grad=True)
        bias = cuda_atom.cuda_tensor(params[1].numpy(), requires_grad=True)

    learnable_parameters.extend([weight])
    learnable_parameters.extend([bias])

    def forward(atom_tensor):
        x_normalized = cuda_atom.layer_norm_(atom_tensor, eps)
        mul_result = cuda_atom.mul(weight, x_normalized)
        apply_layer_norm = cuda_atom.add(mul_result, bias)

        return apply_layer_norm

    return forward, learnable_parameters

def embeddings(num_embeddings, embedding_dim, parameters=None):
    learnable_parameters = init.atom_embedding_weight(num_embeddings, embedding_dim)
    if parameters is not None:
        learnable_parameters = cuda_atom.cuda_tensor(parameters.numpy(), requires_grad=True)

    def forward(indices):
        result = cuda_atom.embeddings_(indices, learnable_parameters)

        return result

    return forward, learnable_parameters

def linear_layer(input_size, output_size, parameters=None, bias=True):
    learnable_parameters = init.atom_kaiming_init(input_size, output_size)
    if parameters is not None:
        if not bias:
            learnable_parameters = [cuda_atom.cuda_tensor(parameters[0].data.numpy(), requires_grad=True)]
        else:
            learnable_parameters = [cuda_atom.cuda_tensor(parameters[0].data.numpy(), requires_grad=True), cuda_atom.cuda_tensor(parameters[1].data.numpy(), requires_grad=True)]

    def forward(data):
        result = cuda_atom.matmul(data, learnable_parameters[0])
        if bias:
            result = cuda_atom.add(result, learnable_parameters[1])

        return result

    return forward, learnable_parameters

def attention_layer(head_size, embedding_dim):
    block_size = 256
    tril = cp.tril(cp.ones((block_size, block_size)))
    learnable_parameters = []

    attn_dropout = dropout(p=0.2)
    query, query_params = linear_layer(embedding_dim, head_size)
    key, key_params = linear_layer(embedding_dim, head_size)
    value, value_params = linear_layer(embedding_dim, head_size)

    softmax_act = atom_act.softmax()

    learnable_parameters.extend(query_params)
    learnable_parameters.extend(key_params)
    learnable_parameters.extend(value_params)

    def forward(data):
        B, T, C = data['shape']

        query_projection = query(data)
        key_projection = key(data)
        value_projection = value(data)
        key_projection['data'] = key_projection['data'].transpose(0, 2, 1)

        qk_projection = cuda_atom.matmul(query_projection, key_projection)
        scale = cuda_atom.cuda_tensor(key_projection['shape'][-1]**-0.5)
        scale_projection = cuda_atom.mul(qk_projection, scale)

        mask = tril[:T, :T] == 0
        scale_projection['data'][:, mask] = -cp.inf

        attention_scores = softmax_act(scale_projection)
        attention_scores = attn_dropout(attention_scores)
        attention_weight = cuda_atom.matmul(attention_scores, value_projection)

        return attention_weight

    return forward, learnable_parameters    

def multi_head_attention_layer(num_heads, head_size, embedding_dim):
    learnable_parameters = []

    attention_blocks = []
    for _ in range(num_heads): 
        attn_block, attn_params = attention_layer(head_size, embedding_dim)
        attention_blocks.append(attn_block)
        learnable_parameters.extend(attn_params)

    dropout_proj = dropout(p=0.2)
    out_projection, params = linear_layer(head_size * num_heads, embedding_dim)
    learnable_parameters.extend(params)

    def forward(data):
        attention_heads_outputs = [block(data) for block in attention_blocks]
        concatenated_attn_heads = cuda_atom.concatenate(attention_heads_outputs, axis=-1)
        out_dropout = dropout_proj(concatenated_attn_heads)

        return out_projection(out_dropout)

    return forward, learnable_parameters

def transformer_block(num_attn_heads, embedding_dim):
    learnable_parameters = []

    head_size =  embedding_dim // num_attn_heads
    multi_heads_attention, mha_params = multi_head_attention_layer(num_attn_heads, head_size, embedding_dim)

    # FeedForward
    linear_1, linear_1_params = linear_layer(embedding_dim, embedding_dim * 4)
    activation_fn = atom_act.relu()
    linear_2, linear_2_params = linear_layer(embedding_dim * 4, embedding_dim)
    linear_2_dropout = dropout(p=0.2)

    # Gradients is successfully propagte in layernorms parameters
    layer_norm1, layer_norm1_params = layer_norm(embedding_dim)
    layer_norm2, layer_norm2_params = layer_norm(embedding_dim)

    #TODO: UGLY!
    learnable_parameters.extend(mha_params)
    learnable_parameters.extend(linear_1_params)
    learnable_parameters.extend(linear_2_params)
    learnable_parameters.extend(layer_norm1_params)
    learnable_parameters.extend(layer_norm2_params)

    def forward(data):
        # Residual Connection
        attention_head_output = multi_heads_attention(layer_norm1(data))
        mha_output = cuda_atom.add(data, attention_head_output)

        # FeedForward acts
        linear_1_out = linear_1(layer_norm2(mha_output))
        activation_out = activation_fn(linear_1_out)
        linear_2_out = linear_2(activation_out)
        ff_out = linear_2_dropout(linear_2_out)

        # Residual Connection
        out = cuda_atom.add(mha_output, ff_out)
        return out

    return forward, learnable_parameters

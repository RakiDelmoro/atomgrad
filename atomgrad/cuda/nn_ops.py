'''Collection of Neural Network operations'''
import math
import torch
import cupy as cp
import atomgrad.cuda.atom as atom
import atomgrad.cuda.params_init as init
import atomgrad.cuda.activations_fn.activations as atom_act

# import atom as atom
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
        if train and p != 0:
            if p == 1:
                return atom.cuda_tensor(cp.zeros_like(atom_tensor['data']), requires_grad=train)

            bool_mask = cp.random.rand(*atom_tensor['shape']) > p
            res = bool_mask * atom_tensor['data'] * (float(1.0 / (1.0 - p)))
            return atom.cuda_tensor(res, requires_grad=train)
        else:
            return atom_tensor

    return forward

def layer_norm(normalized_shape, eps=1e-5):
    learnable_parameters = []

    weight = atom.cuda_tensor(cp.ones(shape=normalized_shape, dtype=cp.float32), requires_grad=True)
    bias = atom.cuda_tensor(cp.zeros(shape=normalized_shape, dtype=cp.float32), requires_grad=True)

    learnable_parameters.extend([weight])
    learnable_parameters.extend([bias])

    def forward(atom_tensor):
        x_normalized = atom.layer_norm_(atom_tensor, eps)
        apply_layer_norm = atom.add(atom.mul(weight, x_normalized), bias)

        return apply_layer_norm

    return forward, learnable_parameters

def embeddings(num_embeddings, embedding_dim):
    parameters = init.atom_embedding_weight(num_embeddings, embedding_dim)

    def forward(indices):
        if cp.any(indices['data'] < 0) or cp.any(indices['data'] >= num_embeddings):
            raise ValueError("Indices out of range [0, num_embeddings-1]")

        result = atom.cuda_tensor(parameters['data'][indices['data'].astype(int)], True)
        result['depends_on'] = parameters

        return result

    return forward, parameters

def linear_layer(input_size, output_size, bias=True, dropout_p=0, train=True):
    learnable_parameters = init.atom_kaiming_init(input_size, output_size)
    
    def forward(data):
        result = atom.matmul(data, learnable_parameters[0])
        if bias:
            result = atom.add(result, learnable_parameters[1])
            if dropout_p > 0:
                # Apply dropout if applicable
                dropped_out = dropout(dropout_p, train)(result)
                return dropped_out

        return result

    return forward, learnable_parameters

def attention_layer(head_size, embedding_dim):
    learnable_parameters = []
    
    query, query_params = linear_layer(embedding_dim, head_size)
    key, key_params = linear_layer(embedding_dim, head_size)
    value, value_params = linear_layer(embedding_dim, head_size)

    learnable_parameters.extend(query_params)
    learnable_parameters.extend(key_params)
    learnable_parameters.extend(value_params)

    def forward(data):
        query_projection = query(data)
        key_projection = key(data)
        value_projection = value(data)
        key_projection['data'] = key_projection['data'].transpose(0, 2, 1)

        qk_projection = atom.matmul(query_projection, key_projection)
        scale = atom.cuda_tensor(key_projection['shape'][-1]**-5)
        scale_projection = atom.mul(qk_projection, scale)
        attention_scores = atom_act.softmax()(scale_projection)
        attention_weight = atom.matmul(attention_scores, value_projection)

        return attention_weight
    
    return forward, learnable_parameters    

def multi_head_attention_layer(num_heads, head_size, embedding_dim):
    learnable_parameters = []
    
    attention_heads = [attention_layer(head_size, embedding_dim) for _ in range(num_heads)]
    out_projection, params = linear_layer(head_size * num_heads, embedding_dim)

    for _, attn_params in attention_heads:
        learnable_parameters.extend(attn_params)

    # learnable_parameters.extend(attn_head_params for _, attn_head_params in attention_heads)
    learnable_parameters.extend(params)

    def forward(data):
        attention_heads_outputs = [attn_head(data) for attn_head, _ in attention_heads]
        concatenated_attn_heads = atom.concatenate(attention_heads_outputs, axis=-1)

        return out_projection(concatenated_attn_heads)
    
    return forward, learnable_parameters

def transformer_block(num_attn_heads, embedding_dim):
    learnable_parameters = []

    head_size =  embedding_dim // num_attn_heads
    multi_heads_attention, mha_params = multi_head_attention_layer(num_attn_heads, head_size, embedding_dim)

    # FeedForward
    linear_1, linear_1_params = linear_layer(embedding_dim, embedding_dim * 4)
    activation_fn = atom_act.relu()
    # linear2 dropout activativated
    linear_2, linear_2_params = linear_layer(embedding_dim * 4, embedding_dim, dropout_p=0.5)

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
        mha_output = atom.add(data, multi_heads_attention(layer_norm1(data)))
        
        # FeedForward acts
        linear_1_out = linear_1(layer_norm2(mha_output))
        activation_out = activation_fn(linear_1_out)
        linear_2_out = linear_2(activation_out)
        
        # Residual Connection
        out = atom.add(mha_output, linear_2_out)

        return out
        
    return forward, learnable_parameters

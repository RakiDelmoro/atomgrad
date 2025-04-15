import cupy as cp
import atomgrad.cuda.nn_ops as cuda_ops
import atomgrad.cuda.atom as cuda_atom
import atomgrad.cuda.activations_fn.activations as atom_act
import atomgrad.cuda.loss_fn.loss_fn_nn as loss_fn

# Test run
VOCAB_SIZE = 10
SEQ_LEN = 3
EMBEDDING_DIM = 10

# Actual Run
# VOCAB_SIZE = 65
# EMBEDDING_DIM = 384
# SEQ_LEN = 256

def char_embeddings(char_params=None, pos_params=None):
    trainable_parameters = []
    # if char_params is not None and pos_params is not None:
        # char_embedding, char_emb_params = cuda_ops.embeddings(num_embeddings=VOCAB_SIZE, embedding_dim=VOCAB_SIZE, parameters=char_params)
        # pos_embedding, pos_emb_params = cuda_ops.embeddings(num_embeddings=SEQ_LEN, embedding_dim=VOCAB_SIZE, parameters=pos_params)
    # else:
    char_embedding, char_emb_params = cuda_ops.embeddings(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, parameters=char_params) #if char_params is None else char_params
    pos_embedding, pos_emb_params = cuda_ops.embeddings(num_embeddings=SEQ_LEN, embedding_dim=EMBEDDING_DIM, parameters=pos_params) #if pos_params is None else pos_params

    trainable_parameters.extend([char_emb_params])
    trainable_parameters.extend([pos_emb_params])

    def forward(data):
        _, T = data['shape']

        tok_embeddings = char_embedding(data)
        pos_embeddings = pos_embedding(cuda_atom.cuda_tensor(cp.arange(T), requires_grad=True))    
        tok_and_pos_embeddings = cuda_atom.add(tok_embeddings, pos_embeddings)

        return tok_and_pos_embeddings

    return forward, trainable_parameters

def attention_layer(head_size, embedding_dim, query_params=None, key_params=None, value_params=None):
    tril = cp.tril(cp.ones((SEQ_LEN, SEQ_LEN)))
    learnable_parameters = []

    attn_dropout = cuda_ops.dropout(p=0.2)
    query, query_params = cuda_ops.linear_layer(embedding_dim, head_size, query_params, bias=False)
    key, key_params = cuda_ops.linear_layer(embedding_dim, head_size, key_params, bias=False)
    value, value_params = cuda_ops.linear_layer(embedding_dim, head_size, value_params, bias=False)

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
        # attention_scores = attn_dropout(attention_scores)
        attention_weight = cuda_atom.matmul(attention_scores, value_projection)

        return attention_weight

    return forward, learnable_parameters

def multi_head_attention_layer(num_heads, head_size, embedding_dim, attn_heads_params=None, linear_proj_params=None):
    learnable_parameters = []

    attention_blocks = []
    for each in range(num_heads):
        if attn_heads_params is not None:
            query_params, key_params, value_params = [attn_heads_params[each].query.weight.detach().cpu()], [attn_heads_params[each].key.weight.detach().cpu()], [attn_heads_params[each].value.weight.detach().cpu()]
        else:
            query_params, key_params, value_params = None, None, None
        
        attn_block, attn_params = attention_layer(head_size, embedding_dim, query_params, key_params, value_params)
        attention_blocks.append(attn_block)
        learnable_parameters.extend(attn_params)

    dropout_proj = cuda_ops.dropout(p=0.2)
    out_projection, attn_proj_params = cuda_ops.linear_layer(head_size * num_heads, embedding_dim, linear_proj_params)
    learnable_parameters.extend(attn_proj_params)

    def forward(data):
        attention_heads_outputs = [block(data) for block in attention_blocks]
        concatenated_attn_heads = cuda_atom.concatenate(attention_heads_outputs, axis=-1)
        # out_dropout = concatenated_attn_heads
        out = out_projection(concatenated_attn_heads)

        # return dropout_proj(out)
        return out

    return forward, learnable_parameters

def feedforward(embedding_dim, linear_1_params=None, linear_2_params=None, layer_norm_params=None):
# def feedforward(embedding_dim):
    learnable_parameters = []

    linear_1, linear_1_params = cuda_ops.linear_layer(embedding_dim, embedding_dim * 4, linear_1_params)
    activation_fn = atom_act.relu()
    linear_2, linear_2_params = cuda_ops.linear_layer(embedding_dim * 4, embedding_dim, linear_2_params)

    ff_dropout = cuda_ops.dropout(p=0.2)
    ff_layer_norm, norm_params = cuda_ops.layer_norm(embedding_dim, params=layer_norm_params)

    learnable_parameters.extend(norm_params)
    learnable_parameters.extend(linear_1_params)
    learnable_parameters.extend(linear_2_params)

    def forward(data):
        norm_out = ff_layer_norm(data)
        linear_1_out = linear_1(norm_out)
        activation_out = activation_fn(linear_1_out)
        linear_2_out = linear_2(activation_out)

        # Apply residual connection
        # out = cuda_atom.add(data, ff_dropout(linear_2(activation_out)))
        out = cuda_atom.add(data, linear_2_out)

        return out
    
    return forward, learnable_parameters

def transformer_block(num_attn_heads, embedding_dim, layer_norm_params, attn_heads_params, attn_out_proj, feedforward_params, ff_layer_norm):
# def transformer_block(num_attn_heads, embedding_dim):
    learnable_parameters = []

    head_size =  embedding_dim // num_attn_heads

    layer_norm1, layer_norm1_params = cuda_ops.layer_norm(embedding_dim, params=layer_norm_params)
    multi_heads_attention, mha_params = multi_head_attention_layer(num_attn_heads, head_size, embedding_dim, attn_heads_params=attn_heads_params, linear_proj_params=attn_out_proj)

    # FeedForward
    feedforward_net, feedforward_params = feedforward(embedding_dim, feedforward_params[0], feedforward_params[1], ff_layer_norm)

    learnable_parameters.extend(layer_norm1_params)
    learnable_parameters.extend(mha_params)
    learnable_parameters.extend(feedforward_params)

    def forward(data):
        # Residual Connection
        attention_head_output = multi_heads_attention(layer_norm1(data))
        mha_output = cuda_atom.add(data, attention_head_output)
        out = feedforward_net(mha_output)

        return out

    return forward, learnable_parameters

def transformer_blocks(num_transformer_blocks, num_attn_heads, embedding_dim, input_layer_norm_params=None, attn_heads_params=None, attn_out_params=None, feed_forward_params=None, feed_forward_layer_norm_params=None):
# def transformer_blocks(num_transformer_blocks, num_attn_heads, embedding_dim):
    learnable_parameters = []

    transformer_blocks = []
    for block in range(num_transformer_blocks):
        transformer_layer, params = transformer_block(num_attn_heads=num_attn_heads, embedding_dim=embedding_dim, layer_norm_params=input_layer_norm_params[block], attn_heads_params=attn_heads_params[block], attn_out_proj=attn_out_params[block], feedforward_params=feed_forward_params[block], ff_layer_norm=feed_forward_layer_norm_params[block])
        # transformer_layer, params = transformer_block(num_attn_heads=num_attn_heads, embedding_dim=embedding_dim)
        transformer_blocks.append(transformer_layer)
        learnable_parameters.extend(params)

    def forward(data):
        embeddings = data
        for each in transformer_blocks:
            embeddings = each(embeddings)

        return embeddings
    
    return forward, learnable_parameters

def atom_transformer(num_transformer_blocks=6, vocab_size=10, embedding_dim=128, num_attn_heads=6):
    learnable_parameters = []

    tok_embeddings, parameters = char_embeddings()
    learnable_parameters.extend(parameters)

    # blocks, transformer_params = transformer_blocks(num_transformer_blocks, num_attn_heads, embedding_dim)

    transformer_blocks = []
    for _ in range(num_transformer_blocks):
        transformer_layer, params = transformer_block(num_attn_heads=num_attn_heads, embedding_dim=embedding_dim)
        transformer_blocks.append(transformer_layer)
        learnable_parameters.extend(params)

    layer_norm, layer_norm_params = cuda_ops.layer_norm(embedding_dim)
    linear_classifier, classifier_params = cuda_ops.linear_layer(embedding_dim, vocab_size)
    learnable_parameters.extend(layer_norm_params)
    learnable_parameters.extend(classifier_params)

    def forward(input_data, target=None):
        B, T = input_data['shape']

        embeddings = tok_embeddings(input_data)
        for each in transformer_blocks:
            embeddings = each(embeddings)
    
        out = linear_classifier(layer_norm(embeddings))

        if target is None:
            loss = None
            gradients = None
        else:
            B, T, C = out['shape']
            reshaped_out = out['data'].reshape(B*T, C)
            out['data'] = reshaped_out
            target = target['data'].reshape(B*T)
            loss, gradients = loss_fn.cross_entropy_loss()(out, target)

        return out, loss, gradients

    return forward, learnable_parameters

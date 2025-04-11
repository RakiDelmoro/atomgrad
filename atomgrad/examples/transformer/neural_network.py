import cupy as cp
import atomgrad.cuda.nn_ops as cuda_ops
import atomgrad.cuda.atom as cuda_atom

def transformer(num_transformer_blocks=4, block_size=256, vocab_size=10, embedding_dim=128, num_attn_heads=4):
    learnable_parameters = []

    char_embedding, char_emb_params = cuda_ops.embeddings(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    pos_embedding, pos_emb_params = cuda_ops.embeddings(num_embeddings=block_size, embedding_dim=embedding_dim)

    learnable_parameters.extend([char_emb_params])
    learnable_parameters.extend([pos_emb_params])

    transformer_blocks = []
    for _ in range(num_transformer_blocks):
        transformer_layer, params = cuda_ops.transformer_block(num_attn_heads=num_attn_heads, embedding_dim=embedding_dim)
        transformer_blocks.append(transformer_layer)
        learnable_parameters.extend(params)

    layer_norm, layer_norm_params = cuda_ops.layer_norm(embedding_dim)
    linear_classifier, classifier_params = cuda_ops.linear_layer(embedding_dim, vocab_size)

    learnable_parameters.extend(layer_norm_params)
    learnable_parameters.extend(classifier_params)

    def forward(input_data, target=None):
        B, T = input_data['shape']
        tok_embeddings = char_embedding(input_data)
        pos_embeddings = pos_embedding(cuda_atom.cuda_tensor(cp.arange(T), requires_grad=True))
    
        embeddings = cuda_atom.add(tok_embeddings, pos_embeddings)

        for each in transformer_blocks:
            embeddings = each(embeddings)

        out = linear_classifier(layer_norm(embeddings))
        return out

    return forward, learnable_parameters

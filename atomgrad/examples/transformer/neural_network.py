import atomgrad.cuda.nn_ops as cuda_ops

def transformer(num_transformer_blocks=4, vocab_size=10, embedding_dim=128, num_attn_heads=4):
    learnable_parameters = []

    char_embedding, embeddings_params = cuda_ops.embeddings(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    learnable_parameters.extend([embeddings_params])

    transformer_blocks = []
    for _ in range(num_transformer_blocks):
        transformer_layer, params = cuda_ops.transformer_block(num_attn_heads=num_attn_heads, embedding_dim=embedding_dim)
        transformer_blocks.append(transformer_layer)
        learnable_parameters.extend(params)

    layer_norm, layer_norm_params = cuda_ops.layer_norm(embedding_dim)
    linear_classifier, classifier_params = cuda_ops.linear_layer(embedding_dim, vocab_size)

    learnable_parameters.extend(layer_norm_params)
    learnable_parameters.extend(classifier_params)

    def forward(data):
        embeddings = char_embedding(data)
        for each in transformer_blocks:
            embeddings = each(embeddings)

        out = linear_classifier(layer_norm(embeddings))
        return out, embeddings

    return forward, learnable_parameters

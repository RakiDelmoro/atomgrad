import cupy as cp
import numpy as np
import atomgrad.atom as atom
import atomgrad.cuda.atom as cuda_atom
import atomgrad.cuda.optimizer as cuda_opt
import atomgrad.cuda.loss_fn.loss_fn_nn as loss_fn
from atomgrad.examples.transformer.neural_network import transformer

BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_EPOCHS = 5000
LEARNING_RATE = 3e-4

EMBEDDING_DIM = 384
NUM_HEAD = 6
NUM_TRANSFORMER_BLOCK = 6
DROPOUT = 0.2

def get_batch(data):
    idx = np.random.randint(len(data) - BLOCK_SIZE, size=(BATCH_SIZE,))
    context = np.stack([data[i:i+BLOCK_SIZE] for i in idx])
    next_context = np.stack([data[i+1:i+BLOCK_SIZE+1] for i in idx])

    return cuda_atom.cuda_tensor(context, requires_grad=True), cuda_atom.cuda_tensor(next_context, requires_grad=True)

    return cuda.context, next_context

def test_runner():
    with open('./dataset/shakespeare.txt', 'r', encoding='utf-8') as f: text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    string_to_int = {ch:i for i, ch in enumerate(chars)}
    int_to_string = {i:ch for i, ch in enumerate(chars)}

    encode_text = lambda text: [string_to_int[ch] for ch in text]
    decode_text = lambda ints: ''.join([int_to_string[i] for i in ints])

    # Train and test splits
    data = np.array(encode_text(text), dtype=np.longlong)
    split = int(0.9*len(data)) # 90% train rest test
    train_data = data[:split]
    test_data = data[split:]

    model, parameters = transformer(embedding_dim=EMBEDDING_DIM, vocab_size=vocab_size, num_transformer_blocks=4)
    loss_func = loss_fn.cross_entropy_loss()
    adam_step, adam_zero_grad = cuda_opt.adam(parameters)

    for _ in range(MAX_EPOCHS):
        train_batched = get_batch(train_data)
        # test_batched = get_batch(test_data)
        model_pred = model(train_batched[0])

        reshaped_model_pred = model_pred['data'].reshape(BATCH_SIZE*BLOCK_SIZE, vocab_size)
        model_pred['data'] = reshaped_model_pred

        loss, grad = loss_func(model_pred, train_batched[1]['data'].reshape(BATCH_SIZE*BLOCK_SIZE))
        grad = grad.reshape(BATCH_SIZE, BLOCK_SIZE, vocab_size)

        adam_zero_grad(parameters)
        atom.backward(model_pred, grad)
        adam_step(train_batched[0]['shape'][0])

        print(loss)

        #TODO: first 6 parameters have zero gradient suggest there's a bug in backward pass
        #TODO: Implement the attention mask in atomgrad

test_runner()

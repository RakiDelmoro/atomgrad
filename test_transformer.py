import cupy as cp
import numpy as np
import atomgrad.atom as atom
import atomgrad.cuda.atom as cuda_atom
import atomgrad.cuda.optimizer as cuda_opt
import atomgrad.cuda.loss_fn.loss_fn_nn as loss_fn
from atomgrad.examples.transformer.neural_network import atom_transformer

import torch
from atomgrad.torch_examples.transformer.neural_network import TorchTransformer

BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_EPOCHS = 5000
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
EMBEDDING_DIM = 384
NUM_HEAD = 6
NUM_TRANSFORMER_BLOCK = 6
DROPOUT = 0.2
EVAL_ITERVAL = 500

def get_batch(data):
    idx = np.random.randint(len(data) - BLOCK_SIZE, size=(BATCH_SIZE,))
    context = np.stack([data[i:i+BLOCK_SIZE] for i in idx])
    next_context = np.stack([data[i+1:i+BLOCK_SIZE+1] for i in idx])

    return cuda_atom.cuda_tensor(context, requires_grad=True), cuda_atom.cuda_tensor(next_context, requires_grad=True)

def atom_evaluate_iterations(model, train_data, test_data):
    train_losses = cp.zeros(EVAL_ITERS)
    test_losses = cp.zeros(EVAL_ITERS)

    for k in range(EVAL_ITERS):
        x_train, y_train = get_batch(train_data)
        x_test, y_test = get_batch(test_data)

        train_logits, train_loss, _ = model(x_train, y_train)
        atom.cleaner(train_logits)
        test_logits, test_loss, _ = model(x_test, y_test)
        atom.cleaner(test_logits)

        train_losses[k] = train_loss.item()
        test_losses[k] = test_loss.item()

        print(f'EVAL: {k}', end='\r', flush=True)

    return cp.mean(train_losses), cp.mean(test_losses)

@torch.no_grad()
def torch_evaluate_iterations(model, train_data, test_data):
    out = {}
    model.eval()

    train_losses = torch.zeros(EVAL_ITERS)
    test_losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
        x_train, y_train = get_batch(train_data)
        x_test, y_test = get_batch(test_data)

        x_train, y_train = torch.tensor(cp.asnumpy(x_train['data']), dtype=torch.long, device='cuda'), torch.tensor(cp.asnumpy(y_train['data']), dtype=torch.long, device='cuda')
        x_test, y_test = torch.tensor(cp.asnumpy(x_test['data']), dtype=torch.long, device='cuda'), torch.tensor(cp.asnumpy(y_test['data']), dtype=torch.long, device='cuda')
        
        train_logits, train_loss = model(x_train, y_train)
        test_logits, test_loss = model(x_test, y_test)
        
        train_losses[k] = train_loss.item()
        test_losses[k] = test_loss.item()

        print(f'EVAL: {k}', end='\r', flush=True)
    model.train()
    return torch.mean(train_losses), torch.mean(test_losses)

def atom_test_runner():
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

    model, parameters = atom_transformer(embedding_dim=EMBEDDING_DIM, vocab_size=vocab_size, num_transformer_blocks=6, num_attn_heads=6)
    loss_func = loss_fn.cross_entropy_loss()
    adam_step, adam_zero_grad = cuda_opt.adam_w(parameters, LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
    #     if epoch % EVAL_ITERVAL == 0 or epoch == MAX_EPOCHS -1:
    #         print('EVALUATING>>>')
    #         train_loss, test_loss = evaluate_iterations(model, train_data, test_data)
    #         print(f"step {iter}: train loss {train_loss:.4f}, val loss {test_loss:.4f}")

        x_batched, y_batched = get_batch(train_data)
        model_pred, loss, grad = model(x_batched, y_batched)
        print(f'EPOCH: {epoch} LOSS: {loss}')
        adam_zero_grad(parameters)
        atom.backward(model_pred, grad.reshape(BATCH_SIZE, BLOCK_SIZE, vocab_size))
        adam_step(x_batched['shape'][0], x_batched['shape'][1])

def torch_test_runner():
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

    model = TorchTransformer()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        # if epoch % EVAL_ITERVAL == 0 or epoch == MAX_EPOCHS -1:
        #     print('EVALUATING>>>')
        #     train_loss, test_loss = torch_evaluate_iterations(model, train_data, test_data)
        #     print(f"step {epoch}: train loss {train_loss:.4f}, val loss {test_loss:.4f}")

        x_batched, y_batched = get_batch(train_data)
        x_batched, y_batched = torch.tensor(cp.asnumpy(x_batched['data']), dtype=torch.long, device='cuda'), torch.tensor(cp.asnumpy(y_batched['data']), dtype=torch.long, device='cuda')
        model_pred, loss = model(x_batched, y_batched)

        # print(f'EPOCH: {epoch} LOSS: {loss}', end='\r', flush=True)
        print(f'EPOCH: {epoch} LOSS: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

# torch_test_runner()
atom_test_runner()

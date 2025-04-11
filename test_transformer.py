import cupy as cp
import atomgrad.atom as atom
from atomgrad.examples.transformer.neural_network import transformer

def test_runner():
    model, model_params = transformer(embedding_dim=512, vocab_size=5, num_transformer_blocks=4)

    x_test = atom.tensor(cp.random.randint(low=0, high=5, size=(1, 10)), device='cuda')
    pred_model, emb = model(x_test)
    grad = cp.random.randn(1, 10, 5)

    atom.backward(pred_model, grad)

    print(emb['grad'])
    print(model_params[-1]['grad'])

    # Backpropagation FIXED
    #TODO: Feed me with real data!

test_runner()

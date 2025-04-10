import cupy as cp
import atomgrad.atom as atom
from atomgrad.examples.transformer.neural_network import transformer

model = transformer(embedding_dim=512, vocab_size=5)

x_test = atom.tensor(cp.random.randint(low=0, high=5, size=(1, 10)), device='cuda')
print(model(x_test)['data'].shape)

# TODO: Train with real texts

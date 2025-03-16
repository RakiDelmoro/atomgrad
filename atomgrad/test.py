import torch
import numpy as np
from tensor import Atom

x = Atom.tensor(np.random.randn(2, 3))
# output size - input_size
y = Atom.tensor(np.random.randn(1, 3))

z = x + 0.1
print(x.data)
print(z.data)

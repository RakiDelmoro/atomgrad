<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)"
          srcset="https://github.com/user-attachments/assets/52d23919-62a4-4a13-bd1b-fc2fb9a33d62">
  <source media="(prefers-color-scheme: dark)"
          srcset="https://github.com/user-attachments/assets/52d23919-62a4-4a13-bd1b-fc2fb9a33d62">
  <img width="50%" height="50%" alt="Image"
       src="https://github.com/user-attachments/assets/52d23919-62a4-4a13-bd1b-fc2fb9a33d62">
</picture>

atomgrad: autograd for numpy/cupy and learn how neural network trains under the hood.

</div>

### Neural networks
```python
import atomgrad.nn as nn
import atomgrad.optim as optim
from atomgrad.tensor import atom

class LinearNet:
    def __init__(self):
        self.linear1, self.linear1_params = nn.linear(784, 128)
        self.linear2, self.linear2_params = nn.linear(128, 10)
    def __call__(self, x: atom):
        return self.linear2(self.linear1(x).relu())
    
model = LinearNet()
loss_fn = nn.cross_entropy()
step, zero_grad = optim.Adam([model.linear1_params, model.linear2_params], lr=0.001)

x, y = atom.rand(shape=(4, 784)), atom.tensor([2, 3, 6, 8])

# Training Loop
for epoch in range(10):
    loss = loss_fn(model(x), y)
    zero_grad()
    loss.backward()
    step()
    print(epoch, loss.data.item())
```








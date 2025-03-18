import atom
import torch

def test_sanity():
    x = atom.Tensor(-4.0, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass satisfy
    assert ymg.data == ypt.data.item()
    # backward pass satisfy
    assert xmg.grad.data == xpt.grad.item()

    print(ymg.data, ypt.data.item())
    print(xmg.grad.data, xpt.grad.item())

test_sanity()
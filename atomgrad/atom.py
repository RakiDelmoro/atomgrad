import numpy as np
from typing import List, Optional, Union, Callable, NamedTuple

class Dependencies(NamedTuple):
    tensor: 'Tensor'
    grad_function: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]
Tensorable = Union['Tensor', float, np.ndarray]

def ensure_is_array(data: Arrayable):
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)
    
def ensure_is_tensor(data: Tensorable):
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data)

class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool=False, depends_on: List[Dependencies]=None):
        self.data = ensure_is_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad: self.zero_grad()

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float32))

    def __repr__(self):
        return f'atom.tensor(data={self.data}, requires_grad={self.requires_grad})'

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == (): grad = Tensor(1.0)
            else: raise RuntimeError("grad must be specified for non-0-tensor")

        # Accumulate gradients
        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_function(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def __sub__(self, other) -> 'Tensor':
        return sub(self, ensure_is_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return sub(ensure_is_tensor(other), self)

    def __matmul__(self, other) -> 'Tensor':
        return matmul(self, ensure_is_tensor(other))
    
    def __rmatmul__(self, other) -> 'Tensor':
        return matmul(ensure_is_tensor(other), self)

    def __mul__(self, other) -> 'Tensor':
        return mul(self, ensure_is_tensor(other))
    
    def __rmul__(self, other) -> 'Tensor':
        return mul(ensure_is_tensor(other), self)

    def __add__(self, other) -> 'Tensor':
        return add(self, ensure_is_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        return add(ensure_is_tensor(other), self)    

'''Tensor Opsss'''

def sub(data1: Tensor, data2: Tensor) -> Tensor:
    return data1 + -data2

def mul(data1: Tensor, data2: Tensor):
    out = data1.data * data2.data
    requires_grad = data1.requires_grad or data2.requires_grad

    depends_on: List[Dependencies] = []

    if data1.requires_grad:
        def grad_fn1(grad: np.ndarray):
            grad = grad * data2.data
            ndims_added = grad.ndim - data1.data.ndim
            # Sum out added dims
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i , dim in enumerate(data2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependencies(data1, grad_fn1))

    if data2.requires_grad:
        def grad_fn2(grad: np.ndarray):
            grad = grad * data1.data
            # Sum out added dims
            ndims_added = grad.ndim - data2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(data2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependencies(data2, grad_fn2))

    return Tensor(out, requires_grad, depends_on)

def add(data1: Tensor, data2: Tensor):
    out = data1.data + data2.data

    requires_grad = data1.requires_grad or data2.requires_grad
    depends_on: List[Dependencies] = []

    if data1.requires_grad:
        def grad_fn1(grad: np.ndarray):
             # Sum out added dims
            ndims_added = grad.ndim - data1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(data1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependencies(data1, grad_fn1))

    if data2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - data2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(data2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
        depends_on.append(Dependencies(data2, grad_fn2))

    return Tensor(out, requires_grad, depends_on)

def matmul(data1: Tensor, data2: Tensor):
    out = np.matmul(data1.data, data2.data.T)
    # If 3d shape don't need to transpose the other data
    if data1.data.ndim == 3 and data2.data.ndim == 3: out = np.matmul(data1.data, data2.data)

    requires_grad = data1.requires_grad or data2.requires_grad
    depends_on: List[Dependencies] = []

    if data1.requires_grad:
        def grad_fn1(grad: np.ndarray):
            return grad @ data2.data.swapaxes(-1, -2) 
        depends_on.append(Dependencies(data1, grad_fn1))

    if data2.requires_grad:
        def grad_fn2(grad: np.ndarray):
            return data1.data.swapaxes(-1, -2) @ grad
        depends_on.append(Dependencies(data2, grad_fn2))

    return Tensor(out, requires_grad, depends_on)

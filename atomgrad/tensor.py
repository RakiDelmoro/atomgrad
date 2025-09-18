import cupy as cp
import numpy as np
from typing import Iterable, Tuple

class atom:
    def __init__(self, data,
                 device: str = 'cpu',
                 requires_grad: bool = False,
                 depends_on: Tuple['atom', ...] = (),
                 operation: str = '',
                 grad_fn=None):

        self.data = np.asarray(data, dtype=np.float32) if device == 'cpu' else cp.asarray(data, dtype=cp.float32)
        self.device = device
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        self.grad = None
        self.requires_grad = requires_grad

        # Graph bookkeeping
        self._depends_on = depends_on
        self._operation = operation
        self._grad_fn = grad_fn

    def __mul__(self, other):
        other = other if isinstance(other, atom) else atom(other)
        assert self.device == other.device, f'Must be same device'
        
        requires_grad = self.requires_grad or other.requires_grad
        result = self.data * other.data

        def grad_fn(grad):
            if self.requires_grad:
                self.grad = cp.zeros_like(self.shape) if self.device == 'cuda' else np.zeros_like(self.shape)
                if self.ndim == 1:
                    self.grad += cp.sum(cp.sum(grad.data * other.data, axis=0), axis=0) if self.device == 'cuda' else np.sum(np.sum(grad.data * other.data, axis=0), axis=0)
                if self.ndim == grad.ndim:
                    self.grad += atom(grad.data * other.data, self.device)
            
            if other.requires_grad:
                other.grad = cp.zeros_like(self.shape) if self.device == 'cuda' else np.zeros_like(self.shape)
                other.grad += atom(grad * self.data, self.device)

        out = atom(result, requires_grad=requires_grad, device=self.device, depends_on=(self, other), operation='+', grad_fn=grad_fn)

        return out

    def __add__(self, other):
        other = other if isinstance(other, atom) else atom(other)
        assert self.device == other.device, f'Must be same device'
        
        requires_grad = self.requires_grad or other.requires_grad
        result = self.data + other.data

        def grad_fn(grad):
            if self.requires_grad:
                self.grad = cp.zeros_like(self.data) if self.device == 'cuda' else np.zeros_like(self.data)
                dim_diff = grad.ndim - self.grad.ndim
                if dim_diff > 0:
                    self.grad += atom(grad.data.sum(axis=tuple(range(dim_diff))), self.device)
                else:
                    self.grad += grad.data
            if other.requires_grad:
                other.grad = cp.zeros_like(self.data) if self.device == 'cuda' else np.zeros_like(self.data)
                dim_diff = grad.ndim - other.grad.ndim
                if dim_diff > 0:
                    other.grad += atom(grad.data.sum(axis=tuple(range(dim_diff))), self.device)
                else:
                    other.grad += grad.data

        out = atom(result, requires_grad=requires_grad, device=self.device, depends_on=(self, other), operation='+', grad_fn=grad_fn)

        return out
    
    def matmul(x1, x2):
        x1 = x1 if isinstance(x2, atom) else atom(x1)
        x2 = x2 if isinstance(x2, atom) else atom(x2)
        assert x1.device == x2.device, f'Must be same device'

        # Gather the shape of two arrays
        x1_shape, x2_shape = x1.shape, x2.shape
        x1_ndim, x2_ndim = x1.ndim, x2.ndim

        if x1_ndim != 3 or x2_ndim != 3:
            result = np.matmul(x1.data, x2.data) if x1.device == 'cpu' else cp.matmul(x1.data, x2.data)
        else:
            if x1_shape[-1] != x2_shape[-1]:
                result = np.matmul(x1.data, x2.data) if x1.device == 'cpu' else cp.matmul(x1.data, x2.data)
            else:
                result = np.matmul(x1.data, x2.data).transpose(0, 2, 1) if x1.device == 'cpu' else cp.matmul(x1.data, x2.data).transpose(0, 2, 1)

        requires_grad = x1.requires_grad or x2.requires_grad

        x1.grad = np.zeros_like(x1.data) if x1.device == 'cpu' else cp.zeros_like(x1.data)
        x2.grad = np.zeros_like(x2.data) if x1.device == 'cpu' else cp.zeros_like(x2.data)

        def grad_fn(grad):
            if x1_ndim != 3 and x2_ndim != 3:
                if x1.requires_grad:
                    x1.grad += np.matmul(grad, x2.data) if x1.device == 'cpu' else cp.matmul(grad, x2.data)
                if x2.requires_grad:
                    x2.grad += np.matmul(grad.T, x1.data) if x1.device == 'cpu' else cp.matmul(grad.T, x1.data)

            elif x1_ndim == 3 and x2_ndim != 3:
                if x1.requires_grad:
                    x1.grad += np.matmul(grad, x2.data) if x1.device == 'cpu' else cp.matmul(grad, x2.data)
                if x2.requires_grad:
                    x2.grad += np.matmul(grad.transpose(0,2,1), x1.data).sum(axis=0) if x1.device == 'cpu' else cp.matmul(grad.transpose(0,2,1), x1.data).sum(axis=0)

            else:
                if grad.shape == (2, 3, 3):
                    print('DEBUG')

                if x1_shape == x2_shape:
                    if x1.requires_grad:
                        x1.grad += np.matmul(grad, x2.data) if x1.device == 'cpu' else cp.matmul(grad, x2.data)
                    if x2.requires_grad:
                        x2.grad += np.matmul(x1.data.transpose(0,2,1), grad).transpose(0,2,1) if x1.device == 'cpu' else cp.matmul(x1.data.transpose(0,2,1), grad).transpose(0,2,1)
                else:
                    if x1.requires_grad:
                        x1.grad += np.matmul(grad, x2.data.transpose(0,2,1)) if x1.device == 'cpu' else cp.matmul(grad, x2.data.transpose(0,2,1))
                    if x2.requires_grad:
                        x2.grad += np.matmul(grad.transpose(0,2,1), x1.data).transpose(0,2,1) if x1.device == 'cpu' else cp.matmul(grad.transpose(0,2,1), x1.data).transpose(0,2,1)

        out = atom(result, requires_grad=requires_grad, device=x1.device, depends_on=(x1, x2), operation='@', grad_fn=grad_fn)
    
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __repr__(self):
        if self.requires_grad != False:
            ret = f"tensor({self.data}, device='{self.device}', requires_grad={self.requires_grad})"
        else:
            ret = f"tensor({self.data}, device='{self.device}')"

        return ret

    @staticmethod
    def zeros(shape: Iterable[int], device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.zeros(shape, dtype=np.float32) if device == 'cpu' else cp.zeros(shape, dtype=cp.float32)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret

    @staticmethod
    def ones(shape: Iterable[int], device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.ones(shape, dtype=np.float32) if device == 'cpu' else cp.ones(shape, dtype=cp.float32)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret
    
    @staticmethod
    def empty(shape: Iterable[int], device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.empty(shape, dtype=np.float32) if device == 'cpu' else cp.empty(shape, dtype=cp.float32)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret
    
    @staticmethod
    def randn(shape: Iterable[int], device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.random.randn(*shape).astype(np.float32) if device == 'cpu' else cp.random.randn(*shape, dtype=cp.float32)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret
    
    @staticmethod
    def rand(shape: Iterable[int], device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.random.rand(*shape).astype(np.float32) if device == 'cpu' else cp.random.rand(*shape, dtype=cp.float32)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret
    
    def backward(self, grad=None):
        if not self.requires_grad: return

        # Topo sort
        topo, visited = [], set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for d in t._depends_on:
                    build_topo(d)
                topo.append(t)
        build_topo(self)

        if grad is None:
            grad_arr = np.ones_like(self.data) if self.device == 'cpu' else cp.ones_like(self.data)
            self.grad = atom(grad_arr, self.device)
        else:
            self.grad = grad

        # gradients dict
        for node in reversed(topo):
            if node._grad_fn is not None:
                node._grad_fn(node.grad)

                node._depends_on = []
                node._grad_fn = None

    def zero_grad(self):
        arr = np.zeros_like(self.data) if self.device == 'cpu' else cp.zeros_like(self.data)
        self.grad = None if self.grad is None else arr


# TODO: Write a function for matrix multiply and other operation need for training Neural Network


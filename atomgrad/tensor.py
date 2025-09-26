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
                self.grad = atom.zeros_like(self, self.device)
                if self.ndim == 1:
                    self.grad += cp.sum(cp.sum(grad.data * other.data, axis=0), axis=0) if self.device == 'cuda' else np.sum(np.sum(grad.data * other.data, axis=0), axis=0)
                if self.ndim == grad.ndim:
                    self.grad += atom(grad.data * other.data, self.device)

            if other.requires_grad:
                other.grad = atom.zeros_like(other, other.device)
                other.grad += atom(grad * self.data, self.device)

        out = atom(result, requires_grad=requires_grad, device=self.device, depends_on=(self, other), operation='*', grad_fn=grad_fn)

        return out

    def __add__(self, other):
        other = other if isinstance(other, atom) else atom(other)
        assert self.device == other.device, f'Must be same device'
        
        requires_grad = self.requires_grad or other.requires_grad
        result = self.data + other.data

        def grad_fn(grad):
            if self.ndim != grad.ndim:
                grad.data = grad.data.reshape(self.shape)
                grad.shape = grad.data.shape
                grad.ndim = grad.data.ndim

            if self.ndim == grad.ndim:
                if self.shape != grad.shape:
                    grad.data = grad.data.reshape(self.shape)
                    grad.shape = grad.data.shape

            if self.requires_grad:
                self.grad = atom.zeros_like(self, self.device)
                dim_diff = grad.ndim - self.grad.ndim
                if dim_diff > 0:
                    self.grad += atom(grad.data.sum(axis=tuple(range(dim_diff))), self.device)
                else:
                    self.grad += grad
            if other.requires_grad:
                other.grad = atom.zeros_like(other, other.device)
                dim_diff = grad.ndim - other.grad.ndim
                if dim_diff > 0:
                    other.grad += atom(grad.data.sum(axis=tuple(range(dim_diff))), self.device)
                else:
                    other.grad += grad

        out = atom(result, requires_grad=requires_grad, device=self.device, depends_on=(self, other), operation='+', grad_fn=grad_fn)

        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, atom) else atom(other, self.device)
        assert self.device == other.device, f'Must be same device'
        requires_grad = self.requires_grad or other.requires_grad

        result = self.data * other.data**-1
        out = atom(result, self.device, requires_grad, depends_on=(self, other), operation='-')

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, atom) else atom(other)
        assert self.device == other.device, f'Must be same device'
        requires_grad = self.requires_grad or other.requires_grad

        result = self.data + (-other.data)
        out = atom(result, self.device, requires_grad, depends_on=(self, other), operation='-')

        return out

    def matmul(x1, x2):
        x1 = x1 if isinstance(x1, atom) else atom(x1)
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

        def grad_fn(grad):
            if x1.ndim != grad.ndim and x2.ndim != grad.ndim:
                grad.data = grad.data.reshape(x2.shape)

            if x1_ndim == 4 and x2_ndim == 4:
                if x1.requires_grad:
                    x1.grad = atom.zeros_like(x1, x1.device)
                    x1.grad += atom.matmul(grad, x2.transpose((0, 1, 3, 2)))
                if x2.requires_grad:
                    x2.grad = atom.zeros_like(x2, x2.device)
                    x2.grad += atom.matmul(grad.transpose((0, 1, 3, 2)), x1).transpose((0, 1, 3, 2))
            elif x1_ndim == 4 and x2_ndim == 2:
                if x1.requires_grad:
                    x1.grad = atom.zeros_like(x1, x1.device)
                    x1.grad += atom.matmul(grad, x2.T())
                if x2.requires_grad:
                    x2.grad = atom.zeros_like(x2, x2.device)
                    x2.grad += atom(np.matmul(grad.data.transpose(0,1,3,2), x1.data).sum(axis=0).sum(axis=0)) if x1.device == 'cpu' else atom(cp.matmul(grad.data.transpose(0,1,3,2), x1.data).sum(axis=0).sum(axis=0))
            elif x1_ndim == 3 and x2_ndim != 3:
                if x1.requires_grad:
                    x1.grad = atom.zeros_like(x1, x1.device)
                    x1.grad += atom.matmul(grad, x2.T())
                if x2.requires_grad:
                    x2.grad = atom.zeros_like(x2, x2.device)
                    x2.grad += atom(np.matmul(grad.data.transpose(0,2,1), x1.data).sum(axis=0)) if x1.device == 'cpu' else atom(cp.matmul(grad.data.transpose(0,2,1), x1.data).sum(axis=0))
            elif x1_ndim == 2 and x2_ndim == 2:
                if x1.requires_grad:
                    x1.grad = atom.zeros_like(x1, x1.device)
                    if x1.device == 'cpu': propagate_grad = np.matmul(grad.data, x2.data.T)
                    else: propagate_grad = cp.matmul(grad.data, x2.data.transpose(0,2,1))
                    x1.grad += atom(propagate_grad)
                if x2.requires_grad:
                    x2.grad = atom.zeros_like(x2, x2.device)
                    if x2.device == 'cpu': propagate_grad = np.matmul(grad.data.T, x1.data).T
                    else: propagate_grad = cp.matmul(grad.data.T, x1.data).T
                    x2.grad += atom(propagate_grad)
            # else:
            #     if x1_shape == x2_shape:
            #         if x1.requires_grad:
            #             x1.grad = atom.zeros_like(x1, x1.device)
            #             if x1.device == 'cpu': propagate_grad = np.matmul(grad.data, x2.data)
            #             else: propagate_grad = cp.matmul(grad.data, x2.data)
            #             x1.grad += atom(propagate_grad)
            #         if x2.requires_grad:
            #             x2.grad = atom.zeros_like(x2, x2.device)
            #             if x2.device == 'cpu': propagate_grad = np.matmul(x1.data.transpose(0,2,1), grad.data).transpose(0,2,1)
            #             else: propagate_grad = cp.matmul(x1.data.transpose(0,2,1), grad.data).transpose(0,2,1)
            #             x2.grad += atom(propagate_grad)
            #     else:
            #         if x1.requires_grad:
            #             x1.grad = atom.zeros_like(x1, x1.device)
            #             if x1.device == 'cpu': propagate_grad = np.matmul(grad.data, x2.data.transpose(0,2,1))
            #             else: propagate_grad = cp.matmul(grad.data, x2.data.transpose(0,2,1))
            #             x1.grad += atom(propagate_grad)
            #         if x2.requires_grad:
            #             x2.grad = atom.zeros_like(x2, x2.device)
            #             if x2.device == 'cpu': propagate_grad = np.matmul(grad.data.transpose(0,2,1), x1.data).transpose(0,2,1)
            #             else: propagate_grad = cp.matmul(grad.data.transpose(0,2,1), x1.data).transpose(0,2,1)
                        # x2.grad += atom(propagate_grad)

        out = atom(result, requires_grad=requires_grad, device=x1.device, depends_on=(x1, x2), operation='@', grad_fn=grad_fn)
    
        return out

    def __rmul__(self, other):
        other = other if isinstance(other, atom) else atom(other)
        assert self.device == other.device, f'Must be same device'
        requires_grad = self.requires_grad or other.requires_grad

        result = other.data * self.data
        out = atom(result, self.device, requires_grad, depends_on=(self, other), operation='-')

        return out

    def __radd__(self, other):
        other = other if isinstance(other, atom) else atom(other)
        assert self.device == other.device, f'Must be same device'
        requires_grad = self.requires_grad or other.requires_grad

        result = other.data + self.data
        out = atom(result, self.device, requires_grad, depends_on=(self, other), operation='-')

        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, atom) else atom(other)
        assert self.device == other.device, f'Must be same device'
        requires_grad = self.requires_grad or other.requires_grad

        result = other.data + (-self.data)
        out = atom(result, self.device, requires_grad, depends_on=(self, other), operation='-')

        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, atom) else atom(other, self.device)
        assert self.device == other.device, f'Must be same device'
        requires_grad = self.requires_grad or other.requires_grad

        result = other.data * self.data**-1
        out = atom(result, self.device, requires_grad, depends_on=(self, other), operation='-')

        return out

    def __repr__(self):
        if self.requires_grad != False:
            if len(self.shape) == 0:
                ret = f"tensor({self.data:.4f}, device='{self.device}', requires_grad={self.requires_grad})"
            else:
                ret = f"tensor({self.data.round(4)}, device='{self.device}', requires_grad={self.requires_grad})"
        else:
            if len(self.shape) == 0:
                ret = f"tensor({self.data:.4f}, device='{self.device}')"
            else:
                ret = f"tensor({self.data.round(4)}, device='{self.device}')"

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
        def grad_fn(grad): ret.grad = grad
        ret._grad_fn = grad_fn if requires_grad else None

        return ret

    @staticmethod
    def rand(shape: Iterable[int], device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.random.rand(*shape).astype(np.float32) if device == 'cpu' else cp.random.rand(*shape, dtype=cp.float32)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret

    @staticmethod
    def uniform(low, high, size=None, device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'
    
        arr = np.random.uniform(low, high, size).astype(np.float32) if device == 'cpu' else cp.random.uniform(low, high, size, dtype=cp.float32)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret
    
    @staticmethod
    def randint(low, high, size=None, device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.random.randint(low, high, size, dtype=np.longlong) if device == 'cpu' else cp.random.randint(low, high, size, dtype=cp.longlong)

        ret = atom(arr, device=device, requires_grad=requires_grad)
        def grad_fn(grad): ret.grad = grad
        ret._grad_fn = grad_fn if requires_grad else None

        return ret

    @staticmethod
    def arange(start, end, step, device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        arr = np.arange(start, end, step) if device == 'cpu' else cp.arange(start, end, step)
        ret = atom(arr, device=device, requires_grad=requires_grad)

        return ret

    @staticmethod
    def ones(shape: Iterable[int], device='cpu', requires_grad=False):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        ret = np.ones(shape) if device == 'cpu' else cp.ones(shape)

        return atom(ret, device=device, requires_grad=requires_grad)
    
    @staticmethod
    def tril(atom_tensor, k=0):
        assert atom_tensor.device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {atom_tensor.device}'

        ret = np.tril(atom_tensor.data, k) if atom_tensor.device == 'cpu' else cp.tril(atom_tensor.data, k)

        return atom(ret, atom_tensor.device, requires_grad=atom_tensor.requires_grad)
    
    def masked_fill(self, mask):
        assert self.device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {self.device}'

        self.data[:, (mask.data == 1)] = -np.inf if self.device == 'cpu' else -cp.inf

        return self
    
    def softmax(self, dim):
        device = self.device
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'

        # Softmax
        max_val_arr = np.max(self.data, axis=dim, keepdims=True) if self.device == 'cpu' else cp.max(self.data, axis=dim, keepdims=True)
        shifted_data = self.data - max_val_arr
        exp = np.exp(shifted_data) if self.device == 'cpu' else cp.exp(shifted_data)
        sum_exp = np.sum(exp, axis=dim, keepdims=True)
        applied_softmax = exp / sum_exp

        out = atom(applied_softmax, self.device, self.requires_grad, [self,], 'softmax')

        def grad_fn(grad):
            if self.requires_grad:
                self.grad = atom.zeros_like(self, self.device)
                if self.ndim == grad.ndim:
                    sum_term = (out.data * grad.data).sum(axis=dim, keepdims=True)
                    deriv = out.data * (grad.data - sum_term)
                    self.grad += atom(deriv, self.device)
                else:
                    grad = np.sum(grad.data, axis=-1)
                    sum_term = (out.data * grad).sum(axis=dim, keepdims=True)
                    deriv = out.data * (grad - sum_term)
                    self.grad += atom(deriv, self.device)

        out._grad_fn = grad_fn
        return out

    def relu(self):
        assert self.device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {self.device}'

        # Relu
        applied_relu = np.maximum(0, self.data) if self.device == 'cpu' else cp.maximum(0, self.data)
        out = atom(applied_relu, self.device, self.requires_grad, [self,], 'relu')

        def grad_fn(grad):
            if self.requires_grad:
                self.grad = atom.zeros_like(self, self.device)
                if self.ndim == grad.ndim:
                    if self.device == 'cpu':
                        deriv = np.where(out.data > 0, 1, 0) * grad.data
                    else:
                        deriv = cp.where(out.data > 0, 1, 0) * grad.data
                    grad_propagated = atom(deriv, self.device)
                    self.grad += grad_propagated
                else:
                    if self.device == 'cpu':
                        deriv = np.where(out.data > 0, 1, 0) * np.sum(grad.data, axis=-1)
                    else:
                        deriv = cp.where(out.data > 0, 1, 0) * cp.sum(grad.data, axis=-1)

                    grad_propagated = atom(deriv, self.device)
                    self.grad += grad_propagated

        out._grad_fn = grad_fn
        return out

    def log_softmax(self, dim):
        assert self.device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {self.device}'

        max_val_arr = np.max(self.data, axis=dim, keepdims=True) if self.device == 'cpu' else cp.max(self.data, axis=dim, keepdims=True)
        shifted = self.data - max_val_arr
        log_sum = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True)) if self.device == 'cpu' else cp.log(cp.sum(cp.exp(shifted), axis=-1, keepdims=True))

        return atom(shifted - log_sum, self.device, self.requires_grad, self._depends_on)

    def one_hot(self, num_classes):
        assert self.device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {self.device}'

        one_hot_arr = np.zeros((len(self.data), num_classes)) if self.device == 'cpu' else cp.zeros((len(self.data), num_classes))
        if self.device == 'cpu':
            one_hot_arr[np.arange(len(self.data)), self.data.astype(np.longlong)] = 1
        else:
            one_hot_arr[cp.arange(len(self.data)), self.data.astype(cp.longlong)] = 1

        return atom(one_hot_arr, self.device, self.requires_grad, self._depends_on)
    
    def T(self):
        assert self.device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {self.device}'
        return atom(self.data.T, self.device, self.requires_grad)
    
    def transpose(self, dims=Iterable[int]):
            result = self.data.transpose(dims)
            def grad_fn(grad):
                if self.requires_grad:
                    self.grad = atom.zeros_like(result, self.device)

                    self.grad += grad

            return atom(result, device=self.device, requires_grad=self.requires_grad, depends_on=[self,], operation='transposed', grad_fn=grad_fn)
    
    def reshape(self, shape=Iterable[int]):
        result = self.data.reshape(shape)
        
        def grad_fn(grad):
            if self.requires_grad:
                self.grad = atom.zeros_like(result, self.device)
                
                if grad.shape != self.grad.shape:
                    grad.data = grad.data.transpose(0, 1, 3, 2)
                    grad.shape = grad.data.shape
                    grad.ndim = grad.data.ndim

                self.grad += grad

        return atom(result, device=self.device, requires_grad=self.requires_grad, depends_on=[self,], operation='reshaped', grad_fn=grad_fn)

    def zeros_like(atom_tensor, device):
        assert device in ['cpu', 'cuda'], f'Tensor must be cpu or cuda, got {device}'
        result = np.zeros_like(atom_tensor.data) if device == 'cpu' else cp.zeros_like(atom_tensor.data)

        return atom(result, device)

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

        for node in reversed(topo):
            if node._grad_fn is not None:
                if node._operation == 'cross_entropy':
                    if grad is None: cross_entropy_grad = node._depends_on[0].softmax(dim=-1) - node._depends_on[1]
                    else: cross_entropy_grad = grad
                    node._grad_fn(cross_entropy_grad)
                else:
                    node._grad_fn(node.grad)

    def zero_grad(self):
        arr = np.zeros_like(self.data) if self.device == 'cpu' else cp.zeros_like(self.data)
        self.grad = None if self.grad is None else arr

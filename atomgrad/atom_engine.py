import numpy as np

class Atom:
    """ atomgrad tensor built at a top of numpy array
        atomgrad tensor make numpy array for automatic gradient calculation
    """

    def __init__(self, data, _parents=()):
        self.data = np.array(data, dtype=np.float32)

        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self.shape = self.data.shape
        self._parents = set(_parents)
        self._backward = lambda: None

    @staticmethod
    def tensor(data, _parents=()):
        return Atom(data, _parents)

    def check_dim(self, x):
        return len(x.data.shape)

    def __add__(self, other):
        other = other if isinstance(other, Atom) else Atom(other)

        # If same number of dimension 
        if self.check_dim(self.data) == self.check_dim(other):
            assert self.data.shape == other.shape, 'Two 2d tensors have to be same shape.'

        out = Atom(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Atom) else Atom(other)

        # If same number of dimension 
        if self.check_dim(self.data) == self.check_dim(other):
            assert self.data.shape == other.shape, 'Two 2d tensors have to be same shape.'

        out = Atom(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Atom) else Atom(other)
        if self.check_dim(self.data) == self.check_dim(other):
            assert self.data.shape == other.shape, 'Two 2d tensors have to be same shape.'

        out = Atom(self.data - other.data, (self, other))

        return out

    def __rsub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other

    def __repr__(self):
        return f'atom.tensor(data={self.data}, grad={self.grad})'

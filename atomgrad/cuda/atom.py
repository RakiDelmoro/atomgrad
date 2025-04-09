import cupy as cp
import atomgrad.cuda.ops as ops

def cuda_tensor(data, requires_grad=False):
    """Create a tensor with data and gradient tracking."""
    return {'data': cp.array(data, dtype=cp.float32), 'shape': cp.array(data).shape, 'grad': cp.zeros_like(data) if requires_grad else None, 'requires_grad': requires_grad, 'grad_fn': None, 'depends_on': []}

'''TENSOR OPS'''

def add(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = cuda_tensor(ops._add(x1['data'], x2['data']), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        if x1['requires_grad']:
            if x1['grad'].ndim == grad.ndim:
                x1['grad'] += grad
            else:
                x1['grad'] += cp.sum(grad, axis=0)
        if x2['requires_grad']:
            if x2['grad'].ndim == grad.ndim:
                x2['grad'] += grad
            else:
                x2['grad'] += cp.sum(grad, axis=0)

    result['grad_fn'] = grad_fn

    return result

def sub(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    return cuda_tensor(ops._sub(x1['data'], x2['data']), requires_grad)

# TODO: Transfer the operation in ops.py file
def broadcasted_mul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    broadcasting_results = []
    for i in range(len(x2)):
        w = x1['data'][:, i].reshape(-1, 1, 1)
        result = w * x2[i]['data']
        broadcasting_results.append(result)

    result = cuda_tensor(broadcasting_results, requires_grad=requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        grad = grad[0]
        # for each in grad:
        if x1['requires_grad']:
            for i in range(len(x2)):
                # Multiply by vk_params[i]['data'], then sum over extra dimensions
                x1['grad'][:, i] += cp.sum(grad * x2[i]['data'][cp.newaxis, :, :], axis=(1, 2))

        if x2[0]['requires_grad']:
            for i, each in enumerate(x2):
                each['grad'] += cp.sum(grad * x1['data'][:, i].reshape(-1, 1, 1), axis=0)
    result['grad_fn'] = grad_fn

    return result

def broadcasted_mul(x1, x2):
    pass

def mul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']
    result = cuda_tensor(ops._mul(x1['data'], x2['data']), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        """Backward function for multiplication."""
        if x1['requires_grad']:
            x1['grad'] += grad * x2['data']
        if x2['requires_grad']:
            x2['grad'] += grad * x1['data']
    result['grad_fn'] = grad_fn

    return result

def matmul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']

    result = cuda_tensor(ops._matmul(x1['data'], x2['data'].T), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        x1_shape = x1['shape']
        x2_shape = x2['shape']

        x1_is_3d = x1['data'].ndim == 3
        x2_is_3d = x2['data'].ndim == 3

        if not x1_is_3d and not x2_is_3d:
            if x1['requires_grad']: x1['grad'] += ((grad @ x2['data'])) 
            if x2['requires_grad']: x2['grad'] += (grad.T @ x1['data']) 
        else:
            if x1_is_3d and not x2_is_3d and len(x2_shape) == 2 and x2_shape[0] == x1_shape[0]:
                if x1['requires_grad']:
                    for i in range(x1_shape[0]): x1['grad'][i] += cp.outer(grad[i], x2['data'][i])
                if x2['requires_grad']:
                    for i in range(x2_shape[0]): x2['grad'][i] += cp.matmul(grad[i], x1['data'][i])
            else:
                if x1['requires_grad']: 
                    if not x1_is_3d: x1['grad'] += grad @ x2['data']
                    else:
                        for i in range(x1_shape[0]): x1['grad'][i] += grad[i][:, cp.newaxis] @ x2['data'][i][cp.newaxis, :]

                if x2['requires_grad']:
                    if not x2_is_3d: x2['grad'] += grad.T @ x1['data']
                    else:
                        for i in range(x2_shape[0]): x2['grad'][i] += x1['data'][i].T @ grad[i]

    result['grad_fn'] = grad_fn

    return result

def sum_tensors(x: list | dict, axis=0):
    if type(x) == list: list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    else: list_atom_data = [atom_tensor for atom_tensor in x['data']]

    result = cuda_tensor(ops._sum_arrays(list_atom_data, axis), requires_grad=True)
    result['depends_on'] = [x]

    def grad_fn(grad):
        for i in range(len(x['data'])):
            x['grad'][i] += grad

    result['grad_fn'] = grad_fn
    return result

def mean_tensor(x: list | dict, axis=0):
    if type(x) == list: list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    else: list_atom_data = [atom_tensor for atom_tensor in x['data']]

    list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    result = cuda_tensor(ops._mean_arrays(list_atom_data, axis), requires_grad=True)
    result['depends_on'] = [i for i in x]

    def grad_fn(grad):
        for each_act in x:
            each_act['grad'] += grad

    result['grad_fn'] = grad_fn
    return result

import cupy as cp
import atomgrad.cuda.ops as ops

#NOTE: uncomment this when running test.py in this directory
# import ops as ops

def cuda_tensor(data, requires_grad=False):
    """Create a tensor with data and gradient tracking."""
    return {'data': cp.array(data, dtype=cp.float32), 'shape': cp.array(data).shape, 'grad': cp.zeros_like(data) if requires_grad else None, 'requires_grad': requires_grad, 'grad_fn': None, 'depends_on': []}

'''TENSOR OPS'''

def add(x1, x2):
    x1_shape = x1['shape']
    x2_shape = x2['shape']

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
                if x2['grad'].ndim == 2:
                    x2['grad'] += grad.sum(axis=0)
                elif x2['grad'].ndim == 1:
                    x2['grad'] += cp.sum(cp.sum(grad, axis=0), axis=0)

    result['grad_fn'] = grad_fn

    return result

def layer_norm_(atom_tensor, eps):
    mean = atom_tensor['data'].mean(axis=-1, keepdims=True)
    std = atom_tensor['data'].std(axis=-1, keepdims=True)

    centered = (atom_tensor['data'] - mean)
    denominator = (std + eps)

    result = cuda_tensor((centered / denominator), requires_grad=atom_tensor['requires_grad'])
    # print(result)
    result['depends_on'] = [atom_tensor]

    def grad_fn(grad):
        mean_grad = grad.mean(axis=-1, keepdims=True)
        sum_grad_centered = (grad * centered).sum(axis=-1, keepdims=True)
        term1 = (grad - mean_grad) / denominator
        term2 = centered * sum_grad_centered / (result['data'].shape[-1] * std * denominator**2)
        atom_tensor['grad'] = term1 - term2

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

def mul(x1, x2):
    requires_grad = x1['requires_grad'] or x2['requires_grad']
    result = cuda_tensor(ops._mul(x1['data'], x2['data']), requires_grad)
    result['depends_on'] = [x1, x2]

    def grad_fn(grad):
        """Backward function for multiplication."""
        if x1['requires_grad']:
            if x1['grad'].ndim == 1:
                x1['grad'] += cp.sum(cp.sum(grad * x2['data'], axis=0), axis=0)
            
            if x1['grad'].ndim == grad.ndim:
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
                        if x2['data'].ndim == 2:
                            for i in range(x1_shape[0]): x1['grad'][i] += cp.matmul(grad[i], x2['data'])
                        else:
                            x1['grad'] += cp.matmul(grad, x2['data'].transpose(0, 2, 1))

                if x2['requires_grad']:
                    if not x2_is_3d: x2['grad'] += cp.sum(cp.matmul(grad.transpose(0, 2, 1), x1['data']), axis=0)
                    else:
                        if x2['grad'].shape == grad.shape:
                            for i in range(x2_shape[0]): x2['grad'][i] += x1['data'][i].T @ grad[i]
                        else:
                            # x2['grad'] += cp.matmul(grad, x1['data'])
                            x2['grad'] += cp.matmul(x1['data'].transpose(0, -1, -2), grad).transpose(0, -1, -2)
                        
    result['grad_fn'] = grad_fn

    return result

def sum_tensors(x: list | dict, axis=0):
    if type(x) == list: list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    else: list_atom_data = [atom_tensor for atom_tensor in x['data']]

    result = cuda_tensor(ops._sum_arrays(list_atom_data, axis), requires_grad=True)
    result['depends_on'] = [x] if type(x) == dict else [each for each in x]

    def grad_fn(grad):
        for i in range(len(x['data'])):
            x['grad'][i] += grad

    result['grad_fn'] = grad_fn
    return result

def concatenate(x: list | dict, axis=-1):
    if type(x) == list: list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    else: list_atom_data = [atom_tensor for atom_tensor in x['data']]

    # list_atom_data = [atom_tensor['data'] for atom_tensor in x]
    result = cuda_tensor(cp.concatenate(list_atom_data, axis=axis), requires_grad=x[0]['requires_grad'])
    result['depends_on'] = [x] if type(x) == dict else [each for each in x]

    def grad_fn(grad):
        if type(x) == list:
            start_idx = 0
            end_idx = x[0]['shape'][-1]
            for each in x:
                each['grad'] += grad[:, :, start_idx:end_idx]

                start_idx += each['shape'][-1]
                end_idx += each['shape'][-1]
        else:
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

def embeddings_(x, parameters):
    x_shape = x['shape']
    num_embeddings = parameters['shape'][0]
    requires_grad = parameters['requires_grad']

    if cp.any(x['data'] < 0) or cp.any(x['data'] >= num_embeddings):
        raise ValueError("Indices out of range [0, num_embeddings-1]")
    
    result = cuda_tensor(parameters['data'][x['data'].astype(int)], requires_grad)
    result['depends_on'] = [parameters]

    def grad_fn(grad):
        if x['data'].ndim == 2:
            indices = x['data'].astype(int).reshape(-1)
        else:
            indices = x['data'].astype(int)

        one_hot = cp.eye(num_embeddings, dtype=grad.dtype)[indices]

        if grad.ndim > 2:
            grad_reshaped = grad.reshape(-1, grad.shape[-1])
            summed_grad = cp.matmul(one_hot.T, grad_reshaped)
            parameters['grad'] += summed_grad / num_embeddings

        else:
            parameters['grad'] += grad / num_embeddings

    result['grad_fn'] = grad_fn
    return result

def dropout_(atom_tensor, p, train=True):
    requires_grad = atom_tensor['requires_grad']

    if train and p != 0:
        if p == 1:
            result = cuda_tensor(cp.zeros_like(atom_tensor['data']), requires_grad=requires_grad)
        else:
            bool_mask = cp.random.rand(*atom_tensor['shape']) > p
            res = bool_mask * atom_tensor['data'] * (float(1.0 / (1.0 - p)))
            result = cuda_tensor(res, requires_grad=requires_grad)
    else:
        result = atom_tensor

    result['depends_on'] = [atom_tensor]

    def grad_fn(grad):
        if requires_grad:
            atom_tensor['grad'] += grad
    
    result['grad_fn'] = grad_fn
    return result

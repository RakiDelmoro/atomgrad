import cupy as cp
import numpy as np
import atomgrad.tensor as tensor

def _add(x1, x2):
    x1_device = x1['device']
    x2_device = x2['device']
    assert x1['device'] == x2['device'], f'x1 and x2 has to be same device: {x1_device}, {x2_device}'

    result = x1['data'] + x2['data']
    return result

def _sub(x1, x2):
    x1_device = x1['device']
    x2_device = x2['device']

    assert x1['device'] == x2['device'], f'x1 and x2 has to be same device: {x1_device}, {x2_device}'
    
    result = x1['data'] - x2['data']
    return result

def _mul(x1, x2):
    x1_device = x1['device']
    x2_device = x2['device']

    assert x1['device'] == x2['device'], f'x1 and x2 has to be same device: {x1_device}, {x2_device}'

    result = x1['data'] * x2['data']
    return result

def _div(x1, x2):
    x1_device = x1['device']
    x2_device = x2['device']

    assert x1['device'] == x2['device'], f'x1 and x2 has to be same device: {x1_device}, {x2_device}'

    result = x1['data'] * x2['data']**-1
    return result

def _mean_arrays(x: list, axis=0):
    x = np.stack(x, axis=0)
    return np.mean(x, axis)

def _sum_arrays(x: list, axis=0):
    x = np.stack(x, axis=0)
    return np.sum(x, axis)

def _matmul(x1, x2):
    x1_device = x1['device']
    x2_device = x2['device']

    assert x1['device'] == x2['device'], f'x1 and x2 has to be same device: {x1_device}, {x2_device}'

    # Check num dim
    x1_shape = x1['data'].shape
    x2_shape = x2['data'].shape

    # Check dimensions
    is_x1_3d = len(x1_shape) == 3
    is_x2_3d = len(x2_shape) == 3

    if is_x1_3d or is_x2_3d:
        result = _matmul_3d(x1, x2['data'])
    else:
        result = np.matmul(x1['data'], x2['data'].T) if x1['device'] == 'cpu' else cp.matmul(x1['data'], x2['data'].T)

    return result

def _matmul_3d(x1, x2):
    x1_device = x1['device']
    x2_device = x2['device']

    assert x1['device'] == x2['device'], f'x1 and x2 has to be same device: {x1_device}, {x2_device}'

    device = x1['device']
    # Check num dim
    x1_shape = x1['data'].shape
    x2_shape = x2['data'].shape

    # Check dimensions
    is_x1_3d = len(x1_shape) == 3
    is_x2_3d = len(x2_shape) == 3

    if not is_x1_3d and not is_x2_3d:
        raise ValueError(f"Should have 3d shape {x1.shape} {x2.shape}")

    if is_x1_3d and not is_x2_3d:
        if len(x2_shape) == 1: result_data = np.matmul(x1, x2)
        elif x2_shape[0] == x1_shape[0]:
            if device == 'cpu': result_data = np.zeros((x1_shape[0], x1_shape[1]))
            else: result_data = cp.zeros((x1_shape[0], x1_shape[1]))
            for i in range(x1_shape[0]): result_data[i] = np.matmul(x1['data'][i], x2['data'][i]) if device == 'cpu' else cp.matmul(x1['data'][i], x2['data'][i])
        else:
            if device == 'cpu':
                result_data = np.zeros((x1_shape[0], x1_shape[1], x2_shape[0]))
            else: result_data = cp.zeros((x1_shape[0], x1_shape[1], x2_shape[0]))
            for i in range(x1_shape[0]): result_data[i] = np.matmul(x1['data'][i], x2['data'].T) if device == 'cpu' else cp.matmul(x1['data'][i], x2['data'].T)

    elif not is_x1_3d and is_x2_3d:
        if len(x1_shape) == 1:
            if device == 'cpu': result_data = np.zeros((x2_shape[0], x2_shape[2]))
            else: result_data = cp.zeros((x2_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1['data'], x2['data'][i]) if device == 'cpu' else cp.matmul(x1['data'], x2['data'][i])
        elif x1_shape[0] == x2_shape[0]:
            if device == 'cpu': result_data = np.zeros((x2_shape[0], x2_shape[2]))
            else: result_data = cp.zeros((x2_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1['data'][i], x2['data'][i]) if device == 'cpu' else cp.matmul(x1['data'][i], x2['data'][i])
        else:
            if device == 'cpu': result_data = np.zeros((x2_shape[0], x1_shape[0], x2_shape[2]))
            else: result_data = cp.zeros((x2_shape[0], x1_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1['data'], x2['data'][i]) if device == 'cpu' else cp.matmul(x1['data'], x2['data'[i]])
    else:
        # Both 3D case - batch dimensions should match
        if x1_shape[0] != x2_shape[0]: raise ValueError("Batch size must match for 3D-3D matmul")

        # Case: (batch, m, n) @ (batch, n, p) -> (batch, m, p)
        if device == 'cpu': result_data = np.zeros((x1_shape[0], x1_shape[1], x2_shape[2]))
        else: result_data = cp.zeros((x1_shape[0], x1_shape[1], x2_shape[2]))
        for i in range(x1_shape[0]):
            result_data[i] = np.matmul(x1[i]['data'], x2[i]['data']) if device == 'cpu' else cp.matmul(x1[i]['data'], x2[i]['data'])

    return result_data

def _broadcast_mul():
    pass

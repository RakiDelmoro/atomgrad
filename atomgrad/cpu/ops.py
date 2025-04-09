import numpy as np

def _add(x1: np.ndarray | int, x2: np.ndarray | int): return x1 + x2

def _sub(x1: np.ndarray | int, x2: np.ndarray | int): return x1 - x2

def _mul(x1: np.ndarray | int, x2: np.ndarray | int): return x1 * x2

def _div(x1: np.ndarray | int, x2: np.ndarray | int): return x1 * x2**-1

def _mean_arrays(x: list, axis=0):
    x = np.stack(x, axis=0)
    return np.mean(x, axis)

def _sum_arrays(x: list, axis=0):
    x = np.stack(x, axis=0)
    return np.sum(x, axis)

def _matmul(x1: np.ndarray, x2: np.ndarray):
    # Check num dim
    x1_shape = x1.shape
    x2_shape = x2.shape

    # Check dimensions
    is_x1_3d = len(x1_shape) == 3
    is_x2_3d = len(x2_shape) == 3

    if is_x1_3d or is_x2_3d:
        return _matmul_3d(x1, x2.T)

    return np.matmul(x1, x2)

def _matmul_3d(x1: np.ndarray, x2: np.ndarray):
    # Check num dim
    x1_shape = x1.shape
    x2_shape = x2.shape

    # Check dimensions
    is_x1_3d = len(x1_shape) == 3
    is_x2_3d = len(x2_shape) == 3

    if not is_x1_3d and not is_x2_3d:
        raise ValueError(f"Should have 3d shape {x1.shape} {x2.shape}")

    if not is_x1_3d and not is_x2_3d: result_data = np.matmul(x1, x2.T)

    elif is_x1_3d and not is_x2_3d:
        if len(x2_shape) == 1: result_data = np.matmul(x1, x2)
        elif x2_shape[0] == x1_shape[0]:
            result_data = np.zeros((x1_shape[0], x1_shape[1]))
            for i in range(x1_shape[0]): result_data[i] = np.matmul(x1[i], x2[i])
        else:
            result_data = np.zeros((x1_shape[0], x1_shape[1], x2_shape[0]))
            for i in range(x1_shape[0]): result_data[i] = np.matmul(x1[i], x2.T)

    elif not is_x1_3d and is_x2_3d:
        if len(x1_shape) == 1:
            result_data = np.zeros((x2_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1, x2[i])
        elif x1_shape[0] == x2_shape[0]:
            result_data = np.zeros((x2_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1[i], x2[i])
        else:
            result_data = np.zeros((x2_shape[0], x1_shape[0], x2_shape[2]))
            for i in range(x2_shape[0]): result_data[i] = np.matmul(x1, x2[i])

    else:
        # Both 3D case - batch dimensions should match
        if x1_shape[0] != x2_shape[0]:
            raise ValueError("Batch size must match for 3D-3D matmul")

        # Case: (batch, m, n) @ (batch, n, p) -> (batch, m, p)
        result_data = np.zeros((x1_shape[0], x1_shape[1], x2_shape[2]))
        for i in range(x1_shape[0]):
            result_data[i] = np.matmul(x1[i], x2[i])

    return result_data

def _broadcast_mul():
    pass

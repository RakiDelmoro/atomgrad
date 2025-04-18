import cupy as cp

def _add(x1: cp.ndarray | int, x2: cp.ndarray | int): return x1 + x2

def _sub(x1: cp.ndarray | int, x2: cp.ndarray | int): return x1 - x2

def _mul(x1: cp.ndarray | int, x2: cp.ndarray | int): return x1 * x2

def _div(x1: cp.ndarray | int, x2: cp.ndarray | int): return x1 * x2**-1

def _mean_arrays(x: list, axis=0):
    x = cp.stack(x, axis=0)
    return cp.mean(x, axis)

def _sum_arrays(x: list, axis=0):
    x = cp.stack(x, axis=0)
    return cp.sum(x, axis)

def _matmul(x1: cp.ndarray, x2: cp.ndarray):
    # Check num dim
    x1_shape = x1.shape
    x2_shape = x2.shape

    # Check dimensions
    is_x1_3d = len(x1_shape) == 3
    is_x2_3d = len(x2_shape) == 3

    if is_x1_3d or is_x2_3d:
        return _matmul_3d(x1, x2.T)

    return cp.matmul(x1, x2)

def _matmul_3d(x1: cp.ndarray, x2: cp.ndarray):
    # Ensure at least one array is 3D
    x1_ndim, x2_ndim = x1.ndim, x2.ndim
    if x1_ndim < 3 and x2_ndim < 3:
        raise ValueError(f"At least one input must be 3D. Shapes: {x1.shape}, {x2.shape}")

    # Promote arrays to 3D by adding leading singleton dimensions
    x1_3d = x1.reshape((1,)*(3 - x1_ndim) + x1.shape) if x1_ndim < 3 else x1
    x2_3d = x2.reshape((1,)*(3 - x2_ndim) + x2.shape) if x2_ndim < 3 else x2

    # Handle dimension alignment
    try:
        # Attempt automatic broadcasting with transpose fallback
        result = cp.matmul(x1_3d, x2_3d.transpose(0, 2, 1) if x1_3d.shape[-1] != x2_3d.shape[-2] else cp.matmul(x1_3d, x2_3d))
    except cp.cuda.cublas.CUBLASError:
        # Fallback to manual alignment if automatic broadcasting fails
        x2_3d = x2_3d.transpose(0, 2, 1)
        if x1_3d.shape[-1] != x2_3d.shape[-2]:
            raise ValueError(f"Incompatible inner dimensions: {x1_3d.shape[-1]} vs {x2_3d.shape[-2]}")
        result = cp.matmul(x1_3d, x2_3d)

    # Squeeze unnecessary dimensions based on original input shapes
    squeeze_dims = []
    if x1_ndim < 3 and x1_3d.shape[0] == 1:
        squeeze_dims.append(0)
    if x2_ndim < 3 and x2_3d.shape[0] == 1:
        squeeze_dims.append(0)
    if x1_ndim == 1 or x2_ndim == 1:
        squeeze_dims.append(-1)
    
    return result.squeeze(axis=tuple(squeeze_dims)) if squeeze_dims else result

def _broadcast_mul():
    pass

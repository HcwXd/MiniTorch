from numba import cuda
import numba
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.3.
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx < out_size:
            out_idx = cuda.local.array(MAX_DIMS, numba.int32)
            in_idx = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(idx, out_shape, out_idx)
            broadcast_index(out_idx, out_shape, in_shape, in_idx)
            out[index_to_position(out_idx, out_strides)] = fn(
                in_storage[index_to_position(in_idx, in_strides)]
            )

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)

        # Instantiate and run the cuda kernel.
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 3.3.
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if idx < out_size:
            out_idx = cuda.local.array(MAX_DIMS, numba.int32)
            a_idx = cuda.local.array(MAX_DIMS, numba.int32)
            b_idx = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(idx, out_shape, out_idx)

            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)

            # cur_o_idx = index_to_position(out_idx, out_strides)
            cur_a_idx = index_to_position(a_idx, a_strides)
            cur_b_idx = index_to_position(b_idx, b_strides)

            out[idx] = fn(a_storage[cur_a_idx], b_storage[cur_b_idx])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def _sum_practice(out, a, size):
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length :math:`n` and out of size :math:`n // blockDIM`
    it should sum up each blockDim values into an out cell.

    [a_1, a_2, ..., a_100]

    |

    [a_1 +...+ a_32, a_32 + ... + a_64, ... ,]

    Note: Each block must do the sum using shared memory!

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    global_idx = cuda.blockIdx.x * BLOCK_DIM + cuda.threadIdx.x
    local_idx = cuda.threadIdx.x

    if global_idx < size:
        shared = cuda.shared.array(BLOCK_DIM, numba.float64)
        shared[local_idx] = a[global_idx]
        cuda.syncthreads()

        s = 1
        while s < BLOCK_DIM:
            if local_idx % (s * 2) != 0 or (global_idx + s) >= size:
                break
            shared[local_idx] += shared[local_idx + s]
            cuda.syncthreads()
            s *= 2

        if local_idx == 0:
            out[cuda.blockIdx.x] = shared[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a):
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):
        BLOCK_DIM = 1024
        # TODO: Implement for Task 3.3.
        out_idx = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(cuda.blockIdx.x, out_shape, out_idx)
        global_idx = index_to_position(out_idx, a_strides) + (
            cuda.threadIdx.x * a_strides[reduce_dim]
        )

        shared = cuda.shared.array(BLOCK_DIM, numba.float64)
        local_idx = cuda.threadIdx.x
        shared[local_idx] = (
            a_storage[global_idx] if local_idx < a_shape[reduce_dim] else reduce_value
        )

        cuda.syncthreads()

        s = 1
        while s < BLOCK_DIM:
            if local_idx % (s * 2) != 0:
                break
            shared[local_idx] = fn(shared[local_idx], shared[local_idx + s])
            cuda.syncthreads()
            s *= 2

        if local_idx == 0:
            out[cuda.blockIdx.x] = shared[0]

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
        out_a = a.zeros(tuple(out_shape))

        threadsperblock = 1024
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )

        return out_a

    return ret


def _mm_practice(out, a, b, size):
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Compute ::

    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `a` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.4.
    global_x = cuda.blockIdx.x * BLOCK_DIM + cuda.threadIdx.x
    global_y = cuda.blockIdx.y * BLOCK_DIM + cuda.threadIdx.y

    local_x = cuda.threadIdx.x
    local_y = cuda.threadIdx.y

    if global_x < size and global_y < size:
        pos = size * global_x + global_y

        shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
        shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

        shared_a[local_x, local_y] = a[pos]
        shared_b[local_x, local_y] = b[pos]
        cuda.syncthreads()

        acc = 0
        for idx in range(size):
            acc += shared_a[local_x, idx] * shared_b[idx, local_y]
        out[pos] = acc


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a, b):

    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.4.
    local_x = cuda.threadIdx.x
    local_y = cuda.threadIdx.y

    global_x, global_y, global_z = cuda.grid(3)
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    out_batch_stride = out_strides[0]
    out_x_stride = out_strides[-2]
    out_y_stride = out_strides[-1]

    a_x_stride = a_strides[-2]
    a_y_stride = a_strides[-1]
    b_x_stride = b_strides[-2]
    b_y_stride = b_strides[-1]

    inner_size = a_shape[-1]

    acc = 0
    for idx in range(0, inner_size, BLOCK_DIM):
        if global_x < a_shape[-2] and (local_y + idx) < a_shape[-1]:
            global_a = (
                global_x * a_x_stride
                + (local_y + idx) * a_y_stride
                + global_z * a_batch_stride
            )
            shared_a[local_x, local_y] = a_storage[global_a]
        else:
            shared_a[local_x, local_y] = 0

        if (local_x + idx) < b_shape[-2] and global_y < b_shape[-1]:
            global_b = (
                (local_x + idx) * b_x_stride
                + global_y * b_y_stride
                + global_z * b_batch_stride
            )
            shared_b[local_x, local_y] = b_storage[global_b]
        else:
            shared_b[local_x, local_y] = 0

        cuda.syncthreads()

        for inner_idx in range(BLOCK_DIM):
            acc += shared_a[local_x, inner_idx] * shared_b[inner_idx, local_y]
        cuda.syncthreads()

    if global_x < out_shape[-2] and global_y < out_shape[-1]:
        out_position = (
            global_x * out_x_stride
            + global_y * out_y_stride
            + global_z * out_batch_stride
        )
        out[out_position] = acc


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    # One block per batch, extra rows, extra col
    blockspergrid = (
        (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        out.shape[0],
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply

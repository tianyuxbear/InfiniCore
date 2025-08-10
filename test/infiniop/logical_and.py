import ctypes
from ctypes import c_uint64
from enum import Enum, auto

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.BOOL,
    InfiniDtype.I8,
    InfiniDtype.I16,
    InfiniDtype.I32,
    InfiniDtype.I64,
    InfiniDtype.BF16,
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.F64,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.BOOL: {"atol": 0, "rtol": 0},
    InfiniDtype.I8: {"atol": 0, "rtol": 0},
    InfiniDtype.I16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.F64: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def logical_and(c, a, b):
    torch.logical_and(a, b, out=c)


def test(
    handle,
    device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    a = TestTensor(shape, a_stride, dtype, device)
    b = TestTensor(shape, b_stride, dtype, device)
    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride:
            return
        c = a
    elif inplace == Inplace.INPLACE_B:
        if c_stride != b_stride:
            return
        c = b
    else:
        c = TestTensor(shape, c_stride, InfiniDtype.BOOL, device, mode="ones")

    if c.is_broadcast():
        return

    print(
        f"Testing LogicalAnd on {InfiniDeviceNames[device]} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    logical_and(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLogicalAndDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLogicalAndWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_logical_and():
        check_error(
            LIBINFINIOP.infiniopLogicalAnd(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),
                None,
            )
        )

    lib_logical_and()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.equal(c.actual_tensor(), c.torch_tensor())

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: logical_and(c.torch_tensor(), a.torch_tensor(), b.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_logical_and(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLogicalAndDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")

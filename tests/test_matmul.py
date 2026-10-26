
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

SHAPES = (
    (1, 1, 1), (1, 1, 3), (1, 3, 1), (3, 1, 2),
    (2, 3, 5), (5, 7, 9), (9, 9, 9), (17, 5, 13),
    (3, 129, 131), (129, 3, 131), (17, 17, 129),
    (130, 135, 133), (2, 0, 3), (0, 4, 7), (2, 3, 0),
)

LIBRARIES = os.environ.get("MINGEMM_TEST_LIBS", "").split()

if not LIBRARIES:
    raise RuntimeError(
        "MINGEMM_TEST_LIBS is not set. "
        "Run tests using 'make test'."
    )

PTR = ctypes.POINTER(ctypes.c_float)


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_matmul(library, m, n, k):
    rng = np.random.default_rng(2026)

    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    c = np.full((m, n), np.nan, dtype=np.float32)

    lib = ctypes.CDLL(str(Path(library).resolve()))
    fn = lib.matmul
    fn.argtypes = [
        PTR, PTR, PTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    fn.restype = None

    fn(
        a.ctypes.data_as(PTR),
        b.ctypes.data_as(PTR),
        c.ctypes.data_as(PTR),
        m, n, k,
    )

    np.testing.assert_allclose(
        c, a @ b,
        rtol=2e-4,
        atol=2e-4,
        err_msg=f"{library}: M={m}, N={n}, K={k}",
    )

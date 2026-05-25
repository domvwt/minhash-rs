"""Rust port of ``datasketch.MinHash``'s core algorithm: batch MinHash
signatures with optional rayon parallelism.

The compiled Rust extension lives in :mod:`minhash_rs._native` and is
re-exported from this module, alongside the pure-Python
:func:`default_permutations` helper that derives the permutation matrix
matching ``datasketch.MinHash(num_perm, seed).permutations``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

import numpy as np

from minhash_rs._native import minhash_batch, minhash_batch_from_text

try:
    __version__ = _pkg_version("minhash-rs")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "Dominic Thorn"

__all__ = [
    "__author__",
    "__version__",
    "default_permutations",
    "minhash_batch",
    "minhash_batch_from_text",
]

_MERSENNE_PRIME = (1 << 61) - 1


def default_permutations(num_perm: int, seed: int = 1) -> np.ndarray:
    """Permutation matrix matching ``datasketch.MinHash(num_perm, seed).permutations``.

    Uses numpy's legacy ``RandomState`` (explicitly preserved for
    bit-stability across numpy versions) with the same ``randint`` bounds
    as ``datasketch``'s internal derivation: ``a`` drawn from
    ``[1, MERSENNE_PRIME)`` and ``b`` from ``[0, MERSENNE_PRIME)``.

    Args:
        num_perm: Number of permutations (MinHash signature length).
        seed: RNG seed. Default ``1`` matches ``datasketch.MinHash``'s default.

    Returns:
        Contiguous ``(2, num_perm)`` uint64 array. Row 0 is ``a``, row 1
        is ``b``. Pass directly to :func:`minhash_batch` or
        :func:`minhash_batch_from_text` via the ``permutations`` argument.
    """
    if num_perm <= 0:
        raise ValueError(f"num_perm must be positive, got {num_perm}")
    # datasketch interleaves the (a, b) draws per permutation rather than
    # drawing all `a`s then all `b`s. Replicate that consumption order so
    # the resulting matrix is byte-identical to
    # `datasketch.MinHash(num_perm, seed).permutations`.
    gen = np.random.RandomState(seed)
    pairs = np.array(
        [
            (
                gen.randint(1, _MERSENNE_PRIME, dtype=np.uint64),
                gen.randint(0, _MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_perm)
        ],
        dtype=np.uint64,
    ).T
    return np.ascontiguousarray(pairs)

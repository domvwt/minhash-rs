"""Benchmark: ``minhash_rs`` vs joblib-parallel ``datasketch.MinHash``.

Compares per-record cost (µs/record) across:

- joblib-parallel ``datasketch`` reference (4 workers)
- ``minhash_batch_from_text`` serial
- ``minhash_batch_from_text`` rayon-parallel (4 threads)

The Python baseline runs on a smaller corpus than the Rust runs because
``joblib`` forks the parent process on Linux; allocating the full Rust
corpus before forking causes 4 workers × inherited CoW pages to exhaust
RAM. Numbers are normalised to µs/record so the speedup ratios are
comparable.

Reproduce:
    pip install datasketch joblib
    maturin develop --release
    python tests/benchmark.py
"""

from __future__ import annotations

import random
import statistics
import string
import time

import numpy as np
from datasketch import MinHash
from joblib import Parallel, delayed

import minhash_rs

NUM_PERM = 128
SEED = 1


def _reference_batch(texts: list[str], num_perm: int, seed: int) -> np.ndarray:
    """Reference: per-record ``datasketch.MinHash`` with the same byte
    2-gram + 3-gram strategy as ``minhash_batch_from_text``'s default.
    """
    out = np.empty((len(texts), num_perm), dtype=np.uint64)
    for i, text in enumerate(texts):
        m = MinHash(num_perm=num_perm, seed=seed)
        if text:
            tb = text.lower().encode("utf-8")
            tl = len(tb)
            if tl < 2:
                m.update(tb)
            else:
                ng = [tb[j : j + 2] for j in range(tl - 1)]
                if tl >= 3:
                    ng.extend([tb[j : j + 3] for j in range(tl - 2)])
                m.update_batch(ng)
        out[i] = m.hashvalues
    return out


def _joblib_baseline(texts: list[str], num_perm: int, seed: int, n_jobs: int = 4) -> np.ndarray:
    """Reference baseline matching horkos's current joblib-parallel path."""
    n = len(texts)
    batch_size = max(100, n // (n_jobs * 2))
    ranges = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_reference_batch)(texts[start:end], num_perm, seed) for start, end in ranges
    )
    return np.vstack(results)


def _corpus(n: int, rng: random.Random) -> list[str]:
    alpha = string.ascii_lowercase
    return ["".join(rng.choices(alpha, k=rng.randint(3, 20))) for _ in range(n)]


def _bench(label: str, fn, runs: int) -> tuple[float, float]:
    timings = []
    fn()  # warm-up
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    median = statistics.median(timings)
    p95 = sorted(timings)[max(0, int(len(timings) * 0.95) - 1)]
    print(f"  {label:35s} median={median*1000:8.2f} ms  p95={p95*1000:8.2f} ms")
    return median, p95


def main() -> None:
    rng = random.Random(20260524)
    perms = minhash_rs.default_permutations(NUM_PERM, SEED)

    print("=== Joblib-parallel datasketch baseline (5,000 records, 4 workers) ===")
    py_corpus = _corpus(5_000, rng)
    py_med, _ = _bench(
        "joblib datasketch (n_jobs=4)",
        lambda: _joblib_baseline(py_corpus, NUM_PERM, SEED, n_jobs=4),
        runs=2,
    )
    py_per_rec_us = py_med * 1e6 / 5_000

    print("\n=== minhash_rs (50,000 records) ===")
    rs_corpus = _corpus(50_000, rng)
    rs_serial_med, _ = _bench(
        "minhash_batch_from_text serial",
        lambda: minhash_rs.minhash_batch_from_text(rs_corpus, perms, NUM_PERM, num_threads=None),
        runs=5,
    )
    rs_parallel_med, _ = _bench(
        "minhash_batch_from_text rayon(4)",
        lambda: minhash_rs.minhash_batch_from_text(rs_corpus, perms, NUM_PERM, num_threads=4),
        runs=5,
    )
    rs_serial_per_rec_us = rs_serial_med * 1e6 / 50_000
    rs_parallel_per_rec_us = rs_parallel_med * 1e6 / 50_000

    print("\n=== Normalised (µs/record) ===")
    print(f"  joblib datasketch (n_jobs=4):    {py_per_rec_us:8.2f} µs/rec  (1× baseline)")
    print(
        f"  minhash_batch_from_text serial:  {rs_serial_per_rec_us:8.2f} µs/rec  "
        f"({py_per_rec_us / rs_serial_per_rec_us:6.1f}×)"
    )
    print(
        f"  minhash_batch_from_text rayon(4):{rs_parallel_per_rec_us:8.2f} µs/rec  "
        f"({py_per_rec_us / rs_parallel_per_rec_us:6.1f}×)"
    )


if __name__ == "__main__":
    main()

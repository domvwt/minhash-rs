"""Bit-equivalence harness: verify ``minhash_rs`` produces uint64
signatures identical to ``datasketch.MinHash`` on the same inputs.

Three checks:

1. ASCII corpus (1000 records, 2 fields) — ``minhash_batch_from_text``
   matches ``datasketch.MinHash.update_batch(...).hashvalues`` per record.
2. Edge corpus (empty strings, single-char, Unicode, punctuation) — same
   equivalence with adversarial inputs.
3. ``default_permutations(num_perm, seed)`` matches
   ``np.stack(datasketch.MinHash(num_perm, seed).permutations).astype(np.uint64)``.

Reproduce:
    maturin develop && python tests/correctness.py

This harness depends on ``datasketch`` (test-only dep — see
``pyproject.toml`` ``[project.optional-dependencies] test``).
"""

from __future__ import annotations

import random
import string
import sys

import numpy as np
from datasketch import MinHash

import minhash_rs

NUM_PERM = 128
SEED = 1


def _reference_signature(text: str, num_perm: int, seed: int) -> np.ndarray:
    """Reference: byte 2-gram + 3-gram MinHash signature for one text via
    ``datasketch``. Mirrors what ``minhash_batch_from_text`` computes for
    the default ``ngram_sizes=[2, 3]`` setting.
    """
    m = MinHash(num_perm=num_perm, seed=seed)
    if text:
        text_bytes = text.lower().encode("utf-8")
        text_len = len(text_bytes)
        if text_len < 2:
            m.update(text_bytes)
        else:
            ngrams = [text_bytes[j : j + 2] for j in range(text_len - 1)]
            if text_len >= 3:
                ngrams.extend([text_bytes[j : j + 3] for j in range(text_len - 2)])
            m.update_batch(ngrams)
    return np.asarray(m.hashvalues, dtype=np.uint64)


def _make_ascii_corpus(n: int, rng: random.Random) -> list[str]:
    alpha = string.ascii_lowercase
    return ["".join(rng.choices(alpha, k=rng.randint(3, 20))) for _ in range(n)]


def _make_edge_corpus() -> list[str]:
    return [
        "",
        "a",
        "ab",
        "abc",
        "café",
        "naïve",
        "日本語",
        "  spaces  ",
        "hyphen-name",
        "O'Brien",
        "very_long_string_with_repeating_repeating_repeating_chars",
        "x" * 200,
    ]


def _compare(label: str, ref: np.ndarray, got: np.ndarray) -> bool:
    same_shape = ref.shape == got.shape
    same_dtype = ref.dtype == got.dtype
    same_bytes = same_shape and same_dtype and np.array_equal(ref, got)
    print(f"  {label}: shape={ref.shape}, dtype={ref.dtype}, match={same_bytes}")
    if not same_bytes and same_shape:
        diffs = np.where(ref != got)
        n_diff = len(diffs[0])
        print(f"    {n_diff} cell(s) differ (first 5):")
        for k in range(min(5, n_diff)):
            i, j = diffs[0][k], diffs[1][k]
            print(f"      [{i},{j}]: ref={ref[i, j]!r} got={got[i, j]!r}")
    return same_bytes


def check_permutations() -> bool:
    print("=== default_permutations vs datasketch ===")
    ours = minhash_rs.default_permutations(NUM_PERM, SEED)
    theirs = np.stack(MinHash(num_perm=NUM_PERM, seed=SEED).permutations).astype(np.uint64)
    return _compare("permutations matrix", theirs, ours)


def check_ascii_corpus() -> bool:
    print("\n=== ASCII corpus (1000 records) ===")
    rng = random.Random(20260524)
    texts = _make_ascii_corpus(1000, rng)
    perms = minhash_rs.default_permutations(NUM_PERM, SEED)
    ref = np.stack([_reference_signature(t, NUM_PERM, SEED) for t in texts])
    serial = minhash_rs.minhash_batch_from_text(texts, perms, NUM_PERM, num_threads=None)
    ok_serial = _compare("rust serial vs datasketch", ref, serial)
    parallel = minhash_rs.minhash_batch_from_text(texts, perms, NUM_PERM, num_threads=4)
    ok_parallel = _compare("rust parallel(4) vs datasketch", ref, parallel)
    return ok_serial and ok_parallel


def check_edge_corpus() -> bool:
    print("\n=== Edge corpus (Unicode, empties, punctuation) ===")
    texts = _make_edge_corpus()
    perms = minhash_rs.default_permutations(NUM_PERM, SEED)
    ref = np.stack([_reference_signature(t, NUM_PERM, SEED) for t in texts])
    got = minhash_rs.minhash_batch_from_text(texts, perms, NUM_PERM, num_threads=None)
    return _compare("rust serial vs datasketch", ref, got)


def check_low_level_primitive() -> bool:
    print("\n=== minhash_batch (low-level primitive) ===")
    # Hand-build the n-gram lists the way minhash_batch_from_text does
    # internally, and verify minhash_batch produces the same output as
    # the convenience wrapper.
    texts = ["hello", "world", "minhash"]
    perms = minhash_rs.default_permutations(NUM_PERM, SEED)
    high = minhash_rs.minhash_batch_from_text(texts, perms, NUM_PERM)
    ngrams_per_record = []
    for t in texts:
        tb = t.lower().encode("utf-8")
        tl = len(tb)
        if tl == 0:
            ngrams_per_record.append([])
        elif tl < 2:
            ngrams_per_record.append([tb])
        else:
            ng = [tb[j : j + 2] for j in range(tl - 1)]
            if tl >= 3:
                ng.extend([tb[j : j + 3] for j in range(tl - 2)])
            ngrams_per_record.append(ng)
    low = minhash_rs.minhash_batch(ngrams_per_record, perms, NUM_PERM)
    return _compare("minhash_batch vs minhash_batch_from_text", high, low)


def main() -> None:
    results = {
        "permutations": check_permutations(),
        "ascii_corpus": check_ascii_corpus(),
        "edge_corpus": check_edge_corpus(),
        "primitive": check_low_level_primitive(),
    }
    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit("FAIL: bit-equivalence broken in at least one check")
    print("\nAll correctness checks passed.")


if __name__ == "__main__":
    main()

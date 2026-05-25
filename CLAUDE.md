# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rust port of `datasketch.MinHash`'s core algorithm, exposed to Python via PyO3. Computes batch MinHash signatures with optional rayon parallelism, **bit-equivalent** to `datasketch` (same SHA1 truncation, Mersenne-prime permutation arithmetic, and numpy `RandomState` permutation matrix).

## Architecture

This is a single-crate PyO3 extension wrapped in a thin Python package.

- **`src/lib.rs`** — entire Rust hot loop. Two `#[pyfunction]`s (`minhash_batch`, `minhash_batch_from_text`) plus the `_native` module entry point. Hashing is `sha1_hash32` (first 4 bytes of SHA1, little-endian u32) → `permute(h, a, b) = (a*h + b) % (2^61 - 1) & (2^32 - 1)` → min reduction across n-grams per permutation. The GIL is released via `py.allow_threads` before the parallel section so a scoped rayon pool can run uncontended.
- **`python/minhash_rs/__init__.py`** — re-exports the Rust functions and provides `default_permutations(num_perm, seed)`, which builds the `(2, num_perm)` uint64 permutation matrix using legacy numpy `RandomState`. The interleaved `(a, b)` draw order is **load-bearing for bit-equivalence with datasketch** — do not refactor to draw all `a`s then all `b`s.
- **Threading model**: `num_threads=None`/`Some(1)` runs serially (no pool spin-up). `Some(n>1)` builds a **scoped** rayon pool just for that call — never touches the global pool. This is so callers already inside `joblib.Parallel` can pass `num_threads=1` without oversubscription.
- **Module layout quirk**: `pyproject.toml` sets `module-name = "minhash_rs._native"` and `python-source = "python"`, so maturin compiles the Rust crate (named `_native` in `Cargo.toml`) into `python/minhash_rs/_native.abi3.so`. Users `import minhash_rs` and get both Rust and Python symbols.

## Commands

### Build
```bash
maturin develop           # debug build, fast iteration
maturin develop --release # release build with LTO (Cargo.toml sets lto=true, codegen-units=1)
```
A debug build is used in CI for speed; benchmarks need `--release`.

### Test (correctness)
```bash
python tests/correctness.py
```
Requires `datasketch` and `joblib` installed (`pip install -e ".[test]" joblib`). The harness verifies bit-equivalence against `datasketch.MinHash` on 1,000 ASCII records + 12 adversarial edge cases, and that `default_permutations` matches `datasketch.MinHash(num_perm, seed).permutations` byte-for-byte. No pytest — it's a `__main__` script.

### Benchmark
```bash
python tests/benchmark.py
```

### Lint
```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```
CI runs both with `-D warnings`; treat clippy warnings as errors.

## Constraints

- **Bit-equivalence with `datasketch` is the core promise.** Any change to `sha1_hash32`, `permute`, `ngrams_of`, or `default_permutations` must be re-verified by running `tests/correctness.py` against `datasketch`. The reference behavior (`ngrams_of` short-text fallback emitting the whole text once for `len < min(ngram_sizes)`) mirrors `datasketch.MinHash.update(text)` for short strings — don't "fix" it to skip short inputs.
- **Crate-level `#![allow(clippy::useless_conversion)]`** in `src/lib.rs` is intentional: PyO3 0.22's `#[pyfunction]` macro expansion triggers the warning on a span that can't be suppressed locally.
- **Python ≥ 3.12, abi3-py312.** PyO3 is pinned to 0.22 to line up with `numpy = "0.22"` (which supports numpy ≥ 1.26 including the 2.x ABI).

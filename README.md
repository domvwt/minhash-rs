# minhash-rs

Rust port of [`datasketch.MinHash`](https://github.com/ekzhu/datasketch)'s core algorithm: batch MinHash signatures with optional rayon parallelism. Bit-equivalent to `datasketch` (same SHA1 + Mersenne-prime permutation arithmetic, same numpy permutation matrix), drop-in compatible.

## Why

`datasketch.MinHash` is pure Python. Its per-record cost is dominated by Python overhead, not the hashing itself. For pipelines that need to hash millions of records (e.g. blocking for entity resolution), this becomes the bottleneck.

`minhash-rs` ports the hot loop to Rust and releases the GIL so multi-record batches can run in a scoped rayon pool. On a typical name-blocking workload (128 permutations, 2-gram + 3-gram bytes, 50,000 records) it is **~178× faster** than `joblib`-parallel `datasketch`. See `tests/benchmark.py`.

## Install

```bash
pip install minhash-rs       # once published to PyPI
# or, in the meantime:
pip install git+https://github.com/domvwt/minhash-rs@v0.1.0
```

Requires Python ≥ 3.12. Wheels include the compiled `.so` (no Rust toolchain needed at install time once PyPI wheels are up; building from source requires `rustc` and `maturin`).

## API

```python
import minhash_rs

# 1. Derive the (2, num_perm) permutation matrix matching
#    datasketch.MinHash(num_perm, seed).permutations.
perms = minhash_rs.default_permutations(num_perm=128, seed=1)

# 2a. High-level: hash a list of text strings end-to-end.
sigs = minhash_rs.minhash_batch_from_text(
    texts,                  # list[str]
    perms,                  # (2, num_perm) uint64 ndarray
    num_perm=128,
    ngram_sizes=[2, 3],     # default
    lowercase=True,         # default
    num_threads=None,       # None = serial; Some(n) = scoped rayon pool
)
# -> (len(texts), num_perm) uint64 ndarray

# 2b. Low-level: hash pre-tokenized byte n-grams (bring your own tokenizer).
sigs = minhash_rs.minhash_batch(
    ngrams_per_record,      # list[list[bytes]]
    perms,
    num_perm=128,
    num_threads=4,
)
```

Both functions return `(n_records, num_perm)` uint64 arrays — the same shape and semantics as stacking `datasketch.MinHash.hashvalues` per record.

## Threading

`num_threads=None` (or `1`) runs serially. `num_threads=n>1` builds a **scoped** rayon pool just for that call (does not touch the global pool, does not oversubscribe if the caller is already inside its own parallel context like `joblib`).

If your caller already parallelizes at the process level (e.g. `joblib.Parallel(n_jobs=n)`), pass `num_threads=1` inside each worker to avoid `n × num_cpus` total threads.

## Benchmarks

128 permutations, byte 2-gram + 3-gram, name-shaped corpus (3–20 ASCII chars per record), single i7-1165G7 box:

| Path                              | µs/record | Speedup |
| --------------------------------- | --------- | ------- |
| `joblib` + `datasketch` (4 jobs)  | 168.3     | 1×      |
| `minhash_batch_from_text` serial  |   1.7     | 98×     |
| `minhash_batch_from_text` 4 threads |   2.3   | 74×     |

At these per-record costs (~1–2 µs) rayon's chunk-dispatch overhead can dominate; **serial often wins** for short text. Threading pays off for longer texts (where the per-record SHA1 + permutation cost is heavier) or when you want to consume more cores without spawning processes. Profile your workload.

See `tests/benchmark.py` for the reproducer.

## Bit-equivalence

`minhash-rs` matches `datasketch.MinHash` byte-for-byte on the same inputs and same seed. Verified on 1,000 random ASCII records + 12 adversarial edge cases (empty strings, single characters, Unicode, punctuation, long strings) in `tests/correctness.py`.

Concretely, both implement:

- `h(x) = struct.unpack('<I', sha1(x).digest()[:4])[0]` (SHA1 truncated to first 4 bytes, little-endian u32)
- `perm_i(h) = ((a_i * h + b_i) % (2**61 - 1)) & (2**32 - 1)`
- `(a_i, b_i)` drawn from `numpy.random.RandomState(seed).randint(...)` over the same ranges as `datasketch` uses internally

## Building from source

```bash
git clone https://github.com/domvwt/minhash-rs
cd minhash-rs
pip install maturin
maturin develop --release
python tests/correctness.py
```

## Credits

This is a port of the MinHash implementation from [`datasketch`](https://github.com/ekzhu/datasketch) by [Eric Zhu](https://github.com/ekzhu) and contributors, distributed under the MIT license. Only the core algorithm is ported here; for the rest of `datasketch`'s probabilistic data structures (LSH, HyperLogLog, etc.), use `datasketch` directly.

## License

MIT — see [LICENSE](LICENSE).

// PyO3's #[pyfunction] proc-macro emits an .into() on the return value
// that clippy flags as useless when the function already returns
// PyResult<...>. The warning fires on the macro-expanded span, so a
// crate-level allow is the only placement that reliably suppresses it.
#![allow(clippy::useless_conversion)]

//! Rust port of `datasketch.MinHash`'s core algorithm: batch MinHash
//! signatures with optional rayon parallelism.
//!
//! ### Bit-equivalence with `datasketch`
//!
//! `datasketch.MinHash` defines its hashvalues as `min_i(perm_i(h(x)))`
//! where:
//! - `h(x) = struct.unpack('<I', sha1(x).digest()[:4])[0]`
//!   (SHA1 truncated to the first 4 bytes, **little-endian** -> u32)
//! - `perm_i(h) = ((a_i * h + b_i) % MERSENNE_PRIME) & MAX_HASH`
//!   where `MERSENNE_PRIME = 2^61 - 1` and `MAX_HASH = 2^32 - 1`
//! - `(a_i, b_i)` come from `numpy.random.RandomState(seed)` randint
//!   draws over `[1, MERSENNE_PRIME)` and `[0, MERSENNE_PRIME)`.
//!
//! The caller passes the `(2, num_perm)` permutation matrix as a numpy
//! array (see `minhash_rs.default_permutations`). Rust owns the hot
//! loop: SHA1 + permutation arithmetic + min-reduction.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;
use sha1::{Digest, Sha1};

/// Mersenne prime `2^61 - 1`, the modulus in `datasketch`'s permutation
/// formula.
const MERSENNE_PRIME: u64 = (1u64 << 61) - 1;

/// `2^32 - 1`, the truncation mask applied after the permutation modulo.
const MAX_HASH: u64 = (1u64 << 32) - 1;

/// `struct.unpack('<I', sha1(data).digest()[:4])[0]` — the first 4 bytes
/// of SHA1 read as a **little-endian** u32, widened to u64.
fn sha1_hash32(data: &[u8]) -> u64 {
    let mut hasher = Sha1::new();
    hasher.update(data);
    let digest = hasher.finalize();
    u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as u64
}

/// Apply one `datasketch`-style permutation: `(a*h + b) % MERSENNE_PRIME & MAX_HASH`.
#[inline(always)]
fn permute(h: u64, a: u64, b: u64) -> u64 {
    (a.wrapping_mul(h).wrapping_add(b) % MERSENNE_PRIME) & MAX_HASH
}

/// Compute one record's MinHash signature from a slice of pre-tokenized
/// byte n-grams. Writes into `out`, which is pre-filled with `MAX_HASH`.
///
/// Empty n-gram list → output stays all-`MAX_HASH` (matches
/// `datasketch.MinHash(num_perm).hashvalues` with no `update` calls).
fn minhash_one(ngrams: &[&[u8]], perms_a: &[u64], perms_b: &[u64], out: &mut [u64]) {
    let num_perm = perms_a.len();
    for ngram in ngrams {
        let h = sha1_hash32(ngram);
        for i in 0..num_perm {
            let p = permute(h, perms_a[i], perms_b[i]);
            if p < out[i] {
                out[i] = p;
            }
        }
    }
}

/// Generate the lowercase byte n-grams of the requested sizes for a
/// single text. Mirrors the production behavior in
/// `horkos.resolution.blockers.ann_blocker._process_batch_equal_weighting`:
/// for each ngram size `k`, emit `text[j..j+k]` for `j in 0..len-k+1`.
/// If `len < min(ngram_sizes)`, emit the whole text once (matches the
/// `datasketch` `update(text)` fallback for short strings).
fn ngrams_of<'a>(text: &'a [u8], ngram_sizes: &[usize]) -> Vec<&'a [u8]> {
    let len = text.len();
    if len == 0 {
        return Vec::new();
    }
    let min_size = *ngram_sizes.iter().min().unwrap_or(&1);
    if len < min_size {
        return vec![text];
    }
    let mut out = Vec::new();
    for &k in ngram_sizes {
        if len < k {
            continue;
        }
        for j in 0..=(len - k) {
            out.push(&text[j..j + k]);
        }
    }
    out
}

/// Run a closure either serially or under a scoped rayon pool. `num_threads=None`
/// means serial; `Some(1)` is also serial (no pool spin-up); `Some(n>1)` builds a
/// scoped pool just for this call (does not touch the global pool).
fn run_pooled<F: FnOnce() + Send>(num_threads: Option<usize>, f: F) -> PyResult<()> {
    match num_threads {
        None | Some(0) | Some(1) => {
            f();
            Ok(())
        }
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")))?;
            pool.install(f);
            Ok(())
        }
    }
}

/// Validate and split a `(2, num_perm)` permutation matrix into two
/// owned `Vec<u64>` slices (a and b coefficients).
fn split_permutations(
    perms: &PyReadonlyArray2<u64>,
    num_perm: usize,
) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let arr = perms.as_array();
    if arr.shape() != [2, num_perm] {
        return Err(PyValueError::new_err(format!(
            "permutations must have shape (2, {}); got {:?}",
            num_perm,
            arr.shape()
        )));
    }
    let a: Vec<u64> = arr.row(0).iter().copied().collect();
    let b: Vec<u64> = arr.row(1).iter().copied().collect();
    Ok((a, b))
}

/// Low-level batch primitive. Given pre-tokenized n-grams per record,
/// return MinHash signatures (uint64) matching
/// `datasketch.MinHash.update_batch(ngrams).hashvalues`.
///
/// # Arguments
/// - `ngrams_per_record`: `list[list[bytes]]`; outer list is records,
///     inner list is the byte n-grams to hash for that record.
/// - `permutations`: numpy `(2, num_perm)` uint64 array; rows are the
///     `(a, b)` coefficients of `datasketch`'s permutation matrix.
/// - `num_perm`: number of permutations per record (typically 128).
/// - `num_threads`: `None`/`Some(1)` runs serially; `Some(n>1)` uses a
///     scoped rayon pool of `n` threads.
///
/// Returns a numpy `(n_records, num_perm)` uint64 array.
#[pyfunction]
#[pyo3(signature = (ngrams_per_record, permutations, num_perm, num_threads=None))]
fn minhash_batch<'py>(
    py: Python<'py>,
    ngrams_per_record: &Bound<'py, PyList>,
    permutations: PyReadonlyArray2<'py, u64>,
    num_perm: usize,
    num_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let n_records = ngrams_per_record.len();
    let (perms_a, perms_b) = split_permutations(&permutations, num_perm)?;

    // Pre-extract Python bytes into owned Vec<Vec<u8>> so the hot loop
    // doesn't need the GIL.
    let mut records: Vec<Vec<Vec<u8>>> = Vec::with_capacity(n_records);
    for i in 0..n_records {
        let row = ngrams_per_record.get_item(i)?;
        let row_list = row.downcast::<PyList>()?;
        let n_ngrams = row_list.len();
        let mut rec: Vec<Vec<u8>> = Vec::with_capacity(n_ngrams);
        for j in 0..n_ngrams {
            let item = row_list.get_item(j)?;
            let b = item.downcast::<PyBytes>()?;
            rec.push(b.as_bytes().to_vec());
        }
        records.push(rec);
    }

    let mut out: Vec<u64> = vec![MAX_HASH; n_records * num_perm];

    py.allow_threads(|| -> PyResult<()> {
        run_pooled(num_threads, || {
            let chunks = out.par_chunks_mut(num_perm);
            let recs = records.par_iter();
            chunks.zip(recs).for_each(|(slot, rec)| {
                let refs: Vec<&[u8]> = rec.iter().map(|v| v.as_slice()).collect();
                minhash_one(&refs, &perms_a, &perms_b, slot);
            });
        })
    })?;

    let arr = numpy::ndarray::Array2::from_shape_vec((n_records, num_perm), out)
        .map_err(|e| PyValueError::new_err(format!("output reshape failed: {e}")))?;
    Ok(arr.into_pyarray_bound(py))
}

/// High-level convenience wrapper. Takes raw text, optionally lowercases
/// it (ASCII-fast then full Unicode), generates byte n-grams of the
/// requested sizes, and computes MinHash signatures in one Rust call —
/// no Python-side n-gram allocation.
///
/// # Arguments
/// - `texts`: `list[str]`. Empty/whitespace strings produce an
///     all-`MAX_HASH` signature (matches `datasketch.MinHash` with no
///     updates).
/// - `permutations`: numpy `(2, num_perm)` uint64 array.
/// - `num_perm`: number of permutations per record.
/// - `ngram_sizes`: list of byte n-gram lengths to use (defaults to
///     `[2, 3]` matching horkos's production setting).
/// - `lowercase`: if true (default), lowercase each text before hashing
///     using Rust's Unicode-aware `to_lowercase`.
/// - `num_threads`: `None`/`Some(1)` serial; `Some(n>1)` scoped rayon
///     pool.
///
/// Returns a numpy `(n_records, num_perm)` uint64 array.
#[pyfunction]
#[pyo3(signature = (texts, permutations, num_perm, ngram_sizes=None, lowercase=true, num_threads=None))]
fn minhash_batch_from_text<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyList>,
    permutations: PyReadonlyArray2<'py, u64>,
    num_perm: usize,
    ngram_sizes: Option<Vec<usize>>,
    lowercase: bool,
    num_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let sizes = ngram_sizes.unwrap_or_else(|| vec![2, 3]);
    if sizes.is_empty() {
        return Err(PyValueError::new_err("ngram_sizes must be non-empty"));
    }
    if sizes.contains(&0) {
        return Err(PyValueError::new_err("ngram_sizes entries must be >= 1"));
    }

    let n_records = texts.len();
    let (perms_a, perms_b) = split_permutations(&permutations, num_perm)?;

    // Extract + lowercase up front so the hot loop can drop the GIL.
    let mut texts_bytes: Vec<Vec<u8>> = Vec::with_capacity(n_records);
    for i in 0..n_records {
        let item = texts.get_item(i)?;
        let s: String = item.extract()?;
        let lowered = if lowercase { s.to_lowercase() } else { s };
        texts_bytes.push(lowered.into_bytes());
    }

    let mut out: Vec<u64> = vec![MAX_HASH; n_records * num_perm];

    py.allow_threads(|| -> PyResult<()> {
        run_pooled(num_threads, || {
            let chunks = out.par_chunks_mut(num_perm);
            let recs = texts_bytes.par_iter();
            chunks.zip(recs).for_each(|(slot, text)| {
                let ngrams = ngrams_of(text, &sizes);
                minhash_one(&ngrams, &perms_a, &perms_b, slot);
            });
        })
    })?;

    let arr = numpy::ndarray::Array2::from_shape_vec((n_records, num_perm), out)
        .map_err(|e| PyValueError::new_err(format!("output reshape failed: {e}")))?;
    Ok(arr.into_pyarray_bound(py))
}

/// Module entry point.
///
/// The compiled extension is named `_native` (per `python-source = "python"`
/// in pyproject.toml) and re-exported from `python/minhash_rs/__init__.py`,
/// so users `import minhash_rs` and get both the Rust functions and the
/// Python `default_permutations` helper.
#[pymodule]
#[pyo3(name = "_native")]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr(
        "__doc__",
        "Compiled Rust extension for minhash_rs: batch MinHash signatures \
         bit-equivalent to datasketch.MinHash, with optional rayon \
         parallelism. Exposes `minhash_batch` (pre-tokenized n-grams) and \
         `minhash_batch_from_text` (raw text + n-gram sizes). Prefer \
         importing from the top-level `minhash_rs` package.",
    )?;
    m.add_function(wrap_pyfunction!(minhash_batch, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_batch_from_text, m)?)?;
    Ok(())
}

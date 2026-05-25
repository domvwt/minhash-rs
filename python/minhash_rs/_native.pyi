from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def minhash_batch(
    ngrams_per_record: list[list[bytes]],
    permutations: NDArray[np.uint64],
    num_perm: int,
    num_threads: int | None = ...,
) -> NDArray[np.uint64]: ...
def minhash_batch_from_text(
    texts: list[str],
    permutations: NDArray[np.uint64],
    num_perm: int,
    ngram_sizes: Sequence[int] | None = ...,
    lowercase: bool = ...,
    num_threads: int | None = ...,
) -> NDArray[np.uint64]: ...

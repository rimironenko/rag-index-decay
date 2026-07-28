"""Single source of truth for all RNG seeds.

Embedding is deterministic given a pinned model revision + fp32, so the only
seeds that matter are NumPy (chunk/document sampling) and Faker (synthetic
PII generation). torch is seeded too, defensively, even though BGE-M3
inference itself has no training-time randomness.

Known residual nondeterminism: HNSW build order (pgvector, Chroma) is not
controlled by any of these seeds. Mitigate in engines/*_adapter.py by
sorting all upserts by `chunk_id` before insertion and building each index
single-threaded where the engine's config allows it.
"""

import random
import warnings

import numpy as np
from faker import Faker

MASTER_SEED = 42


def seed_everything(seed: int = MASTER_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as exc:
            warnings.warn(f"torch deterministic algorithms not fully available: {exc}")
    except ImportError:
        pass


if __name__ == "__main__":
    seed_everything()
    print(f"seeded with MASTER_SEED={MASTER_SEED}")

"""Your code. One stub per milestone; build them in order and keep each one small."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from thilimarketbasket.dataset import (
    Dataset,
    generate_dataset as _gen_dataset,
    load_dataset,
    save_dataset,
)
from thilimarketbasket.generate import (
    background_baskets,
    default_planted,
    make_catalog,
    plant_all,
    popularity,
)

from app.storage import Store


def generate_basket_dataset(
    n_baskets: int = 10_000,
    seed: int = 0,
    min_support: float = 0.0005,
    max_size: int = 5,
    compute_truth: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dataset:
    """Generate a synthetic dataset of shopping baskets with planted itemsets.

    Uses the data generation technique from "Teach a Language Model What Sells Together":
    1. Realistic grocery catalog with brands and pack sizes.
    2. Item sales popularity distribution.
    3. Background baskets sampled according to product popularity and basket size distribution.
    4. Planted natural and surprising itemsets with known target supports.
    5. Itemset planting avoiding duplicate SKUs.
    6. Optional exact mining of frequent itemsets.
    """
    if compute_truth:
        dataset = _gen_dataset(n_baskets=n_baskets, seed=seed, min_support=min_support, max_size=max_size)
    else:
        catalog = make_catalog(seed=seed)
        shares = popularity(catalog)
        sample_size = 200_000 if n_baskets >= 10_000 else max(10_000, n_baskets * 2)
        sample = background_baskets(sample_size, catalog, shares, seed=seed + 10_000)
        planted = default_planted(catalog, shares, sample, seed=seed)
        raw_baskets = background_baskets(n_baskets, catalog, shares, seed=seed)
        baskets = plant_all(raw_baskets, planted, catalog, seed=seed)
        dataset = Dataset(
            catalog=catalog,
            shares=shares,
            planted=planted,
            baskets=baskets,
            truth={},
            min_support=min_support,
            seed=seed,
        )

    if output_dir is not None:
        save_dataset(dataset, output_dir)

    return dataset


def load_basket_dataset(directory: Union[str, Path]) -> Dataset:
    """Load a previously saved basket dataset from disk."""
    return load_dataset(directory)


# --- Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
# Hint: Use the data generation technique from "Teach a Language Model What Sells Together".
def milestone_1(
    n_baskets: int = 10_000,
    seed: int = 0,
    output_dir: Optional[Union[str, Path]] = "data/synthetic_baskets",
    compute_truth: bool = False,
    min_support: float = 0.0005,
    max_size: int = 5,
) -> Dataset:
    """Generate a synthetic dataset of shopping baskets with planted itemsets."""
    dataset = generate_basket_dataset(
        n_baskets=n_baskets,
        seed=seed,
        min_support=min_support,
        max_size=max_size,
        compute_truth=compute_truth,
        output_dir=output_dir,
    )

    store = Store()
    store.put(
        "milestone_1",
        {
            "n_baskets": len(dataset.baskets),
            "seed": seed,
            "n_items": len(dataset.catalog),
            "n_planted": len(dataset.planted),
            "output_dir": str(output_dir) if output_dir else None,
            "has_truth": bool(dataset.truth),
        },
    )
    return dataset


# --- Milestone 2: Implement the n-gram model to predict next items in a basket.
# Hint: Apply the n-gram model building technique from "Build a Next-Word Predictor from Scratch".
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
# Hint: Adapt the language model training process from "Teach a Language Model What Sells Together".
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
# Hint: Calculate standard next-item prediction metrics like accuracy or rank.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Present a comparison of the models' performance.
# Hint: Summarize and compare the evaluation results from both models.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")

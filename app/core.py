"""Your code. One stub per milestone; build them in order and keep each one small."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


class NGramModel:
    """N-gram language model for predicting the next item in a retail shopping basket.

    Based on 'Build a Next-Word Predictor from Scratch':
    - Extracts n-grams from basket sequences up to order n.
    - Estimates conditional probabilities using frequency counts.
    - Uses Stupid Backoff and smoothing for unseen contexts and candidates.
    - Provides candidate scoring and next-item predictions.
    """

    def __init__(self, n: int = 2, smoothing: float = 0.01, backoff_alpha: float = 0.4):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n
        self.smoothing = smoothing
        self.backoff_alpha = backoff_alpha
        self.vocabulary: List[str] = []
        self.item_to_idx: Dict[str, int] = {}
        # counts[k] stores context tuple of length (k-1) -> Counter of next items
        self.counts: Dict[int, Dict[Tuple[str, ...], Counter]] = {k: defaultdict(Counter) for k in range(1, n + 1)}
        self.context_totals: Dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self.total_unigrams: int = 0

    def fit(self, baskets: Sequence[Sequence[str]], bidirectional: bool = True) -> "NGramModel":
        """Train the n-gram model on shopping baskets.

        Args:
            baskets: List of shopping baskets (each a sequence of item names).
            bidirectional: If True, also trains on reversed basket sequences so
                item associations in unordered baskets are learned symmetrically.
        """
        vocab_set = set()
        for basket in baskets:
            vocab_set.update(basket)

        self.vocabulary = sorted(vocab_set)
        self.item_to_idx = {item: i for i, item in enumerate(self.vocabulary)}

        # Reset counts
        self.counts = {k: defaultdict(Counter) for k in range(1, self.n + 1)}
        self.context_totals = {k: Counter() for k in range(1, self.n + 1)}
        self.total_unigrams = 0

        def process_sequence(seq: Sequence[str]):
            m = len(seq)
            if m == 0:
                return
            # Unigrams
            for item in seq:
                self.counts[1][()][item] += 1
                self.total_unigrams += 1
            self.context_totals[1][()] = self.total_unigrams

            # Higher orders k from 2 to self.n
            for k in range(2, self.n + 1):
                for i in range(m - k + 1):
                    ctx = tuple(seq[i : i + k - 1])
                    nxt = seq[i + k - 1]
                    self.counts[k][ctx][nxt] += 1
                    self.context_totals[k][ctx] += 1

        for basket in baskets:
            if not basket:
                continue
            process_sequence(basket)
            if bidirectional and len(basket) > 1:
                process_sequence(list(reversed(basket)))

        return self

    def score_candidates(
        self,
        context: Sequence[str],
        candidates: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Compute probability scores for all candidates given the context sequence.

        Uses Stupid Backoff: tries longest context up to n-1, backing off to
        shorter contexts with a backoff discount factor down to smoothed unigrams.
        """
        if candidates is None:
            cands = self.vocabulary
        else:
            cands = list(candidates)

        n_cands = len(cands)
        if n_cands == 0:
            return np.array([], dtype=float)

        if not self.vocabulary:
            return np.ones(n_cands, dtype=float) / max(1, n_cands)

        # Baseline: smoothed unigram probabilities
        unigram_counts = self.counts.get(1, {}).get((), Counter())
        denom_unigram = self.total_unigrams + self.smoothing * len(self.vocabulary)
        unigram_scores = np.array(
            [(unigram_counts.get(c, 0) + self.smoothing) / denom_unigram for c in cands],
            dtype=float,
        )

        max_ctx_len = min(len(context), self.n - 1)
        if max_ctx_len == 0:
            return unigram_scores

        scores = np.zeros(n_cands, dtype=float)
        matched = np.zeros(n_cands, dtype=bool)

        # Try contexts from longest down to 1
        current_factor = 1.0
        for ctx_len in range(max_ctx_len, 0, -1):
            ctx = tuple(context[-ctx_len:])
            k = ctx_len + 1
            if k in self.counts and ctx in self.counts[k]:
                next_counts = self.counts[k][ctx]
                total = self.context_totals[k][ctx]
                if total > 0:
                    for i, c in enumerate(cands):
                        if not matched[i] and c in next_counts:
                            scores[i] = current_factor * (next_counts[c] / total)
                            matched[i] = True

            current_factor *= self.backoff_alpha

        # For unmatched candidates, fill with backed-off unigram score
        unmatched_mask = ~matched
        if np.any(unmatched_mask):
            scores[unmatched_mask] = current_factor * unigram_scores[unmatched_mask]

        return scores

    def predict_next(
        self,
        context: Sequence[str],
        top_k: int = 1,
        exclude: Sequence[str] = (),
        candidates: Optional[Sequence[str]] = None,
        return_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Predict the most likely next item(s) given a partial basket context.

        Args:
            context: The items currently in the basket in sequential order.
            top_k: How many top candidates to return.
            exclude: Items to exclude from predictions (e.g. items already in basket).
            candidates: Optional candidate pool (defaults to all vocabulary items).
            return_scores: If True, returns list of (item_name, score) tuples.
        """
        if candidates is None:
            cands = list(self.vocabulary)
        else:
            cands = list(candidates)

        if not cands:
            return []

        scores = self.score_candidates(context, candidates=cands)
        exclude_set = set(exclude)
        if exclude_set:
            for i, c in enumerate(cands):
                if c in exclude_set:
                    scores[i] = -np.inf

        top_indices = np.argsort(-scores, kind="stable")[:top_k]
        if return_scores:
            return [(cands[i], float(scores[i])) for i in top_indices if scores[i] != -np.inf]
        return [cands[i] for i in top_indices if scores[i] != -np.inf]

    def save(self, path: Union[str, Path]) -> Path:
        """Save model counts and configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized_counts = {}
        for k in self.counts:
            serialized_counts[str(k)] = {
                "\t".join(ctx): dict(counter)
                for ctx, counter in self.counts[k].items()
            }
        serialized_totals = {
            str(k): {"\t".join(ctx): total for ctx, total in self.context_totals[k].items()}
            for k in self.context_totals
        }
        data = {
            "n": self.n,
            "smoothing": self.smoothing,
            "backoff_alpha": self.backoff_alpha,
            "vocabulary": self.vocabulary,
            "total_unigrams": self.total_unigrams,
            "counts": serialized_counts,
            "context_totals": serialized_totals,
        }
        path.write_text(json.dumps(data))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NGramModel":
        """Load a saved NGramModel from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        model = cls(n=data["n"], smoothing=data.get("smoothing", 0.01), backoff_alpha=data.get("backoff_alpha", 0.4))
        model.vocabulary = data["vocabulary"]
        model.item_to_idx = {item: i for i, item in enumerate(model.vocabulary)}
        model.total_unigrams = data["total_unigrams"]

        model.counts = {int(k): defaultdict(Counter) for k in data["counts"]}
        for k_str, ctx_map in data["counts"].items():
            k = int(k_str)
            for ctx_str, counter_dict in ctx_map.items():
                ctx = tuple(ctx_str.split("\t")) if ctx_str else ()
                model.counts[k][ctx] = Counter(counter_dict)

        model.context_totals = {int(k): Counter() for k in data.get("context_totals", {})}
        for k_str, ctx_map in data.get("context_totals", {}).items():
            k = int(k_str)
            for ctx_str, total in ctx_map.items():
                ctx = tuple(ctx_str.split("\t")) if ctx_str else ()
                model.context_totals[k][ctx] = total

        return model


# --- Milestone 2: Implement the n-gram model to predict next items in a basket.
# Hint: Apply the n-gram model building technique from "Build a Next-Word Predictor from Scratch".
def milestone_2(
    n: int = 2,
    dataset: Optional[Dataset] = None,
    data_dir: Optional[Union[str, Path]] = "data/synthetic_baskets",
    model_path: Optional[Union[str, Path]] = "data/ngram_model.json",
    bidirectional: bool = True,
) -> NGramModel:
    """Implement and train an n-gram model to predict next items in a basket."""
    if dataset is None:
        p = Path(data_dir) if data_dir else None
        if p and (p / "baskets.jsonl").exists():
            dataset = load_basket_dataset(p)
        else:
            dataset = milestone_1(output_dir=data_dir)

    model = NGramModel(n=n)
    model.fit(dataset.baskets, bidirectional=bidirectional)

    if model_path is not None:
        model.save(model_path)

    store = Store()
    store.put(
        "milestone_2",
        {
            "n": n,
            "vocab_size": len(model.vocabulary),
            "n_baskets": len(dataset.baskets),
            "model_path": str(model_path) if model_path else None,
            "bidirectional": bidirectional,
        },
    )
    return model


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

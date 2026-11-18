"""Your code. One stub per milestone; build them in order and keep each one small."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


class SmallLanguageModel:
    """A small causal language model for predicting next items in basket sequences.

    Adapted from 'Teach a Language Model What Sells Together':
    - Causal self-attention with masked attention over item tokens.
    - Autoregressive next-token prediction objective with cross-entropy loss.
    - Optimized with Adam optimizer.
    - Evaluates next-item candidate scoring and predictions.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        d_model: int = 32,
        max_len: int = 32,
        seed: int = 42,
    ):
        self.vocabulary = list(vocab)
        self.item_to_id: Dict[str, int] = {item: i for i, item in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)
        self.d_model = d_model
        self.max_len = max_len
        self.seed = seed

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.E = rng.normal(0, scale, (self.vocab_size, d_model)).astype(np.float32)
        self.P = rng.normal(0, scale, (max_len, d_model)).astype(np.float32)
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
        self.W_out = rng.normal(0, scale, (d_model, self.vocab_size)).astype(np.float32)
        self.b_out = np.zeros(self.vocab_size, dtype=np.float32)

        self.param_names = ["E", "P", "W_q", "W_k", "W_v", "W_o", "W_out", "b_out"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.param_names}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.param_names}
        self.t = 0

    def forward(self, token_ids: Sequence[int]):
        T = min(len(token_ids), self.max_len)
        ids = np.array(token_ids[:T], dtype=np.int32)
        X = self.E[ids] + self.P[:T]
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        scores = (Q @ K.T) / np.sqrt(self.d_model)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[mask] = -1e9
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        A = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        H = A @ V
        Z = H @ self.W_o
        O = Z + X
        logits = O @ self.W_out + self.b_out
        return ids, X, Q, K, V, A, H, Z, O, logits

    def train_step(
        self,
        token_ids: Sequence[int],
        lr: float = 0.005,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        if len(token_ids) < 2:
            return 0.0
        ids, X, Q, K, V, A, H, Z, O, logits = self.forward(token_ids)
        T = len(ids)
        targets = ids[1:]
        pred_logits = logits[:-1]
        l_max = np.max(pred_logits, axis=-1, keepdims=True)
        exp_l = np.exp(pred_logits - l_max)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        T_pred = len(targets)
        loss = -float(np.mean(np.log(np.maximum(probs[np.arange(T_pred), targets], 1e-12))))

        dlogits = np.zeros_like(logits)
        dlogits[:-1] = probs
        dlogits[np.arange(T_pred), targets] -= 1.0
        dlogits /= T_pred

        dW_out = O.T @ dlogits
        db_out = np.sum(dlogits, axis=0)
        dO = dlogits @ self.W_out.T
        dZ = dO
        dH = dZ @ self.W_o.T
        dW_o = H.T @ dZ
        dV = A.T @ dH
        dA = dH @ V.T
        dS = A * (dA - np.sum(dA * A, axis=-1, keepdims=True)) / np.sqrt(self.d_model)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dS[mask] = 0.0
        dQ = dS @ K
        dK = dS.T @ Q
        dW_q = X.T @ dQ
        dW_k = X.T @ dK
        dW_v = X.T @ dV
        dX = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T + dO

        dP = np.zeros_like(self.P)
        dP[:T] = dX
        dE = np.zeros_like(self.E)
        np.add.at(dE, ids, dX)

        self.t += 1
        grads = {
            "E": dE,
            "P": dP,
            "W_q": dW_q,
            "W_k": dW_k,
            "W_v": dW_v,
            "W_o": dW_o,
            "W_out": dW_out,
            "b_out": db_out,
        }
        for name in self.param_names:
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            param = getattr(self, name)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        return loss

    def compute_loss(self, token_ids: Sequence[int]) -> float:
        if len(token_ids) < 2:
            return 0.0
        ids, _, _, _, _, _, _, _, _, logits = self.forward(token_ids)
        targets = ids[1:]
        pred_logits = logits[:-1]
        l_max = np.max(pred_logits, axis=-1, keepdims=True)
        exp_l = np.exp(pred_logits - l_max)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        T_pred = len(targets)
        return -float(np.mean(np.log(np.maximum(probs[np.arange(T_pred), targets], 1e-12))))

    def fit(
        self,
        baskets: Sequence[Sequence[str]],
        steps: int = 500,
        batch_size: int = 16,
        lr: float = 0.005,
        val_baskets: Optional[Sequence[Sequence[str]]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        seed: int = 42,
    ) -> List[Dict[str, Union[int, float]]]:
        """Pretrain the small language model on shopping baskets."""
        rng = np.random.default_rng(seed)
        history = []
        recent_losses = []

        valid_baskets = [b for b in baskets if len(b) >= 2]
        if not valid_baskets:
            raise ValueError("No baskets with at least 2 items found for training.")

        for step in range(1, steps + 1):
            batch_indices = rng.choice(len(valid_baskets), size=min(batch_size, len(valid_baskets)), replace=False)
            step_losses = []
            for idx in batch_indices:
                basket = valid_baskets[idx]
                perm = rng.permutation(len(basket))
                shuffled_items = [basket[i] for i in perm]
                token_ids = [self.item_to_id[it] for it in shuffled_items if it in self.item_to_id]
                if len(token_ids) >= 2:
                    l = self.train_step(token_ids, lr=lr)
                    step_losses.append(l)

            if step_losses:
                recent_losses.append(float(np.mean(step_losses)))

            if step % 50 == 0 or step == steps:
                avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                record: Dict[str, Union[int, float]] = {"step": step, "loss": avg_loss}
                if val_baskets:
                    val_losses = []
                    for vb in val_baskets[:50]:
                        val_ids = [self.item_to_id[it] for it in vb if it in self.item_to_id]
                        if len(val_ids) >= 2:
                            val_losses.append(self.compute_loss(val_ids))
                    record["val_loss"] = float(np.mean(val_losses)) if val_losses else avg_loss
                history.append(record)
                recent_losses = []
                if log_fn:
                    val_str = f" val_loss={record['val_loss']:.4f}" if "val_loss" in record else ""
                    log_fn(f"Step {step}/{steps} - loss: {avg_loss:.4f}{val_str}")

        return history

    def score_candidates(
        self,
        context: Sequence[str],
        candidates: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Compute score (log probability) for each candidate given context items."""
        if candidates is None:
            cands = self.vocabulary
        else:
            cands = list(candidates)

        n_cands = len(cands)
        if n_cands == 0:
            return np.array([], dtype=float)

        token_ids = [self.item_to_id[it] for it in context if it in self.item_to_id]
        if not token_ids:
            logits = self.b_out
        else:
            *_, all_logits = self.forward(token_ids)
            logits = all_logits[-1]

        l_max = np.max(logits)
        log_sum_exp = l_max + np.log(np.sum(np.exp(logits - l_max)))
        log_probs = logits - log_sum_exp

        scores = np.zeros(n_cands, dtype=float)
        for i, cand in enumerate(cands):
            if cand in self.item_to_id:
                scores[i] = log_probs[self.item_to_id[cand]]
            else:
                scores[i] = -1e9
        return scores

    def predict_next(
        self,
        context: Sequence[str],
        top_k: int = 1,
        exclude: Sequence[str] = (),
        candidates: Optional[Sequence[str]] = None,
        return_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Predict the next item(s) in a shopping basket."""
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
        """Save model weights and vocabulary to an .npz archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vocabulary=np.array(self.vocabulary, dtype=object),
            d_model=np.array([self.d_model]),
            max_len=np.array([self.max_len]),
            seed=np.array([self.seed]),
            E=self.E,
            P=self.P,
            W_q=self.W_q,
            W_k=self.W_k,
            W_v=self.W_v,
            W_o=self.W_o,
            W_out=self.W_out,
            b_out=self.b_out,
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SmallLanguageModel":
        """Load a saved SmallLanguageModel from an .npz archive."""
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            vocab = list(data["vocabulary"])
            d_model = int(data["d_model"][0])
            max_len = int(data["max_len"][0])
            seed = int(data["seed"][0]) if "seed" in data else 42
            model = cls(vocab=vocab, d_model=d_model, max_len=max_len, seed=seed)
            for param in model.param_names:
                setattr(model, param, data[param].copy())
        return model


# --- Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
# Hint: Adapt the language model training process from "Teach a Language Model What Sells Together".
def milestone_3(
    steps: int = 500,
    batch_size: int = 16,
    d_model: int = 32,
    lr: float = 0.005,
    dataset: Optional[Dataset] = None,
    data_dir: Optional[Union[str, Path]] = "data/synthetic_baskets",
    model_path: Optional[Union[str, Path]] = "data/lm_model.npz",
    seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> SmallLanguageModel:
    """Pretrain a small language model on the basket sequences."""
    if dataset is None:
        p = Path(data_dir) if data_dir else None
        if p and (p / "baskets.jsonl").exists():
            dataset = load_basket_dataset(p)
        else:
            dataset = milestone_1(output_dir=data_dir)

    vocab = sorted({item for b in dataset.baskets for item in b})
    model = SmallLanguageModel(vocab=vocab, d_model=d_model, seed=seed)

    history = model.fit(
        dataset.baskets,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        log_fn=log_fn,
    )

    if model_path is not None:
        model.save(model_path)

    store = Store()
    final_loss = history[-1]["loss"] if history else None
    store.put(
        "milestone_3",
        {
            "steps": steps,
            "batch_size": batch_size,
            "d_model": d_model,
            "lr": lr,
            "vocab_size": len(model.vocabulary),
            "final_loss": final_loss,
            "model_path": str(model_path) if model_path else None,
        },
    )
    return model


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
# Hint: Calculate standard next-item prediction metrics like accuracy or rank.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Present a comparison of the models' performance.
# Hint: Summarize and compare the evaluation results from both models.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")

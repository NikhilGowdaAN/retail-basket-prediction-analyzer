"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import numpy as np
import pytest

from app.core import milestone_1, milestone_2, load_basket_dataset, NGramModel
from app.storage import Store


# Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
def test_milestone_1(tmp_path):
    out_dir = tmp_path / "synthetic_baskets"
    dataset = milestone_1(
        n_baskets=10_000,
        seed=42,
        output_dir=out_dir,
        compute_truth=False,
    )

    # 1. Baskets generated
    assert len(dataset.baskets) == 10_000
    assert all(isinstance(b, list) and len(b) > 0 for b in dataset.baskets[:100])
    assert all(isinstance(item, str) for item in dataset.baskets[0])

    # 2. Catalog generated with SKUs
    assert len(dataset.catalog) > 0
    catalog_names = {item.name for item in dataset.catalog}
    for b in dataset.baskets[:50]:
        for item in b:
            assert item in catalog_names

    # 3. Planted itemsets exist (natural and surprising)
    assert len(dataset.planted) == 32
    assert any(p.kind == "natural" for p in dataset.planted)
    assert any(p.kind == "surprising" for p in dataset.planted)

    # 4. Planted itemsets are realised in the baskets with expected support
    high_support_planted = [p for p in dataset.planted if p.support >= 0.05]
    assert len(high_support_planted) > 0
    for p in high_support_planted:
        planted_set = set(p.items)
        count = sum(1 for b in dataset.baskets if planted_set <= set(b))
        realised_support = count / len(dataset.baskets)
        assert realised_support >= 0.04

    # 5. Output files written and loadable
    assert (out_dir / "baskets.jsonl").exists()
    assert (out_dir / "catalog.json").exists()
    assert (out_dir / "planted.json").exists()
    assert (out_dir / "meta.json").exists()

    loaded = load_basket_dataset(out_dir)
    assert len(loaded.baskets) == 10_000
    assert len(loaded.planted) == 32
    assert len(loaded.catalog) == len(dataset.catalog)

    # 6. Store updated with milestone 1 metadata
    store = Store()
    meta = store.get("milestone_1")
    assert meta is not None
    assert meta["n_baskets"] == 10_000
    assert meta["n_planted"] == 32


def test_cli_generate_command(tmp_path, capsys):
    from app.cli import main
    out_dir = tmp_path / "cli_baskets"
    main(["generate", "--baskets", "10000", "--seed", "7", "--output-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert "Generated 10,000 baskets" in out
    assert (out_dir / "baskets.jsonl").exists()
    assert (out_dir / "meta.json").exists()


# Milestone 2: Implement the n-gram model to predict next items in a basket.
def test_milestone_2(tmp_path):
    data_dir = tmp_path / "baskets"
    model_path = tmp_path / "ngram_model.json"

    # Train model via milestone_2
    model = milestone_2(n=2, data_dir=data_dir, model_path=model_path)
    assert model.n == 2
    assert len(model.vocabulary) > 0
    assert model_path.exists()

    # Verify model loading
    loaded = NGramModel.load(model_path)
    assert loaded.n == 2
    assert len(loaded.vocabulary) == len(model.vocabulary)

    # Verify predictions on planted itemsets
    # Load dataset to inspect planted pairs
    dataset = load_basket_dataset(data_dir)
    high_support_pairs = [p for p in dataset.planted if len(p.items) == 2 and p.support >= 0.01]
    assert len(high_support_pairs) > 0

    item_a, item_b = high_support_pairs[0].items
    # Given item_a, predict next items (excluding item_a)
    top_preds = model.predict_next([item_a], top_k=10, exclude=[item_a])
    assert item_b in top_preds

    # Verify backoff on unseen context
    unseen_context = ["UnknownItemABC_XYZ"]
    fallback_preds = model.predict_next(unseen_context, top_k=5)
    assert len(fallback_preds) == 5
    # Should fall back to top unigrams
    top_unigram = model.predict_next([], top_k=1)[0]
    assert top_unigram in fallback_preds

    # Verify varying n
    unigram_model = NGramModel(n=1).fit(dataset.baskets[:500])
    assert unigram_model.n == 1
    trigram_model = NGramModel(n=3).fit(dataset.baskets[:500])
    assert trigram_model.n == 3

    scores = trigram_model.score_candidates([item_a, item_b])
    assert len(scores) == len(trigram_model.vocabulary)
    assert np.all(scores >= 0)

    # Verify store metadata
    store = Store()
    meta = store.get("milestone_2")
    assert meta is not None
    assert meta["n"] == 2
    assert meta["vocab_size"] == len(model.vocabulary)


def test_cli_ngram_commands(tmp_path, capsys):
    from app.cli import main
    model_path = tmp_path / "cli_ngram.json"
    main(["train-ngram", "--n", "2", "--data-dir", "data/synthetic_baskets", "--model-path", str(model_path)])
    out = capsys.readouterr().out
    assert "Trained 2-gram model successfully" in out
    assert model_path.exists()

    main(["predict-ngram", "--model-path", str(model_path), "--context", "Crunchy Crust Bread 400 g", "--top-k", "3"])
    pred_out = capsys.readouterr().out
    assert "Top Predictions:" in pred_out


# Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Present a comparison of the models' performance.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import json
import numpy as np
import pytest

from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    load_basket_dataset,
    NGramModel,
    SmallLanguageModel,
    evaluate_model,
    format_comparison_report,
)
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
def test_milestone_3(tmp_path):
    data_dir = tmp_path / "baskets"
    model_path = tmp_path / "lm_model.npz"

    # Pretrain model via milestone_3
    model = milestone_3(
        steps=50,
        batch_size=8,
        d_model=16,
        data_dir=data_dir,
        model_path=model_path,
        seed=42,
    )
    assert len(model.vocabulary) > 0
    assert model_path.exists()

    # Verify model loading
    loaded = SmallLanguageModel.load(model_path)
    assert loaded.vocab_size == model.vocab_size
    assert loaded.d_model == model.d_model

    # Verify candidate scoring
    dataset = load_basket_dataset(data_dir)
    sample_basket = dataset.baskets[0]
    scores = model.score_candidates(sample_basket[:2])
    assert len(scores) == len(model.vocabulary)
    assert np.all(np.isfinite(scores))

    # Verify next-item prediction
    top_preds = model.predict_next(sample_basket[:2], top_k=5, exclude=sample_basket[:2])
    assert len(top_preds) == 5
    for item in sample_basket[:2]:
        assert item not in top_preds

    # Verify store metadata
    store = Store()
    meta = store.get("milestone_3")
    assert meta is not None
    assert meta["steps"] == 50
    assert meta["d_model"] == 16


def test_cli_lm_commands(tmp_path, capsys):
    from app.cli import main
    model_path = tmp_path / "cli_lm.npz"
    main(["train-lm", "--steps", "20", "--batch-size", "4", "--d-model", "16", "--data-dir", "data/synthetic_baskets", "--model-path", str(model_path)])
    out = capsys.readouterr().out
    assert "Pretraining complete." in out
    assert model_path.exists()

    main(["predict-lm", "--model-path", str(model_path), "--context", "Crunchy Crust Bread 400 g", "--top-k", "3"])
    pred_out = capsys.readouterr().out
    assert "Top LM Predictions:" in pred_out


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
def test_milestone_4(tmp_path):
    data_dir = tmp_path / "baskets"
    ngram_path = tmp_path / "ngram.json"
    lm_path = tmp_path / "lm.npz"
    results_path = tmp_path / "eval_results.json"

    # Run milestone_4 evaluation
    results = milestone_4(
        data_dir=data_dir,
        ngram_path=ngram_path,
        lm_path=lm_path,
        n_cases=30,
        seed=42,
        output_path=results_path,
    )

    assert results["n_cases"] == 30
    assert "ngram" in results
    assert "lm" in results

    # Verify metrics for both models
    for key in ("ngram", "lm"):
        m = results[key]
        assert "accuracy" in m
        assert "hits@1" in m
        assert "hits@5" in m
        assert "hits@10" in m
        assert "hits@50" in m
        assert "mrr" in m
        assert "median_rank" in m
        assert "mean_rank" in m
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["mrr"] <= 1.0
        assert m["median_rank"] >= 1.0
        assert m["mean_rank"] >= 1.0

    # Verify ranks list
    assert len(results["ngram_ranks"]) == 30
    assert len(results["lm_ranks"]) == 30

    # Verify results file saved
    assert results_path.exists()
    saved = json.loads(results_path.read_text())
    assert saved["n_cases"] == 30
    assert "ngram" in saved

    # Verify store recorded evaluation
    store = Store()
    meta = store.get("milestone_4")
    assert meta is not None
    assert meta["n_cases"] == 30
    assert "ngram_accuracy" in meta
    assert "lm_accuracy" in meta


def test_cli_evaluate_command(tmp_path, capsys):
    from app.cli import main
    eval_file = tmp_path / "cli_eval.json"
    main(["evaluate", "--cases", "15", "--data-dir", "data/synthetic_baskets", "--output-path", str(eval_file)])
    out = capsys.readouterr().out
    assert "Accuracy (Hits@1)" in out
    assert "MRR" in out
    assert "Median Rank" in out
    assert eval_file.exists()


# Milestone 5: Present a comparison of the models' performance.
def test_milestone_5(tmp_path):
    report_file = tmp_path / "comparison_report.md"

    # Mock evaluation results
    mock_eval = {
        "n_cases": 20,
        "seed": 42,
        "ngram": {
            "accuracy": 0.10,
            "hits@1": 0.10,
            "hits@5": 0.20,
            "hits@10": 0.30,
            "hits@50": 0.50,
            "mrr": 0.15,
            "median_rank": 40.0,
            "mean_rank": 85.0,
        },
        "lm": {
            "accuracy": 0.05,
            "hits@1": 0.05,
            "hits@5": 0.15,
            "hits@10": 0.25,
            "hits@50": 0.60,
            "mrr": 0.11,
            "median_rank": 35.0,
            "mean_rank": 75.0,
        },
        "ngram_ranks": [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "lm_ranks": [3, 4, 5, 8, 15, 40, 80, 120, 180, 250, 2, 3, 5, 7, 14, 28, 50, 100, 200, 400],
    }

    comp = milestone_5(
        eval_results=mock_eval,
        report_path=report_file,
    )

    assert comp["n_cases"] == 20
    assert "win_rates" in comp
    assert "ngram_wins" in comp["win_rates"]
    assert "lm_wins" in comp["win_rates"]
    assert "ties" in comp["win_rates"]
    assert len(comp["key_findings"]) > 0
    assert "report_text" in comp
    assert report_file.exists()

    content = report_file.read_text()
    assert "Model Comparison Report" in content
    assert "Predictive Performance Metrics" in content
    assert "Head-to-Head Case Comparison" in content
    assert "Key Findings & Behavioral Analysis" in content

    # Check store
    store = Store()
    meta = store.get("milestone_5")
    assert meta is not None
    assert meta["n_cases"] == 20
    assert "ngram_wins" in meta


def test_cli_compare_command(tmp_path, capsys):
    from app.cli import main
    report_file = tmp_path / "cli_report.md"
    main(["compare", "--results-path", "data/evaluation_results.json", "--report-path", str(report_file)])
    out = capsys.readouterr().out
    assert "Model Comparison Report" in out
    assert "Head-to-Head Case Comparison" in out
    assert report_file.exists()


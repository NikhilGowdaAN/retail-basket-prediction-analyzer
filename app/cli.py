"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retail-basket-prediction-analyzer", description="Retail analysts use this to compare next-item prediction accuracy between an n-gram model and a small LM, both trained on simulated shopping baskets data.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    
    gen_parser = sub.add_parser("generate", help="generate synthetic basket dataset with planted itemsets")
    gen_parser.add_argument("--baskets", "-n", type=int, default=10000, help="number of baskets to generate (default: 10000)")
    gen_parser.add_argument("--seed", "-s", type=int, default=0, help="random seed for repeatability (default: 0)")
    gen_parser.add_argument("--output-dir", "-o", type=str, default="data/synthetic_baskets", help="directory to save dataset files (default: data/synthetic_baskets)")
    gen_parser.add_argument("--min-support", type=float, default=0.0005, help="minimum support for itemset mining (default: 0.0005)")
    gen_parser.add_argument("--compute-truth", action="store_true", help="run exact frequent itemsets miner")

    train_ng = sub.add_parser("train-ngram", help="train n-gram next-item prediction model on baskets")
    train_ng.add_argument("--n", type=int, default=2, help="n-gram order (default: 2 for bigram)")
    train_ng.add_argument("--data-dir", type=str, default="data/synthetic_baskets", help="path to dataset directory")
    train_ng.add_argument("--model-path", type=str, default="data/ngram_model.json", help="path to save model")
    train_ng.add_argument("--no-bidirectional", action="store_true", help="disable bidirectional training")

    pred_ng = sub.add_parser("predict-ngram", help="predict next item in a basket using trained n-gram model")
    pred_ng.add_argument("--context", "-c", type=str, required=True, help="semicolon-separated item names in basket")
    pred_ng.add_argument("--model-path", type=str, default="data/ngram_model.json", help="path to trained model")
    pred_ng.add_argument("--top-k", "-k", type=int, default=5, help="number of predictions to return")
    pred_ng.add_argument("--allow-duplicates", action="store_true", help="allow predicting items already in context")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Retail Basket Prediction Analyzer is ready. Open README.md for your milestones.")
    elif args.command == "generate":
        from app.core import milestone_1
        print(f"Generating {args.baskets:,} baskets (seed={args.seed})...")
        ds = milestone_1(
            n_baskets=args.baskets,
            seed=args.seed,
            output_dir=args.output_dir,
            compute_truth=args.compute_truth,
            min_support=args.min_support,
        )
        avg_items = sum(len(b) for b in ds.baskets) / len(ds.baskets) if ds.baskets else 0
        print(f"Generated {len(ds.baskets):,} baskets with {len(ds.planted)} planted itemsets.")
        print(f"Catalog contains {len(ds.catalog):,} SKUs.")
        print(f"Average items per basket: {avg_items:.1f}")
        if args.output_dir:
            print(f"Saved dataset to {args.output_dir}")
    elif args.command == "train-ngram":
        from app.core import milestone_2
        print(f"Training {args.n}-gram model on baskets from {args.data_dir}...")
        model = milestone_2(
            n=args.n,
            data_dir=args.data_dir,
            model_path=args.model_path,
            bidirectional=not args.no_bidirectional,
        )
        print(f"Trained {model.n}-gram model successfully.")
        print(f"Vocabulary: {len(model.vocabulary):,} items.")
        print(f"Saved model to {args.model_path}")
    elif args.command == "predict-ngram":
        from pathlib import Path
        from app.core import NGramModel, milestone_2
        p = Path(args.model_path)
        if p.exists():
            model = NGramModel.load(p)
        else:
            print(f"Model file {args.model_path} not found. Training default model first...")
            model = milestone_2(model_path=args.model_path)

        context_items = [item.strip() for item in args.context.split(";") if item.strip()]
        exclude = () if args.allow_duplicates else context_items
        predictions = model.predict_next(context_items, top_k=args.top_k, exclude=exclude, return_scores=True)
        print(f"Context: {context_items}")
        print("Top Predictions:")
        for rank, (item, score) in enumerate(predictions, 1):
            print(f"  {rank}. {item} (score: {score:.5f})")


if __name__ == "__main__":
    main()

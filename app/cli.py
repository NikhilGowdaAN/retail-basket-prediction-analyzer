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


if __name__ == "__main__":
    main()

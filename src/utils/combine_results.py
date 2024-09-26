import json
import os
import pandas as pd
import argparse
import torch


def main(args):
    # Read in the results
    results = []

    if args.replace_i:
        base = args.results[0]
        paths = [
            base.replace("{i}", str(idx)) for idx in range(args.min_i, args.max_i + 1)
        ]
    else:
        paths = [r for r in args.results]

    if args.torch:
        results = [torch.load(f) for f in paths]
    else:
        results = [json.load(open(f)) for f in paths]
    print(paths)

    if args.no_metrics:
        # Combine samples
        all_samples = []
        for r in results:
            if type(r) == dict and "samples" in r:
                all_samples += r["samples"]
            else:
                assert type(r) == list
                all_samples += r

        print(f"Combined {len(all_samples)} samples")

        if type(results[0]) == dict:
            # For chunked outputs, config should be the same, so use the first one
            res = results[0]
            res["samples"] = all_samples
            res.pop("metrics", None)
        else:
            res = all_samples
        if args.torch:
            torch.save(res, args.output)
        else:
            json.dump(res, open(args.output, "w"), indent=4)

    else:
        # Flatten metrics
        run_info = [k for k in results[0].keys() if k not in ["samples", "metrics"]]
        sep = "--"
        metric_keys = []
        for metric_group, metric_dict in results[0]["metrics"].items():
            for metric_name, value in metric_dict.items():
                metric_keys.append(f"{metric_group}{sep}{metric_name}")

        rows = []
        for result in results:
            row = {k: v for k, v in result.items() if k in run_info}
            for metric_group, metric_dict in result["metrics"].items():
                for metric_name, value in metric_dict.items():
                    row[f"{metric_group}{sep}{metric_name}"] = value
            rows.append(row)

        results = pd.DataFrame(rows)
        results.to_csv(args.output, index=False)

    print(f"Saved results to:\n{args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine results from multiple runs")
    parser.add_argument(
        "results",
        type=str,
        nargs="+",
        help='JSON results files to combine, or one JSON file name with "{i}" if using --replace-i (or .pth files if using --torch)',
    )
    parser.add_argument(
        "--output", type=str, default="combined_results.csv", help="Output file name"
    )
    parser.add_argument(
        "--torch", action="store_true", help="Use torch to load results"
    )

    # For chunked results
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Don't include metrics in combined file, just combine samples.",
    )
    parser.add_argument(
        "--replace-i",
        action="store_true",
        help='Replace  "{i}" with integers from --min-i to --max-i (inclusive)',
    )
    parser.add_argument(
        "--min-i", type=int, default=0, help="Minimum value for --replace-i"
    )
    parser.add_argument(
        "--max-i", type=int, default=4, help="Maximum value for --replace-i"
    )
    args = parser.parse_args()
    main(args)

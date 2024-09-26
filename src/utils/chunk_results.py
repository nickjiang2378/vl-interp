import json
import os
import pandas as pd
import argparse
import torch


def main(args):
    results = json.load(open(args.results))
    if type(results) == dict:
        if "samples" not in results:
            print(
                f'Key "samples" not found in {args.results}. Keys are: {results.keys()}'
            )
            raise KeyError
        samples = results["samples"]
    else:
        assert (
            type(results) == list
        ), f"Results file {args.results} is not a list or dict. Is: {type(results)}"
        samples = results

    total_samples = len(samples)
    print(f"Total samples: {total_samples}")

    N = args.N
    if args.mode == "samples":
        # Chunk by samples - each output file gets N samples
        for chunk_number, i in enumerate(range(0, total_samples, N)):
            chunk = samples[i : i + N]
            chunk_name = (args.results).replace(".json", f"_chunk-{chunk_number}.json")
            json.dump(chunk, open(chunk_name, "w"))
            print(f"Saved chunk with samples {i}-{i+N} to {chunk_name}")
    elif args.mode == "files":
        # Create N files, each with 1/N of the samples
        for chunk_number in range(N):
            chunk = samples[chunk_number::N]
            chunk_name = (args.results).replace(".json", f"_chunk-{chunk_number}.json")
            json.dump(chunk, open(chunk_name, "w"))
            print(f"Saved chunk with {len(chunk)} samples to {chunk_name}")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine results from multiple runs")
    parser.add_argument("results", type=str, help="JSON result file to chunk")
    parser.add_argument(
        "N",
        type=int,
        help="Either number of files to chunk into, or max number of samples per file. Usage depends on value of --mode.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["samples", "files"],
        default="files",
        help="Whether to chunk by samples or files. Gives interpretation to N.",
    )
    args = parser.parse_args()
    main(args)

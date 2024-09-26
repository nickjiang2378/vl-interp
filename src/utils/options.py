import argparse

from src.caption import CAPTION_ENGINES_CLI


def add_vision_args(parser):
    parser.add_argument("--vision-only", action="store_true", help="Compute vision features only, and save to output-path.")

def add_thresholding_args(parser):
    parser.add_argument(
        "--threshold-type", type=str, default=None, help="Decision rule for threshold-based decoding.",
        choices=["entropy", "vocab"]
    )
    parser.add_argument(
        "--distribution-type", type=str, default="MI", help="Type of equation for logits.",
        choices=["MI", "CAD"]
    )

    # Entropy
    parser.add_argument("--entropy-alpha", type=float, default=0.1, help="If using entropy thresholding model, weight for language-only model outputs.")
    parser.add_argument(
        "--entropy-ratio-threshold", type=float, default=0.5,
        help="If using entropy thresholding model, threshold to switch between original (< threshold) and modified (>= threshold) outputs."
    )
    parser.add_argument("--entropy-topk", type=int, default=-1, help="Top language-only outputs to use. If -1, uses full distribution.")
    parser.add_argument("--entropy-renormalize", action="store_true", help="If using entropy thresholding model, renormalize modifed outputs.")

    # Vocab
    parser.add_argument(
        "--vocab-label-file", type=str, default="./data/grounding_labels/grounding_labels_v0.pth",
        help="Path to .pth file containing dictionary mapping tokens to groundedness labels."
    )

def add_hallucination_args(parser):
    parser.add_argument("--hc-mode", action="store_true", help="Compute word confidences for hallucination evaluation.")
    parser.add_argument("--hc-conf", nargs="*", default=["logit"], help="Confidences to compute.")
    parser.add_argument(
        "--hc-save-attentions",
        action="store_true",
        help="With --hc-mode, save attention tensors with filename `output_json_path`.replace('.json', '_attentions.pth')."
    )
    parser.add_argument("--language-only", action="store_true", help="Save logit distributions from an only-language conditioned model.")
    parser.add_argument("--language-only-sparse", action="store_true", help="If using --language-only, don't save distributions, just statistics (e.g, selected logit value).")

def add_verification_args(parser):
    parser.add_argument("--verifier-type", type=str, default="openclip-ViTG", help="Type of verifier for generation_type = iterative.")
    parser.add_argument("--verifier-threshold", type=float, default=None, help="Threshold for verifier. Positive samples are >= threshold.")

def add_generation_args(parser):
    parser.add_argument("--num-captions", type=int, default=1, help="Number of captions to generate for each image.")
    parser.add_argument("--caption-do-sample", action="store_true", help="If set, sample instead of greedy decoding.")
    parser.add_argument("--caption-nucleus-sampling", action="store_true", help="If set, use nucleus sampling.")
    parser.add_argument("--caption-temperature", type=float, default=1.0, help="Temperature to use if sampling captions.")
    parser.add_argument("--caption-beams", type=int, default=16, help="Number of beams.")
    parser.add_argument("--caption-max-length", type=int, default=24, help="Maximum length of caption.")
    parser.add_argument("--caption-topp", type=float, default=-1, help="Top-p for nucleus sampling, if using --caption-nucleus.")
    parser.add_argument("--generation-type", type=str, default="normal", help="Type of decoding.", choices=["normal", "iterative"])

def add_batch_parser(parser):
    parser.add_argument("--batch-eval", action="store_true", help="Run eval in batches. Provides multi-GPU support.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for batched eval.")
    parser.add_argument("--batch-workers", type=int, default=8, help="Number of workers for batched eval.")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs used.")


def get_main_parser():
    parser = argparse.ArgumentParser("Main")
    parser.add_argument("dataset_json_path", type=str, help="Path to dataset JSON file.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--debug", action="store_true", help="Select debug mode for logging.")
    parser.add_argument(
        "--caption-engine",
        choices=(CAPTION_ENGINES_CLI.keys()),
        default="ofa",
        help="The underlying captioning model to use.",
    )
    parser.add_argument("--caption-key", type=str, default="caption", help="The key to use for the captions.")
    parser.add_argument("--reference-key", type=str, default=None, help="The key to use for the references.")
    parser.add_argument("--image-path-key", type=str, default="image_path", help="The key to use for the image path.")
    parser.add_argument("--prompt-key", type=str, default=None, help="The key to use for prompt inputs, e.g., question in VQA.")
    parser.add_argument("--image-root-dir", type=str, default=None, help="The root directory for the images.")
    parser.add_argument("--overwrite-captions", action="store_true", help="Whether to overwrite the captions if they already exist.")
    parser.add_argument("--return-embeds", action="store_true", help="Return tokens along with caption.")

    parser.add_argument(
        "--pure-llm", action="store_true",
        help="Use only LLM (no Qformer) for language prior."
    )

    # ** Evaluation **
    #parser.add_argument("--clair", action="store_true", help="Compute CLAIR score.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Metrics to compute.")

    # ** Outputs **
    parser.add_argument("--output-path", type=str, default="output.json", help="The path to save the output to.")

    # Logging results
    #parser.add_argument("--output-csv-path", type=str, default=None, help="The path to a CSV that stores metrics for each run. Columns are metric names, as well as `name`, defined by the `name` option. If None, no results are logged.")
    parser.add_argument("--name", type=str, default=None, help="The name of the run. If not provided, will be inferred from the `output_path`. Used as an entry for the `name` column in the CSV.")

    add_generation_args(parser)
    add_hallucination_args(parser)
    add_batch_parser(parser)
    add_thresholding_args(parser)
    add_vision_args(parser)
    add_verification_args(parser)

    return parser




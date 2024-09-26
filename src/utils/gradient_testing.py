import torch
from typing import Any, Dict, List
import json
from PIL import Image
from tqdm import tqdm
import os

from src.caption import CAPTION_ENGINES_CLI



def forward(image, captioner, args, sample):

    prompt = "Write a detailed description."
    inputs = captioner.processor(image, prompt)

    if args.overwrite_captions:
        num_beams = 5
        max_length = 256
        temperature = 1.0
        baseline_caption, inputs_embeds, inputs_query, outputs = captioner.model.generate(
            inputs,
            num_beams=num_beams,
            temperature=temperature,
            max_length=max_length,
            return_dict=True,
            return_embeds=True,
        )
        tokens = outputs[0][0]

    else:
        baseline_caption = sample[args.caption_key]
        if type(baseline_caption) == list:
            baseline_caption = baseline_caption[0]
        with torch.no_grad():
            inputs_query = captioner.model.get_query_embeddings(inputs).float()
        tokens = torch.Tensor(sample["tokens"]).long().to(inputs_query.device)

    print(baseline_caption)

    gradients, inputs_embeds = captioner.take_grads(baseline_caption, inputs_query, tokens)
    gradients = [g.detach().cpu() for g in gradients]
    #gradients = None

    return gradients, tokens.detach().cpu().tolist(), inputs_query.detach().cpu(), inputs_embeds.detach().cpu()


def iterate_samples(samples, captioner, args):
    for sample in tqdm(samples):

        gradients, tokens, queries, inputs_embeds = forward(
            Image.open(os.path.join(args.image_root_dir or ".", f"{sample[args.image_path_key]}")).convert("RGB"),
            captioner,
            args,
            sample
        )
        if "gradients" not in sample:
            sample["gradients"] = gradients
        if "tokens" not in sample:
            sample["tokens"] = tokens
        # if "queries" not in sample:
        #     sample["queries"] = queries
        if "inputs_embeds" not in sample:
            sample["inputs_embeds"] = inputs_embeds

    return samples



def main(args):

    with open(args.dataset_json_path, "r") as f:
        samples: List[Dict[str, Any]] = json.load(f)
        if isinstance(samples, dict):
            samples = samples["samples"]  # type: ignore

    print('Loading model...')
    captioner = CAPTION_ENGINES_CLI["instruct-blip"](
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print('Iterating samples...')
    samples = iterate_samples(samples, captioner, args)

    if args.output_path:
        print('Saving samples...')
        torch.save(samples, args.output_path)


if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--dataset-json-path', type=str,
        default='data/coco/hallucination_annotations/subsets_coco_ofa_hallucinate_db_04-04-23/true_hallucinations.json'
    )
    argparser.add_argument('--output-path', type=str, default=None)
    argparser.add_argument('--image-root-dir', type=str, default='data/coco/val2014')
    argparser.add_argument('--image-path-key', type=str, default='image_id')
    argparser.add_argument('--caption-key', type=str, default='baseline')
    argparser.add_argument('--overwrite-captions', action="store_true")
    main(args=argparser.parse_args())

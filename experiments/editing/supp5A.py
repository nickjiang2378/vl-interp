# Supp 5A: Llava on training images

from experiments.blip_utils import *
from experiments.llava_utils import *
from tqdm import tqdm
import os
import pickle
import torch
import random
import argparse

def main(outfile):
  os.chdir("/home/nickj/vl-hallucination")

  evaluator = pickle.load(open('/home/nickj/vl-hallucination/experiments/chair.pkl', "rb"))
  coco_classes = torch.load("/home/nickj/vl-hallucination/coco_classes.pt")

  vocab_embeddings = torch.load("/home/nickj/vl-hallucination/vocab_embeddings_llava.pt")
  vocabulary = torch.load("/home/nickj/vl-hallucination/vocabulary.pt")

  id_to_token = dict()
  for word in vocabulary:
    id_to_token[vocabulary[word]] = word

  coco_imgs = torch.load("/home/nickj/vl-hallucination/vl_data/train_coco_img_ids.pt")

  # Load the LlaVA model
  model_path = "liuhaotian/llava-v1.5-7b"

  model_name = get_model_name_from_path(model_path)
  tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
  )

  coco_img_to_subtractions = dict()
  # coco_img_to_subtractions = torch.load("/home/nickj/vl-hallucination/experiment6A_results.pt")
  img_count = 0

  for coco_img in tqdm(coco_imgs, desc = "Supp"):
    img_count += 1

    if coco_img in coco_img_to_subtractions:
      continue

    if img_count % 100 == 0:
      torch.save(coco_img_to_subtractions, outfile)

    img_path = coco_img_id_to_path(coco_img, validation = False)

    images_tensor, images, image_sizes = generate_images_tensor(model, img_path, image_processor)

    # Generate the original caption
    original_caption = run_llava_model(model, model_name, images_tensor, image_sizes, tokenizer)

    # Compute the hallucinations
    original_chair_evals = evaluator.compute_hallucinations(coco_img, original_caption)

    coco_img_to_subtractions[coco_img] = {
      "caption": original_caption,
      "chair_evals": original_chair_evals
    }

  torch.save(coco_img_to_subtractions, outfile)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Supp 5A: Llava on training images")
  parser.add_argument('--outfile', type=str, required=True, help="Output file to save the results")
  args = parser.parse_args()
  main(args.outfile)

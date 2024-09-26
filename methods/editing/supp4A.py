# Supp4A: run the training dataset over InstructBLIP

import json
import pickle
from io import BytesIO
import numpy as np
import sys
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import requests
import random
from PIL import Image
from io import BytesIO
import torch
from transformers import LlamaTokenizer
from tqdm import tqdm
import time
import argparse
import os

try:
  from src.caption.instruct_blip_engine import InstructBLIPVicuna7B, InstructBLIPVicuna13B
except:  # first call for some reason fails
  pass

from src.caption.instruct_blip_engine import InstructBLIPVicuna7B, InstructBLIPVicuna13B
from experiments.blip_utils import get_image_embeddings, coco_img_id_to_path, run_model

def main(model_size, outfile):
  os.chdir("/home/nickj/vl-hallucination")

  coco_imgs = torch.load("/home/nickj/vl-hallucination/vl_data/train_coco_img_ids.pt")
  evaluator = pickle.load(open('/home/nickj/vl-hallucination/experiments/chair.pkl', 'rb'))

  if model_size == '7b':
    captioner = InstructBLIPVicuna7B(device="cuda", return_embeds=True)
  elif model_size == '13b':
    captioner = InstructBLIPVicuna13B(device="cuda", return_embeds=True)
  else:
    raise ValueError("Invalid model size. Choose either '7b' or '13b'.")

  progress_bar = tqdm(total=len(coco_imgs), desc="Processing")
  img_to_embeddings = dict()
  count = 0
  for img_id in coco_imgs:
    progress_bar.update(1)
    count += 1
    if img_id in img_to_embeddings:
      continue

    img_path = coco_img_id_to_path(img_id, validation=False)
    image_embeddings = get_image_embeddings(img_path, captioner)

    caption = run_model(image_embeddings, captioner)

    chair_evals = evaluator.compute_hallucinations(img_id, caption)

    img_to_embeddings[img_id] = {
      "image_embeddings": image_embeddings,
      "caption": caption,
      "chair_evals": chair_evals
    }

    if count % 100 == 0:
      torch.save(img_to_embeddings, outfile)

  torch.save(img_to_embeddings, outfile)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run InstructBLIP on the training dataset")
  parser.add_argument('--model_size', type=str, choices=['7b', '13b'], required=True, help="Choose between '7b' or '13b' model size")
  parser.add_argument('--outfile', type=str, required=True, help="Output file to save the results")
  args = parser.parse_args()
  main(args.model_size, args.outfile)

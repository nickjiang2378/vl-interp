# Supp3A: Run InstructBlip7B over the entirety of the validation COCO dataset to get the image embeddings, caption, and CHAIR evals

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

try:
  from src.caption.instruct_blip_engine import InstructBLIPVicuna7B
except: # first call for some reason fails
  pass

from src.caption.instruct_blip_engine import InstructBLIPVicuna7B
from experiments.blip_utils import get_image_embeddings, coco_img_id_to_path
import os

os.chdir("/home/nickj/vl-hallucination")

coco_img_ids = torch.load("/home/nickj/vl-hallucination/coco_img_ids.pt")
evaluator = pickle.load(open('/home/nickj/vl-hallucination/experiments/chair.pkl', 'rb'))

captioner = InstructBLIPVicuna7B(device="cuda", return_embeds=True)

progress_bar = tqdm(total=len(coco_img_ids), desc="Processing")
img_to_embeddings = dict()
# img_to_embeddings = torch.load("/home/nickj/vl-hallucination/supp3A_results.pt")
count = 0
for img_id in coco_img_ids:
  progress_bar.update(1)
  count += 1
  if img_id in img_to_embeddings:
    continue

  img_path = coco_img_id_to_path(img_id)
  image_embeddings = get_image_embeddings(img_path, captioner)

  test_input = {"image": torch.ones(1).to("cuda"), "prompt": captioner.prompt}
  inputs_embeds = image_embeddings[:, :32]
  atts_llm = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)
  out = captioner.model.generate(  # type: ignore
    test_input,
    num_beams=5,
    max_length=200,
    return_embeds=True,
    atts_vision=atts_llm,
    inputs_vision=inputs_embeds, # stick input embeddings from vision here
    return_dict=True
    # pure_llm = True # meaning only text model used
  )

  chair_evals = evaluator.compute_hallucinations(img_id, out[0][0])

  img_to_embeddings[img_id] = {
    "image_embeddings": inputs_embeds,
    "caption": out[0][0],
    "chair_evals": chair_evals
  }

  if count % 10:
    torch.save(img_to_embeddings, "supp3A_results.pt")

torch.save(img_to_embeddings, "supp3A_results.pt")



import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.chdir(os.environ["VL_ROOT_DIR"])

import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch

try:
  from src.caption.instruct_blip_engine import InstructBLIPVicuna7B
except: # first call for some reason fails
  pass

from src.caption.instruct_blip_engine import InstructBLIPVicuna7B
from methods.blip_utils import get_image_embeddings, get_vocab_embeddings_blip, get_hidden_text_embedding, run_blip_model
from methods.utils import coco_img_id_to_name

device = "cuda:0"
captioner = InstructBLIPVicuna7B(device=device, return_embeds=True)

vocab_embeddings = get_vocab_embeddings_blip(captioner)

img_path = os.path.join("./images", coco_img_id_to_name(562150))
img_embeddings = get_image_embeddings(img_path, captioner)

caption = run_blip_model(img_embeddings, captioner)
print(caption)
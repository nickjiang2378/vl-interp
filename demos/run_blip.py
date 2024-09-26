import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.chdir("/home/nickj/vl")

import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from sklearn.decomposition import SparseCoder
from transformers import LlamaTokenizer

try:
  from src.caption.instruct_blip_engine import InstructBLIPVicuna13B, InstructBLIPVicuna7B
except: # first call for some reason fails
  pass

from src.caption.instruct_blip_engine import InstructBLIPVicuna13B, InstructBLIPVicuna7B
from experiments.blip_utils import get_image_embeddings, coco_img_id_to_path, get_vocab_embeddings_blip, get_hidden_text_embedding, run_blip_model

device = "cuda:0"
captioner = InstructBLIPVicuna7B(device=device, return_embeds=True)

def memory_hook(module, input, output):
    print(f"Memory after {module}: {torch.cuda.memory_allocated() / 1024**3} GB")

#for name, layer in captioner.model.named_modules():
#    layer.register_forward_hook(memory_hook)

vocab_embeddings = get_vocab_embeddings_blip(captioner)

img_path = coco_img_id_to_path(60532, validation = False)
img_embeddings = get_image_embeddings(img_path, captioner)

caption = run_blip_model(img_embeddings, captioner)
print(caption)
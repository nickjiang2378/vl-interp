# Experiment 13A: run hallucination reduction over OPERA

import os
import json
from experiments.llava_utils import get_model_name_from_path, load_llava_state, load_pretrained_model
from experiments.blip_utils import get_phrase_embedding, load_blip_state, get_image_embeddings
from tqdm import tqdm
import os
import pickle
from experiments.utils import generate_mass_edit_hook, generate_mass_edit_pre_hook
import torch
import random
import numpy as np

torch.set_grad_enabled(False)

os.chdir(os.environ["VL_ROOT_DIR"])

def coco_img_id_to_path_val(img_id, validation = True):
  file_name = f"COCO_val2014_{(12 - len(str(img_id))) * '0' + str(img_id)}"
  img_path = f"/home/nickj/vl-hallucination/data/coco/val2014/{file_name}.jpg"
  return img_path

def read_jsonl(file_path):
  data = []
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
      data.append(json.loads(line.strip()))
  return data

evaluator = pickle.load(open('./experiments/chair.pkl', "rb"))

def run_main(model_type, img_layer_index, text_layer_index, threshold, path_to_detections, weight = 1, remove_hallucinations = True, remove_gt = False, sample_size = 500, output_file="experiment12A_results.pt"):
  # Process OPERA data
  detections = torch.load(path_to_detections)
  file_path = f"./experiments/long_experiments/opera/{model_type}_opera.jsonl"
  opera_data_raw = read_jsonl(file_path)

  opera_data = dict()
  for row in opera_data_raw:
    img_path = coco_img_id_to_path_val(row["image_id"])
    if not os.path.exists(img_path):
      print(row["image_id"])
      raise Exception("image id from opera not found!")
    opera_data[row["image_id"]] = row

  if model_type == "llava7b":
    # Load the LlaVA model
    loaded_state = load_llava_state(model_type, train = True)
  elif model_type.startswith("blip"):
    loaded_state = load_blip_state(model_type, train = True)
  else:
    raise Exception(f"model type {model_type} not supported")

  vocabulary, vocab_embeddings, tokenizer, data, execute_model, register_hook, hidden_layer_embedding, register_pre_hook, model = loaded_state["vocabulary"], loaded_state["vocab_embeddings"], loaded_state["tokenizer"], loaded_state["data"], loaded_state["execute_model"], loaded_state["register_hook"], loaded_state["hidden_layer_embedding"], loaded_state["register_pre_hook"], loaded_state["model"]

  id_to_token = dict()
  for word in vocabulary:
    id_to_token[vocabulary[word]] = word

  output_file_path = f"./vl_results/{output_file}"
  if os.path.exists(output_file_path):
    results = torch.load(output_file_path)
  else:
    results = dict()

  random.seed(1)

  cache = dict() # Applies for a text layer index

  results = dict()

  img_count = 0
  for coco_img in tqdm(list(opera_data.keys()), desc=f"Experiment 13"):
    img_count += 1
    if img_count % 100 == 0:
      torch.save(results, f"./vl_results/{output_file}")

    # Collect the text embeddings that correspond with the hallucinations
    text_embeddings = []
    for pair in detections[coco_img]["object_detection_results"]:
      caption_word, coco_class = pair
      if detections[coco_img]["object_detection_results"][pair]["pred_hallucination"]:
        if text_layer_index != -1:
          if caption_word not in cache:
            cache[caption_word] = hidden_layer_embedding(caption_word, layer = text_layer_index)
          text_embeddings.append(cache[caption_word])
        else:
          text_embeddings.append(get_phrase_embedding(caption_word, vocab_embeddings, tokenizer))

    if len(text_embeddings) == 0:
      print("Continuing")
      continue

    text_embeddings = torch.stack(text_embeddings, dim = 0)

    image_embeddings = None
    if model_type.startswith("blip"):
      coco_path = coco_img_id_to_path_val(coco_img)
      image_embeddings = get_image_embeddings(coco_path, model)

    # Create a hook that will make the residual embedding orthogonal to these text embeddings
    if img_layer_index != -1:
      if model_type.startswith("llava"):
        edit_embeddings_hook = generate_mass_edit_hook(text_embeddings, start_edit_index=35, end_edit_index=-12, layer=img_layer_index, weight = weight, minimum_size=576)
      else:
        edit_embeddings_hook = generate_mass_edit_hook(text_embeddings, start_edit_index=0, end_edit_index=32, layer=img_layer_index, weight = weight, minimum_size=32)
      hook = register_hook(edit_embeddings_hook, layer = img_layer_index)
    else:
      if model_type.startswith("llava"):
        edit_embeddings_hook = generate_mass_edit_pre_hook(text_embeddings, start_edit_index=35, end_edit_index=-12, layer=-1, weight = weight, minimum_size=576)
      else:
        edit_embeddings_hook = generate_mass_edit_pre_hook(text_embeddings, start_edit_index=0, end_edit_index=32, layer=-1, weight = weight, minimum_size=32)
      hook = register_pre_hook(edit_embeddings_hook, layer = 0)

    # Rerun the model with the hooks enabled
    new_caption = execute_model(coco_img, image_embeddings)

    # Compute the hallucinations
    new_chair_eval = evaluator.compute_hallucinations(coco_img, new_caption)

    # Store the chair results
    results[coco_img] = {
      "caption": new_caption,
      "chair_evals": new_chair_eval
    }

    # Remove the hook
    hook.remove()

  torch.save(results, f"./vl_results/{output_file}")

import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="Experiment 12A")
  parser.add_argument('--model_type', type=str, required=True, help='Type of model to use (e.g., llava7b, blip13b)')
  parser.add_argument('--img_layer_index', type=int, required=True, help='Index of the image layer to edit')
  parser.add_argument('--text_layer_index', type=int, required=True, help='Index of the text layer to edit')
  parser.add_argument('--weight', type=float, help='weight for the edit hook')
  parser.add_argument('--threshold', type=float, required=True, help='Threshold value for the experiment')
  parser.add_argument('--path_to_detections', type=str, required=True, help='Path to the detections file')

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  model_type = args.model_type
  img_layer_index = args.img_layer_index
  text_layer_index = args.text_layer_index
  weight = args.weight
  threshold = args.threshold
  path_to_detections = args.path_to_detections

  output_file = f"experiment12A_{model_type}_il_{img_layer_index}_tl_{text_layer_index}_w_{weight}_th_{threshold}.pt"
  print(f"Executing with parameters: model_type={model_type}, img_layer_index={img_layer_index}, text_layer_index={text_layer_index}, weight={weight}, threshold={threshold}, path_to_detections={path_to_detections}, output_file={output_file}")
  run_main(model_type=model_type, threshold=threshold, img_layer_index=img_layer_index, text_layer_index=text_layer_index, weight=weight, path_to_detections=path_to_detections, output_file=output_file)

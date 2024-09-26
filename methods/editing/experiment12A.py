# Experiment 12A: perform weight ablations

from experiments.llava_utils import get_model_name_from_path, load_llava_state, load_pretrained_model
from experiments.blip_utils import get_phrase_embedding, load_blip_state
from tqdm import tqdm
import os
import pickle
from experiments.utils import generate_mass_edit_hook, generate_mass_edit_pre_hook
import torch
import random
import numpy as np

torch.set_grad_enabled(False)

os.chdir(os.environ["VL_ROOT_DIR"])

evaluator = pickle.load(open('./experiments/chair.pkl', "rb"))

def run_main(model_type, img_layer_index, text_layer_index, start_weight=1, end_weight=10, step_size = 1.0, sample_hallucinations = True, remove_hallucinations = True, remove_gt = False, sample_size = 500, output_file="experiment12A_results.pt"):
  if model_type == "llava7b":
    # Load the LlaVA model
    loaded_state = load_llava_state(model_type, train = True)
  elif model_type.startswith("blip"):
    loaded_state = load_blip_state(model_type, train = True)
  else:
    raise Exception(f"model type {model_type} not supported")

  vocabulary, vocab_embeddings, tokenizer, data, execute_model, register_hook, hidden_layer_embedding, register_pre_hook = loaded_state["vocabulary"], loaded_state["vocab_embeddings"], loaded_state["tokenizer"], loaded_state["data"], loaded_state["execute_model"], loaded_state["register_hook"], loaded_state["hidden_layer_embedding"], loaded_state["register_pre_hook"]

  id_to_token = dict()
  for word in vocabulary:
    id_to_token[vocabulary[word]] = word

  output_file_path = f"./vl_results/{output_file}"
  if os.path.exists(output_file_path):
    results = torch.load(output_file_path)
  else:
    results = dict()

  random.seed(1)

  if sample_hallucinations:
    # Find coco_imgs that have hallucinations
    hallucinated_imgs = []
    for coco_img in data:
      if len(data[coco_img]["chair_evals"]["mscoco_hallucinated_words"]) > 0:
        hallucinated_imgs.append(coco_img)
    sampled_coco_img_ids = random.sample(hallucinated_imgs, min(sample_size, len(hallucinated_imgs)))

    if sample_size > len(hallucinated_imgs):
      print(f"[Warning]: Sample size {sample_size} is greater than the number of images with hallucinations ({len(hallucinated_imgs)})")
  else:
    sampled_coco_img_ids = random.sample(list(data.keys()), sample_size)

  cache = dict() # Applies for a text layer index
  for weight in np.arange(start_weight, end_weight, step_size): # hardcoded for now

    results[weight] = dict()

    img_count = 0
    for coco_img in tqdm(sampled_coco_img_ids, desc=f"Experiment 12A: weight ablations (weight {weight})"):
      img_count += 1
      if img_count % 100 == 0:
        torch.save(results, f"./vl_results/{output_file}")

      # Collect the text embeddings that correspond with the hallucinations
      text_embeddings = []
      if remove_hallucinations:
        for caption_word, coco_class in set(data[coco_img]["chair_evals"]["mscoco_hallucinated_words"]):
          if text_layer_index != -1:
            if caption_word not in cache:
              cache[caption_word] = hidden_layer_embedding(caption_word, layer = text_layer_index)
            text_embeddings.append(cache[caption_word])
          else:
            text_embeddings.append(get_phrase_embedding(caption_word, vocab_embeddings, tokenizer))

      if remove_gt:
        for caption_word, coco_class in set(data[coco_img]["chair_evals"]["recall_words"]):
          if text_layer_index == -1:
            if caption_word not in cache:
              cache[caption_word] = hidden_layer_embedding(caption_word, layer = text_layer_index)
            text_embeddings.append(cache[caption_word])
          else:
            text_embeddings.append(get_phrase_embedding(caption_word, vocab_embeddings, tokenizer))

      if len(text_embeddings) == 0:
        print("Continuing")
        continue

      text_embeddings = torch.stack(text_embeddings, dim = 0)

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
      new_caption = execute_model(coco_img)

      # Compute the hallucinations
      new_chair_eval = evaluator.compute_hallucinations(coco_img, new_caption)

      # Store the chair results
      results[weight][coco_img] = {
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
  parser.add_argument('--start_weight', type=float, help='Starting weight for the edit hook')
  parser.add_argument('--end_weight', type=float, help='Ending weight for the edit hook')
  parser.add_argument('--step_size', type=float, default=1.0, help='Step size for the weight increments')
  parser.add_argument('--sample_hallucinations', action='store_true', default=True, help='Sample only images with hallucinations (default: True)')
  parser.add_argument('--hallucinations', action='store_true', default=False, help='Remove hallucinations (default: True)')
  parser.add_argument('--gt', action='store_true', default=False, help='Remove ground truth (default: False)')
  parser.add_argument('--sample_size', type=int, default=500, help='Number of images to sample')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  model_type = args.model_type
  img_layer_index = args.img_layer_index
  text_layer_index = args.text_layer_index
  start_weight = args.start_weight
  end_weight = args.end_weight
  step_size = args.step_size
  sample_hallucinations = args.sample_hallucinations
  remove_hallucinations = args.hallucinations
  remove_gt = args.gt

  assert remove_hallucinations or remove_gt, "either hallucinations or gt should be true"
  sample_size = args.sample_size
  output_file = f"experiment12A_{model_type}_il_{img_layer_index}_tl_{text_layer_index}_sw_{start_weight}_ew_{end_weight}_ss_{step_size}_sz_{sample_size}_sh_{sample_hallucinations}_rh_{remove_hallucinations}_rg_{remove_gt}.pt"
  print(f"Executing with parameters: model_type={model_type}, img_layer_index={img_layer_index}, text_layer_index={text_layer_index}, start_weight={start_weight}, end_weight={end_weight}, step_size={step_size}, sample_hallucinations={sample_hallucinations}, remove_hallucinations={remove_hallucinations}, remove_gt={remove_gt}, sample_size={sample_size}, output_file={output_file}")
  run_main(model_type=model_type, img_layer_index=img_layer_index, text_layer_index=text_layer_index, start_weight=start_weight, end_weight=end_weight, step_size=step_size, sample_hallucinations=sample_hallucinations, remove_hallucinations=remove_hallucinations, remove_gt=remove_gt, sample_size=sample_size, output_file=output_file)

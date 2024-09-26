# Experiment 10A: Mass removing hallucinations (4.1.2) on Llava

from experiments.llava_utils import get_model_name_from_path, load_llava_state, load_pretrained_model
from experiments.blip_utils import get_phrase_embedding, load_blip_state
from tqdm import tqdm
import os
import pickle
from experiments.utils import generate_mass_edit_hook, generate_mass_edit_pre_hook
import torch
import random

torch.set_grad_enabled(False)

os.chdir(os.environ["VL_ROOT_DIR"])

evaluator = pickle.load(open('./experiments/chair.pkl', "rb"))
coco_classes = torch.load("./coco_classes.pt")

coco_img_ids = torch.load("./vl_data/llava7b_train_ids_with_hallucinations.pt")

def run_main(model_type, remove_hallucinations = True, remove_gt = False, weight_factor = 1, sample_size = 500, output_file="experiment10A_results.pt"):
  if model_type == "llava7b":
    # Load the LlaVA model
    loaded_state = load_llava_state(model_type, train = True)
  elif model_type.startswith("blip"):
    loaded_state = load_blip_state(model_type, train = True)
  else:
    raise Exception(f"model type {model_type} not supported")

  vocabulary, vocab_embeddings, data, execute_model, register_pre_hook, tokenizer = loaded_state["vocabulary"], loaded_state["vocab_embeddings"], loaded_state["data"], loaded_state["execute_model"], loaded_state["register_pre_hook"], loaded_state["tokenizer"]

  id_to_token = dict()
  for word in vocabulary:
    id_to_token[vocabulary[word]] = word

  output_file_path = f"./vl_results/{output_file}"
  if os.path.exists(output_file_path):
    results = torch.load(output_file_path)
  else:
    results = dict()

  random.seed(1)
  sampled_coco_img_ids = random.sample(list(data.keys()), sample_size)

  img_count = 0
  cache = dict()
  for coco_img in tqdm(sampled_coco_img_ids, desc=f"Experiment 10A: mass editing"):
    img_count += 1
    if img_count % 100 == 0:
      torch.save(results, f"./vl_results/{output_file}")

    if coco_img in results:
      continue

    # Collect the text embeddings that correspond with the hallucinations
    text_embeddings = []
    if remove_hallucinations:
      for caption_word, coco_class in set(data[coco_img]["chair_evals"]["mscoco_hallucinated_words"]):
        text_embeddings.append(get_phrase_embedding(caption_word, vocab_embeddings, tokenizer))

    if remove_gt:
      for caption_word, coco_class in set(data[coco_img]["chair_evals"]["recall_words"]):
        text_embeddings.append(get_phrase_embedding(caption_word, vocab_embeddings, tokenizer))

    if len(text_embeddings) == 0:
      print("Continuing")
      continue

    text_embeddings = torch.stack(text_embeddings, dim = 0)

    # Create a hook that will make the residual embedding orthogonal to these text embeddings
    if model_type.startswith("llava"):
      edit_embeddings_hook = generate_mass_edit_pre_hook(text_embeddings, start_edit_index=35, end_edit_index=-12, layer=0, weight = weight_factor, minimum_size=576)
    else:
      edit_embeddings_hook = generate_mass_edit_pre_hook(text_embeddings, start_edit_index=0, end_edit_index=32, layer=0, weight = weight_factor, minimum_size=32)
    hook = register_pre_hook(edit_embeddings_hook, layer = 0)

    # Rerun the model with the hooks enabled
    new_caption = execute_model(coco_img)

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
  parser = argparse.ArgumentParser(description="Experiment 10A")
  parser.add_argument('--model_type', type=str, required=True, help='Type of model to use (e.g., llava7b, blip13b)')
  parser.add_argument('--weight', type=int, default=1, help='Weight factor for the edit hook')
  parser.add_argument('--sample_size', type=int, default=5000, help='Number of images to sample')
  parser.add_argument('--experiment_name', type=str, default='', help='Output file name for saving results')
  parser.add_argument('--hallucinations', action='store_true', default=False, help='Include hallucinations (default: True)')
  parser.add_argument('--gt', action='store_true', default=False, help='Include ground truth (default: False)')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  model_type = args.model_type
  weight = args.weight
  sample_size = args.sample_size
  experiment_name = args.experiment_name
  hallucinations = args.hallucinations
  gt = args.gt
  assert hallucinations or gt, "Either hallucinations or gt must be true!"
  output_file = f"experiment10A_{model_type}_w{weight}_ss{sample_size}_h{hallucinations}_gt{gt}_{experiment_name}.pt"
  print(f"Executing with parameters: model_type={model_type}, weight_factor={weight}, sample_size={sample_size}, hallucinations={hallucinations}, gt={gt}, output_file={output_file}")
  run_main(model_type=model_type, weight_factor=weight, sample_size=sample_size, remove_hallucinations=hallucinations, remove_gt=gt, output_file=output_file)

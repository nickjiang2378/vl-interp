# Experiment 11A: Ablate along every image and text hidden layer

from experiments.llava_utils import get_model_name_from_path, load_llava_state, load_pretrained_model
from experiments.blip_utils import get_phrase_embedding, load_blip_state
from tqdm import tqdm
import os
import pickle
from experiments.utils import generate_mass_edit_hook
import torch
import random

torch.set_grad_enabled(False)

os.chdir(os.environ["VL_ROOT_DIR"])

evaluator = pickle.load(open('./experiments/chair.pkl', "rb"))

def run_main(model_type, start_img_layer, end_img_layer, step_size = 1, diagonal = False, sample_hallucinations = True, remove_hallucinations = True, remove_gt = False, weight_factor = 1, sample_size = 500, output_file="experiment10A_results.pt"):
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

  for text_layer_index in range(-1, 31, step_size): # hardcoded for now
    cache = dict() # Applies for a text layer index
    for img_layer_index in range(start_img_layer, end_img_layer, step_size):
      if diagonal and img_layer_index != text_layer_index:
        continue
      if img_layer_index not in results:
        results[img_layer_index] = dict()
      if text_layer_index not in results[img_layer_index]:
        results[img_layer_index][text_layer_index] = dict()

      ablation_results = results[img_layer_index][text_layer_index]

      img_count = 0
      for coco_img in tqdm(sampled_coco_img_ids, desc=f"Experiment 11A: mass editing (img {img_layer_index}, text {text_layer_index})"):
        img_count += 1
        if img_count % 100 == 0:
          torch.save(results, f"./vl_results/{output_file}")

        if coco_img in ablation_results:
          continue

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
        ablation_results[coco_img] = {
          "caption": new_caption,
          "chair_evals": new_chair_eval
        }

        # Remove the hook
        hook.remove()

      results[img_layer_index][text_layer_index] = ablation_results

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
  parser.add_argument('--start_img_layer', type=int, required=True, help='Start index for image layer')
  parser.add_argument('--end_img_layer', type=int, required=True, help='End index for image layer (-1 for last layer)')
  parser.add_argument('--sample_only_hallucinations', action='store_true', default=True, help='Sample only images with hallucinations (default: True)')
  parser.add_argument('--diagonal', action='store_true', default=False, help='Run only on layers where image and text layer indices are the same')
  parser.add_argument('--step_size', type=int, default=1, help='Step size for iterating through layers')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  model_type = args.model_type
  weight = args.weight
  sample_size = args.sample_size
  experiment_name = args.experiment_name
  hallucinations = args.hallucinations
  gt = args.gt
  start_img_layer = args.start_img_layer
  end_img_layer = args.end_img_layer
  sample_only_hallucinations = args.sample_only_hallucinations
  diagonal = args.diagonal
  step_size = args.step_size
  assert hallucinations or gt, "Either hallucinations or gt must be true!"
  output_file = f"experiment11_{model_type}_is_{start_img_layer}_ie_{end_img_layer}_w{weight}_ss{sample_size}_h{hallucinations}_gt{gt}_soh{sample_only_hallucinations}_{experiment_name}.pt"
  print(f"Executing with parameters: model_type={model_type}, weight_factor={weight}, sample_size={sample_size}, hallucinations={hallucinations}, gt={gt}, start_img_layer={start_img_layer}, end_img_layer={end_img_layer}, sample_only_hallucinations={sample_only_hallucinations}, diagonal={diagonal}, step_size={step_size}, output_file={output_file}")
  run_main(model_type=model_type, weight_factor=weight, sample_size=sample_size, remove_hallucinations=hallucinations, remove_gt=gt, output_file=output_file, start_img_layer=start_img_layer, end_img_layer=end_img_layer, sample_hallucinations=sample_only_hallucinations, diagonal=diagonal, step_size=step_size)

# Experiment 8A: try the remove hallucinations and preserve recall method on Llava

from experiments.llava_utils import get_model_name_from_path, load_pretrained_model, hidden_layer_embedding, generate_remove_h_preserve_recall_hook, generate_images_tensor, run_llava_model
from experiments.blip_utils import coco_img_id_to_path
from tqdm import tqdm
import os
import pickle
import torch
import random

os.chdir("/home/nickj/vl-hallucination")

evaluator = pickle.load(open('/home/nickj/vl-hallucination/experiments/chair.pkl', "rb"))
coco_classes = torch.load("/home/nickj/vl-hallucination/coco_classes.pt")

vocab_embeddings = torch.load("/home/nickj/vl-hallucination/vocab_embeddings_llava.pt")
vocabulary = torch.load("/home/nickj/vl-hallucination/vocabulary.pt")

id_to_token = dict()
for word in vocabulary:
  id_to_token[vocabulary[word]] = word

coco_img_ids = torch.load("/home/nickj/vl-hallucination/llava7B_coco_imgs_with_hallucinations.pt")

data = torch.load("/home/nickj/vl-hallucination/llava7B_coco.pt")

# Load the LlaVA model
model_path = "liuhaotian/llava-v1.5-7b"

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
  model_path, None, model_name
)
results = dict()
# results = torch.load("/home/nickj/vl-hallucination/experiment8A_results.pt")

def run_main(weight_factor = 1, sample_size = 500, output_file="experiment8A_results.pt"):
  random.seed(1)
  sampled_coco_img_ids = random.sample(coco_img_ids, sample_size)

  img_count = 0
  cache = dict()
  for coco_img in tqdm(sampled_coco_img_ids, desc=f"Experiment 8A"):
    img_count += 1
    if img_count % 100 == 0:
      torch.save(results, f"/home/nickj/vl-hallucination/{output_file}")

    if coco_img in results:
      continue

    # Collect the text embeddings that correspond with the hallucinations
    h_embeddings = []
    for caption_word, coco_class in set(data[coco_img]["hallucinations"]["mscoco_hallucinated_words"]):
      if caption_word in cache:
        h_embeddings.append(cache[caption_word])
      else:
        text_embedding = hidden_layer_embedding(caption_word, model, vocab_embeddings, tokenizer, layer=0)
        h_embeddings.append(text_embedding)
        cache[caption_word] = text_embedding

    recall_embeddings = []
    for recall_word, coco_class in set(data[coco_img]["hallucinations"]["recall_words"]):
      if recall_word in cache:
        recall_embeddings.append(cache[recall_word])
      else:
        text_embedding = hidden_layer_embedding(recall_word, model, vocab_embeddings, tokenizer, layer=0)
        recall_embeddings.append(text_embedding)
        cache[recall_word] = text_embedding

    if len(h_embeddings) == 0 or len(recall_embeddings) == 0:
      print("Continuing")
      continue

    h_embeddings = torch.concat(h_embeddings, dim = 0)
    recall_embeddings = torch.concat(recall_embeddings, dim = 0)
    # Create a hook that will make the residual embedding orthogonal to these text embeddings
    edit_embeddings_hook = generate_remove_h_preserve_recall_hook(recall_embeddings, h_embeddings, start_edit_index=35, end_edit_index=-12, layer=0, weight = weight_factor)
    hook = model.get_model().layers[0].register_forward_hook(edit_embeddings_hook)

    # Rerun the model with the hooks enabled
    img_path = coco_img_id_to_path(coco_img)
    images_tensor, images, image_sizes = generate_images_tensor(model, img_path, image_processor)

    # Generate the new caption
    new_caption = run_llava_model(model, model_name, images_tensor, image_sizes, tokenizer)

    # Compute the hallucinations
    new_chair_eval = evaluator.compute_hallucinations(coco_img, new_caption)

    # Store the chair results
    results[coco_img] = {
      "caption": new_caption,
      "chair_evals": new_chair_eval
    }

    # Remove the hook
    hook.remove()

  torch.save(results, f"/home/nickj/vl-hallucination/{output_file}")

import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="Experiment 8A")
  parser.add_argument('--weight', type=int, default=5, help='Weight factor for the edit hook')
  parser.add_argument('--sample_size', type=int, default=len(coco_img_ids), help='Number of images to sample')
  parser.add_argument('--output_file', type=str, default='experiment8A_results.pt', help='Output file name for saving results')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  weight = args.weight
  sample_size = args.sample_size
  output_file = args.output_file
  run_main(weight_factor=weight, sample_size=sample_size, output_file=output_file)

# Supp 1C: Compute the heatmaps for ~500 images in the COCO dataset (the same as in supp1A) using cosine similarities as the metric

import torch
from experiments.blip_utils import get_heatmap, InstructBLIPVicuna13B, reshape_hidden_states, run_model
from tqdm import tqdm
import pickle
import os

os.chdir("/home/nickj/vl-hallucination")

data = torch.load("/home/nickj/vl-hallucination/BLIP13B_coco.pt")
supp1A = torch.load("/home/nickj/vl-hallucination/supp1A_results.pt")
supp1A_img_ids = list(supp1A.keys())

cos = torch.nn.CosineSimilarity(dim = 1)
# img_to_cs_heatmap = dict()
img_to_cs_heatmap = torch.load("/home/nickj/vl-hallucination/supp1C_results.pt")

evaluator = pickle.load(open('/home/nickj/vl-hallucination/experiments/chair.pkl', "rb"))
captioner = InstructBLIPVicuna13B(device="cuda", return_embeds=True)

vocab_embeddings = torch.load("/home/nickj/vl-hallucination/vocab_embeddings.pt")
vocabulary = torch.load("/home/nickj/vl-hallucination/vocabulary.pt")
id_to_token = dict()
for word in vocabulary:
    id_to_token[vocabulary[word]] = word

progress_bar = tqdm(total=len(supp1A_img_ids), desc="Supp1C Processing")
cnt = 0
for img_id in supp1A_img_ids:
    cnt += 1
    progress_bar.update(1)
    if img_id in img_to_cs_heatmap:
        continue

    image_embeddings = data[img_id]["image_embeddings"]
    # out = run_model(captioner, image_embeddings[:, :32])
    # hidden_states = reshape_hidden_states(out[4].hidden_states)
    word_to_cs = []
    for caption_word in supp1A[img_id]:
        # Get heatmap for original image embeddings
        inputs_embeds = image_embeddings[:, :32]

        # Get the original heatmap
        heatmap, out = get_heatmap(inputs_embeds, caption_word, captioner, vocab_embeddings)
        word_to_cs.append([heatmap, caption_word, out[0][0]])

    img_to_cs_heatmap[img_id] = word_to_cs

    if cnt % 10 == 0:
        torch.save(img_to_cs_heatmap, "supp1C_results.pt")

if cnt % 10 == 0:
    torch.save(img_to_cs_heatmap, "supp1C_results.pt")

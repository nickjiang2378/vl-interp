# Supp 2A: Extract the summed normalized cosine similarities > 0.9 for the coco images

import torch
import pickle
from tqdm import tqdm
from experiments.blip_utils import *
import os

os.chdir("/home/nickj/vl-hallucination")
os.environ["CUDA_VISIBLE_DEVICES"]="2"

print(torch.cuda.current_device())

data = torch.load("/home/nickj/vl-hallucination/hallucination_data.pt")
coco_classes = torch.load("coco_classes.pt")

evaluator = pickle.load(open('/home/nickj/vl-hallucination/experiments/chair.pkl', "rb"))

img_count = 0
avg_recall_total = 0
avg_h_total = 0
avg_h_count = 0

imgs_to_cs_scores = dict()

progress_bar = tqdm(total=len(data), desc="Processing")
for img_id in data:
    img_count += 1
    progress_bar.update(1)

    eval = evaluator.compute_hallucinations(img_id, data[img_id]["caption"])
    gt_classes = eval["mscoco_gt_words"]
    hallucinated_classes = [pair[1] for pair in eval["mscoco_hallucinated_words"]]

    interesting_words = []
    caption_to_score = dict()
    for caption_word, coco_class in eval["recall_words"]:
        interesting_words.append((caption_word, coco_class, True))
    for caption_word, coco_class in eval["mscoco_hallucinated_words"]:
        interesting_words.append((caption_word, coco_class, False))
    for coco_class in coco_classes:
        if coco_class not in gt_classes and coco_classes not in hallucinated_classes:
            interesting_words.append((coco_class, coco_class, False))

    # Compute the min and max cosine similarities
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    ranges = []
    for i in range(32):
        cosine_similarity = cos(data[img_id]["image_embeddings"][:, i], vocab_embeddings.squeeze(0))
        ranges.append([torch.min(cosine_similarity), torch.max(cosine_similarity)])

    # Find the average cosine similarities normalized
    for caption_word, coco_class, is_gt in interesting_words:
        selected_vocab_embeddings = []
        selected_vocab_embeddings.append(get_phrase_embedding(caption_word, vocab_embeddings))
        selected_vocab_embeddings = torch.cat(selected_vocab_embeddings, dim = 0)

        total = 0
        for i in range(32):
            cosine_similarity = cos(data[img_id]["image_embeddings"][:, i], selected_vocab_embeddings)
            normalized_sim = (float(cosine_similarity[0]) - ranges[i][0]) / (ranges[i][1] - ranges[i][0])
            if normalized_sim > 0.9:
                total += normalized_sim
        caption_to_score[caption_word] = (total, coco_class, is_gt)

    imgs_to_cs_scores[img_id] = caption_to_score

    if img_count % 100 == 0:
        torch.save(imgs_to_cs_scores, "supp1B.pt")

torch.save(imgs_to_cs_scores, "supp1B.pt")
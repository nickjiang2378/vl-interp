# Supplemental method 1A: for each image, compute a set of embeddings based on cosine similarity such that when you zero out the embeddings, the caption loses the word

try:
  from src.caption.instruct_blip_engine import InstructBLIPVicuna13B
except: # first call for some reason fails
  pass

from src.caption.instruct_blip_engine import InstructBLIPVicuna13B
from experiments.blip_utils import seq_to_tokens
import torch
import pickle
from tqdm import tqdm
import os
import random

os.chdir("/home/nickj/vl-hallucination")

captioner = InstructBLIPVicuna13B(device="cuda", return_embeds=True)
evaluator = pickle.load(open('/home/nickj/vl-hallucination/experiments/chair.pkl', "rb"))
data = torch.load("/home/nickj/vl-hallucination/hallucination_data.pt")

vocab_embeddings = torch.load("/home/nickj/vl-hallucination/vocab_embeddings.pt")
vocabulary = torch.load("/home/nickj/vl-hallucination/vocabulary.pt")
id_to_token = dict()
for word in vocabulary:
    id_to_token[vocabulary[word]] = word

def check_if_word_exists(remove_indices, image_embeddings, target_word, img_id):
    test_input = {"image": torch.ones(1).to("cuda"), "prompt": captioner.prompt}
    inputs_embeds = image_embeddings.clone()
    for i in remove_indices:
        inputs_embeds[:, i] = 0

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
    evals = evaluator.compute_hallucinations(img_id, out[0][0])

    word_pairs = evals["recall_words"] + evals["mscoco_hallucinated_words"]
    for word, coco_class in word_pairs:
        if word == target_word:
            return True
    return False

img_to_embed_sets = dict()
img_count = 0
progress_bar = tqdm(total=len(data), desc="Supp1A Processing")
rand_img_ids = random.choices(list(data.keys()), k=500)
for img_id in rand_img_ids:
    img_count += 1
    if img_count > 500:
        break
    progress_bar.update(1)

    image_embeddings = data[img_id]["image_embeddings"]
    evals = evaluator.compute_hallucinations(img_id, data[img_id]["caption"])

    word_pairs = evals["recall_words"] + evals["mscoco_hallucinated_words"]

    embed_sets = dict() # maps word to minimum set of embeddings that contain this concept

    for caption_word, coco_class in word_pairs:
        if caption_word in embed_sets:
            continue
        # use binary search to find the minimum set of embeddings that remove the target word from the caption
        start = 0
        end = 31

        token = seq_to_tokens([caption_word], id_to_token, only_first = True)[0]

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(image_embeddings[:, :32].squeeze(0), vocab_embeddings[:, vocabulary[token]])
        sorted_sim, ind_sim = torch.sort(cosine_similarity, descending=True)

        iters = 0 # just in case binary search goes infinitely
        while start < end and iters < 7:
            iters += 1
            mid = (start + end) // 2
            removes_word_from_caption = not check_if_word_exists(ind_sim[:mid+1], image_embeddings, caption_word, img_id)

            if removes_word_from_caption:
                end = mid
            else:
                start = mid + 1
        if start != end:
            print(f"Start not eeqla to end on {caption_word}, {start, end} for img {img_id}")
        embed_sets[caption_word] = ind_sim[:end+1]

    img_to_embed_sets[img_id] = embed_sets

    if img_count % 10 == 0:
        torch.save(img_to_embed_sets, "supp1A_results.pt")

torch.save(img_to_embed_sets, "supp1A_results.pt")






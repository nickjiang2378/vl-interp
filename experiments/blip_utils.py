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
from experiments.utils import subtract_projections

# captioner = InstructBLIPVicuna13B(device="cuda:1", return_embeds=True)

llm_tokenizer = LlamaTokenizer.from_pretrained(
            "/home/spetryk/large_model_checkpoints/llm/vicuna-7b-v1.1", use_fast=False, truncation_side="left"
        )
TOKEN_UNDERSCORE = chr(9601) # a lot of tokens have _rain, etc. but the underscore in front is not normal

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def get_image_embeddings(image_file, captioner):
    image = load_image(image_file)
    baseline_caption, inputs_embeds, inputs_query, outputs = captioner(image, prompt=captioner.prompt, return_embeds=True)
    image_embeddings = inputs_query
    # image_embeddings = image_embeddings / torch.norm(image_embeddings)
    return image_embeddings

def coco_img_id_to_path(img_id, validation = True):
    if validation:
        file_name = f"COCO_val2014_{(12 - len(str(img_id))) * '0' + str(img_id)}"
        img_path = f"/home/nickj/vl-hallucination/vl_data/val2014/{file_name}.jpg"
        return img_path
    else:
        file_name = f"COCO_train2014_{(12 - len(str(img_id))) * '0' + str(img_id)}"
        img_path = f"/home/nickj/vl-hallucination/vl_data/train2014/{file_name}.jpg"
        return img_path

def get_vocab_embeddings_blip(captioner, device="cuda"):
    vocab = captioner.tokenizer.get_vocab()
    llm_tokens = torch.tensor(list(vocab.values()), dtype=torch.long).unsqueeze(0).to(device)
    token_embeddings = captioner.model.llm_model.get_input_embeddings()(llm_tokens)
    # token_embeddings = token_embeddings / torch.norm(token_embeddings)
    return token_embeddings.to(device)

def lasso_decomposition(image_embedding, vocab_embeddings, vocabulary, l1_penalty = 3, device = "cuda:0"):
    """
    Runs LASSO decomposition on the provided image embedding,
    returning a stats summary as well as the decomposition sorted by decreasing weight magnitude

    image_embedding: (1, D), where D = dimension of each image embedding (ex. 5120)
    vocab_embeddings: (V, D), where V = vocab size
    vocabulary: Dict mapping vocab token to index into vocab_embeddings
    l1_penalty: higher value -> increased sparsity
    """

    # Use the vocab embeddings as the atomic representation
    A = vocab_embeddings.squeeze(0).T.to(torch.float64).to(device)

    # Invert vocabulary mapping
    id_to_token = dict()
    for key in vocabulary:
        id_to_token[vocabulary[key]] = key

    # Run sparse coder
    coder = SparseCoder(dictionary = A.T.to("cpu"), transform_algorithm="lasso_cd", transform_alpha = l1_penalty)
    weights = coder.transform(image_embedding.to("cpu"))
    weights_cuda = torch.from_numpy(weights).to(device)

    top_tokens = []
    for j in range(weights.shape[1]):
        top_tokens.append((weights[0][j], id_to_token[j]))

    top_tokens.sort(key = lambda e: -abs(e[0]))

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    reconstruction = A @ weights_cuda.T

    stats = {
        "cosine_similarity": float(cos(reconstruction.squeeze(1), image_embedding.squeeze(0))),
        "nonzero": np.sum(weights > 0),
        "decomposition": top_tokens,
        "reconstruction": reconstruction
    }

    return stats

def lasso_decomposition_no_cuda(image_embedding, vocab_embeddings, vocabulary, l1_penalty = 3):
    """
    Runs LASSO decomposition on the provided image embedding,
    returning a stats summary as well as the decomposition sorted by decreasing weight magnitude

    image_embedding: (1, D), where D = dimension of each image embedding (ex. 5120)
    vocab_embeddings: (V, D), where V = vocab size
    vocabulary: Dict mapping vocab token to index into vocab_embeddings
    l1_penalty: higher value -> increased sparsity
    """

    # Use the vocab embeddings as the atomic representation
    A = vocab_embeddings.squeeze(0).T.to(torch.float64).to("cpu")

    image_embedding = image_embedding.to("cpu")

    # Invert vocabulary mapping
    id_to_token = dict()
    for key in vocabulary:
        id_to_token[vocabulary[key]] = key

    # Run sparse coder
    coder = SparseCoder(dictionary = A.T, transform_algorithm="lasso_cd", transform_alpha = l1_penalty)
    weights = coder.transform(image_embedding)
    weights_cuda = torch.from_numpy(weights)

    top_tokens = []
    for j in range(weights.shape[1]):
        top_tokens.append((weights[0][j], id_to_token[j]))

    top_tokens.sort(key = lambda e: -abs(e[0]))

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    reconstruction = A @ weights_cuda.T

    stats = {
        "cosine_similarity": float(cos(reconstruction.squeeze(1), image_embedding.squeeze(0))),
        "nonzero": np.sum(weights > 0),
        "decomposition": top_tokens,
        "reconstruction": reconstruction
    }

    return stats

def seq_to_tokens(strings, id_to_token, tokenizer = llm_tokenizer, remove_first = True, only_first = False, remove_underscore=True):
    search_tokens = []
    for string in strings:
        start_ind = 1 if remove_first else 0
        for id in tokenizer(string)["input_ids"][start_ind:]:
            if id not in search_tokens:
                search_tokens.append(id_to_token[id])
                if remove_underscore and id_to_token[id][0] == TOKEN_UNDERSCORE and id_to_token[id][1:] in id_to_token.values():
                    search_tokens.append(id_to_token[id][1:])

            if only_first:
                break

    return search_tokens

def vocab_similarity(token1, token2, vocabulary, vocab_embeddings):
    ind1, ind2 = vocabulary[token1], vocabulary[token2]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return float(cos(vocab_embeddings[:, ind1], vocab_embeddings[:, ind2])[0])

def get_phrase_embedding(phrase, vocab_embeddings, tokenizer = llm_tokenizer, remove_first = True):
    # returns size (1, 5120)
    text_embeddings = []
    for token_id in tokenizer(phrase)["input_ids"]:
        text_embeddings.append(vocab_embeddings[:, token_id])
    if remove_first:
        text_embeddings = text_embeddings[1:]
    phrase_embedding = torch.sum(torch.concat(text_embeddings), dim = 0, keepdim = True) / len(text_embeddings)
    return phrase_embedding

# reshape so that image embeddings and text embeddings put together
def reshape_hidden_states(hidden_states):
    # converts hidden states into shape (layer #s, beam count, hidden state #s, dims)
    all_embeddings = []
    for i in range(len(hidden_states)):
        if i == 0: # only get the image embeddings
            all_embeddings.append(torch.stack(hidden_states[i])[:, :, :32])
        else:
            all_embeddings.append(torch.stack(hidden_states[i]))
    all_hidden_states = torch.concat(all_embeddings, dim = 2)
    return all_hidden_states

def run_blip_model(inputs_embeds, captioner, caption_only = True):
    # Run image embeddings into the model
    test_input = {"image": torch.ones(1).to("cuda"), "prompt": captioner.prompt}
    atts_llm = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)

    with torch.no_grad():
        out = captioner.model.generate(  # type: ignore
                    test_input,
                    num_beams=5,
                    max_length=500,
                    return_embeds=True,
                    atts_vision=atts_llm,
                    inputs_vision=inputs_embeds, # stick input embeddings from vision here
                    return_dict=True,
                    temperature=1,
                )
        if caption_only:
            return out[0][0]
        return out

def string_to_token_ids(string, tokenizer = llm_tokenizer):
    return tokenizer(string)["input_ids"]

def get_heatmap(inputs_embeds, caption_word, captioner, vocab_embeddings):
    out = run_blip_model(captioner, inputs_embeds)

    hidden_states = reshape_hidden_states(out[4].hidden_states)
    cs_scores = []

    cos = torch.nn.CosineSimilarity(dim = 1)

    for embedding_ind in range(32):
        cs = cos(hidden_states[:, 0, embedding_ind], get_phrase_embedding(caption_word, vocab_embeddings))
        cs_scores.append(cs)
    return torch.stack(cs_scores), out

def get_hidden_text_embedding(coco_class, captioner, img_embeddings, vocab_embeddings, layer = 10, device = "cuda"):
    test_input = {"image": torch.ones(1).to("cuda"), "prompt": coco_class}
    inputs_embeds = img_embeddings.clone().to(device)

    atts_llm = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(device)
    with torch.no_grad():
        out = captioner.model.generate(  # type: ignore
            test_input,
            num_beams=5,
            max_length=10,
            return_embeds=True,
            atts_vision=atts_llm,
            inputs_vision=inputs_embeds, # stick input embeddings from vision here
            return_dict=True,
            pure_llm = True, # meaning only text model used
        )

        hidden_states = reshape_hidden_states(out[4].hidden_states)

        # Extract the size of the tokens split up
        token_ids = string_to_token_ids(coco_class)
        dist = torch.norm(hidden_states[0, 0, len(token_ids) - 1] - vocab_embeddings[0, token_ids[len(token_ids) - 1]])
        if dist > 0.1:
            print(f"Coco class {coco_class} didn't match: {dist}")

        return hidden_states[layer, 0, len(token_ids) - 1].unsqueeze(0)

def load_blip_state(model_type, train, device="cuda"):
    if model_type == "blip7b":
        captioner = InstructBLIPVicuna7B(device=device, return_embeds=True)
    elif model_type == "blip13b":
        captioner = InstructBLIPVicuna13B(device=device, return_embeds=True)
    vocabulary = captioner.tokenizer.get_vocab()
    vocab_embeddings = get_vocab_embeddings_blip(captioner, device=device)

    if train:
        data = torch.load(f"/home/nickj/vl-hallucination/vl_data/{model_type}_train.pt")
    else:
        data = torch.load(f"/home/nickj/vl-hallucination/vl_data/{model_type}_val.pt")

    def execute_model(coco_img, image_embeddings=None):
        if image_embeddings != None:
            return run_blip_model(image_embeddings, captioner)
        else:
            return run_blip_model(data[coco_img]["image_embeddings"].to(device), captioner)

    register_hook = lambda hook, layer: captioner.model.llm_model.model.layers[layer].register_forward_hook(hook)
    register_pre_hook = lambda pre_hook, layer: captioner.model.llm_model.model.layers[layer].register_forward_pre_hook(pre_hook)
    hidden_layer_embedding = lambda text, layer: get_hidden_text_embedding(text, captioner, torch.zeros((1, 32, vocab_embeddings.shape[2])), vocab_embeddings, layer, device=device)

    return {
        "vocabulary": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "data": data,
        "tokenizer": captioner.tokenizer,
        "execute_model": execute_model,
        "register_hook": register_hook,
        "register_pre_hook": register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model": captioner,
    }
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch

import pickle
import torch
import tqdm
import json
from collections import defaultdict

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import requests
from io import BytesIO
import re
import numpy as np
import matplotlib.pyplot as plt
from experiments.blip_utils import coco_img_id_to_path, string_to_token_ids
from experiments.utils import subtract_projection, subtract_projections

from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import LogitsProcessorList
import os

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_vocab_embeddings_llava(llm_model, tokenizer, device="cuda"):
    vocab = tokenizer.get_vocab()
    llm_tokens = torch.tensor(list(vocab.values()), dtype=torch.long).unsqueeze(0).to(device)
    token_embeddings = llm_model.get_input_embeddings()(llm_tokens)
    return token_embeddings

def generate_text_prompt(model, model_name):
    qs = "Write a detailed description."
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    return conv

def generate_images_tensor(model, img_path, image_processor):
    image_files = [img_path]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    return images_tensor, images, image_sizes

def prompt_to_img_input_ids(prompt, tokenizer):
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    return input_ids

def run_llava_model(model, model_name, images_tensor, image_sizes, tokenizer, images_embeds = None, hidden_states = False):
    conv = generate_text_prompt(model, model_name)
    input_ids = prompt_to_img_input_ids(conv.get_prompt(), tokenizer)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = 1.0
    top_p = None
    num_beams = 5
    max_new_tokens = 500

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=images_tensor,
            # images_embeds = images_embeds,
            # apply_on_image_features = lambda image_embeddings: subtract_projection(image_embeddings, get_phrase_embedding("umbrella", vocab_embeddings, tokenizer)),
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            # use_cache=True,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            # output_attentions=False,
            output_hidden_states=hidden_states,
            return_dict_in_generate=True,
            image_sizes = image_sizes,
        )

    if hidden_states:
        return input_ids, output

    outputs = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()

    return outputs

def retrieve_logit_lens_llava(state, img_path):
    images_tensor, images, image_sizes = generate_images_tensor(state["model"], img_path, state["image_processor"])
    input_ids, output = run_llava_model(state["model"], state["model_name"], images_tensor, image_sizes, state["tokenizer"], images_embeds=None, hidden_states = True)

    input_token_len = input_ids.shape[1]
    output_ids = output.sequences
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    o = state["tokenizer"].batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]    
    caption = o.strip()

    hidden_states = torch.stack(output.hidden_states[0])

    logits_warper = TopKLogitsWarper(top_k=50, filter_value=float('-inf'))
    logits_processor = LogitsProcessorList([])

    with torch.no_grad():
        curr_layer_logits = state["model"].lm_head(hidden_states)
        logit_scores = torch.nn.functional.log_softmax(
            curr_layer_logits, dim=-1
        )
        logit_scores_processed = logits_processor(input_ids, logit_scores)
        logit_scores = logits_warper(input_ids, logit_scores_processed)
        softmax_probs = torch.nn.functional.softmax(logit_scores, dim=-1)
    
    softmax_probs = softmax_probs.detach().cpu().numpy()

    image_token_index = input_ids.tolist()[0].index(-200)
    softmax_probs = softmax_probs[:, :, image_token_index:image_token_index+(24*24)]
    # transpose to (vocab_dim, num_layers, num_tokens, num_beams)
    softmax_probs = softmax_probs.transpose(3, 0, 2, 1)
    # maximum over all beams
    softmax_probs = softmax_probs.max(axis=3)
    return caption, softmax_probs

def remove_h_and_preserve_recall(image_embeddings, recall_embeddings, h_embeddings, weight = 1):
    # recall_embeddings: Tensor[# embeddings, # dims]
    # h_embeddings: Tensor[# embeddings, # dims]
    # image_embeddings: Tensor[1, # embeddings, # dims]

    image_embeddings = image_embeddings.clone()
    h_embeddings_no_recall = subtract_projections(h_embeddings.unsqueeze(1), recall_embeddings.unsqueeze(1))
    return subtract_projections(image_embeddings, h_embeddings_no_recall, weight = weight)

def reshape_llava_prompt_hidden_layers(hidden_states):
    prompt_hidden_states = hidden_states[0] # shape is (# layers, # beams, # prompt tokens, # dim size)
    first_beam_layers = torch.stack(list(prompt_hidden_states), dim = 0)[:, 0]
    return first_beam_layers

def get_hidden_layer_embedding(target_word, model, vocab_embeddings, tokenizer, layer = 5, device = "cuda"):
    # Tokenize the target word into input ids
    token_ids = string_to_token_ids(target_word)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stop_str = target_word
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = 1.0
    top_p = None
    num_beams = 5
    max_new_tokens = 10
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=False,
            stopping_criteria=[stopping_criteria],
        )

    hidden_states = reshape_llava_prompt_hidden_layers(output["hidden_states"])

    dist = torch.norm(hidden_states[0, len(token_ids) - 1] - vocab_embeddings[0, token_ids[len(token_ids) - 1]])
    if dist > 0.1:
        print(f"Validation check failed: caption word {target_word} didn't match: {dist}")

    return hidden_states[layer, len(token_ids) - 1].unsqueeze(0)

# def generate_llava_edit_hook(text_embeddings, start_edit_index, end_edit_index, layer, weight = 1, minimum_size = 576):
#     def edit_embeddings(module, input, output):
#         new_output = list(output)
#         if new_output[0].shape[1] > minimum_size:
#             print(f"Editing layer {layer}")
#             new_output[0][:, start_edit_index: end_edit_index] = subtract_projections(new_output[0][:, start_edit_index:end_edit_index], text_embeddings, weight = weight)
#         return tuple(new_output)
#     return edit_embeddings

def generate_remove_h_preserve_recall_hook(recall_embeddings, h_embeddings, start_edit_index, end_edit_index, layer, weight = 1, minimum_size = 576):
    def edit_embeddings(module, input, output):
        new_output = list(output)
        if new_output[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_output[0][:, start_edit_index: end_edit_index] = remove_h_and_preserve_recall(new_output[0][:, start_edit_index:end_edit_index], recall_embeddings, h_embeddings, weight = weight)
        return tuple(new_output)
    return edit_embeddings

def get_caption_from_llava(coco_img, model, model_name, tokenizer, image_processor, train = True):
    img_path = coco_img_id_to_path(coco_img, validation=not train)
    images_tensor, images, image_sizes = generate_images_tensor(model, img_path, image_processor)

    # Generate the new caption
    new_caption = run_llava_model(model, model_name, images_tensor, image_sizes, tokenizer)

    return new_caption

def load_llava_state(model_type, train, device="cuda"):
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
      model_path, None, model_name
    )
    if train:
        data = torch.load(f"/home/nickj/vl-hallucination/vl_data/{model_type}_train.pt")
    else:
        data = torch.load(f"/home/nickj/vl-hallucination/vl_data/{model_type}_val.pt")

    vocabulary = tokenizer.get_vocab()
    vocab_embeddings = get_vocab_embeddings_llava(model, tokenizer, device=device)

    execute_model = lambda coco_img, image_embeddings=None: get_caption_from_llava(coco_img, model, model_name, tokenizer, image_processor, train=train)
    register_hook = lambda hook, layer: model.get_model().layers[layer].register_forward_hook(hook)
    register_pre_hook = lambda pre_hook, layer: model.get_model().layers[layer].register_forward_pre_hook(pre_hook)
    hidden_layer_embedding = lambda text, layer: get_hidden_layer_embedding(text, model, vocab_embeddings, tokenizer, layer, device=device)

    return {
        "vocabulary": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "data": data,
        "tokenizer": tokenizer,
        "execute_model": execute_model,
        "register_hook": register_hook,
        "register_pre_hook": register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model": model,
        "model_name": model_name,
        "image_processor": image_processor,
    }
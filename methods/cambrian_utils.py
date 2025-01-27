import torch
import sys
# ADD PATH TO CAMBRIAN REPOSITORY
sys.path.append('/home/anish/vl-interp/cambrian')

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import process_images, tokenizer_image_token
from methods.utils import load_images, string_to_token_ids
from PIL import Image
import requests

def projection(image_embeddings, text_embedding):
    return (image_embeddings @ text_embedding.T)[0, :, 0] / (
        text_embedding @ text_embedding.T
    ).squeeze()

def subtract_projection(image_embeddings, text_embedding, weight=1):
    image_embeddings = image_embeddings.clone()
    proj = projection(image_embeddings, text_embedding)
    for i in range(image_embeddings.shape[1]):
        if proj[i] > 0:
            image_embeddings[:, i] -= weight * proj[i] * text_embedding
            # image_embeddings[:, i] += weight * proj[i] * text_embedding
    return image_embeddings

def subtract_projections(image_embeddings, text_embeddings, weight=1):
    # text_embeddings: (# embeds, 1, # dim size)
    img_embeddings = image_embeddings.clone()
    for text_embedding in text_embeddings:
        img_embeddings = subtract_projection(img_embeddings, text_embedding, weight)
    return img_embeddings

def generate_mass_edit_hook(
    text_embeddings, start_edit_index, end_edit_index, layer, weight=1, minimum_size=32
):
    def edit_embeddings(module, input, output):
        new_output = list(output)
        if new_output[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_output[0][:, start_edit_index:end_edit_index] = subtract_projections(
                new_output[0][:, start_edit_index:end_edit_index],
                text_embeddings,
                weight=weight,
            )
        return tuple(new_output)

    return edit_embeddings

def get_vocab_embeddings_cambrian(llm_model, tokenizer, device="cuda"):
    vocab = tokenizer.get_vocab()
    llm_tokens = torch.tensor(list(vocab.values()), dtype=torch.long).unsqueeze(0).to(device)
    token_embeddings = llm_model.get_input_embeddings()(llm_tokens)
    return token_embeddings

def generate_text_prompt(model, text_prompt):
    qs = text_prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # Using llama_3 for Cambrian-8b
    conv_mode = "llama_3"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv

def generate_images_tensor(model, img_path, image_processor):
    image_files = [img_path]
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    
    return images_tensor, images, image_sizes

def prompt_to_img_input_ids(prompt, tokenizer):
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    return input_ids

def run_cambrian_model(
    model,
    images_tensor,
    image_sizes,
    tokenizer,
    text_prompt=None,
    hidden_states=False,
    temperature=1.0
):
    if text_prompt is None:
        text_prompt = "Please describe this image in detail."

    conv = generate_text_prompt(model, text_prompt)
    input_ids = prompt_to_img_input_ids(conv.get_prompt(), tokenizer)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            num_beams=5,
            max_new_tokens=512,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=hidden_states
        )

    if hidden_states:
        return input_ids, output

    outputs = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
    return outputs

def reshape_cambrian_prompt_hidden_layers(hidden_states):
    prompt_hidden_states = hidden_states[0]
    first_beam_layers = torch.stack(list(prompt_hidden_states), dim=0)[:, 0]
    return first_beam_layers

def get_hidden_text_embedding(
    target_word, model, vocab_embeddings, tokenizer, layer=22, device="cuda"
):
    token_ids = string_to_token_ids(target_word)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            temperature=1.0,
            num_beams=5,
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=False
        )

    hidden_states = reshape_cambrian_prompt_hidden_layers(output["hidden_states"])

    # Validation check
    dist = torch.norm(
        hidden_states[0, len(token_ids) - 1] 
        - vocab_embeddings[0, token_ids[len(token_ids) - 1]]
    )
    if dist > 0.1:
        print(f"Validation check failed: caption word {target_word} didn't match: {dist}")

    return hidden_states[layer, len(token_ids) - 1].unsqueeze(0)

def get_caption_from_cambrian(
    img_path, model, tokenizer, image_processor, text_prompt=None
):
    images_tensor, images, image_sizes = generate_images_tensor(
        model, img_path, image_processor
    )

    new_caption = run_cambrian_model(
        model,
        images_tensor,
        image_sizes,
        tokenizer,
        text_prompt=text_prompt,
    )

    return new_caption

def process(image, question, tokenizer, image_processor, model_config):
    conv_mode = "llama_3" 
    qs = question

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, image_tensor, image_size, prompt

def retrieve_logit_lens_cambrian(state, img_path):
    """Get logit lens analysis for an image using Cambrian."""
    device = state["model"].device
    
    question = "Please describe this image in detail."
    
    # Use process() function for image processing
    if isinstance(img_path, str):
        if img_path.startswith('http'):
            image = Image.open(requests.get(img_path, stream=True).raw)
        else:
            image = Image.open(img_path)
    else:
        image = img_path

    input_ids, image_tensor, image_sizes, prompt = process(
        image, 
        question, 
        state["tokenizer"], 
        state["image_processor"], 
        state["model"].config
    )
    
    input_ids = input_ids.to(device=device, non_blocking=True)

    with torch.inference_mode():
        outputs = state["model"].generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=1.0,
            num_beams=5,
            max_new_tokens=512,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

    caption = state["tokenizer"].decode(outputs.sequences[0], skip_special_tokens=True)

    x = torch.stack(outputs.hidden_states[0]).max(dim=1).values.to(device)

    logits = []
    for i in range(x.shape[0]):
        with torch.no_grad():
            logits.append(state["model"].lm_head(x[i, :, :]).detach().cpu())
    
    logit_lens = torch.stack(logits)
    
    with torch.no_grad():
        softmax_probs = torch.softmax(logit_lens, dim=-1)
    
    softmax_probs = softmax_probs.permute(2, 0, 1)
    
    try:
        image_token_region = softmax_probs[:, :, 91:-12]
    except Exception as e:
        print(f"Warning: Error slicing token region: {e}")
        print(f"Softmax probs shape: {softmax_probs.shape}")
        image_token_region = softmax_probs
    
    image_token_region = image_token_region.detach().cpu().float().numpy()
    return caption, image_token_region

def load_cambrian_state(device="cuda"):
    model_path = "nyu-visionx/cambrian-8b"
    model_name = model_path
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    vocabulary = tokenizer.get_vocab()
    vocab_embeddings = get_vocab_embeddings_cambrian(model, tokenizer, device=device)

    execute_model = lambda img_path, text_prompt=None: get_caption_from_cambrian(
        img_path, model, tokenizer, image_processor, text_prompt
    )
    register_hook = lambda hook, layer: model.model.layers[layer].register_forward_hook(hook)
    register_pre_hook = lambda pre_hook, layer: model.model.layers[layer].register_forward_pre_hook(pre_hook)
    hidden_layer_embedding = lambda text, layer: get_hidden_text_embedding(
        text, model, vocab_embeddings, tokenizer, layer, device=device
    )

    return {
        "vocabulary": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "tokenizer": tokenizer,
        "execute_model": execute_model,
        "register_hook": register_hook,
        "register_pre_hook": register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model": model,
        "model_name": model_name,
        "image_processor": image_processor,
    }
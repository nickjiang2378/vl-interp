import torch
from PIL import Image
import requests
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def get_logits(model, output):
    """Extract and process logits from model output."""
    x = torch.stack(output.hidden_states[0]).max(dim=1).values

    logits = []
    for i in tqdm(range(x.shape[0])):
        with torch.no_grad():
            logits.append(model.language_model.lm_head(x[i, :, :]).detach().cpu())
    
    logit_lens = torch.stack(logits)
    with torch.no_grad():
        probs = torch.softmax(logit_lens, dim=-1)
    
    raw_probs = probs.permute(2, 0, 1)
    raw_probs = raw_probs[:, :, 5:-14]
    probs = torch.max(raw_probs, dim=1).values
    probs = torch.max(probs, dim=1).values
    return raw_probs, probs

def get_generated_logits(model, output):
    """Get logits from generated output."""
    logits = []
    for i in range(1, len(output.hidden_states)):
        logits.append(torch.stack(output.hidden_states[i]).max(dim=1).values)
    
    logits = torch.stack(logits)
    logits = logits.squeeze(2)[:, -1, :]

    real_logits = model.language_model.lm_head(logits).detach().cpu()
    softmax_logits = torch.softmax(real_logits, dim=-1)

    return softmax_logits.permute(1, 0)

def get_text_hidden_state_at_layer(text, model, processor, layer):
    """Get hidden state representation for text at specified layer."""
    token_ids = processor.tokenizer.encode(text)
    input_ids = torch.tensor(token_ids).unsqueeze(0).to("cuda")

    with torch.inference_mode():
        outputs = model.language_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            max_new_tokens=5,
        )

    hidden_states = outputs.hidden_states
    return hidden_states[layer][0, -1, :].unsqueeze(0)

def projection(image_embeddings, text_embedding):
    """Calculate projection of image embeddings onto text embedding."""
    return (image_embeddings @ text_embedding.T)[0, :, 0] / (
        text_embedding @ text_embedding.T
    ).squeeze()

def subtract_projection(image_embeddings, text_embedding, weight=1):
    """Subtract text projection from image embeddings."""
    image_embeddings = image_embeddings.clone()
    proj = projection(image_embeddings, text_embedding)
    for i in range(image_embeddings.shape[1]):
        if proj[i] > 0:
            image_embeddings[:, i] -= weight * proj[i] * text_embedding
    return image_embeddings

def subtract_projections(image_embeddings, text_embeddings, weight=1):
    """Subtract multiple text projections from image embeddings."""
    img_embeddings = image_embeddings.clone()
    for text_embedding in text_embeddings:
        img_embeddings = subtract_projection(img_embeddings, text_embedding, weight)
    return img_embeddings

def generate_mass_edit_hook(
    text_embeddings, start_edit_index, end_edit_index, layer, weight=1, minimum_size=32
):
    """Generate hook for editing embeddings during forward pass."""
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

def generate_caption(model, processor, image_path, prompt=None):
    """Generate caption for an image."""
    if isinstance(image_path, str):
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
    else:
        image = image_path

    if prompt is None:
        prompt = "Please describe this image in detail."

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_hidden_states=True,
            temperature=1.0,
            num_beams=5
        )
    
    caption = processor.decode(output.sequences[0], skip_special_tokens=True)
    return caption, output

def retrieve_logit_lens_llava_next(state, img_path):
    """Get logit lens analysis for an image using LLaVA-Next."""
    device = state["model"].device
    
    if isinstance(img_path, str):
        if img_path.startswith('http'):
            image = Image.open(requests.get(img_path, stream=True).raw)
        else:
            image = Image.open(img_path)
    else:
        image = img_path

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Write a detailed description."},
            ],
        },
    ]
    
    prompt = state["processor"].apply_chat_template(conversation, add_generation_prompt=True)
    inputs = state["processor"](images=image, text=prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = state["model"].generate(
            **inputs,
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_hidden_states=True,
            temperature=1.0,
            num_beams=5
        )

    caption = state["processor"].decode(output.sequences[0], skip_special_tokens=True).strip()

    x = torch.stack(output.hidden_states[0]).max(dim=1).values.to(device)

    logits = []
    for i in range(x.shape[0]):
        with torch.no_grad():
            logits.append(state["model"].language_model.lm_head(x[i, :, :]).detach().cpu())
    
    logit_lens = torch.stack(logits)
    
    with torch.no_grad():
        softmax_probs = torch.softmax(logit_lens, dim=-1)
    
    softmax_probs = softmax_probs.permute(2, 0, 1)
    
    try:
        image_token_region = softmax_probs[:, :, 5:-14]
    except Exception as e:
        print(f"Warning: Error slicing token region: {e}")
        print(f"Softmax probs shape: {softmax_probs.shape}")
        image_token_region = softmax_probs
    
    image_token_region = image_token_region.detach().cpu().numpy()
    return caption, image_token_region

def load_llava_next_state(model_path="llava-hf/llava-v1.6-vicuna-7b-hf", device="cuda"):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    vocabulary = processor.tokenizer.get_vocab()
    vocab_embeddings = model.get_input_embeddings()(
        torch.tensor(list(vocabulary.values()), dtype=torch.long).unsqueeze(0).to(device)
    )

    execute_model = lambda img_path, text_prompt=None: generate_caption(
        model, processor, img_path, text_prompt
    )[0]
    
    register_hook = lambda hook, layer: model.language_model.model.layers[layer].register_forward_hook(hook)
    register_pre_hook = lambda pre_hook, layer: model.language_model.model.layers[layer].register_forward_pre_hook(pre_hook)
    hidden_layer_embedding = lambda text, layer: get_text_hidden_state_at_layer(
        text, model, processor, layer
    )

    return {
        "vocabulary": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "tokenizer": processor.tokenizer,
        "execute_model": execute_model,
        "register_hook": register_hook,
        "register_pre_hook": register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model": model,
        "model_name": model_path,
        "processor": processor,
    }

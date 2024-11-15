import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from transformers import LlamaTokenizer
from methods.utils import load_image, string_to_token_ids

try:
    from src.caption.instruct_blip_engine import InstructBLIPVicuna7B
except:  # first call for some reason fails
    pass

from src.caption.instruct_blip_engine import InstructBLIPVicuna7B

TOKEN_UNDERSCORE = chr(
    9601
)  # a lot of tokens have _rain, etc. but the underscore in front is not normal


def get_image_embeddings(image_file, captioner, prompt = None):
    if prompt == None:
        prompt = captioner.prompt

    image = load_image(image_file)
    baseline_caption, inputs_embeds, inputs_query, outputs = captioner(
        image, prompt=prompt, return_embeds=True
    )
    image_embeddings = inputs_query
    return image_embeddings


def get_vocab_embeddings_blip(captioner, device="cuda"):
    vocab = captioner.tokenizer.get_vocab()
    llm_tokens = (
        torch.tensor(list(vocab.values()), dtype=torch.long).unsqueeze(0).to(device)
    )
    token_embeddings = captioner.model.llm_model.get_input_embeddings()(llm_tokens)
    return token_embeddings.to(device)


# reshape so that image embeddings and text embeddings put together
def reshape_hidden_states(hidden_states):
    # converts hidden states into shape (layer #s, beam count, hidden state #s, dims)
    all_embeddings = []
    for i in range(len(hidden_states)):
        if i == 0:  # only get the image embeddings
            all_embeddings.append(torch.stack(hidden_states[i])[:, :, :32])
        else:
            all_embeddings.append(torch.stack(hidden_states[i]))
    all_hidden_states = torch.concat(all_embeddings, dim=2)
    return all_hidden_states


def run_blip_model(inputs_embeds, captioner, text_prompt=None, caption_only=True):
    if text_prompt == None:
        text_prompt = captioner.prompt

    # Run image embeddings into the model
    test_input = {"image": torch.ones(1).to("cuda"), "prompt": text_prompt}
    atts_llm = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
        inputs_embeds.device
    )

    with torch.no_grad():
        out = captioner.model.generate(  # type: ignore
            test_input,
            num_beams=5,
            max_length=512,
            return_embeds=True,
            atts_vision=atts_llm,
            inputs_vision=inputs_embeds,  # stick input embeddings from vision here
            return_dict=True,
            temperature=1,
        )
        if caption_only:
            return out[0][0]
        return out


def retrieve_logit_lens_blip(state, img_path, text_prompt = None):
    input_embeds = get_image_embeddings(img_path, state["model"], prompt = text_prompt)
    out = run_blip_model(input_embeds, state["model"], caption_only=False, text_prompt=text_prompt)
    caption = out[0][0]
    hidden_states = torch.stack(out[4]["hidden_states"][0])[:, :, :32]
    with torch.no_grad():
        softmax_probs = torch.nn.functional.softmax(
            state["model"].model.llm_model.lm_head(hidden_states.half()), dim=-1
        )
    # max over beams
    softmax_probs = softmax_probs.max(axis=1).values
    softmax_probs = softmax_probs.cpu().numpy()
    # reshape into (vocab_size, num_layers, num_tokens)
    softmax_probs = softmax_probs.transpose(2, 0, 1)
    return caption, softmax_probs


def get_hidden_text_embedding(
    input_text, captioner, img_embeddings, vocab_embeddings, tokenizer, layer=10, device="cuda"
):
    test_input = {"image": torch.ones(1).to("cuda"), "prompt": input_text}
    inputs_embeds = img_embeddings.clone().to(device)

    atts_llm = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(device)
    with torch.no_grad():
        out = captioner.model.generate(  # type: ignore
            test_input,
            num_beams=5,
            max_length=10,
            return_embeds=True,
            atts_vision=atts_llm,
            inputs_vision=inputs_embeds,  # stick input embeddings from vision here
            return_dict=True,
            pure_llm=True,  # meaning only text model used
        )

        hidden_states = reshape_hidden_states(out[4].hidden_states)

        # Extract the size of the tokens split up
        token_ids = string_to_token_ids(input_text, tokenizer)
        dist = torch.norm(
            hidden_states[0, 0, len(token_ids) - 1]
            - vocab_embeddings[0, token_ids[len(token_ids) - 1]]
        )
        if dist > 0.1:
            print(f"Coco class {input_text} didn't match: {dist}")

        return hidden_states[layer, 0, len(token_ids) - 1].unsqueeze(0)


def load_blip_state(device="cuda"):
    captioner = InstructBLIPVicuna7B(device=device, return_embeds=True)

    vocabulary = captioner.tokenizer.get_vocab()
    vocab_embeddings = get_vocab_embeddings_blip(captioner, device=device)

    def execute_model(img_path, image_embeddings=None, text_prompt = None):
        if image_embeddings == None:
            image_embeddings = get_image_embeddings(img_path, captioner, prompt=text_prompt).to(device)
        return run_blip_model(image_embeddings, captioner, text_prompt=text_prompt)

    register_hook = lambda hook, layer: captioner.model.llm_model.model.layers[
        layer
    ].register_forward_hook(hook)
    register_pre_hook = lambda pre_hook, layer: captioner.model.llm_model.model.layers[
        layer
    ].register_forward_pre_hook(pre_hook)
    hidden_layer_embedding = lambda text, layer: get_hidden_text_embedding(
        text,
        captioner,
        torch.zeros((1, 32, vocab_embeddings.shape[2])),
        vocab_embeddings,
        captioner.tokenizer,
        layer,
        device=device,
    )

    return {
        "vocabulary": vocabulary,
        "vocab_embeddings": vocab_embeddings,
        "tokenizer": captioner.tokenizer,
        "execute_model": execute_model,
        "register_hook": register_hook,
        "register_pre_hook": register_pre_hook,
        "hidden_layer_embedding": hidden_layer_embedding,
        "model": captioner,
    }

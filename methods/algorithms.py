import torch
from .utils import get_device_from_module

def get_phrase_embedding(phrase, vocab_embeddings, tokenizer, remove_first=True):
    # returns size (1, 5120)
    text_embeddings = []
    for token_id in tokenizer(phrase)["input_ids"]:
        text_embeddings.append(vocab_embeddings[:, token_id])
    if remove_first:
        text_embeddings = text_embeddings[1:]
    phrase_embedding = torch.sum(
        torch.concat(text_embeddings), dim=0, keepdim=True
    ) / len(text_embeddings)
    return phrase_embedding


def projection(image_embeddings, text_embedding):
    return (image_embeddings @ text_embedding.T)[0, :, 0] / (
        text_embedding @ text_embedding.T
    ).squeeze()


def subtract_projection(image_embeddings, text_embedding, weight=1, device = None):
       # if device is None, don't move the embeddings to any device - keep their pre-existing device configs in tact
    if device != None:
        image_embeddings = image_embeddings.to(device)
        text_embedding = text_embedding.to(device)
    image_embeddings = image_embeddings.clone()
    proj = projection(image_embeddings, text_embedding)
    for i in range(image_embeddings.shape[1]):
        if proj[i] > 0:
            image_embeddings[:, i] -= weight * proj[i] * text_embedding
    return image_embeddings


def subtract_projections(image_embeddings, text_embeddings, weight=1, device = None):
    # text_embeddings: (# embeds, 1, # dim size)
    # if device is None, don't move the embeddings to any device - keep their pre-existing device configs in tact
    img_embeddings = image_embeddings.clone()
    for text_embedding in text_embeddings:
        img_embeddings = subtract_projection(img_embeddings, text_embedding, weight, device=device)
    return img_embeddings


def remove_all_hooks(model):
    # Iterate over all modules in the model
    for module in model.modules():
        # Clear forward hooks
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        # Clear backward hooks (if any)
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()
        # Clear forward pre-hooks (if any)
        if hasattr(module, "_forward_pre_hooks"):
            module._forward_pre_hooks.clear()


def generate_mass_edit_hook(
    text_embeddings, start_edit_index, end_edit_index, layer, weight=1, minimum_size=32
):
    if len(text_embeddings) == 0:
        print("No text embeddings found. Note that no editing will occur.")
    def edit_embeddings(module, input, output):
        device = get_device_from_module(module)
        new_output = list(output)
        if new_output[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_output[0][:, start_edit_index:end_edit_index] = subtract_projections(
                new_output[0][:, start_edit_index:end_edit_index],
                text_embeddings,
                weight=weight,
                device=device
            )
        return tuple(new_output)

    return edit_embeddings


def generate_mass_edit_pre_hook(
    text_embeddings, start_edit_index, end_edit_index, layer, weight=1, minimum_size=32
):
    if len(text_embeddings) == 0:
        print("No text embeddings found. Note that no editing will occur.")
    def edit_embeddings(module, input):
        device = get_device_from_module(module)
        new_input = list(input)
        if new_input[0].shape[1] > minimum_size:
            print(f"Editing layer {layer}")
            new_input[0][:, start_edit_index:end_edit_index] = subtract_projections(
                new_input[0][:, start_edit_index:end_edit_index],
                text_embeddings,
                weight=weight,
                device=device
            )
        return tuple(new_input)

    return edit_embeddings


def internal_confidence(tokenizer, softmax_probs, class_):
    class_token_indices = tokenizer.encode(class_)[1:]
    return softmax_probs[class_token_indices].max()


def internal_confidence_heatmap(tokenizer, softmax_probs, class_):
    class_token_indices = tokenizer.encode(class_)[1:]
    return softmax_probs[class_token_indices].max(axis=0).T


def internal_confidence_segmentation(tokenizer, softmax_probs, class_, num_patches=24):
    class_token_indices = tokenizer.encode(class_)[1:]
    return (
        softmax_probs[class_token_indices]
        .max(axis=0)
        .max(axis=0)
        .reshape(num_patches, num_patches)
        .astype(float)
    )

import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import requests


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


def display_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Display image
        plt.imshow(img)
        plt.axis("off")  # Hide axes
        plt.show()


def string_to_token_ids(string, tokenizer):
    return tokenizer(string)["input_ids"]


def coco_img_id_to_name(img_id, train=False):
    if train:
        return f"COCO_train2014_{(12 - len(str(img_id))) * '0' + str(img_id)}"
    else:
        return f"COCO_val2014_{(12 - len(str(img_id))) * '0' + str(img_id)}"

def display_chair_results(chair_evals):
    print(f"Hallucinations: {chair_evals['mscoco_hallucinated_words']}\nCorrectly detected (recall) objects: {chair_evals['recall_words']}\nGround truth: {chair_evals['mscoco_gt_words']}")

def get_device_from_module(module):
    return next(module.parameters()).device
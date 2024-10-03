import torch
from PIL import Image
from io import BytesIO
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

def string_to_token_ids(string, tokenizer):
    return tokenizer(string)["input_ids"]

def coco_img_id_to_name(img_id, train = False):
    if train:
        return f"COCO_train2014_{(12 - len(str(img_id))) * '0' + str(img_id)}"
    else:
        return f"COCO_val2014_{(12 - len(str(img_id))) * '0' + str(img_id)}"
import os
import string
from typing import List, Optional, Tuple, Callable, Dict, Any
from collections import defaultdict

import torch
from git.repo import Repo
from PIL import Image
from torchvision import transforms
import numpy as np
from scipy.stats import entropy

from src.caption.base import CaptionEngine
from src.caption.utils import postprocess_caption

from .ofa import OFAModel, OFATokenizer

_OFA_PATHS = {
    "large-caption": "https://huggingface.co/OFA-Sys/OFA-large-caption",
}

_OFA_DEFAULT_PROMPT = " what does the image describe?"


def _get_ofa_model(model: str) -> Tuple[str, str]:
    if os.path.exists(model):
        # We can load the model locally
        return model, model

    if model not in _OFA_PATHS:
        raise ValueError(f"Invalid OFA model: {model}, should be one of {list(_OFA_PATHS.keys())}")

    git_repo = _OFA_PATHS[model]

    # If the repo is already cloned, we can use it
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "src", model)
    if os.path.exists(cache_dir):
        return cache_dir, cache_dir

    # Clone the repo into a cache directory
    os.makedirs(cache_dir, exist_ok=True)
    repo = Repo.clone_from(git_repo, cache_dir, branch="main")
    repo.git.checkout("main")
    return cache_dir, cache_dir


class OFACaptionEngine(CaptionEngine):
    def __init__(
        self, model: str = "large-caption", device: Optional[str] = None, prompt: str = _OFA_DEFAULT_PROMPT
    ):

        tokenizer_path, model_path = _get_ofa_model(model)
        self.tokenizer = OFATokenizer.from_pretrained(tokenizer_path)
        self.model = OFAModel.from_pretrained(model_path, device=device, use_cache=True)
        self.model = self.model.to(device or "cpu").eval()
        self.prompt = prompt
        self.device = device or "cpu"
        self.processor = self._preprocess_image_simple

    def _preprocess_image_simple(self, raw_image: Image.Image) -> torch.Tensor:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        patch_resize_transform = transforms.Compose(
            [
                lambda image: image.convert("RGB"),
                transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),  # type: ignore
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return patch_resize_transform(raw_image)

    def _preprocess_image(self, raw_image: Image.Image) -> torch.Tensor:
        return self._preprocess_image_simple(raw_image).unsqueeze(0).to(self.device)

    def _get_language_prompt(
        self,
    ) -> torch.Tensor:
        return self.tokenizer([self.prompt], return_tensors="pt").input_ids.to(self.device)

    def __call__(self, raw_image: Image.Image, n_captions: int = 1, temperature: Optional[float] = 1.0) -> List[str]:

        patch_img = self._preprocess_image(raw_image)
        inputs = self._get_language_prompt()

        output_captions = self.get_baseline_caption(raw_image)[0]
        if n_captions > 1:
            for _ in range(n_captions - 1):
                # Sample from the model
                gen = self.model.generate(  # type: ignore
                    inputs,
                    patch_images=patch_img,
                    do_sample=True,
                    top_p=0.9,
                    temperature=temperature,
                    no_repeat_ngram_size=3,
                    max_length=256,
                )
                output_captions.append(self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0])

        return [postprocess_caption(caption.strip()) for caption in output_captions]

    def get_baseline_caption(self, image) -> List[str]:
        if isinstance(image, Image.Image):
            image = self._preprocess_image(image)
        else:
            assert isinstance(image, torch.Tensor)

        prefix = self._get_language_prompt().repeat(image.shape[0], 1)

        baseline_gen = self.model.generate(  # type: ignore
            prefix,
            patch_images=image,
            num_beams=8,
            no_repeat_ngram_size=3,
            max_length=256,
        )
        baseline_caption = self.tokenizer.batch_decode(baseline_gen, skip_special_tokens=True)
        baseline_caption = [postprocess_caption(c.strip()) for c in baseline_caption]

        return baseline_caption


    def get_baseline_gen(self, force_caption: Optional[str], raw_image: Image.Image):
        if force_caption is not None:
            # Lowercase & remove punctuation
            force_caption = force_caption.translate(str.maketrans("", "", string.punctuation)).lower()
            encoded = self.tokenizer.encode(force_caption, return_tensors="pt").to(self.device) # Size = 1 x sequence length
            max_length = encoded.shape[1]
            num_beams = 1
            no_repeat_ngram_size = 0
            def prefix_allowed_tokens_fn(batch_id, sent):
                next_tok_id = len(sent)
                return encoded[batch_id][next_tok_id].tolist()
        else:
            no_repeat_ngram_size = 3
            prefix_allowed_tokens_fn = None
            max_length = 256

        patch_image = self._preprocess_image(raw_image)
        prefix = self._get_language_prompt()

        baseline_gen = self.model.generate(  # type: ignore
            prefix,
            patch_images=patch_image,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_length=max_length,
            return_dict_in_generate=True,
            output_attentions=True,
            output_scores=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        baseline_caption = self.tokenizer.batch_decode(baseline_gen.sequences, skip_special_tokens=True)
        baseline_caption = postprocess_caption(baseline_caption[0].strip())
        return baseline_gen, baseline_caption

    def _get_generated_attention(self, gen):
        cross_attention = gen.cross_attentions # does not include BOS
        self_attention = gen.decoder_attentions # does not include BOS
        return {'self_attention': self_attention, 'cross_attention': cross_attention}

    def get_forced_output_distributions(self, raw_image: Image, encoded_caption: torch.Tensor, vocab_size: int, language_only: bool = False) -> torch.Tensor:
        distributions = [] # Will be list of len(encoded_caption shape - 1), each entry vocab_size
        prefix = self._get_language_prompt()
        patch_image = self._preprocess_image(raw_image)

        if language_only:
            raise NotImplementedError

        for i in range(1, encoded_caption.shape[1]):
            def prefix_allowed_tokens_fn(batch_id, sent):
                if sent.shape[0] < i:
                    tokens = encoded_caption[batch_id][sent.shape[0]].tolist()
                else:
                    tokens = None
                return tokens

            gen = self.model.generate(  # type: ignore
                prefix,
                patch_images=patch_image,
                num_beams=1,
                no_repeat_ngram_size=0,
                max_length=encoded_caption.shape[1] + 1,
                return_dict_in_generate=True,
                output_attentions=False,
                output_scores=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
            distributions.append(gen.scores[i-1][0])
        return torch.stack(distributions)




from typing import Dict, List, Literal, Optional, Union, Any
import logging
from PIL import Image
import numpy as np
import torch

from src.caption.base import CaptionEngine

from src.caption.llava.LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from src.caption.llava.LLaVA.llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from src.caption.llava.LLaVA.llava.conversation import conv_templates, SeparatorStyle
from src.caption.llava.LLaVA.llava.model.builder import load_pretrained_model


_LLAVA_DEFAULT_PROMPT = "Provide a detailed description of the given image."


class NameSpace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LLaVA(CaptionEngine):
    def __init__(
        self,
        model: str = "llava-7b",
        device: Optional[str] = None,
        prompt: str = _LLAVA_DEFAULT_PROMPT,
        **kwargs,
    ):
        super().__init__()
        logging.info(f"Using LLaVA model {model}")

        # Hard-coded model dirs for now
        if "7b" in model:
            # model_dir = "/checkpoint/spetryk/llm/llava/llava-v1.5-7b"
            model_dir = "/shared/spetryk/large_model_checkpoints/lmm/llava-v1.5-7b/"
        elif "13b" in model:
            # model_dir = "/checkpoint/spetryk/llm/llava/llava-v1.5-13b"
            model_dir = "/shared/spetryk/large_model_checkpoints/lmm/llava-v1.5-13b/"
        else:
            raise ValueError(f"Unknown model {model}")

        self.model_name = get_model_name_from_path(model_dir)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_dir, model_base=None, model_name=self.model_name
        )
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
        image_aspect_ratio = "pad"  # Default from llava
        self.image_processor_config = NameSpace(image_aspect_ratio=image_aspect_ratio)
        self.device = device

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        # elif "mpt" in self.model_name.lower():
        #     conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        self.conv_mode = conv_mode

    def generate(
        self,
        inputs,
        do_sample=False,
        num_beams=16,
        max_length=256,
        temperature=1.0,
        topp=-1,
        prefix_allowed_tokens_fn=None,
    ):
        """
        inputs: dictionary with keys "input_ids", "stopping_criteria", and optionally "image_tensor" if using image input
        """
        input_ids = inputs["input_ids"]

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=inputs.get("image_tensor", None),
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_length,
                use_cache=False,
                stopping_criteria=[inputs["stopping_criteria"]],
                return_dict_in_generate=True,
                output_scores=True,
                top_k=None,
                top_p=topp,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        output_ids = outputs[0]
        # output_ids = outputs.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        out_ids = output_ids[:, input_token_len:]
        return outputs, out_ids

    def get_baseline_caption(
        self,
        inputs,
        do_sample=False,
        num_beams=16,
        max_length=256,
        temperature=1.0,
        topp=-1,
        return_embeds=False,
    ) -> List[str]:
        """
        inputs: dictionary with keys "input_ids", "stopping_criteria", and optionally "image_tensor" if using image input
        """

        outputs = self.generate(
            inputs,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            temperature=temperature,
            topp=topp,
        )
        output_ids = outputs[1]
        # output_ids = outputs.sequences

        caption = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        token_ids = output_ids[0][:-1]  # remove EOS
        token_ids = [token_ids]
        logging.info(caption)

        if return_embeds:
            return caption, token_ids
        return caption

    def _preprocess_image(
        self, raw_image: Image.Image, prompt: Optional[str] = None
    ) -> torch.Tensor:
        image_tensor = process_images(
            [raw_image], self.image_processor, self.image_processor_config
        )
        if type(image_tensor) is list:
            image_tensor = [
                image.to(self.model.device, dtype=torch.float16)
                for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        return image_tensor

    def processor(self, image: Image.Image, prompt=None, add_image=True):
        if add_image:
            image_tensor = self._preprocess_image(image)
        else:
            image_tensor = None
        conv = conv_templates[self.conv_mode].copy()
        qs = prompt if prompt is not None else _LLAVA_DEFAULT_PROMPT
        if add_image:
            if self.model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        new_prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                new_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        inputs = {
            "input_ids": input_ids,
            "image_tensor": image_tensor,
            "stopping_criteria": stopping_criteria,
        }

        return inputs

    def __call__(
        self,
        raw_image: Image.Image,
        n_captions: int = 1,
        do_sample=False,
        num_beams=16,
        max_length=256,
        temperature=1.0,
        topp=-1,
        prompt=None,
        return_embeds=False,
        generation_type="normal",
    ) -> List[str]:

        inputs = self.processor(raw_image, prompt)

        return self.get_baseline_caption(
            inputs,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            temperature=temperature,
            return_embeds=return_embeds,
            topp=topp,
        )

    def get_forced_output_distributions(
        self,
        raw_image: Image,
        encoded_caption: torch.Tensor,
        vocab_size: int,
        prompt: Optional[str] = None,
        language_only: bool = False,
        pure_llm: bool = False,
    ) -> torch.Tensor:

        distributions = (
            []
        )  # Will be list of len(encoded_caption shape - 1), each entry vocab_size

        inputs = self.processor(raw_image, prompt=prompt, add_image=not language_only)

        if language_only:
            inputs.pop("image_tensor")

        from tqdm import tqdm

        N_input_ids = inputs["input_ids"].shape[1]

        for i in tqdm(range(0, encoded_caption.shape[1])):

            def prefix_allowed_tokens_fn(batch_id, sent):
                # diff_idx = (sent.shape[0] - N_input_ids) + 1
                diff_idx = sent.shape[0] - N_input_ids
                if diff_idx < i:
                    tokens = encoded_caption[batch_id][diff_idx].tolist()
                else:
                    tokens = None
                return tokens

            gen, gen_ids = self.generate(  # type: ignore
                inputs,
                num_beams=1,
                max_length=i + 1,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

            distributions.append(gen[1][i][0].detach().cpu())

        return torch.stack(distributions)


class LLaVA7B(LLaVA):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="llava-7b", device=device, **kwargs)


class LLaVA13B(LLaVA):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="llava-13b", device=device, **kwargs)

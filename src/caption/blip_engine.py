from typing import Dict, List, Literal, Optional, Union, Any
import logging

import torch
import transformers
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers.image_processing_utils import BatchFeature

from packaging import version
from PIL import Image
import string

from src.caption.base import CaptionEngine
from src.caption.utils import postprocess_caption
from src.utils.pytorch import select_device

from src.caption.lavis.models import load_model_and_preprocess


_BLIP_DEFAULT_PROMPT = "this is a picture of"
# _INSTRUCT_BLIP_DEFAULT_PROMPT = "Describe this image in detail."
_INSTRUCT_BLIP_DEFAULT_PROMPT = "Write a short description for the image."


class BLIP2CaptionEngine(CaptionEngine):
    def __init__(
        self,
        model: str = "Salesforce/blip2-opt-2.7b-coco",
        device: Optional[str] = None,
        prompt: str = _BLIP_DEFAULT_PROMPT,
        **kwargs,
    ):
        logging.info(f"Using BLIP2 model {model}")
        self.vision_only = kwargs.get("vision_only", False)

        if model == "Salesforce/blip2-opt-2.7b":
            model_type = "pretrain_opt2.7b"
            model_name = "blip2_opt"
            tokenizer = "self.model.opt_tokenizer"
            start_token = 2
        elif model == "Salesforce/blip2-opt-2.7b-coco":
            model_type = "caption_coco_opt2.7b"
            model_name = "blip2_opt"
            tokenizer = "self.model.opt_tokenizer"
            start_token = 2
        elif model == "Salesforce/blip2-opt-6.7b":
            model_type = "pretrain_opt6.7b"
            model_name = "blip2_opt"
            tokenizer = "self.model.opt_tokenizer"
            start_token = 2
        elif model == "Salesforce/blip2-opt-6.7b-coco":
            model_type = "caption_coco_opt6.7b"
            model_name = "blip2_opt"
            tokenizer = "self.model.opt_tokenizer"
            start_token = 2
        else:
            raise ValueError(f"Unknown BLIP2 model {model}")

        model, vis_processors, _ = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=device,
            vision_only=self.vision_only,
        )
        self.model = model
        self.vis_processors = vis_processors
        self.tokenizer = eval(tokenizer)

        # if not torch.distributed.is_initialized():
        self.model.to(device or "cpu").eval()
        self.prompt = prompt
        self.device = device or "cpu"
        self.pure_llm = kwargs.get("pure_llm", False)
        self._init_cross_attention()
        self.vision_only = kwargs.get("vision_only", False)
        self.start_token = start_token

    def _init_cross_attention(self):
        """Save original cross-attention settings, in case of turning off cross-attention."""
        self.layer_idx2original_cross_attention = {}
        # For loading model from Blip2ForConditionalGeneration.from_pretrained
        # for idx, layer in enumerate(self.model.qformer.encoder.layer):

        # For loading OPT model from load_model_and_preprocess
        for idx, layer in enumerate(self.model.Qformer.bert.encoder.layer):
            self.layer_idx2original_cross_attention[idx] = layer.has_cross_attention

    def processor(self, image, prompt=None):
        if prompt is None:
            prompt = self.prompt
        inputs = {"image": self._preprocess_image(image), "prompt": prompt}
        return inputs

    def _preprocess_image(
        self, raw_image: Image.Image, prompt: Optional[str] = None
    ) -> torch.Tensor:
        return self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

    def get_baseline_caption(
        self,
        inputs,
        do_sample=False,
        num_beams=16,
        max_length=256,
        temperature=1.0,
        topp=-1,
        return_embeds=False,
        return_tokens=False,
    ) -> List[str]:

        out = self.model.generate(  # type: ignore
            inputs,
            num_beams=num_beams,
            temperature=temperature,
            max_length=max_length,
            top_p=topp,
            use_nucleus_sampling=topp > 0,
            return_embeds=return_embeds,
        )

        if return_embeds:
            baseline_caption, inputs_embeds, inputs_query, outputs = out
        else:
            baseline_caption = out

        baseline_caption = [postprocess_caption(b.strip()) for b in baseline_caption]

        if return_embeds:
            return baseline_caption, inputs_embeds, inputs_query, outputs
        if return_tokens:
            return baseline_caption, out[1]
        return baseline_caption

    def get_baseline_gen(self, force_caption: Optional[str], raw_image: Image.Image):
        if force_caption is not None:
            # Lowercase & remove punctuation
            force_caption = force_caption.translate(
                str.maketrans("", "", string.punctuation)
            ).lower()
            encoded = self.get_encoded_caption(force_caption)
            max_length = encoded.shape[1]
            num_beams = 1
            no_repeat_ngram_size = 0

            def prefix_allowed_tokens_fn(batch_id, sent):
                next_tok_id = len(sent)
                return encoded[batch_id][next_tok_id].tolist()

        else:
            num_beams = 16
            no_repeat_ngram_size = 3
            prefix_allowed_tokens_fn = None
            max_length = 256

        inputs = self._preprocess_image(raw_image)

        baseline_gen = self.model.generate(  # type: ignore
            **inputs,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_length=max_length,
            return_dict_in_generate=True,
            output_attentions=True,
            output_scores=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        baseline_caption = self.tokenizer.batch_decode(
            baseline_gen.sequences, skip_special_tokens=True
        )
        baseline_caption = postprocess_caption(baseline_caption[0].strip())
        return baseline_gen, baseline_caption

    def _get_generated_attention(self, gen):
        # Generation object is a type of [X]DecoderOnlyOutput (e.g., GreedyDecoderOnlyOutput),
        # which only has self attention.
        return {"self_attention": gen.attentions}

    def __call__(
        self,
        raw_image: Image.Image,
        n_captions: int = 1,
        do_sample=False,
        num_beams=16,
        max_length=256,
        temperature=1.0,
        prompt=None,
        topp=-1,
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
            topp=topp,
            return_embeds=return_embeds,
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
        inputs = self.processor(raw_image, prompt)

        if language_only:
            self._disable_cross_attention()

        for i in range(1, encoded_caption.shape[1]):

            def prefix_allowed_tokens_fn(batch_id, sent):
                if sent.shape[0] < i:
                    tokens = encoded_caption[batch_id][sent.shape[0]].tolist()
                else:
                    tokens = None
                return tokens

            gen = self.model.generate(  # type: ignore
                inputs,
                num_beams=1,
                # max_length=encoded_caption.shape[1] + 1,
                max_length=i + 1,
                return_dict=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                pure_llm=False if not language_only else self.pure_llm,
            )

            distributions.append(gen[1].scores[i - 1][0])

        if language_only:
            self._enable_cross_attention()

        return torch.stack(distributions)

    def _disable_cross_attention(self):
        """
        Turn off cross-attention in model QFormer layers. Used to obtain a caption conditioned only on language, not the image.
        Modifies self.model in-place.
        """
        # See notes in self._init_cross_attention()
        # for layer in self.model.qformer.encoder.layer:
        for layer in self.model.Qformer.bert.encoder.layer:
            layer.has_cross_attention = False

    def _enable_cross_attention(self):
        """
        Retores cross-attention in model QFormer layers to the original settings.
        Modifies self.model in-place.
        """
        # See notes in self._init_cross_attention()
        # for idx, layer in enumerate(self.model.qformer.encoder.layer):
        for idx, layer in enumerate(self.model.Qformer.bert.encoder.layer):
            layer.has_cross_attention = self.layer_idx2original_cross_attention[idx]


class BLIP2COCOLarge(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(
            model="Salesforce/blip2-opt-6.7b-coco", device=device, **kwargs
        )


class BLIP2COCOBase(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(
            model="Salesforce/blip2-opt-2.7b-coco", device=device, **kwargs
        )


class BLIP2COCOT5Large(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(
            model="Salesforce/blip2-flan-t5-xl-coco", device=device, **kwargs
        )


class BLIP2Large(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="Salesforce/blip2-opt-6.7b", device=device, **kwargs)


class BLIP2Base(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="Salesforce/blip2-opt-2.7b", device=device, **kwargs)


class BLIP2T5Large(BLIP2CaptionEngine):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="Salesforce/blip2-flan-t5-xl", device=device, **kwargs)


# class InstructBLIP(BLIP2CaptionEngine):
#     def __init__(
#         self, model: str = "instruct-blip", device: Optional[str] = None, prompt: str = _INSTRUCT_BLIP_DEFAULT_PROMPT
#     ):
#         logging.info('Using InstructBLIP model')
#         model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
#         self.model = model
#         self.vis_processors = vis_processors
#         breakpoint()
#         self.tokenizer = self._processor.tokenizer
#         self.model = Blip2ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16)
#         if not torch.distributed.is_initialized():
#             self.model.to(device or "cpu").eval()
#         self.prompt = prompt
#         self.device = device or "cpu"


#     def processor(self, image, prompt=None):
#         if prompt is None:
#             prompt = self.prompt
#         kwargs = {"return_tensors": "pt", "text": prompt}
#         return self._processor(image, **kwargs)

#     def _preprocess_image(self, raw_image: Image.Image, prompt: Optional[str] = None) -> torch.Tensor:
#         return self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

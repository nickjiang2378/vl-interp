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

_INSTRUCT_BLIP_DEFAULT_PROMPT = "Write a detailed description."
# _INSTRUCT_BLIP_DEFAULT_PROMPT = "Write a short description for the image."


class InstructBLIP(CaptionEngine):
    def __init__(
        self,
        model: str = "instruct-blip/vicuna7b",
        device: Optional[str] = None,
        prompt: str = _INSTRUCT_BLIP_DEFAULT_PROMPT,
        **kwargs,
    ):
        super().__init__()
        logging.info(f"Using InstructBLIP model {model}")

        self.vision_only = kwargs.get("vision_only", False)

        if model == "instruct-blip/vicuna7b":
            model_type = "vicuna7b"
            model_name = "blip2_vicuna_instruct"
            tokenizer = "self.model.llm_tokenizer"
            start_token = 1
        elif model == "instruct-blip/vicuna13b":
            model_type = "vicuna13b"
            model_name = "blip2_vicuna_instruct"
            tokenizer = "self.model.llm_tokenizer"
            start_token = 1
        elif model == "instruct-blip/flant5xl":
            model_type = "flant5xl"
            model_name = "blip2_t5_instruct"
            tokenizer = "self.model.t5_tokenizer"
            start_token = 0
        elif model == "instruct-blip/flant5xxl":
            model_type = "flant5xxl"
            model_name = "blip2_t5_instruct"
            tokenizer = "self.model.t5_tokenizer"
            start_token = 0
        else:
            raise ValueError(f"Unknown InstructBLIP model {model}")

        model, vis_processors, _ = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=device,
            vision_only=self.vision_only,
        )
        self.model = model
        self.vis_processors = vis_processors

        # if not torch.distributed.is_initialized():
        #     self.model.to(device or "cpu").eval()
        self.prompt = prompt
        self.device = device or "cpu"
        self.pure_llm = kwargs.get("pure_llm", False)
        self.return_embeds = kwargs.get("return_embeds", False)
        self.stop_token = 29889  # Token for period after sentence
        self.start_token = start_token  # First token predicted by model, after prompt. Used for getting confidences in self.get_caption_distributions

        if self.vision_only:
            return

        self.tokenizer = eval(tokenizer)

        self._init_cross_attention()

    def _init_cross_attention(self):
        """Save original cross-attention settings, in case of turning off cross-attention."""
        self.layer_idx2original_cross_attention = {}
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

    def get_vision_features(self, inputs) -> torch.Tensor:
        raise NotImplementedError
        # breakpoint()
        # with self.maybe_autocast():
        #     image_embeds = self.ln_vision(self.visual_encoder(image))

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
        output_hidden_states=True,
    ) -> List[str]:

        out = self.model.generate(  # type: ignore
            inputs,
            num_beams=num_beams,
            temperature=temperature,
            max_length=max_length,
            return_embeds=return_embeds,
            top_p=topp,
            use_nucleus_sampling=topp > 0,
        )
        if return_embeds:
            baseline_caption, inputs_embeds, inputs_query, atts_query, outputs = out
        else:
            baseline_caption = out

        baseline_caption = [postprocess_caption(b.strip()) for b in baseline_caption]
        print(baseline_caption)

        if return_embeds:
            return baseline_caption, inputs_embeds, inputs_query, outputs
        if return_tokens:
            return baseline_caption, out[1]
        return baseline_caption

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

        from tqdm import tqdm

        # Initialize vision outputs to None; they will be computed on
        # first iteration and saved.
        inputs_vision = None
        atts_vision = None

        for i in tqdm(range(1, encoded_caption.shape[1])):

            def prefix_allowed_tokens_fn(batch_id, sent):
                if sent.shape[0] < i:
                    tokens = encoded_caption[batch_id][sent.shape[0]].tolist()
                else:
                    tokens = None
                return tokens

            gen = self.model.generate(  # type: ignore
                inputs,
                num_beams=1,
                max_length=i + 1,
                return_dict=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                pure_llm=False if not language_only else self.pure_llm,
                return_embeds=True,
                inputs_vision=inputs_vision,
                atts_vision=atts_vision
            )

            inputs_vision, atts_vision = gen[2], gen[3]

            distributions.append(gen[-1].scores[i - 1][0].detach().cpu())

        if language_only:
            self._enable_cross_attention()

        return torch.stack(distributions)

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

        if generation_type == "normal":
            return self.get_baseline_caption(
                inputs,
                do_sample=do_sample,
                num_beams=num_beams,
                max_length=max_length,
                temperature=temperature,
                return_embeds=return_embeds,
                topp=topp,
            )
        elif generation_type == "iterative":
            return self.get_caption_iterative_filtering(
                inputs,
                do_sample=do_sample,
                num_beams=num_beams,
                max_length=max_length,
                temperature=temperature,
                raw_image=raw_image,
            )

    def _disable_cross_attention(self):
        """
        Turn off cross-attention in model QFormer layers. Used to obtain a caption conditioned only on language, not the image.
        Modifies self.model in-place.
        """
        for layer in self.model.Qformer.bert.encoder.layer:
            layer.has_cross_attention = False

    def _enable_cross_attention(self):
        """
        Retores cross-attention in model QFormer layers to the original settings.
        Modifies self.model in-place.
        """
        for idx, layer in enumerate(self.model.Qformer.bert.encoder.layer):
            layer.has_cross_attention = self.layer_idx2original_cross_attention[idx]

    def take_grads(
        self,
        caption: str,
        inputs_query,
        tokens,
    ):

        prompt_tokens = self.tokenizer(
            self.prompt, padding="longest", return_tensors="pt"
        ).to(
            inputs_query.device
        )  # has keys input_ids and attention_mask of 1s

        gradients = []
        final_inputs_embeds = None
        for i, token in enumerate(tokens[:-1]):
            llm_tokens = torch.cat(
                (prompt_tokens["input_ids"], tokens[: i + 1].unsqueeze(0)), dim=1
            )

            # input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            inputs_embeds = self.model.llm_model.get_input_embeddings()(llm_tokens)
            inputs_embeds = torch.cat([inputs_query, inputs_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
                inputs_embeds.device
            )
            # attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            input_ids = None

            model_inputs = self.model.llm_model.prepare_inputs_for_generation(
                input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
            model_inputs["inputs_embeds"].requires_grad = True
            # model_inputs['inputs_embeds']: shape input_sequence_length x input_token_dimension (e.g., 42 x 4096)

            out = self.model.llm_model(
                **model_inputs,
                return_dict=True,
                output_attentions=True,
                output_hidden_states=True,
            )

            next_token_logits = out.logits[0, -1, :]
            # print(tokens[i+1], torch.topk(next_token_logits, 10)[1])

            N_query = self.model.query_tokens.shape[1]
            # queries = model_inputs['inputs_embeds'][0, :N_query, :]

            selected_index = tokens[i + 1]
            grads = torch.autograd.grad(
                next_token_logits[selected_index], model_inputs["inputs_embeds"]
            )[0][
                0
            ]  # same shape as inputs_embeds

            gradients.append(grads)
            final_inputs_embeds = model_inputs["inputs_embeds"]

        return gradients, final_inputs_embeds


class InstructBLIPVicuna7B(InstructBLIP):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="instruct-blip/vicuna7b", device=device, **kwargs)


class InstructBLIPVicuna13B(InstructBLIP):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="instruct-blip/vicuna13b", device=device, **kwargs)


class InstructBLIPFlanT5XL(InstructBLIP):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="instruct-blip/flant5xl", device=device, **kwargs)


class InstructBLIPFlanT5XXL(InstructBLIP):
    def __init__(self, device: Optional[str] = None, **kwargs):
        super().__init__(model="instruct-blip/flant5xxl", device=device, **kwargs)

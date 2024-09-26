from typing import Dict, List, Literal, Optional, Union, Any, Tuple
import logging
from PIL import Image
import string
import warnings
import torch
from torch import nn
import torch.distributed as dist

import transformers
from transformers import PreTrainedModel, LogitsProcessor
from transformers.modeling_outputs import ModelOutput
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers.generation.utils import (
    GenerationMixin,
    BeamSearchOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.beam_search import BeamScorer

from src.caption.base import CaptionEngine
from src.caption.blip_engine import BLIP2CaptionEngine, _BLIP_DEFAULT_PROMPT
from src.caption.instruct_blip_engine import InstructBLIP, _INSTRUCT_BLIP_DEFAULT_PROMPT
from src.caption.utils import postprocess_caption
from src.utils.pytorch import torch_int_div

from src.caption.lavis.models import load_model_and_preprocess


class EntropyThresholdBLIP2Engine(BLIP2CaptionEngine):
    def __init__(
            self,
            model_name: str = "Salesforce/blip2-opt-2.7b-coco",
            device: Optional[str] = None,
            threshold_type: Literal['entropy', 'vocab'] = 'entropy',
            distribution_type: Literal['MI', 'CAD'] = 'MI',
            threshold: Optional[float] = 1.0,
            vocab_label_file: Optional[str] = None,
            pure_llm: Optional[bool] = False,

            # Distribution modification parameters
            alpha: Optional[float] = 1.0,
            topk: Optional[int] = -1,
            renormalize: Optional[bool] = False,

            **kwargs
    ):
        self.device = device or "cpu"
        self.model = EntropyThresholdBLIP2Model(
            model_name,
            self.device,
            threshold_type = threshold_type,
            distribution_type = distribution_type,
            threshold = threshold,
            alpha = alpha,
            topk = topk,
            renormalize = renormalize,
            pure_llm = pure_llm,
            vocab_label_file = vocab_label_file
        )
        self.prompt = _BLIP_DEFAULT_PROMPT
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor_kwargs = {"return_tensors": "pt", "text": self.prompt}

        self.tokenizer = self.processor.tokenizer
        self.threshold_type = threshold_type
        self.distribution_type = distribution_type
        self.vocab_size = self.tokenizer.vocab_size
        self.threshold = threshold
        self.alpha = alpha
        self.topk = topk
        self.pure_llm = pure_llm


    def _get_captioner_class(self):
        raise NotImplementedError

    def _disable_cross_attention(self):
        self.model._disable_cross_attention()

    def _enable_cross_attention(self):
        self.model._enable_cross_attention()

    def _preprocess_image(self, raw_image: Image.Image, prompt: Optional[str] = None) -> torch.Tensor:
        return self.processor(raw_image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

    def get_caption_distributions(
        self,
        raw_image: Image,
        force_caption: str,
        remove_punctuation: bool = False,
    ) -> Dict[str, Any]:

        vocab_size = self.tokenizer.vocab_size
        if remove_punctuation:
            force_caption = force_caption.translate(str.maketrans("", "", string.punctuation)).lower()
        encoded = self.get_encoded_caption(force_caption) # Size = 1 x sequence length
        tokens_decoded = self.tokenizer.convert_ids_to_tokens(encoded[0])[1:] # remove BOS

        full_distributions = self.get_forced_output_distributions(raw_image, encoded, vocab_size, language_only=False)

        return {
            'caption': force_caption,
            'encoded_caption': encoded.cpu(),
            'tokens_decoded': tokens_decoded,
            'full_distributions': full_distributions.cpu(),
            'language_distributions': None
        }


class EntropyThresholdInstructBLIPEngine(InstructBLIP):
    def __init__(
            self,
            model_name: str = "instruct-blip",
            device: Optional[str] = None,
            threshold_type: Literal['entropy', 'vocab'] = 'entropy',
            distribution_type: Literal['MI', 'CAD'] = 'MI',
            threshold: Optional[float] = 1.0,
            vocab_label_file: Optional[str] = None,
            pure_llm: Optional[bool] = False,

            # Distribution modification parameters
            alpha: Optional[float] = 1.0,
            topk: Optional[int] = -1,
            renormalize: Optional[bool] = False,

            **kwargs
    ):
        self.device = device or "cpu"
        self.model = EntropyThresholdInstructBLIPModel(
            self.device,threshold = threshold, alpha = alpha, topk = topk, renormalize = renormalize, pure_llm = pure_llm,
            vocab_label_file = vocab_label_file, threshold_type = threshold_type, distribution_type = distribution_type
        )
        self.vis_processors = self.model.vis_processors
        self.tokenizer = self.model.model.llm_tokenizer
        self.prompt = _INSTRUCT_BLIP_DEFAULT_PROMPT
        self.threshold = threshold
        self.alpha = alpha
        self.topk = topk
        self.pure_llm = pure_llm

    def _get_captioner_class(self):
        raise NotImplementedError

    def get_caption_distributions(
        self,
        raw_image: Image,
        force_caption: str,
        remove_punctuation: bool = False,
    ) -> Dict[str, Any]:

        vocab_size = self.tokenizer.vocab_size
        if remove_punctuation:
            force_caption = force_caption.translate(str.maketrans("", "", string.punctuation)).lower()
        encoded = self.get_encoded_caption(force_caption) # Size = 1 x sequence length
        tokens_decoded = self.tokenizer.convert_ids_to_tokens(encoded[0])[1:] # remove BOS

        full_distributions = self.get_forced_output_distributions(raw_image, encoded, vocab_size, language_only=False)

        return {
            'caption': force_caption,
            'encoded_caption': encoded.cpu(),
            'tokens_decoded': tokens_decoded,
            'full_distributions': full_distributions.cpu(),
            'language_distributions': None
        }


class ThresholdModel(nn.Module):
    def __init__(self):
        super().__init__()

    def _find_modify_indices(self, next_token_logits, log_softmax):
        raise NotImplementedError

    def _compute_modified_distribution(self, modify_indices, next_token_logits, log_softmax, language_next_token_logits):
        raise NotImplementedError

    def _forward(
        self,
            extra_language_qformer_output=None,
        extra_qformer_input_ids=None,
        extra_running_input_ids=None,
        **kwargs
    ) -> ModelOutput:
        # Compute logits for the original model
        outputs = self.model.language_model.original_forward(**kwargs)

        # Moved here, from beam search
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = self.model.language_model.adjust_logits_during_generation(
            next_token_logits, cur_len=extra_running_input_ids.shape[1]
        )
        log_softmax = nn.functional.log_softmax(next_token_logits, dim=-1) # (batch_size * num_beams, vocab_size)

        modify_indices = self._find_modify_indices(next_token_logits, log_softmax)

        if len(modify_indices) > 0:
            # Prepare inputs
            llm_input_ids = extra_running_input_ids[modify_indices] if self.pure_llm else torch.cat(
                    (extra_qformer_input_ids[modify_indices], extra_running_input_ids[modify_indices]), dim=1
            )
            updated_inputs = self._get_language_model_inputs(
                llm_input_ids,
                None,
                query_output=None if self.pure_llm else extra_language_qformer_output[modify_indices],
            )

            # Use non-cached forward pass for language-only model
            kwargs['use_cache'] = False
            kwargs.update(updated_inputs)
            kwargs.pop('input_ids', None)
            kwargs.pop('past_key_values', None)

            model_inputs = self.model.language_model.prepare_inputs_for_generation(llm_input_ids, **kwargs)
            model_inputs.pop('extra_running_input_ids', None)

            # Compute logits without image conditioning
            language_outputs = self.model.language_model.original_forward(**model_inputs)
            language_next_token_logits = language_outputs.logits[:, -1, :]

            modified = self._compute_modified_distribution(modify_indices, next_token_logits, log_softmax, language_next_token_logits)

            # Make everything past the original distribution top-k logits a minimum value
            topk = self.topk
            if topk > -1:
                k = language_next_token_logits.shape[-1] - topk
                bottom_k_values, bottom_k_indices = torch.topk(log_softmax[modify_indices], k=k, dim=-1, largest=False)
                bottom_k_mask = torch.zeros_like(log_softmax[modify_indices], dtype=torch.bool)
                bottom_k_mask.scatter_(1, bottom_k_indices, True)
                modified[bottom_k_mask] = -torch.inf

            if self.renormalize:
                modified = torch.nn.functional.log_softmax(modified, dim=-1)
            modified = modified.type(outputs.logits.dtype)

        outputs.logits[:, -1, :] = log_softmax
        if len(modify_indices) > 0:
            outputs.logits[modify_indices, -1, :] = modified

        return outputs


    def _find_modify_indices_entropy(self, next_token_logits, log_softmax) -> torch.Tensor:

        # Compute entropy of full distribution
        softmax = nn.functional.softmax(next_token_logits, dim=-1)
        log_softmax = nn.functional.log_softmax(next_token_logits, dim=-1) # (batch_size * num_beams, vocab_size)
        entropy = -torch.sum(softmax * log_softmax, dim=-1)
        max_entropy = torch.log(torch.tensor(softmax.shape[-1], dtype=torch.float16, device=softmax.device))
        entropy_ratio = entropy / max_entropy

        # If entropy_ratio is < threshold, use full distribution. Else, compute `full - alpha * lang`,
        # where lang is the output distribution from the language-only model.
        modify_indices = torch.where(entropy_ratio >= self.threshold)[0]

        return modify_indices

    def _find_modify_indices_vocab(self, next_token_logits, log_softmax) -> torch.Tensor:

        # Compute groundedness score and compare to threshold
        softmax = nn.functional.softmax(next_token_logits, dim=-1)
        groundedness = (self.binary_grounding * softmax[:, :-1]).sum(dim=-1) # Remove extra last column to match vocab size

        # If groundedness is < threshold, use full distribution. Else, compute `full - alpha * lang`,
        # where lang is the output distribution from the language-only model.
        modify_indices = torch.where(groundedness >= self.threshold)[0]

        return modify_indices

    def _compute_modified_distribution_MI(self, modify_indices, next_token_logits, log_softmax, language_next_token_logits):
        language_log_softmax = nn.functional.log_softmax(language_next_token_logits, dim=-1)
        return log_softmax[modify_indices] - self.alpha * language_log_softmax

    def _compute_modified_distribution_CAD(self, modify_indices, next_token_logits, log_softmax, language_next_token_logits):
        cad = (1 + self.alpha) * next_token_logits[modify_indices] - self.alpha * language_next_token_logits
        return nn.functional.log_softmax(cad, dim=-1)

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        """
        Most of code is copied directly from transformers.generation.utils.GenerateMixin.beam_search.
        Small modification: Original beam search always ran a log_softmax on the logits. We don't want to do this if the
        entropy objective is used. Instead, rely on language model forward to do log_softmax.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.language_model.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.language_model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.model.language_model.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.model.language_model.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.language_model.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.model.language_model.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model.language_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.model.language_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.model.language_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_scores = outputs.logits[:, -1, :] # Already post log-softmax, if that's being used.
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.language_model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model.language_model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.language_model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self.model.language_model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.language_model.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.model.language_model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.model.language_model.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]




class EntropyThresholdBLIP2Model(ThresholdModel):
    def __init__(
            self,
            model_name: str,
            device: str,
            threshold_type: Literal['entropy', 'vocab'] = 'entropy',
            distribution_type: Literal['MI', 'CAD'] = 'MI',
            alpha: float = 1.0,
            topk: int = -1,
            renormalize: bool = False,
            threshold: float = 1.0,
            vocab_label_file: Optional[str] = None,
            pure_llm: bool = False
    ):
        super().__init__()
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        # if not torch.distributed.is_initialized():
        self.model.to(device).eval()
        self.model.language_model.original_forward = self.model.language_model.forward
        self.model.language_model.forward = self._forward

        if threshold_type == 'entropy':
            self._find_modify_indices = self._find_modify_indices_entropy
        elif threshold_type == 'vocab':
            self._find_modify_indices = self._find_modify_indices_vocab
            self.grounding_labels = torch.load(vocab_label_file)
            binary_grounding = [1 if self.grounding_labels[i] == 'grounded' else 0 for i in range(len(self.grounding_labels))]
            self.binary_grounding = torch.Tensor(binary_grounding).to(device)

        if distribution_type == 'MI':
            self._compute_modified_distribution = self._compute_modified_distribution_MI
        elif distribution_type == 'CAD':
            self._compute_modified_distribution = self._compute_modified_distribution_CAD


        self.model.language_model.prepare_inputs_for_generation = self._prepare_inputs_for_generation
        self.model.language_model.beam_search = self._beam_search
        self.alpha = alpha
        self.threshold = threshold
        self.topk = topk
        self.renormalize = renormalize
        self.pure_llm = pure_llm
        self._init_cross_attention()


    def _init_cross_attention(self):
        """Save original cross-attention settings, in case of turning off cross-attention."""
        self.layer_idx2original_cross_attention = {}
        for idx, layer in enumerate(self.model.qformer.encoder.layer):
            self.layer_idx2original_cross_attention[idx] = layer.has_cross_attention

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self.model._preprocess_accelerate()

        # 1. Get image embeddings
        batch_size = pixel_values.shape[0]
        image_embeds = self.model.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # 2. Get language model inputs with full QFormer
        query_output = self._get_qformer_output(image_embeds, image_attention_mask)
        language_inputs = self._get_language_model_inputs(
            input_ids,
            attention_mask,
            query_output=query_output,
        )
        inputs_embeds, attention_mask = language_inputs["inputs_embeds"], language_inputs["attention_mask"]

        # 3. Precompute QFormer output with language-only QFormer
        self._disable_cross_attention()
        language_query_outputs = self._get_qformer_output(image_embeds, image_attention_mask)
        self._enable_cross_attention()

        # 4. Create dict of extra tensors needed for getting language-only logits during decoding
        extra_kwargs = {
            "extra_language_qformer_output": language_query_outputs,
            "extra_qformer_input_ids": input_ids,
            "extra_running_input_ids": None,
        }

        # 5. Generate with custom language model
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
            **extra_kwargs,
        )
        return outputs

    def _prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        extra_kwargs = {k:v for k,v in kwargs.items() if k.startswith('extra_')}
        extra_kwargs['extra_running_input_ids'] = input_ids

        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        model_inputs.update(extra_kwargs)

        return model_inputs

    def _get_qformer_output(self, image_embeds, image_attention_mask):
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state
        query_output = self.model.language_projection(query_output)
        return query_output

    def _get_language_model_inputs(self, input_ids, attention_mask, query_output=None):
        if query_output is None:
            raise NotImplementedError
        #language_model_inputs = self.model.language_projection(query_output)
        language_attention_mask = torch.ones(
            query_output.size()[:-1], dtype=torch.long, device=query_output.device
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.model.config.text_config.bos_token_id]])
                .repeat(query_output.shape[0], 1)
                .to(language_attention_mask.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # Concatenate query embeddings with prompt/partially generated output embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([query_output, inputs_embeds.to(query_output.device)], dim=1)
        return {'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask}

    def _disable_cross_attention(self):
        """
        Turn off cross-attention in model QFormer layers. Used to obtain a caption conditioned only on language, not the image.
        Modifies self.model in-place.
        """
        for layer in self.model.qformer.encoder.layer:
            layer.has_cross_attention = False

    def _enable_cross_attention(self):
        """
        Retores cross-attention in model QFormer layers to the original settings.
        Modifies self.model in-place.
        """
        for idx, layer in enumerate(self.model.qformer.encoder.layer):
            layer.has_cross_attention = self.layer_idx2original_cross_attention[idx]



class EntropyThresholdBLIP2COCOBase(EntropyThresholdBLIP2Engine):
    def __init__(self, model_name = "Salesforce/blip2-opt-2.7b-coco", **kwargs):
        super().__init__(model_name, **kwargs)

class EntropyThresholdBLIP2Base(EntropyThresholdBLIP2Engine):
    def __init__(self, model_name = "Salesforce/blip2-opt-2.7b", **kwargs):
        super().__init__(model_name, **kwargs)


class EntropyThresholdInstructBLIPModel(ThresholdModel):
    def __init__(
            self,
            device: str,
            threshold_type: Literal['entropy', 'vocab'] = 'entropy',
            distribution_type: Literal['MI', 'CAD'] = 'MI',
            vocab_label_file: Optional[str] = None,
            alpha: float = 1.0,
            topk: int = -1,
            renormalize: bool = False,
            threshold: float = 1.0,
            pure_llm: bool = False
    ):
        super().__init__()

        model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
        self.model = model
        self.vis_processors = vis_processors

        if not torch.distributed.is_initialized():
            self.model.to(device).eval()

        self.model.language_model = self.model.llm_model
        self.model.language_model.original_forward = self.model.language_model.forward
        self.model.language_model.forward = self._forward
        if threshold_type == 'entropy':
            self._find_modify_indices = self._find_modify_indices_entropy
        elif threshold_type == 'vocab':
            self._find_modify_indices = self._find_modify_indices_vocab
            self.grounding_labels = torch.load(vocab_label_file)
            binary_grounding = [1 if self.grounding_labels[i] == 'grounded' else 0 for i in range(len(self.grounding_labels))]
            self.binary_grounding = torch.Tensor(binary_grounding).to(device)

        if distribution_type == 'MI':
            self._compute_modified_distribution = self._compute_modified_distribution_MI
        elif distribution_type == 'CAD':
            self._compute_modified_distribution = self._compute_modified_distribution_CAD

        self.model.language_model.prepare_inputs_for_generation = self._prepare_inputs_for_generation
        self.model.language_model.beam_search = self._beam_search
        self.alpha = alpha
        self.threshold = threshold
        self.topk = topk
        self.renormalize = renormalize
        self.pure_llm = pure_llm
        self._init_cross_attention()

    def _init_cross_attention(self):
        """Save original cross-attention settings, in case of turning off cross-attention."""
        self.layer_idx2original_cross_attention = {}
        for idx, layer in enumerate(self.model.Qformer.bert.encoder.layer):
            self.layer_idx2original_cross_attention[idx] = layer.has_cross_attention

    def _disable_cross_attention(self):
        for layer in self.model.Qformer.bert.encoder.layer:
            layer.has_cross_attention = False

    def _enable_cross_attention(self):
        for idx, layer in enumerate(self.model.Qformer.bert.encoder.layer):
            layer.has_cross_attention = self.layer_idx2original_cross_attention[idx]

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        return_dict=False,
        prefix_allowed_tokens_fn=None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        self.model.llm_tokenizer.padding_side = "left"

        if type(samples) == list:
            # Hack - using batch eval makes a list of samples. Convert to single dictionary with
            # stacked values.
            combined = {}
            for k,v in samples[0].items():
                if type(v) == torch.Tensor:
                    combined[k] = torch.cat([s[k] for s in samples], dim=0)
                else:
                    combined[k] = [s[k] for s in samples]
            samples = combined

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.model.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.model.query_tokens.expand(bs, -1, -1)
        if self.model.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.model.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.model.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            raise NotImplementedError("Video data is not supported yet.")
        else:
            with self.model.maybe_autocast():
                image_embeds = self.model.ln_vision(self.model.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            query_output = self._get_qformer_output(
                text_Qformer, Qformer_atts, query_tokens, image_embeds, image_atts
            )

            self._disable_cross_attention()
            language_query_output = self._get_qformer_output(
                text_Qformer, Qformer_atts, query_tokens, image_embeds, image_atts
            )
            self._enable_cross_attention()

            atts_llm = torch.ones(query_output.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.model.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        extra_kwargs = {
            "extra_language_qformer_output": language_query_output,
            "extra_qformer_input_ids": llm_tokens.input_ids,
            "extra_running_input_ids": None,
        }

        with self.model.maybe_autocast():
            inputs_embeds = self.model.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([query_output, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.model.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=return_dict,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                output_scores=True,
                **extra_kwargs,
            )

        tokens = outputs[0] if return_dict else outputs

        tokens[tokens == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.model.llm_tokenizer.batch_decode(tokens, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        if return_dict:
            return output_text, outputs
        return output_text


    def _prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        extra_kwargs = {k:v for k,v in kwargs.items() if k.startswith('extra_')}
        extra_kwargs['extra_running_input_ids'] = input_ids

        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        model_inputs.update(extra_kwargs)
        return model_inputs

    def _get_qformer_output(self, text_Qformer, Qformer_atts, query_tokens, image_embeds, image_atts):
        if self.model.qformer_text_input:
            query_output = self.model.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        inputs_llm = self.model.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        return inputs_llm

    def _get_language_model_inputs(self, input_ids, attention_mask, query_output=None):
        if input_ids is None:
            breakpoint()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Concatenate query embeddings with prompt/partially generated output embeddings
        inputs_embeds = self.model.llm_model.get_input_embeddings()(input_ids)

        if query_output is not None:
            inputs_embeds = torch.cat([query_output, inputs_embeds], dim=1)
            atts_llm = torch.ones(query_output.size()[:-1], dtype=torch.long).to(query_output.device)
            attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return {'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask, 'position_ids': position_ids}



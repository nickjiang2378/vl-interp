from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from collections import defaultdict
import logging
from PIL import Image
import torch
import string
from torchvision import transforms
import numpy as np
from scipy.stats import entropy


class CaptionEngineAbstract(ABC):
    @abstractmethod
    def __init__(self, device: Optional[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        raw_image: Image,
        n_captions: int = 1,
        temperature: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> List[str]:
        # Takes an RGB image and returns a list of captions
        raise NotImplementedError()

    @abstractmethod
    def get_baseline_caption(self, inputs: Any) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_caption_hallucination_mode(
        self,
        raw_image: Image.Image,
        force_caption: Optional[str] = None,
        hc_confs: List[str] = None,
        return_attentions: bool = False,
    ) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_encoded_caption(self, caption: str) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_forced_output_distributions(
        self,
        raw_image: Image.Image,
        encoded_caption: torch.Tensor,
        vocab_size: int,
        language_only: bool = False,
    ) -> torch.Tensor:
        """
        Output the logit distributions for a teacher-forced caption.
        args:
            raw_image: PIL Image input
            encoded_caption: 1 x seq_len+1 Tensor of encoded caption, includes BOS & EOS
            vocab_size: int, size of vocabulary
            language_only: bool, if True, return the distribution of logits that are conditioned only on the language prefix and not the image at each time step.

        return seq_len x vocab_size Tensor of logit distributions at each time step of teacher-forced encoded_caption. Includes EOS but not BOS.
        """
        raise NotImplementedError


class CaptionEngine(CaptionEngineAbstract):
    def __init__(self, *args, **kwargs):
        self.start_token = None
        self.verifier_initialized = False
        self.verification_threshold = None

    def initialize_verifier(self, verifier_type, verification_threshold):
        logging.info(
            f"Initializing verifier {verifier_type} with threshold {verification_threshold} ..."
        )
        assert verification_threshold is not None, "Must provide verification threshold"
        if verifier_type == "openclip-ViTG":
            import open_clip

            model_type, pretrained = ("ViT-bigG-14", "laion2b_s39b_b160k")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_type, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_type)
            model.to(self.device)
            self.verifier = model
            self.verifier_type = verifier_type
            self.verification_threshold = verification_threshold
            self.verifier_preprocess = preprocess
            self.verifier_tokenizer = tokenizer
            self.verifier_initialized = True
        else:
            raise NotImplementedError(
                f"Verifier type {verifier_type} not implemented for initialize_verifier."
            )

    def verify_caption(self, raw_image: Image.Image, sentence_list: List[str]):
        """
        Given list of sentences, return list of same length of 0s and 1s, where 0 = reject, 1 = pass.
        """
        if "openclip" in self.verifier_type:
            # Get image embedding
            image_input = (
                self.verifier_preprocess(raw_image).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                image_features = self.verifier.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Get text embeddings
            text_tokenized = self.verifier_tokenizer(sentence_list).to(self.device)
            with torch.no_grad():
                text_features = self.verifier.encode_text(text_tokenized)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarities and threshold for decisions
            similarities = (100.0 * image_features @ text_features.T)[
                0
            ]  # Length of sentence_list
            similarities = similarities.cpu()
            decisions = (similarities > self.verification_threshold).int().tolist()
            print(sentence_list)
            print(decisions)

        else:
            raise NotImplementedError(
                f"Verifier type {self.verifier_type} not implemented for verify_caption."
            )

        return decisions

    def get_caption_iterative_filtering(
        self,
        inputs,
        do_sample=False,
        num_beams=16,
        max_length=256,
        temperature=1.0,
        raw_image=None,
    ) -> List[str]:
        assert (
            self.verifier_initialized
        ), "Must initialize verifier before using iterative filtering."

        caption, inputs_embeds, inputs_query, outputs = self.get_baseline_caption(
            inputs,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            temperature=temperature,
            return_embeds=True,
        )
        tokens = outputs[0]
        stop_idxs = torch.where(tokens == self.stop_token)[0]  # Indices of stop tokens
        current_start = 0
        sentences = []
        for stop_idx in stop_idxs:
            sentences.append(tokens[current_start : stop_idx + 1])
            current_start = stop_idx + 1
        decoded_sentences = [
            self.tokenizer.decode(sentence) for sentence in sentences
        ]  # List[str] of sentences

        # Get reject/accept decisions for each sentence (0 = reject, 1 = accept)
        decisions = self.verify_caption(raw_image, decoded_sentences)

        return caption

    def get_encoded_caption(self, caption: str) -> torch.Tensor:
        return self.tokenizer.encode(caption, return_tensors="pt").to(self.device)

    def compute_sparse_distribution_metrics(
        self,
        full_distributions: torch.Tensor,
        language_distributions: torch.Tensor,
        encoded_caption: torch.Tensor,
    ) -> Dict[str, Any]:

        """
        N = sequence length
        full_distributions: N x vocab size
        language_distributions: N x vocab size
        encoded_caption: N
        """
        N = full_distributions.shape[0]
        assert language_distributions.shape[0] == N
        assert encoded_caption.shape[0] == N

        full_logits = [
            d[token].item() for d, token in zip(full_distributions, encoded_caption)
        ]
        language_logits = [
            d[token].item() for d, token in zip(language_distributions, encoded_caption)
        ]

        full_softmax_scores = torch.softmax(full_distributions.float(), dim=-1)
        language_softmax_scores = torch.softmax(language_distributions.float(), dim=-1)

        full_entropies = [entropy(d) for d in full_softmax_scores]
        language_entropies = [entropy(d) for d in language_softmax_scores]
        kl_divs_qk_lang = [
            entropy(VL, qk=L)
            for VL, L in zip(full_softmax_scores, language_softmax_scores)
        ]
        kl_divs_qk_full = [
            entropy(L, qk=VL)
            for VL, L in zip(full_softmax_scores, language_softmax_scores)
        ]

        out = {
            "full_logit": full_logits,
            "language_logit": language_logits,
            "full_entropy": full_entropies,
            "language_entropy": language_entropies,
            "kl_div_qk_lang": kl_divs_qk_lang,
            "kl_div_qk_full": kl_divs_qk_full,
        }
        return out

    def get_caption_distributions(
        self,
        raw_image: Image,
        force_caption: str,
        tokens: Optional[List[int]] = None,
        prompt: Optional[str] = None,
        remove_punctuation=False,
        sparse=False,
    ) -> Dict[str, Any]:

        vocab_size = self.tokenizer.vocab_size

        if tokens is None:
            if remove_punctuation:
                force_caption = force_caption.translate(
                    str.maketrans("", "", string.punctuation)
                ).lower()
            encoded = self.get_encoded_caption(
                force_caption
            )  # Size = 1 x sequence length
            if self.start_token is not None and encoded[0][0] != self.start_token:
                # Add start_token to beginning of tokens
                encoded = torch.cat(
                    (torch.tensor([[self.start_token]]).to(self.device), encoded), dim=1
                )
            tokens_decoded = self.tokenizer.convert_ids_to_tokens(encoded[0])
        else:
            if self.start_token is not None and tokens[0] != self.start_token:
                # Add start_token to beginning of tokens
                tokens = [self.start_token] + tokens
            encoded = torch.tensor(tokens).long().unsqueeze(0).to(self.device)
            tokens_decoded = self.tokenizer.convert_ids_to_tokens(tokens)

        # encoded begins with self.start_token (and tokens_decoded starts with decoded version)

        full_distributions = self.get_forced_output_distributions(
            raw_image, encoded, vocab_size, language_only=False, prompt=prompt
        )
        language_distributions = self.get_forced_output_distributions(
            raw_image, encoded, vocab_size, language_only=True, prompt=prompt
        )

        full_distributions = full_distributions.cpu()
        language_distributions = language_distributions.cpu()
        encoded = encoded.cpu()[0]

        if self.start_token is not None:
            encoded = encoded[1:]  # Remove start_token

        assert (
            full_distributions.shape == language_distributions.shape
        ), f"Shape mismatch. full_distributions.shape: {full_distributions.shape}, language_distributions.shape: {language_distributions.shape}"

        sparse_metrics = self.compute_sparse_distribution_metrics(
            full_distributions, language_distributions, encoded
        )

        out = {
            "caption": force_caption,
            "encoded_caption": encoded,
            "tokens_decoded": tokens_decoded,
        }
        out.update(sparse_metrics)
        if not sparse:
            out.update(
                {
                    "full_distributions": full_distributions,
                    "language_distributions": language_distributions,
                }
            )

        return out

    def get_caption_hallucination_mode(
        self,
        raw_image: Image,
        force_caption: Optional[str] = None,
        hc_confs: List[str] = ["logit"],
        return_attentions: Optional[bool] = False,
    ) -> Dict[str, Any]:
        assert len(hc_confs) > 0, "hc_confs should not be empty"

        baseline_gen, baseline_caption = self.get_baseline_gen(force_caption, raw_image)

        # Get word confidences for each hc_confs method.
        # For now, assume the aggregation is "mean" over multiple tokens that make up one word.
        # confidence_data: Dict[str, Dict[str, List[float]]]. maps hc_conf to dictionary of format:
        #  {'all_confidences': List[float], 'word_confidences': List[List[float]], 'word_confidences_aggregated': List[float]}
        confidence_data = {}
        for hc_conf in hc_confs:
            confidence_data[hc_conf] = {
                "all_confidences": [],
                "word_confidences": [],
                "word_confidences_aggregated": [],
            }
        tokens = baseline_gen.sequences[0]  # includes BOS
        attentions = self._get_generated_attention(baseline_gen)
        word_indices = self.tokens_to_word_indices(tokens)

        vocab_size = self.tokenizer.vocab_size
        encoded = self.get_encoded_caption(force_caption)  # Size = 1 x sequence length
        full_logit_distributions = self.get_forced_output_distributions(
            raw_image, encoded, vocab_size
        )  # no BOS, includes EOS
        tokens_decoded = self.tokenizer.convert_ids_to_tokens(tokens)[1:]  # remove BOS

        # Filter tokens for those that correspond to words.
        tokens = tokens[word_indices >= 0]

        # Only take values that correspond to words, to align tensors with `tokens`.
        # Don't use the BOS in word_indices.
        logit_distributions = [
            full_logit_distributions[i]
            for i in range(len(word_indices) - 1)
            if word_indices[i + 1] >= 0
        ]
        if return_attentions:
            for attention_type, attention in attentions.items():
                attention = [
                    attention[i]
                    for i in range(len(word_indices) - 1)
                    if word_indices[i + 1] >= 0
                ]
                attentions[attention_type] = attention

        # Align word indices with `tokens` (removes `-1` entries, e.g. for EOS).
        word_indices = word_indices[word_indices >= 0]

        # Remove punctuation and split caption into words.
        remove_chars = string.punctuation
        word_list = baseline_caption.translate(
            str.maketrans("", "", remove_chars)
        ).split()

        if return_attentions:
            word_to_attentions = {}
            for attention_type, attention in attentions.items():
                word_to_att = self._get_word_to_attentions(
                    attention, tokens, word_list, word_indices
                )
                word_to_attentions[attention_type] = word_to_att
        else:
            word_to_attentions = None

        for hc_conf in hc_confs:
            confs = self._get_confidences(hc_conf, tokens, logit_distributions)
            (
                word_confidences_lists,
                word_confidences_aggregated,
            ) = self.aggregate_confidences_by_words(confs, word_indices)

            confidence_data[hc_conf]["all_confidences"] = confs
            confidence_data[hc_conf]["word_confidences"] = word_confidences_lists
            confidence_data[hc_conf][
                "word_confidences_aggregated"
            ] = word_confidences_aggregated

        return {
            "baseline_caption": baseline_caption,
            "word_list": word_list,
            "confidence_data": confidence_data,
            "word_to_attentions": word_to_attentions,
            "logit_distributions": full_logit_distributions,
            "encoded_caption": encoded,
            "tokens_decoded": tokens_decoded,
        }

    def _get_confidences(self, hc_conf, tokens, logits):
        softmax_scores = [
            torch.nn.functional.softmax(logits[i], dim=-1) for i in range(len(tokens))
        ]
        if hc_conf == "logit":
            confs = [logits[i][tokens[i]].item() for i in range(len(tokens))]
        elif hc_conf == "softmax":
            confs = [softmax_scores[i][tokens[i]].item() for i in range(len(tokens))]
        elif hc_conf == "entropy":
            confs = [
                float(-entropy(softmax_scores[i].cpu())) for i in range(len(tokens))
            ]
        else:
            raise ValueError(f"Unknown hc_conf: {hc_conf}")
        return confs

    def _get_word_to_attentions(
        self, attention, tokens, word_list, word_indices
    ) -> Dict[str, List[Any]]:
        """
        Returns a dictionary mapping each word to a list of attentions of len(# tokens in word), where each element
               is a tuple of len(# layers), and each element is a tensor of size:
                (num_heads x gen_len x num_enc_tokens) for cross-attention, or
                (num_heads x gen_len x gen_len) for self-attention.
        """
        word_to_attentions = {(i, w): [] for (i, w) in enumerate(word_list)}
        index_to_word = {i: w for i, w in enumerate(word_list)}
        for i in range(len(tokens)):
            word_index = word_indices[i].item()
            word = index_to_word[word_index]
            att = attention[i]  # tuple of len(# layers)
            att = [a[0].detach().cpu() for a in att]
            word_to_attentions[(word_index, word)].append(att)

        return word_to_attentions

    def aggregate_confidences_by_words(
        self,
        confidences: torch.Tensor,
        word_indices: torch.Tensor,
        agg_fn: Callable[[torch.Tensor], float] = lambda x: x.mean().item(),
    ) -> List[float]:
        """
        Aggregates confidences by word, using the given aggregation function.

        Args:
            confidences (torch.Tensor[float]): A tensor of confidences, of shape (sequence_length,).
            word_indices (torch.Tensor[long]): A tensor of word indices, of shape (sequence_length,).
            agg_fn (Callable[[torch.Tensor], float]): An aggregation function that takes in a tensor and returns a single value (e.g., mean).
        Returns:
            word_confidences (List[List[float]]): A list of lists of unaggregated word confidences, of length num_words.
            word_confidences_aggregated (List[float]): A list of aggregated word confidences, of length num_words.
        """
        grouped = defaultdict(list)
        for word_index, conf in zip(word_indices, confidences):
            grouped[word_index.item()].append(conf)

        word_confidences = [grouped[idx] for idx in sorted(grouped.keys())]

        # Convert each list of confidences to a tensor, and apply the aggregation function.
        word_confidences_aggregated = [
            agg_fn(torch.Tensor(confidences)) for confidences in grouped.values()
        ]

        if -float("inf") in word_confidences_aggregated:
            breakpoint()

        return word_confidences, word_confidences_aggregated

    def tokens_to_word_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Maps a list of token IDs to a list of integers, where each integer corresponds to the index of the word that the token belongs to.
        Ignored tokens are mapped to -1, which are punctuation, BOS, EOS, or PAD.

        Args:
            tokens (torch.Tensor[int]): A tensor of token IDs, of shape (sequence_length,).
        Returns:
            word_indices (torch.Tensor[long]): A tensor of word indices, of shape (sequence_length,).
        """
        ignore = [
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.pad_token,
            "Ġ",  # 'space' token, don't count if it appears on its own (happens rarely)
        ]
        ignore.extend(list(string.punctuation))

        tokens_decoded = self.tokenizer.convert_ids_to_tokens(tokens)
        # tokens_decoded = [self.tokenizer.decode(torch.Tensor([t])) for t in tokens][1:]

        current_word_index = 0
        word_indices = []

        for token in tokens_decoded:
            if token in ignore:
                word_indices.append(-1)
            else:
                if token.startswith("Ġ"):
                    current_word_index += 1
                word_indices.append(current_word_index)

        return torch.Tensor(word_indices).long().to(tokens.device)

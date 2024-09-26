# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Any, Dict, List

from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.cider.cider import Cider
from vdtk.metrics.rouge.rouge import Rouge
from vdtk.metrics.tokenizer.ptbtokenizer import PTBTokenizer



def _cm_kys(samples: List[Dict[str, Any]], c_key: str, ref_key: str) -> List[Dict[str, Any]]:
    pass

class NGramMetrics():
    def __call__(
        self, samples: List[Dict[str, Any]], candidate_key: str, reference_key: str#, image_key: str, image_root_dir: str
    ) -> List[Dict[str, Any]]:
        # Compute the BLEU, ROUGE, and CIDEr scores
        chyp = {i: [d[candidate_key]] for i, d in enumerate(samples)}
        cref = {i: d[reference_key] for i, d in enumerate(samples)}

        tokenizer = PTBTokenizer()

        # Tokenize the hypotheses and references
        chyp_tok = tokenizer.tokenize(chyp)
        cref_tok = tokenizer.tokenize(cref)

        # Compute the BLEU, ROUGE, and CIDEr scores
        c_bleu_score, c_b_full = Bleu().compute_score(cref_tok, chyp_tok)
        c_rouge_score, c_r_full = Rouge().compute_score(cref_tok, chyp_tok)
        c_cider_score, c_c_full = Cider().compute_score(cref_tok, chyp_tok)

        # Update the scores
        for i, d in enumerate(samples):
            if d.get("metrics") is None:
                d["metrics"] = {}
            if d["metrics"].get("ngram") is None:
                d["metrics"]["ngram"] = {}

            d["metrics"]["ngram"]["bleu_1"] = float(c_b_full[0][i])
            d["metrics"]["ngram"]["bleu_2"] = float(c_b_full[1][i])
            d["metrics"]["ngram"]["bleu_3"] = float(c_b_full[2][i])
            d["metrics"]["ngram"]["bleu_4"] = float(c_b_full[3][i])
            d["metrics"]["ngram"]["rouge"] = float(c_r_full[i])
            d["metrics"]["ngram"]["cider"] = float(c_c_full[i])

        return samples

    def aggregate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Compute the BLEU, ROUGE, and CIDEr scores
        return {
            "bleu_1": float(sum([d["metrics"]["ngram"]["bleu_1"] for d in samples])) / (len(samples) + 1e-12),
            "bleu_2": float(sum([d["metrics"]["ngram"]["bleu_2"] for d in samples])) / (len(samples) + 1e-12),
            "bleu_3": float(sum([d["metrics"]["ngram"]["bleu_3"] for d in samples])) / (len(samples) + 1e-12),
            "bleu_4": float(sum([d["metrics"]["ngram"]["bleu_4"] for d in samples])) / (len(samples) + 1e-12),
            "rouge": float(sum([d["metrics"]["ngram"]["rouge"] for d in samples])) / (len(samples) + 1e-12),
            "cider": float(sum([d["metrics"]["ngram"]["cider"] for d in samples])) / (len(samples) + 1e-12),
        }



    
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    parser.add_argument("--candidate_key", type=str, default="baseline")
    parser.add_argument("--reference_key", type=str, default="references")
    #parser.add_argument("--image_key", type=str, default="image_id")
    #parser.add_argument("--image_root_dir", type=str, default=".")
    parser.add_argument("--save_file", type=str, default=None)

    args = parser.parse_args()

    samples = json.load(open(args.results_file))['samples']
    print(len(samples))
    metric = NGramMetrics()
    samples = metric(samples, args.candidate_key, args.reference_key)
    aggregated_metrics = metric.aggregate(samples)

    for k,v in aggregated_metrics.items():
        print(f"{k}: {v}")

    if args.save_file is not None:
        json.dump(aggregated_metrics, open(args.save_file, "w"))

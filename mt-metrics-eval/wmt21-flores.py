from re import M
from mt_metrics_eval import data
import bert_score
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
model_types = ["bert-base-multilingual-cased", "google/byt5-small"]
for model_type in model_types:
    logging.info(f"{model_type}")
    scorer = bert_score.scorer.BERTScorer(model_type=model_type)

    def MyMetric(out, ref, scorer):
        """Return a scalar score for given output/reference texts."""
        (P, R, F) = scorer.score(out, ref, batch_size=4)
        F = torch.mean(F, dim=0)
        return float(F[0])

    datasets = "wmt21.flores"
    lang_pairs = ['bn-hi', 'hi-bn', 'xh-zu', 'zu-xh']
    for lang_pair in lang_pairs:
        logging.info(f"{datasets} {lang_pair}")
        evs = data.EvalSet(datasets, lang_pair)

        sys_scores, doc_scores, seg_scores = {}, {}, {}
        ref = evs.ref
        for s, out in evs.sys_outputs.items():
            seg_scores[s] = [MyMetric([o], [r], scorer) for o, r in zip(out, ref)]

        # Official WMT correlations.
        logging.info(f"seg KendallLike: {evs.Correlation('seg', seg_scores).KendallLike()}")


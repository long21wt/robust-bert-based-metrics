import os
import torch
import bert_score
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from collections import defaultdict


def load_data(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            lines.append(l)
    return lines


def get_wmt19_seg_data(lang_pair, attack="no_attack"):
    src, tgt = lang_pair.split('-')
    rr_data = pd.read_csv(
        "wmt19/wmt19-metrics-task-package/manual-evaluation/RR-seglevel.csv", sep=' ')
    rr_data_lang = rr_data[rr_data['LP'] == lang_pair]
    systems = set(rr_data_lang['BETTER'])
    systems.update(list(set(rr_data_lang['WORSE'])))
    systems = list(systems)
    sentences = {}
    for system in systems:
        if lang_pair == "zh-en":
            with open("wmt19/wmt19-metrics-task-package/input/"
            "system-outputs/newstest2019/{}/newstest2019.{}".format(lang_pair, system)) as f:
                sentences[system] = f.read().split("\n")
        else:
            with open("wmt19/wmt19-metrics-task-package/input/"
            "system-outputs/newstest2019/{}/newstest2019.{}.{}".format(lang_pair, system, lang_pair)) as f:
                sentences[system] = f.read().split("\n")

    attack_method, pertubation = attack.split('_')
    if attack == "no_attack":
        with open("wmt19/wmt19-metrics-task-package/input/"
                "references/{}".format('newstest2019-{}{}-ref.{}'.format(src, tgt, tgt))) as f:
            references = f.read().split("\n")
    else:
        with open("wmt19/wmt19-metrics-task-package/input/"
                "attacked-references/{}/{}/{}".format(attack_method,
                                                      pertubation,
                                                      'newstest2019-{}{}-ref.{}'.format(src, tgt, tgt))) as f:
            references = f.read().split("\n")

    ref, cand_better, cand_worse = [], [], []
    
    for _, row in rr_data_lang.iterrows():
        cand_better += [sentences[row['BETTER']][row['SID']-1]]
        cand_worse += [sentences[row['WORSE']][row['SID']-1]]
        ref += [references[row['SID']-1]]

    return ref, cand_better, cand_worse


def kendell_score(scores_better, scores_worse):
    total = len(scores_better)
    correct = torch.sum(scores_better > scores_worse).item()
    incorrect = total - correct
    return (correct - incorrect) / total


def get_wmt19_seg_score(lang_pair, scorer, attack, cache=False, batch_size=64):
    filename = "cache_score/19/{}/{}/wmt19_seg_to_{}_{}.pkl".format(scorer.model_type, attack, *lang_pair.split('-'))
    if cache:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                print("loaded from cache")
                return pkl.load(f)
    else:
        refs, cand_better, cand_worse = get_wmt19_seg_data(lang_pair, attack=attack)
        scores_better = list(scorer.score(cand_better, refs, batch_size=batch_size))
        scores_worse = list(scorer.score(cand_worse, refs, batch_size=batch_size))

        if cache:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pkl.dump((scores_better, scores_worse), f)
        return scores_better, scores_worse,


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="wmt19", help="path to wmt19 data")
    parser.add_argument("-m", "--model", help="models to tune")
    parser.add_argument("-l", "--log_file", default="wmt19_log.csv", help="log file path")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-a", "--attack", default="no_attack")
    parser.add_argument("-n", "--num_layers", default=1)
    parser.add_argument(
        "--lang_pairs",
        default="en-cs",
        help="language pairs used for tuning",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    print(f'model_type, {args.lang_pairs}, avg')
    if not os.path.exists(args.log_file):
        with open(args.log_file, 'w') as f:
            print(f'model_type, {args.lang_pairs}, avg', file=f)
    
    print(args.model, args.attack)

    scorer = bert_score.scorer.BERTScorer(model_type=args.model, num_layers=args.num_layers)
    results = defaultdict(dict)

    scores_better, scores_worse = get_wmt19_seg_score(
        args.lang_pairs, scorer, attack=args.attack, cache=False, batch_size=args.batch_size)

    results[args.lang_pairs][f"{args.model} F"] = kendell_score(scores_better[2], scores_worse[2])

    temp = results[args.lang_pairs][f"{args.model} F"]
    results["avg"][f"{args.model} F"] = np.mean(temp)
    msg = f"{args.model} F: {results[args.lang_pairs][f'{args.model} F']}"

    print(msg)
    with open(args.log_file, "a") as f:
        print(msg, file=f)

        del scorer


if __name__ == "__main__":
    main()

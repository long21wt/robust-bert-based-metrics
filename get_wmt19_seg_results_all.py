import numpy as np
import pandas as pd
import pickle as pkl
from collections import defaultdict
import os
import torch
import argparse
from transformers import AutoTokenizer
from collections import defaultdict
import bert_score

import argparse

def load_data(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            lines.append(l)
    return lines

def count_unk(sen: list, model: str) -> int :
    count, temp = 0, 0
    for word in sen:
        if model == 'bert-base-uncased' or model == 'bert-large-uncased' or model == 'bert-base-multilingual-cased':
            if "'[UNK]'" in word:
                count += 1
            if '##' in word:
                temp += 1
            else:
                if temp != 0:
                    count += 1
                    temp = 0
        if model == 'roberta-large':
            if 'Ġ' not in word:
                temp += 1
            else:
                if temp != 0:
                    count += 1
                    temp = 0
    if model == 'roberta-large' and 'Ġ' not in sen[0] and 'Ġ' in sen[1]:
        count -= 1
    if len(sen) >= 2:
        if '##' in sen[-1]:
            count += 1
    return count

def dict_count_unk_output(lang_pair: str, model, references):

    tokenizer = AutoTokenizer.from_pretrained(model) 

    def load_metadata(lp):
        files_path = []
        for root, _, files in os.walk(lp):
            print(root)
            for file in files:
                if '.hybrid' not in file:
                    raw = file.split('.')
                    lp = raw[-1]
                    system = '.'.join(raw[1:-1])
                    files_path.append((os.path.join(root, file), system))
        return files_path
   
    num = []
    for reference in references:
        tokenized = ["'" + token + "'" for token in tokenizer.tokenize(reference)]
        reference_count_unk = count_unk(tokenized, model)
        num.append(reference_count_unk)
    mean = np.mean(num)
    print(f'ref: {mean}')
    output_dict = defaultdict(dict)
    all_meta_data = load_metadata(os.path.join('wmt19/wmt19-metrics-task-package/input/system-outputs/newstest2019/', lang_pair))   
    for meta_data in range(len(all_meta_data)):
        count_unk_meta_data = []
        path, system = all_meta_data[meta_data]
        translations = load_data(path)
        for translation in translations:
            tokenized = ["'" + token + "'" for token in tokenizer.tokenize(translation)]
            translation_count_unk = count_unk(tokenized, model)
            count_unk_meta_data.append(translation_count_unk)
        if lang_pair == "zh-en":
            system = f"{system}.{lang_pair}"
        output_dict[system]['chunk_0'] = []
        output_dict[system]['chunk_1'] = []
        for index, count in enumerate(count_unk_meta_data):
            try:
                num_rep = num[index]
            except IndexError:
                num_rep = 0
            if (count + num_rep) / 2 < mean:
                output_dict[system]['chunk_0'].append((index, count, num_rep))            
            else:
                output_dict[system]['chunk_1'].append((index, count, num_rep))
    return output_dict

def get_wmt19_seg_data(lang_pair, model, split, attack="no_attack"):
    src, tgt = lang_pair.split('-')
    RRdata = pd.read_csv(
        "wmt19/wmt19-metrics-task-package/manual-evaluation/RR-seglevel.csv", sep=' ')
    RRdata_lang = RRdata[RRdata['LP'] == lang_pair] # there is a typo in this data. One column name is missing in the header
    #RRdata_lang = RRdata[RRdata.index == lang_pair]
    systems = set(RRdata_lang['BETTER'])
    systems.update(list(set(RRdata_lang['WORSE'])))
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
                "attacked-references/{}/{}/{}".format(attack_method, pertubation, 'newstest2019-{}{}-ref.{}'.format(src, tgt, tgt))) as f:
            references = f.read().split("\n")

    ref, cand_better, cand_worse = [], [], []
    if split:    
        ref_chunk_0, cand_better_chunk_0, cand_worse_chunk_0 = [], [], []
        ref_chunk_1, cand_better_chunk_1, cand_worse_chunk_1 = [], [], []
        output = dict_count_unk_output(lang_pair, model, references)
        chunk_0_temp, chunk_1_temp, chunk_0_total_unk, chunk_1_total_unk = [], [], [], []
    
    for _, row in RRdata_lang.iterrows():
        cand_better += [sentences[row['BETTER']][row['SID']-1]]
        cand_worse += [sentences[row['WORSE']][row['SID']-1]]
        ref += [references[row['SID']-1]]
        if split:
            len_ref = len(ref)
            for chunk in output[row['BETTER']]['chunk_0']:
                if row['SID'] - 1 == chunk[0]:
                    chunk_0_temp.append((chunk[1], (chunk[1] + chunk[2]) / 2, [sentences[row['BETTER']][row['SID']-1]], \
                        [sentences[row['WORSE']][row['SID']-1]], [references[row['SID']-1]]))
            for chunk in output[row['BETTER']]['chunk_1']:
                if row['SID'] - 1 == chunk[0]:
                    chunk_1_temp.append((chunk[1], (chunk[1] + chunk[2]) / 2, [sentences[row['BETTER']][row['SID']-1]], \
                        [sentences[row['WORSE']][row['SID']-1]], [references[row['SID']-1]]))
    if split:
        chunk_0_temp.sort(key=lambda x: x[1])
        chunk_1_temp.sort(key=lambda x: x[1], reverse=True)
        half_len_ref = int(round(len_ref / 2))
        if len(chunk_0_temp) > half_len_ref:
            for ele in chunk_0_temp[:half_len_ref]:
                cand_better_chunk_0 += ele[2]
                cand_worse_chunk_0 += ele[3]
                ref_chunk_0 += ele[4]
                chunk_0_total_unk.append(ele[1] * 2)
            for ele in chunk_0_temp[half_len_ref:]:
                cand_better_chunk_1 += ele[2]
                cand_worse_chunk_1 += ele[3]
                ref_chunk_1 += ele[4]
                chunk_1_total_unk.append(ele[1] * 2)
            for ele in chunk_1_temp:
                cand_better_chunk_1 += ele[2]
                cand_worse_chunk_1 += ele[3]
                ref_chunk_1 += ele[4]
                chunk_1_total_unk.append(ele[1] * 2)
        else:
            for ele in chunk_1_temp[:half_len_ref]:
                cand_better_chunk_1 += ele[2]
                cand_worse_chunk_1 += ele[3]
                ref_chunk_1 += ele[4]
                chunk_1_total_unk.append(ele[1] * 2)
            for ele in chunk_1_temp[half_len_ref:]:
                cand_better_chunk_0 += ele[2]
                cand_worse_chunk_0 += ele[3]
                ref_chunk_0 += ele[4]
                chunk_0_total_unk.append(ele[1] * 2)
            for ele in chunk_0_temp:
                cand_better_chunk_0 += ele[2]
                cand_worse_chunk_0 += ele[3]
                ref_chunk_0 += ele[4]
                chunk_0_total_unk.append(ele[1] * 2)
        
        num_unk_chunk_0 = sum(chunk_0_total_unk)
        num_unk_chunk_1 = sum(chunk_1_total_unk)
        len_ref_chunk_0 = len(ref_chunk_0)
        len_ref_chunk_1 = len(ref_chunk_1)
        print(f'ratio size: {len_ref_chunk_1 / len_ref_chunk_0}, total: {len_ref}, chunk_0: {len_ref_chunk_0} chunk_1: {len_ref_chunk_1}')
        print(f'chunk_0 number of unk tokens: {num_unk_chunk_0}, chunk_1 number of unk tokens: {num_unk_chunk_1}')
        try:
            print(f'ratio chunk_0 token: {num_unk_chunk_0 / len_ref_chunk_0}, ratio_chunk_1 token: {num_unk_chunk_1 / len_ref_chunk_1}')
        except ZeroDivisionError:
            print("Chunk 0 have zero unk token")         
        return ref, cand_better, cand_worse, \
            ref_chunk_0, cand_better_chunk_0, cand_worse_chunk_0, \
            ref_chunk_1, cand_better_chunk_1, cand_worse_chunk_1
    else: 
        return ref, cand_better, cand_worse


def kendell_score(scores_better, scores_worse):
    scores_better = torch.mean(scores_better, dim=0)
    scores_worse = torch.mean(scores_worse, dim=0)
    total = len(scores_better)
    correct = torch.sum(scores_better > scores_worse).item()
    incorrect = total - correct
    return (correct - incorrect) / total


def get_wmt19_seg_score(lang_pair, evaluation, scorer, model, split, attack, cache=False, batch_size=64):
    filename = "cache_score/19/{}/{}/{}/wmt19_seg_to_{}_{}.pkl".format(evaluation, scorer.model_type, attack, *lang_pair.split('-'))
    if cache:
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                print("loaded from cache")
                return pkl.load(f)
    else:
        if split:
            refs, cand_better, cand_worse, \
            ref_chunk_0, cand_better_chunk_0, cand_worse_chunk_0, \
            ref_chunk_1, cand_better_chunk_1, cand_worse_chunk_1 = get_wmt19_seg_data(lang_pair, model, split, attack=attack)
            scores_better = list(scorer.score(cand_better, refs, batch_size=batch_size))
            scores_worse = list(scorer.score(cand_worse, refs, batch_size=batch_size))
            scores_better_chunk_0 = list(scorer.score(cand_better_chunk_0, ref_chunk_0, batch_size=batch_size))
            scores_worse_chunk_0 = list(scorer.score(cand_worse_chunk_0, ref_chunk_0, batch_size=batch_size))
            scores_better_chunk_1 = list(scorer.score(cand_better_chunk_1, ref_chunk_1, batch_size=batch_size))
            scores_worse_chunk_1 = list(scorer.score(cand_worse_chunk_1, ref_chunk_1, batch_size=batch_size))
           
            if cache:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wb") as f:
                    pkl.dump((scores_better, scores_worse, \
                        scores_better_chunk_0, scores_worse_chunk_0, \
                        scores_better_chunk_1, scores_worse_chunk_1), f)
            return scores_better, scores_worse, \
                scores_better_chunk_0, scores_worse_chunk_0, \
                scores_better_chunk_1, scores_worse_chunk_1
        else:
            refs, cand_better, cand_worse = get_wmt19_seg_data(lang_pair, model, split, attack=attack)
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
    parser.add_argument("-e", "--evaluation", default="bert-score")
    parser.add_argument("-m", "--model", help="models to tune")
    parser.add_argument("-l", "--log_file", default="wmt19_log.csv", help="log file path")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-s", "--split", default=False)
    parser.add_argument("-a", "--attack", default="no_attack")
    parser.add_argument(
        "--lang_pairs",
        default="fi-en",
        help="language pairs used for tuning",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    print(f'model_type, {args.lang_pairs}, avg')
    if not os.path.exists(args.log_file):
        with open(args.log_file, 'w') as f:
            print(f'model_type, {args.lang_pairs}, avg', file=f)
    
    print(args.model, args.attack, args.evaluation)
    if args.evaluation == 'bert-score':
        scorer = bert_score.scorer.BERTScorer(model_type=args.model, all_layers=True)

    results = defaultdict(dict)
    chunk_0 = defaultdict(dict)
    chunk_1 = defaultdict(dict)

    if args.split:
        scores_better, scores_worse, \
            scores_better_chunk_0, scores_worse_chunk_0, \
            scores_better_chunk_1, scores_worse_chunk_1 = get_wmt19_seg_score(args.lang_pairs, args.evaluation, scorer, args.model, split=args.split, \
                attack=args.attack, cache=False, batch_size=args.batch_size)
    else:
        scores_better, scores_worse = get_wmt19_seg_score(args.lang_pairs, args.evaluation, scorer, args.model, split=False, \
            attack=args.attack, cache=False, batch_size=args.batch_size)
        
    if args.evaluation == 'bart-score':
        results[args.lang_pairs][f"{args.model}"] = kendell_score(torch.FloatTensor(scores_better), torch.FloatTensor(scores_worse))
    else:
        results[args.lang_pairs][f"{args.model} F"] = kendell_score(scores_better[2], scores_worse[2])

    if args.split:
        chunk_0[args.lang_pairs][f"{args.model} F"] = kendell_score(scores_better_chunk_0[2], scores_worse_chunk_0[2])
        chunk_1[args.lang_pairs][f"{args.model} F"] = kendell_score(scores_better_chunk_1[2], scores_worse_chunk_1[2])

    if args.evaluation == 'bart-score':
        temp = results[args.lang_pairs][f"{args.model}"]
        results["avg"][f"{args.model}"] = np.mean(temp)
        msg = f"{args.model} {results[args.lang_pairs][f'{args.model}']}"
    else:
        temp = results[args.lang_pairs][f"{args.model} F"]
        results["avg"][f"{args.model} F"] = np.mean(temp)
        msg = f"{args.model} F: {results[args.lang_pairs][f'{args.model} F']}"

        if args.split:
            temp_chunk_0 = chunk_0[args.lang_pairs][f"{args.model} F"]
            temp_chunk_1 = chunk_1[args.lang_pairs][f"{args.model} F"]
            chunk_0["avg"][f"{args.model} F"] = np.mean(temp_chunk_0)
            chunk_1["avg"][f"{args.model} F"] = np.mean(temp_chunk_1)
            msg += f" chunk_0: {chunk_0[args.lang_pairs][f'{args.model} F']}"
            msg += f" chunk_1: {chunk_1[args.lang_pairs][f'{args.model} F']}"

    print(msg)
    with open(args.log_file, "a") as f:
        print(msg, file=f)

        del scorer


if __name__ == "__main__":
    main()

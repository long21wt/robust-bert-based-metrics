# Layer or Representation Space: What makes BERT-based Metrics Robust?

## Citation
```bib
@inproceedings{vu-etal-2022-layer,
    title = "Layer or Representation Space: What Makes {BERT}-based Evaluation Metrics Robust?",
    author = "Vu, Doan Nam Long  and
      Moosavi, Nafise Sadat  and
      Eger, Steffen",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.300",
    pages = "3401--3411",
    abstract = "The evaluation of recent embedding-based evaluation metrics for text generation is primarily based on measuring their correlation with human evaluations on standard benchmarks. However, these benchmarks are mostly from similar domains to those used for pretraining word embeddings. This raises concerns about the (lack of) generalization of embedding-based metrics to new and noisy domains that contain a different vocabulary than the pretraining data. In this paper, we examine the robustness of BERTScore, one of the most popular embedding-based metrics for text generation. We show that (a) an embedding-based metric that has the highest correlation with human evaluations on a standard benchmark can have the lowest correlation if the amount of input noise or unknown tokens increases, (b) taking embeddings from the first layer of pretrained models improves the robustness of all metrics, and (c) the highest robustness is achieved when using character-level embeddings, instead of token-based embeddings, from the first layer of the pretrained model.",
}
```
## Evaluation on different ratios of unknown tokens.

Compute score on WMT19

```python
python get_wmt19_seg_results.py --model bert-base-uncased --attack visual_0.3 --num_layers 9 --lang_pairs fi-en
```
where 

```
--model : your model name (e.g., bert-base-uncased, google/byt5-small)

--attack : default "no-attack", attack name (visual, intrude, disemvowel, keyboard-typo, phonetic) with pertubation level (0.1-0.3) e.g., visual-0.3 

--num_layers : the number of layers for computing the score, by default first layer are selected.

--lang_pairs : WMT19 language pairs (fi-en, gu-en, kk-en, zh-en, lt-en, de-en, ru-en)
```

default setting for `num_layers` in BERTScore is:
    
- `"bert-base-uncased": 9`
- `"bert-large-uncased": 18`


## Evaluation on low-resource language pairs

```python
cd mt-metrics-eval
python wmt21-flores.py
```
## Impact of Character-level Embeddings

```python
python get_wmt19_seg_results.py --model google/byt5-small --attack visual_0.3 --num_layers 1
```

Similar to previous experiment, default byt5 setting for `num_layers` in BERTScore is: 

- `"google/byt5-small" : 1`

- `"google/byt5-base": 17`

- `"google/byt5-large": 30`


## Impact of the Selected Hidden Layer

**Note**: mean of aggregation setting is extremly resource consuming on WMT19 dataset.

```python
python get_wmt19_seg_results_all.py --model bert-base-base --attack visual_0.3 --lang_pairs fi-en
```

## Acknowledgement

- We would like to thank the authors of BERTScore and WMT21 for their scripts to reproduce the score.
- We would like to thank Ubiquitous Knowledge Processing (UKP) Lab for providing computational resources to finish the paper.

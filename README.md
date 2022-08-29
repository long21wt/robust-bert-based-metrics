# Layer or Representation Space: What makes BERT-based Metrics Robust?

## Evaluation on different ratios of unknown tokens.

Compute score on WMT19

```python
python get_wmt19_seg_results.py --model bert-base-large --attack visual_0.3 --num_layers 9 --lang_pairs fi-en
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

```
python get_wmt19_seg_results.py --model google/byt5-small --attack visual_0.3 --num_layers 1
```

Similar to previous experiment, default byt5 setting for `num_layers` in BERTScore is: 

- `"google/byt5-small" : 1`

- `"google/byt5-base": 17`

- `"gooel/byt5-large": 30`


## Impact of the Selected Hidden Layer

**Note**: mean of aggregation setting is extremly resource consuming on WMT19 dataset.

```
python get_wmt19_seg_results.py --model bert-base-base --attack visual_0.3 --lang_pairs fi-en
```

## Acknowledgement

- We would like to thank the authors of BERTScore and WMT21 for their scripts to reproduce the score.
- We would like to thank Ubiquitous Knowledge Processing (UKP) Lab for providing computational resources to finish the paper.

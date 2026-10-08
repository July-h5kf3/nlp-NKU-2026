# Homework 1 results (NYT test, 1152 documents)

## Main results

| Method | Accuracy | Macro-F1 |
|---|---|---|
| Binary BoW | 0.9896 | 0.9741 |
| Word frequency BoW | 0.9913 | 0.9779 |
| TF-IDF BoW (extra) | 0.9896 | 0.9751 |
| GloVe 6B 100d | 0.9852 | 0.9646 |
| Word2Vec AG News (seed 42) | 0.9757 | 0.9410 |
| Word2Vec NYT train split (seed 42) | 0.9792 | 0.9491 |
| Word2Vec NYT all text, leaky ablation (seed 42) | 0.9835 | 0.9589 |
| Word2Vec AG News (3 seeds) | 0.9760 ± 0.0005 | 0.9415 ± 0.0016 |
| Word2Vec NYT train split (3 seeds) | 0.9806 ± 0.0018 | 0.9524 ± 0.0037 |
| Word2Vec NYT all text (3 seeds) | 0.9818 ± 0.0017 | 0.9546 ± 0.0042 |
| BERT max_length 64, last epoch (3 seeds) | 0.9826 ± 0.0009 | 0.9615 ± 0.0006 |
| BERT max_length 64, best dev epoch (3 seeds) | 0.9826 ± 0.0009 | 0.9611 ± 0.0012 |

## Per-class F1

| Method | business | politics | sports |
|---|---|---|---|
| Binary BoW | 0.9580 | 0.9655 | 0.9988 |
| Word frequency BoW | 0.9658 | 0.9684 | 0.9994 |
| TF-IDF BoW (extra) | 0.9619 | 0.9650 | 0.9983 |
| GloVe 6B 100d | 0.9444 | 0.9517 | 0.9977 |
| Word2Vec AG News (seed 42) | 0.9103 | 0.9161 | 0.9965 |
| Word2Vec NYT train split (seed 42) | 0.9193 | 0.9310 | 0.9971 |
| Word2Vec NYT all text, leaky ablation (seed 42) | 0.9333 | 0.9452 | 0.9983 |
| BERT-64 last epoch seed 42 | 0.9481 | 0.9444 | 0.9936 |
| BERT-64 last epoch seed 43 | 0.9444 | 0.9412 | 0.9971 |
| BERT-64 last epoch seed 44 | 0.9448 | 0.9444 | 0.9954 |
| BERT-64 best dev epoch seed 42 | 0.9437 | 0.9416 | 0.9948 |
| BERT-64 best dev epoch seed 43 | 0.9444 | 0.9412 | 0.9971 |
| BERT-64 best dev epoch seed 44 | 0.9474 | 0.9452 | 0.9948 |

## BERT max_length sweep (3 seeds, mean ± sample std)

| max_length | last acc | last macro-F1 | best-dev acc | best-dev macro-F1 | best epochs |
|---|---|---|---|---|---|
| 32 | 0.9664 ± 0.0018 | 0.9323 ± 0.0035 | 0.9664 ± 0.0018 | 0.9323 ± 0.0035 | 3,3,3 |
| 64 | 0.9826 ± 0.0009 | 0.9615 ± 0.0006 | 0.9826 ± 0.0009 | 0.9611 ± 0.0012 | 2,3,2 |
| 96 | 0.9855 ± 0.0013 | 0.9664 ± 0.0026 | 0.9855 ± 0.0013 | 0.9664 ± 0.0026 | 3,3,3 |
| 128 | 0.9870 ± 0.0009 | 0.9693 ± 0.0020 | 0.9870 ± 0.0009 | 0.9693 ± 0.0020 | 3,3,3 |
| 192 | 0.9881 ± 0.0018 | 0.9724 ± 0.0048 | 0.9881 ± 0.0018 | 0.9724 ± 0.0048 | 3,3,3 |
| 256 | 0.9916 ± 0.0005 | 0.9790 ± 0.0012 | 0.9916 ± 0.0005 | 0.9790 ± 0.0012 | 3,3,3 |
| 384 | 0.9905 ± 0.0017 | 0.9779 ± 0.0037 | 0.9899 ± 0.0010 | 0.9770 ± 0.0027 | 3,3,1 |
| 512 | 0.9931 ± 0.0009 | 0.9838 ± 0.0018 | 0.9931 ± 0.0009 | 0.9838 ± 0.0018 | 3,3,3 |

## BoW inverse regularization C (val macro-F1 / test macro-F1)

| Method | C=0.01 | C=0.1 | C=1 | C=10 | C=100 |
|---|---|---|---|---|---|
| binary | 0.9596 / 0.9681 | 0.9573 / 0.9741 | 0.9579 / 0.9741 | 0.9579 / 0.9741 | 0.9579 / 0.9741 |
| frequency | 0.9561 / 0.9756 | 0.9575 / 0.9779 | 0.9575 / 0.9779 | 0.9575 / 0.9779 | 0.9575 / 0.9779 |
| tfidf | 0.2857 / 0.2857 | 0.8770 / 0.9272 | 0.9569 / 0.9751 | 0.9636 / 0.9775 | 0.9674 / 0.9812 |

## Word embedding coverage on NYT test

| Embedding | vocab | token OOV | type OOV |
|---|---|---|---|
| emb_glove | 400000 | 0.0195 | 0.2793 |
| emb_ag_seed42 | 25376 | 0.0576 | 0.6024 |
| emb_nyt_seed42 | 31657 | 0.0246 | 0.4268 |
| emb_nyt_all_seed42 | 35447 | 0.0186 | 0.3633 |

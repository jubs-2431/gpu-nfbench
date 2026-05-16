# V2 Gold Generalization

## Leave-one-repository-out

| model/mode | accuracy | macro F1 |
| --- | ---: | ---: |
| candidate_weak_label | 0.505 | 0.426 |
| bm25_knn | 0.493 | 0.478 |
| naive_bayes | 0.550 | 0.475 |
| tfidf_logistic | 0.653 | 0.630 |
| tfidf_linear_svm | 0.668 | 0.648 |
| bigram_tfidf_logistic | 0.663 | 0.633 |
| expanded_gold_vote_ensemble | 0.668 | 0.647 |

## Chronological 80/20

| model/mode | accuracy | macro F1 |
| --- | ---: | ---: |
| candidate_weak_label | 0.469 | 0.325 |
| bm25_knn | 0.502 | 0.454 |
| naive_bayes | 0.598 | 0.475 |
| tfidf_logistic | 0.711 | 0.690 |
| tfidf_linear_svm | 0.732 | 0.697 |
| bigram_tfidf_logistic | 0.766 | 0.738 |
| expanded_gold_vote_ensemble | 0.745 | 0.722 |

## Decision Trees

This repository includes lessons and practices related to `Decision Trees` and `Random Forest` classifier algorithms.
Machine learning metrics are also reviewed, including:

### Information gain:

#### Entropy:

For categorical attributes,

$$
Entropy \ = \sum^{c}_{i} \ -p_{i}\cot log_{2}{p_{i}}
$$

#### Gini index:

$$
Gini \ = 1-\sum^{c}_{i} p_{i}^2
$$

### Precision:
Percent of positive detected cases, this work to measure model quality in classification.

#### Recall:
Understand rate of true positives.

$$
Recall \ = \frac{TP}{TP+FN}
$$

#### Specificity:
Proportion of true negatives.

$$
Specificity \ = \frac{TN}{TN+FP}
$$

#### F1-score

$$
Score \ = \frac{2\cdot precision \cdot recall}{precision \ + \ recall}
$$
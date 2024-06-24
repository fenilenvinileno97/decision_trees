## Decision Trees

This repository includes lessons and practices related to `Decision Trees` and `Random Forest` classifier algorithms.
Machine learning metrics are also reviewed, including:

### Information gain:

#### Entropy:

For categorical attributes,

$Entropy \ = {\sum_{c}}^{i} \ -p_{i} \cdot log_{2}{p_{i}}$

#### Gini index:

$Gini \ = 1-{\sum_{c}}^{i} \ p_{i}^2$

### Precision:
The number of correctly-identified members of a class divided by all the times the model predicted that class.

#### Recall:
Understand rate of true positives.

$Recall \ = \frac{TP}{TP+FN}$

#### Specificity:
Proportion of true negatives.

$Specificity \ = \frac{TN}{TN+FP}$

#### F1-score
It combines precision and recall into one metric. If precision and recall are both high, F1 will be high, too. If they are both low, F1 will be low. If one is high and the other low, F1 will be low. F1 is a quick way to tell whether the classifier is actually good at identifying members of a class, or if it is finding shortcuts (e.g., just identifying everything as a member of a large class).

$F1 \ Score \ = \frac{2\cdot precision \cdot recall}{precision \ + \ recall}$

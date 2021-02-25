# Active Learning Summary
In this repository, previous works of active learning were categorized. 

- [Active Learning Summary](#active-learning-summary)
- [At the Beginning](#at-the-beginning)
- [Brief Introduction](#brief-introduction)
  - [Scenarios](#scenarios)
    - [Pool-based AL](#pool-based-al)
    - [Stream-based AL](#stream-based-al)
    - [Query Synthesis](#query-synthesis)
- [Previous Works](#previous-works)
  - [Query Strategy](#query-strategy)
    - [Heterogeneity/Disagreement-Based Models](#heterogeneitydisagreement-based-models)
      - [Uncertainty](#uncertainty)
      - [Margin](#margin)
      - [Entropy](#entropy)
      - [Ensemble Models](#ensemble-models)
    - [Performance-Based Models](#performance-based-models)
      - [Information or Performance Gain](#information-or-performance-gain)
      - [Expected Error Reduction](#expected-error-reduction)
      - [Expected Variance Reduction](#expected-variance-reduction)
    - [Representativeness-imparted Models](#representativeness-imparted-models)
      - [Information Density Frameworks](#information-density-frameworks)
    - [Meta Active Learning](#meta-active-learning)
    - [Others](#others)
  - [Problem Settings Change (Combine AL with other settings)](#problem-settings-change-combine-al-with-other-settings)
    - [Multi-Class Active Learning](#multi-class-active-learning)
    - [Multi-Task Active Learning](#multi-task-active-learning)
      - [Multi-Label Active Learning](#multi-label-active-learning)
      - [With pre-defined constrains](#with-pre-defined-constrains)


# At the Beginning

Active learning is used to reduce the annotation cost in machine learning process.
There have been several surveys for this topic.
The main ideas and the scenarios are introduced in these surveys.

- Active learning: theory and applications [[2001]](https://ai.stanford.edu/~koller/Papers/Tong:2001.pdf.gz)
- **Active Learning Literature Survey (Recommend to read)**[[2009]](https://minds.wisconsin.edu/handle/1793/60660)
- A survey on instance selection for active learning [[2012]](https://link.springer.com/article/10.1007/s10115-012-0507-8)
- Active Learning: A Survey [[2014]](https://www.taylorfrancis.com/books/e/9780429102639/chapters/10.1201/b17320-27)

In the rest of this note, we collect the works in the following categories.
A short summary for each work might be provided.

# Brief Introduction
## Scenarios
### Pool-based AL
### Stream-based AL
### Query Synthesis

# Previous Works

## Query Strategy
### Heterogeneity/Disagreement-Based Models
#### Uncertainty
Uncertainty sampling simply queries an instance of which the predicted class value possesses a minimum probability among all candidate instances.
- Heterogeneous uncertainty sampling for supervised learning [1994, ICML]
- A sequential algorithm for training text classifiers [1994, SIGIR]
- [Support Vector Machine Active Learning with Applications to Text Classification [2001, JMLR]](http://www.jmlr.org/papers/v2/tong01a.html): 
  Version space reduction with SVM.
#### Margin
Margin in classification problems is calculated as diﬀerence between the ﬁrst and second highest class probability.
- Selective sampling with redundant views [2000, AAAI]
#### Entropy
- [Committee-based sampling for training probabilistic classifiers [1995, ICML]](https://dl.acm.org/doi/10.1145/1015330.1015385)
- Employing em and pool-based active learning for text classification [1998, ICML]
- Diverse ensembles for active learning [ICML, 2004]: 
  Use margins to measure ensemble disagreement but generalizes the idea to multi-class problems
- Active learning for probability estimation using jensen-shannon divergence [2005, ECML]
#### Ensemble Models
- Query learning strategies using boosting and bagging [1998, ICML]: 
  Query-by-Bagging, Query-by-Boosting.
- Selective sampling with redundant views [2000, AAAI]: 
  Co-testing.
### Performance-Based Models
#### Information or Performance Gain
#### Expected Error Reduction
#### Expected Variance Reduction
### Representativeness-imparted Models
#### Information Density Frameworks
### Meta Active Learning
### Others


-------------------
## Problem Settings Change (Combine AL with other settings)
### Multi-Class Active Learning 
- [Active Hidden Markov Models for Information Extraction [2001, International Symposium on Intelligent Data Analysis]](https://link.springer.com/chapter/10.1007/3-540-44816-0_31):
  Margin methods compares the most likely and second likely labels. 
- [Diverse ensembles for active learning [ICML, 2004]](https://dl.acm.org/doi/10.1145/1015330.1015385): 
  Use margins to measure ensemble disagreement but generalizes the idea to multi-class problems. 
  C4.5 as base learner.
- [Multi-class Ensemble-Based Active Learning [2006, ECML]](https://link.springer.com/chapter/10.1007/11871842_68): 
  Extract the most valuable samples by margin-based disagreement, uncertainty, sampling-based disagreement, or specific disagreement. 
  C4.5 as base learner.
- [Multi-Class Active Learning for Image Classification [CVPR, 2009]](https://ieeexplore.ieee.org/abstract/document/5206627):
  A comparison of BvSB and Entropy for active multi-classification. 
  They use one-vs-one SVM.
  **Well accepted solution for multi-class classification.**
  (Very simple paper, just a comparison, nothing else)
- [Active learning for large multi-class problems [CVPR, 2009]](https://ieeexplore.ieee.org/abstract/document/5206651):
  Introduce a probabilistic variant of the K-nearest neighbor (KNN) method for classiﬁcation that can be further used for active learning in multi-class scenarios.
- [A batch-mode active learning technique based on multiple uncertainty for SVM classifier [2011, Geoscience and Remote Sensing Letters]](https://ieeexplore.ieee.org/abstract/document/6092438/):
  Batch-mode SVM-based method. 
  Not only consider the smallest distances to the decision hyperplanes but also take into account the distances to other hyperplanes.
- [Scalable Active Learning for Multiclass Image Classification [TPAMI, 2012]](https://ieeexplore.ieee.org/abstract/document/6127880/): 
  Convert an active multi-class classification problem into a series active binary classification problem. 
  One-vs-one SVM. 
  Oracle answer if the query selection match the class of the other selected image
- [An active learning-based SVM multi-class classification model [2015, Pattern Recognition]](https://www.sciencedirect.com/science/article/pii/S003132031400497X):
  Use ove-vs-rest SVM, and select from the three types of unclear region (CBA, CCA, CNA).


### Multi-Task Active Learning  
Works under this category normally have multiple tasks on single domain.
When each task in multi-task learning is a classification task, MTAL degenerate to multi-label active learning.

#### Multi-Label Active Learning
- Multi-label svm active learning for image classification. [ICIP, 2004]:
  First MLAL work.
  Query all the labels of the selected instance.
- Two-dimensional active learning for image classiﬁcation [CVPR, 2008]:
  Authors show that querying instance-label pairs is more effective.
  Use most generation error reduction criteria. iteratively select the ones to minimize Multi-Labeled Bayesian Error Bound.
- Effective multi-label active learning for text classification [SIGKDD, 2009]:
  Multi-label text classification and query all the labeled of the instance.
  Approximated by the size of version space, and the reduction rate of the size of version space is optimized with Support Vector Machines (SVM). 
  Expected loss for multi-label data is approximated by summing up losses on all labels according to the most conﬁdent result of label prediction.
- Optimal batch selection for active learning in multi-label classification [ACMMM, 2011]:
  Select a batch of points that each individual point furnishes high information and the selected batch of points have minimal redundancy.
  Design an uncertainty vector and uncertainty matrix to measure the redundancy between unlabeled points.
  Use SVM as underlying model.
- Active learning with multi-label svm classification [IJCAI, 2013]:
  Query all the labels of the selected instance.
  Measure the informativeness of an instance by combining the label cardinality inconsistency and the separation margin with a tradeoff parameter.
- Active query driven by uncertainty and diversity for incremental multi-label learning [ICDM, 2013]:
  Exploit both uncertainty and diversity in the instance space and label space with an incremental multi-label classification model. 
  Pick instance then pick label.
- Multilabel Image Classification via High-Order Label Correlation Driven Active Learning [TIP, 2014]:
  Consider high order correlations (more than pairwise correlation).
  Query item-label pair.
- Active learning by querying informative and representative examples [NIPS 2010/TPAMI, 2014 (with multi-label)]:
  Count unlabeled instances into query object(loss) function (so that representative), and select the instance would have the lowest loss.
- Multi-Label Active Learning: Query Type Matters [IJCAI, 2015]：
  Iteratively select one instance along with a pair of labels, and then query their relevance ordering
- Multi-Label Deep Active Learning with Label Correlation [ICIP, 2018]:
  CNN is used to produce high level representation of the image and the LSTM models the label dependencies.
  Then select k samples furnishing the maximum entropies to form batch B.
- Effective active learning strategy for multi-label learning [Neural Computing, 2018]:
  An review paper.

#### With pre-defined constrains
- Multi-Task Active Learning with Output Constraints [AAAI, 2010]: 
  Query item-label pairs.
  They design a reward function to calculate VOI (value of information).
  Constrains are known and provided at the beginning.
  Use Naive Bayes as underlying model.
- Cost-Effective Active Learning for Hierarchical Multi-Label Classification [IJCAI, 2018]:
  Hierarchical structure are predefined.
  Use batch mode selection.
  Query item-label pairs, and the informativeness of instance-label pair counted in the contribution of ancestor and dependent.
  Use one-vs-all linear SVM.


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
  - [<h3 id="others-725">Others</h3>](#h3-id%22others-725%22othersh3)
  - [Models](#models)
    - [SVM/LR](#svmlr)
    - [Bayesian/Probabilistic](#bayesianprobabilistic)
  - [<h3 id="neural-network-723">Neural Network</h3>](#h3-id%22neural-network-723%22neural-networkh3)
  - [Theoretical Support for Active Learning](#theoretical-support-for-active-learning)
  - [Might fill this slot at the end.](#might-fill-this-slot-at-the-end)
  - [Problem Settings Change (Combine AL with other settings)](#problem-settings-change-combine-al-with-other-settings)
    - [Multi-Class Active Learning](#multi-class-active-learning)
    - [Multi-Label Active Learning](#multi-label-active-learning)
    - [Multi-Task Active Learning](#multi-task-active-learning)
    - [Multi-Domain Active Learning](#multi-domain-active-learning)
    - [Active Domain Adaptation](#active-domain-adaptation)
    - [Active Learning for Recommendation](#active-learning-for-recommendation)
    - [Active Learning for Remote Sensing Image Classification](#active-learning-for-remote-sensing-image-classification)
    - [Active Meta-Learning](#active-meta-learning)
    - [Semi-Supervised Active Learning](#semi-supervised-active-learning)
    - [Active Reinforcement Learning](#active-reinforcement-learning)
  - [<h3 id="generative-adversarial-network-with-active-learning-719">Generative Adversarial Network with Active Learning</h3>](#h3-id%22generative-adversarial-network-with-active-learning-719%22generative-adversarial-network-with-active-learningh3)
  - [Practical Considerations](#practical-considerations)
    - [Batch mode selection](#batch-mode-selection)
    - [Varying Costs](#varying-costs)
    - [Noise Labelers](#noise-labelers)
    - [Multiple Labelers](#multiple-labelers)
- [Applications](#applications)
- [Libraries/Toolboxes](#librariestoolboxes)
- [Related Scholars](#related-scholars)


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
- [Support Vector Machine Active Learning with Applications to Text Classiﬁcation [2001, JMLR]](http://www.jmlr.org/papers/v2/tong01a.html): 
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
## Models
### SVM/LR
Most common models, we won't waste time here.
Most of classic strategies are based on these models.
### Bayesian/Probabilistic
- Employing EM and Pool-Based Active Learning for Text Classiﬁcation [[1998. ICML]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.10&rep=rep1&type=pdf): 
  EM + Query-by-Committee (QBC-with-EM)
### Neural Network
-------------------
## Theoretical Support for Active Learning
Not really familiar with this.
Might fill this slot at the end.
-------------------
## Problem Settings Change (Combine AL with other settings)
### Multi-Class Active Learning 
- [Diverse ensembles for active learning [ICML, 2004]](https://dl.acm.org/doi/10.1145/1015330.1015385): 
  Use margins to measure ensemble disagreement but generalizes the idea to multi-class problems. 
  C4.5 as base learner.
- [Multi-class Ensemble-Based Active Learning [2006, ECML]](https://link.springer.com/chapter/10.1007/11871842_68): 
  Extract the most valuable samples by margin-based disagreement, uncertainty, sampling-based disagreement, or specific disagreement. 
  C4.5 as base learner.
- [A batch-mode active learning technique based on multiple uncertainty for SVM classifier [2011, Geoscience and Remote Sensing Letters]](https://ieeexplore.ieee.org/abstract/document/6092438/):
  Batch-mode SVM-based method. 
  Not only consider the smallest distances to the decision hyperplanes but also take into account the distances to other hyperplanes.
- [Scalable Active Learning for Multiclass Image Classification [2012, TPAMI]](https://ieeexplore.ieee.org/abstract/document/6127880/): 
  Convert an active multi-class classiﬁcation problem into a series active binary classiﬁcation problem. 
  One-vs-one SVM. 
  Oracle answer if the query selection match the class of the other selected image
- [An active learning-based SVM multi-class classification model [2015, Pattern Recognition]](https://www.sciencedirect.com/science/article/pii/S003132031400497X):
  Use ove-vs-rest SVM, and select from the three types of unclear region (CBA, CCA, CNA).
### Multi-Label Active Learning  
### Multi-Task Active Learning
### Multi-Domain Active Learning
### Active Domain Adaptation
### Active Learning for Recommendation
- A survey of active learning in collaborative filtering recommender systems
### Active Learning for Remote Sensing Image Classification
- A survey of active learning algorithms for supervised remote sensing image classification
### Active Meta-Learning
### Semi-Supervised Active Learning
### Active Reinforcement Learning
### Generative Adversarial Network with Active Learning
---------------------
## Practical Considerations
### Batch mode selection
- Batch Mode Active Learning and Its Application to Medical Image Classiﬁcation [[2006, ICML]](https://dl.acm.org/doi/10.1145/1143844.1143897): Largest reduction in the Fisher information + submodular functions. Multi-class classification. Kernel logistic regressions (KLR) and the support vector machines (SVM).
- Semi-Supervised SVM Batch Mode Active Learning for Image Retrieval [CVPR, 2008]
### Varying Costs
### Noise Labelers
### Multiple Labelers

# Applications
# Libraries/Toolboxes
# Related Scholars
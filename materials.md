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


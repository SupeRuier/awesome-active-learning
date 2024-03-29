# AL strategies on Different models

In this chapter, we care about how to apply AL on specific models.

# Models

## SVM/LR

- Discover-Then-Rank Unlabeled Support Vectors in the Dual Space for Multi-Class Active Learning [2023, ICML]

## Bayesian/Probabilistic/Gaussian Progress
- Employing EM and Pool-Based Active Learning for Text Classification [[1998. ICML]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.10&rep=rep1&type=pdf): 
  EM + Query-by-Committee (QBC-with-EM)
- [Active learning with Gaussian Processes for object categorization [2007, ICCV]](https://ieeexplore.ieee.org/abstract/document/4408844): Consider both the distance from the boundary as well as the variance in selecting the points; this is only possible due to the availability of the predictive distribution in GP regression. A significant boost in classification performance is possible, especially when the amount of training data for a category is ultimately very small.(303 citations)
- Active instance sampling via matrix partition [2010, NeurIPS]: Gaussian Process. Maximizing a natural mutual information criterion between the labeled and unlabeled instances. No comparison with others.(69 citations)
- [Bayesian active learning for classification and preference learning [2011]](https://arxiv.org/abs/1112.5745):
  Propose an approach that expresses information gain in terms of predictive entropies, and apply this method to the Gaussian Process Classifier (GPC).
  This method is referred as *BALD*.
  Capture how strongly the model predictions for a given data point and the model parameters are coupled, implying that finding out about the true label of data points with high mutual information would also inform us about the true model parameters.
- Adaptive active learning for image classification [CVPR, 2013]
- Active Learning for Probabilistic Hypotheses Using the Maximum Gibbs Error Criterion [2013, NeurIPS]
- Safe active learning for time-series modeling with gaussian processes [2018, NeurIPS]
- Actively learning gaussian process dynamics
- [Deeper Connections between Neural Networks and Gaussian Processes Speed-up Active Learning [IJCAI, 2019]](https://arxiv.org/abs/1902.10350)
- Depth Uncertainty Networks for Active Learning [2021, NeurIPS]
- BALanCe: Deep Bayesian Active Learning via Equivalence Class Annealing [2021]
- Efficient and Reliable Probabilistic Interactive Learning with Structured Outputs [2022]: CRISPs, A Property-driven Approach for Guaranteed Reliable Inference.
- Deep Bayesian Active Learning via Equivalence Class Annealing [2022, ICML workshop]
- Fast Bayesian Updates for Deep Learning with a Use Case in Active Learning [2022]

## Decision Trees
- [Active Learning with Direct Query Construction [KDD, 2008]](https://dl.acm.org/doi/pdf/10.1145/1401890.1401950)

## Neural Network

We have systematically reviewed the works for AL with neural networks.
Please see the details [here](/../contents/deep_AL.md).

## $k$-Nearest Neighbors

- A Two-Stage Active Learning Algorithm for Nearest Neighbors [2022]
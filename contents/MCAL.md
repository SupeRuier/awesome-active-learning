# AL with Multi-class classification

Multi-class problem is really common in real life.
Each instance has one label from multiple classes (more than 2).
In multi-class classification, conventional methods use one-vs-all and one-vs-rest methods to tackle this problem.
Of course, several other models could handle multi-class problem naturally, such as Neural Network.
(AL for neural networks is quite complex, so here in this section, the strategies are mostly for non-deep models. We have a whole section for AL in deep models [here](deep_AL.md).)
There are not many remarkable works in this multi-class active learning fields.

## Uncertainty-based sampling
The uncertainty should be revealed from the outputs of the multi-class model.

There are mainly two types of strategies. 
We need to mention that these works are rely on the prediction of probability $f(y_i|x)$.
- Classification margin **(BvSB)**: 
  Best vs second best. 
  This is the **well accepted** methods.
  Select the instance whose probability to be classified into to the two most likely classes are most close.
- Classification entropy: 
  Select the instance which has the largest output entropy.

Works:
1. [Multi-class Ensemble-Based Active Learning [2006, ECML]](https://link.springer.com/chapter/10.1007/11871842_68): 
  Extract the most valuable samples by margin-based disagreement, uncertainty, sampling-based disagreement, or specific disagreement. 
  C4.5 as base learner.
2. [Multi-Class Active Learning for Image Classification [CVPR, 2009]](https://ieeexplore.ieee.org/abstract/document/5206627): A comparison of **BvSB** and **Entropy** for active multi-classification. They use one-vs-one SVM to evaluate the probability for each class. (338 citations). Also see [Scalable Active Learning for Multiclass Image Classification [TPAMI, 2012]](https://ieeexplore.ieee.org/abstract/document/6127880/). (104 citations)
3. [An active learning-based SVM multi-class classification model [2015, Pattern Recognition]](https://www.sciencedirect.com/science/article/pii/S003132031400497X): Use ove-vs-rest SVM, and select from the three types of unclear region (CBA, CCA, CNA). Allow the addition of new classes. (65 citations)

## Representative-imparted sampling

- [Integrating Bayesian and Discriminative Sparse Kernel Machines for Multi-class Active Learning [2019, NPIS]](https://papers.nips.cc/paper/2019/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf):
  By further augmenting the SVM with a RVM, the KMC model is able to choose data samples that provide a good coverage of the entire data space (by maximizing the data likelihood) while giving special attention to the critical areas for accurate classification (by maximizing the margins of decision boundaries).
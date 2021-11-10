# Pool-Based Active Learning for Classification

In this chapter, we use our taxonomy to classify different types of AL strategies.
In each section and each type of strategy, we will make a short description at the beginning, then provide a more detail.
And at the end, we will list the representative works under the category (with a short note).

We note that here we doesn't take batch mode as a dimension in our taxonomy.
If you are only care about how to apply batch selection, please check [**here**](subfields/AL_combinations.md).
And the classification problems here include binary and multi-class classification (even some works can only be applied to binary classification).
There also are some works focus on multi-class classification settings, please check [**here**](subfields/MCAL.md).

- [Pool-Based Active Learning for Classification](#pool-based-active-learning-for-classification)
- [Taxonomy](#taxonomy)
- [Categories](#categories)
  - [Informativeness](#informativeness)
    - [Uncertainty-based sampling](#uncertainty-based-sampling)
    - [Disagreement-based sampling](#disagreement-based-sampling)
  - [Expected Improvements](#expected-improvements)
  - [Representativeness-impart sampling](#representativeness-impart-sampling)
    - [Cluster-based sampling](#cluster-based-sampling)
    - [Density-based sampling](#density-based-sampling)
    - [Alignment-based sampling](#alignment-based-sampling)
    - [Expected loss on unlabeled data](#expected-loss-on-unlabeled-data)
    - [Inconsistency of the neighbors](#inconsistency-of-the-neighbors)
  - [Learn to Score](#learn-to-score)
  - [Others](#others)

# Taxonomy

In pool based AL, the strategy is in fact a scoring function for each instance to judge how much information it contains for the current task.
Previous works calculate their scores in different ways.
We summarize them into the following catagories.

| Score                     | Description                                       | Comments                                                                                              |
| ------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Informativeness           | Uncertainty by the model prediction               | Neglect the underlying distribution.                                                                  |
| Representativeness-impart | Represent the underlying distribution             | Normally used with informativeness. This type of methods may have overlaps with batch-mode selection. |
| Expected Improvements     | The improvement of the model's performance        | The evaluations usually take time.                                                                    |
| Learn to score            | Learn a evaluation function directly.             |                                                                                                       |
| Others                    | Could not classified into the previous categories |                                                                                                       |


# Categories

## Informativeness

### Uncertainty-based sampling

This is the most basic strategy of AL.
It aims to select the instances which are most uncertain to the current model.
There are basically three sub-strategies here.

- Classification uncertainty
  - Select the instance close to the decision boundary.
- Classification margin
  - Select the instance whose probability to be classified into to the two most likely classes are most close.
- Classification entropy
  - Select the instance whose have the largest classification entropy among all the classes.

The equations and details could see [here](https://minds.wisconsin.edu/handle/1793/60660).

Works:
- [Heterogeneous uncertainty sampling for supervised learning [1994, ICML]](https://www.sciencedirect.com/science/article/pii/B978155860335650026X): Most basic **Uncertainty** strategy. Could be used with probabilistic classifiers. (1071 citations)
- [Support Vector Machine Active Learning with Applications to Text Classification [2001, JMLR]](http://www.jmlr.org/papers/v2/tong01a.html): Version space reduction with SVM. （2643 citations）
- [How to measure uncertainty in uncertainty sampling for active learning [2021, Machine Learning]](https://link.springer.com/content/pdf/10.1007/s10994-021-06003-9.pdf)

### Disagreement-based sampling

This types of methods need a group of models.
The sampling strategy is basing on the output of the models.
The group of models are called committees, so this type of works are also named Query-By-Committee (QBC).
The intuition is that if the group of committees are disagree with the label of an unlabeled instance, it should be informative in the current stage.

- Disagreement measurement
  - Vote entropy
  - Consensus entropy

Works:
- [Query by committee [1992, COLT]](https://dl.acm.org/doi/abs/10.1145/130385.130417): **QBC**. The idea is to build multiple classifiers on the current labeled data by using the Gibbs algorithm. The selection is basing on the disagreement of classifiers. (1466 citations)
- [Query learning strategies using boosting and bagging [1998, ICML]](https://www.researchgate.net/profile/Naoki_Abe2/publication/221345332_Query_Learning_Strategies_Using_Boosting_and_Bagging/links/5441464b0cf2e6f0c0f60abf.pdf): Avoid Gibbs algorithm in QBC. Ensemble learning with diversity query.  (433 citations)
- [Diverse ensembles for active learning [ICML, 2004]](https://dl.acm.org/doi/10.1145/1015330.1015385): Previous QBC are hard to make classifiers very different from each other. This method use DECORATE to build classifiers. C4.5 as base learner. Outperform 4. Select the instance with the highest JS divergence. (339 citations)(Delete)
- [Bayesian active learning for classification and preference learning [2011, Arxiv]](https://arxiv.xilesou.top/abs/1112.5745): Bayesian Active Learning by Disagreement (**BALD**). Seek the x for which the parameters under the posterior (output by using the parameters) disagree about the outcome (output by using the labeled dataset) the most. (149 citations)
- [The power of ensembles for active learning in image classification [2018, CVPR]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf)
- [Consistency-Based Semi-supervised Active Learning: Towards Minimizing Labeling Cost [2021, Springer]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-58607-2_30.pdf): A semi-supervised AL method.

## Expected Improvements

Our learning purpose is to reduce the generalization error at the end (in other word, have a better performance at the end).
From this perspective, we can select the instances which could improve the performance for each selection stage.
Because we don't know the true label of the instance we are going to selecting, normally the expected performance is calculated for each instance.
These methods normally need to be retrained for each unlabeled instances in the pool, so it could be really time consuming.

- Expected improvement
  - Error Reduction: Most directly, reduce the generalization error.
  - Variance Reduction: We can still reduce generalization error indirectly by minimizing output variance.
  - Expected Model Change: If the selected instance will bring the largest model change, it could be considered as the most valuable instance.

Works:
- Toward optimal active learning through sampling estimation of error reduction [2001, ICML]: Error Reduction
- [Combining active learning and semisupervised learning using Gaussian fields and harmonic functions [2003, ICML]](http://mlg.eng.cam.ac.uk/zoubin/papers/zglactive.pdf): Greedily select queries from the unlabeled data to minimize the estimated expected classification error. GRF as basic model. This is also the most computationally expensive query framework since it needs to iterate over the whole unlabeled pool. (517 citations)
- [Active learning for logistic regression: an evaluation [2007, Machine Learning]](https://link.springer.com/article/10.1007/s10994-007-5019-5): Variance Reduction
- [An analysis of active learning strategies for sequence labeling tasks [2008, CEMNL]](https://www.aclweb.org/anthology/D08-1112.pdf): Expected gradient change in gradient based optimization.(659 citations)
- [Influence Selection for Active Learning [2021]](https://arxiv.org/pdf/2108.09331.pdf): Select instances with the most positive influence on the model. It was divided to an expected gradient length and another term.
- Model-Change Active Learning in Graph-Based Semi-Supervised Learning [2021]
- [Uncertainty-aware active learning for optimal Bayesian classifier [2021, ICLR]](https://openreview.net/pdf?id=Mu2ZxFctAI):
  ELR tends to stuck in local optima; BALD tends to be overly explorative.
  Propose an acquisition function based on a weighted form of mean objective cost of uncertainty (MOCU).

## Representativeness-impart sampling

Previous introduced works seldomly consider the data distributions.
So those strategies are more focusing on the decision boundary, and the representativeness of the data is neglected.
Therefore, many works take the representativeness of the data into account.
Basically, it measures how much the labeled instances are aligned with the unlabeled instances in distribution.
We note that there aren't many works only consider the representativeness of the data.
More commonly, the representativeness and informativeness are considered together to sample instances.

### Cluster-based sampling

The simplest idea is to use cluster structure to guide the selection.
The cluster could either be applied on the original features or the learned embeddings.

- Cluster-based sampling: 
  - Pre-cluster
  - Hierarchical sampling
  - Cluster on other types of embedding

Works:
- [Active learning using pre-clustering [2004, ICML]](https://dl.acm.org/doi/abs/10.1145/1015330.1015349): (483 citations)
- [Hierarchical Sampling for Active Learning [2008, ICML]](https://dl.acm.org/doi/pdf/10.1145/1390156.1390183): Take into account both the uncertainty and the representativeness. The performance heavily depends on the quality of clustering results. (388 citations)
- [Ask-n-Learn: Active Learning via Reliable Gradient Representations for Image Classification [2020]](https://arxiv.org/pdf/2009.14448.pdf): Use kmeans++ on the learned gradient embeddings to select instances.

### Density-based sampling

These types of strategies take into account the distribution and local density.
The intuition is that the location with more density is more likely to be queried.
i.e. the selected instances and the unlabeled instances should have similar distributions.

- Density-based sampling: 
  - **Information density**
  - **RALF**
  - **k-Center-Greedy (Core-set)**: Only consider the representativeness.

Works:
- [An analysis of active learning strategies for sequence labeling tasks [2008, CEMNL]](https://www.aclweb.org/anthology/D08-1112.pdf): **Information density** framework. The main idea is that informative instances should not only be those which are uncertain, but also those which are “representative” of the underlying distribution (i.e., inhabit dense regions of the input space).(659 citations)
- [RALF: A Reinforced Active Learning Formulation for Object Class Recognition [2012, CVPR]](https://ieeexplore.ieee.org/abstract/document/6248108/): **RALF**. A time-varying combination of exploration and exploitation sampling criteria. Include graph density in the exploitation strategies. (59 citations)
- [Active learning for convolutional neural networks: A core-set approach [ICLR, 2018]](https://arxiv.org/abs/1708.00489):
  Core-set loss is simply the difference between average empirical loss over the set of points which have labels for and the average empirical loss over the entire dataset including unlabelled points. Optimize the upper bound of core-set loss could be considered as a k-center problem in practice. Doesn't need to know the out put of the current model.
- [Minimax Active Learning [2020]](https://arxiv.org/pdf/2012.10467.pdf): Develop a semi-supervised minimax entropy-based active learning algorithm that leverages both uncertainty and diversity in an adversarial manner.
- [Multiple-criteria Based Active Learning with Fixed-size Determinantal Point Processes [2021]](https://arxiv.org/pdf/2107.01622.pdf)

### Alignment-based sampling

This type of works directly takes into account the measurement of distribution alignment between labeled/selected data and unlabeled data.
i.e. The labeled and the unlabeled instances should hard to be distinguished.
There are adversarial works and non-adversarial works.

Types:
- **Adversarial based**
- **non-adversarial based**

Works:
- [Exploring Representativeness and Informativeness for Active Learning [2017, IEEE TRANSACTIONS ON CYBERNETICS]](https://ieeexplore.ieee.xilesou.top/abstract/document/7329991): Optimization based. The representativeness is measured by fully investigating the triple similarities that include the similarities between a query sample and the unlabeled set, between a query sample and the labeled set, and between any two candidate query samples. For representativeness, our goal is also to find the sample that makes the distribution discrepancy of unlabeled data and labeled data small. For informativeness, use BvSB. (85 citations)
- [Discriminative Active Learning [2019, Arxiv]](https://arxiv.org/pdf/1907.06347.pdf):
  Make the labeled and unlabeled pool indistinguishable.
- Agreement-Discrepancy-Selection: Active Learning with Progressive Distribution Alignment [2021]
- Dual Adversarial Network for Deep Active Learning [2021, ECCV]: DAAL.

### Expected loss on unlabeled data

Many works only score the instance by the expected performance on the labeled data and the selected data.
Some other works also take into account the expected loss on the rest unlabeled data as a measurement of representativeness.

- Expected loss on unlabeled data: 
  - **QUIRE**
  - **ALDR+**

Works:
- [Active Learning by Querying Informative and Representative Examples [2010, NeurIPS]](http://papers.nips.cc/paper/4176-active-learning-by-querying-informative-and-representative-examples): **QUIRE**. Optimization based. Not only consider the loss in the labeled data (uncertainty) but also consider the loss in the unlabeled data (representations, correct labels would leads small value of overall evaluation function.). This methods is very computationally expensive. (370 citations)
- [Efﬁcient Active Learning by Querying Discriminative and Representative Samples and Fully Exploiting Unlabeled Data [2020, TNNLS]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9178457): **ALDR+**. **This paper also provide a new taxonomy in AL classification**, which includes three parts: criteria for querying samples, exploiting unlabeled data and acceleration. In this paper, they provide a method take all three parts into account.

### Inconsistency of the neighbors

Some works believe the data points that are similar in the model feature space and yet the model outputs maximally different predictive likelihoods should be quired.

- Inconsistency of the neighbors
  - CAL

Works:
- Active Learning by Acquiring Contrastive Examples [2021, Arxiv]: CAL

## Learn to Score

All the mentioned sampling strategies above are basing on heuristic approaches.
Their intuitions are clear, but might perform differently in different datasets.
So some researchers purposed that we can learn a sampling strategy from the sampling process.

- Learn to score
  - Learn a strategy selection method: select from heuristics
  - Learn a score function: learn a score function
  - Learn a AL policy (as a MDP process)

Works:
- [Active learning by learning [2015, AAAI]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9636): **ALBL**. A single human-designed philosophy is unlikely to work on all scenarios. Given an appropriate choice for the multi-armed bandit learner, take the importance-weighted-accuracy as reward function (an unbiased estimator for the test accuracy). It is possible to estimate the performance of different strategies on the fly. SVM as underlying classifier.(41 citations)
- [Learning active learning from data [2017, NeurIPS]](http://papers.nips.cc/paper/7010-learning-active-learning-from-data): **LAL**. Train a random forest regressor that predicts the expected error reduction for a candidate sample in a particular learning state. Previous works they cannot go beyond combining pre-existing hand-designed heuristics. Random forest as basic classifiers. (Not clear how to get test classiﬁcation loss l. It is not explained in both the paper and the code.)(73 citations)
- [Learning how to Active Learn: A Deep Reinforcement Learning Approach [2017, Arxiv]](https://arxiv.org/abs/1708.02383): **PAL**. Use RL to learn how to select instance. Even though the strategy is learned and applied in a stream manner, the stream is made by the data pool. So under my angle, it could be considered as a pool-based method. (92)
- [Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, ACL]](https://www.aclweb.org/anthology/P18-1174.pdf): Learn an AL policy using imitation learning, mapping situations to most informative query datapoints. (8 citations)
- Meta-Learning Transferable Active Learning Policies by Deep Reinforcement Learning [2018, Arxiv]
- [Learning Loss for Active Learning [2019, CVPR]](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html): Attach a small parametric module, named “loss prediction module,” to a target network, and learn it to predict target losses of unlabeled inputs. 
- [Learning to Rank for Active Learning: A Listwise Approach [2020]](https://arxiv.org/pdf/2008.00078.pdf): Have an additional loss prediction model to predict the loss of instances beside the classification model. Then the loss is calculated by the ranking instead of the ground truth loss of the classifier.
- [Deep Reinforcement Active Learning for Medical Image Classiﬁcation [2020, MICCAI]](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_4): Take the prediction probability of the whole unlabeled set as the state. The action as the strategy is to get a rank of unlabeled set by a actor network. The reward is the different of prediction value and true label of the selected instances. Adopt a critic network with parameters θ cto approximate the Q-value function.
- [ImitAL: Learning Active Learning Strategies from Synthetic Data [2021]](https://arxiv.org/pdf/2108.07670.pdf): An imitation learning approach.
- Cartography Active Learning [2021]: CAL. Select the instances that are the closest to the decision boundary between ambiguous and hard-to-learn instances.

## Others
There still are other works uses innovative heuristics.
It is a little bit hard to classify those works for now.
So we put these works under this section.
These works might be classified later.

Self-paced:
- [Self-Paced Active Learning: Query the Right Thing at the Right Time [AAAI 2019]](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4445): 
  Borrow the idea in the self-paced learning that the learning from easy instances to hard instances would improve the performance.
- [Combination of Active Learning and Self-Paced Learning for Deep Answer Selection with Bayesian Neural Network [2020, ECAI]](http://ecai2020.eu/papers/449_paper.pdf):
  Just a combination.
  Not use the idea of the self-paced learning into AL.
- [Self-paced active learning for deep CNNs via effective loss function [2021, Neurocomputing]](https://reader.elsevier.com/reader/sd/pii/S0925231220317963?token=725A4F15C8F8012A11EFEC2DE94118DE4E4FF42C2413D930CC38AF3093BBD3B8C946420F084CCCBB78DBD1F424D709D2):
  The selection criteria selects instances with more discrepancy at the beginning and instances with more uncertainty later.
  They also ass a similarity classification loss function in their model to ensure the effectiveness of the representations.

Utilize historical evaluation results:
- [Looking Back on the Past: Active Learning with Historical Evaluation Results [TKDE, 2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9298849&tag=1):
  Model the AL process as a ranking problem and use the learned rank results to select instance.



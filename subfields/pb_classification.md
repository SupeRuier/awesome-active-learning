# Pool based active learning for classification

We could divide the works of pool-based active learning into for different problem settings.

|                 | Binary classification     | Multi-class classification |
| --------------- | ------------------------- | -------------------------- |
| Non-batch mode: | (A). Most of the AL works | (B). Generalize from (A)   |
| Batch mode:     | (C). Improve over (A)     | (D). Combine (B) and (C)   |

The previous table is an problem oriented taxonomy.
In this chapter, we would use technique-oriented taxonomy to classify the current works under each subproblem.
In each section and each type of strategy, we will make a short description at the beginning, then provide a more detail taxonomy.
And at the end, we will list the famous works (with a short note).

- [Pool based active learning for classification](#pool-based-active-learning-for-classification)
- [AL with Binary classification](#al-with-binary-classification)
  - [Uncertainty-based sampling](#uncertainty-based-sampling)
  - [Disagreement-based sampling](#disagreement-based-sampling)
  - [Expected-improvement-based](#expected-improvement-based)
  - [Representativeness-impart sampling](#representativeness-impart-sampling)
  - [Learn how to sample / Learn how to active learn](#learn-how-to-sample--learn-how-to-active-learn)
  - [Other works](#other-works)
- [AL with Multi-class classification](#al-with-multi-class-classification)
  - [Uncertainty-based sampling](#uncertainty-based-sampling-1)
- [Batch mode classification](#batch-mode-classification)
  - [Heuristic-diversity](#heuristic-diversity)
  - [Optimization-based](#optimization-based)
  - [Greedy Selection](#greedy-selection)

# AL with Binary classification

We combine the batch mode and the non-batch mode together, because even the batch mode strategies are also base on the following ideas.
For the batch mode works, we will add (*batch) after the paper names.


## Uncertainty-based sampling
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

## Disagreement-based sampling
This types of methods need a group of models.
The sampling strategy is basing on the output of the models.
The group of models are called committees, so this type of works are also named Query-By-Committee (QBC).
The intuition is that if the group of committees are disagree with the label of an unlabeled instance, it should be informative to the current stage.

- Disagreement measurement
  - Vote entropy
  - Consensus entropy

Works:
- [Query by committee [1992, COLT]](https://dl.acm.org/doi/abs/10.1145/130385.130417): **QBC**. The idea is to build multiple classifiers on the current labeled data by using the Gibbs algorithm. The selection is basing on the disagreement of classifiers. (1466 citations)
- [Query learning strategies using boosting and bagging [1998, ICML]](https://www.researchgate.net/profile/Naoki_Abe2/publication/221345332_Query_Learning_Strategies_Using_Boosting_and_Bagging/links/5441464b0cf2e6f0c0f60abf.pdf): Avoid Gibbs algorithm in QBC. Ensemble learning with diversity query.  (433 citations)
- [Diverse ensembles for active learning [ICML, 2004]](https://dl.acm.org/doi/10.1145/1015330.1015385): Previous QBC are hard to make classifiers very different from each other. This method use DECORATE to build classifiers. C4.5 as base learner. Outperform 4. Select the instance with the highest JS divergence. (339 citations)(Delete)
- [Bayesian active learning for classification and preference learning [2011, Arxiv]](https://arxiv.xilesou.top/abs/1112.5745): Bayesian Active Learning by Disagreement (**BALD**). Seek the x for which the parameters under the posterior (output by using the parameters) disagree about the outcome (output by using the labeled dataset) the most. (149 citations)

## Expected-improvement-based

Our learning purpose is to reduce the generalization error at the end (in other word, have a better performance at the end).
From this perspective, we can select the instances which could improve the performance for each selection stage.
Because we don't know the true label of the instance we are going to selecting, normally the expected performance is calculated for each instance.
These methods normally need to be retrained for each unlabeled instances in the pool, so it might be really time consuming.

- Expected improvement
  - Error Reduction: Most directly, reduce the generalization error.
  - Variance Reduction: We can still reduce generalization error indirectly by minimizing output variance.
  - Expected Model Change: If the selected instance will bring the largest model change, it could be considered as the most valuable instance.

Works:
- Toward optimal active learning through sampling estimation of error reduction [2001, ICML]: Error Reduction
- [Combining active learning and semisupervised learning using Gaussian fields and harmonic functions [2003, ICML]](http://mlg.eng.cam.ac.uk/zoubin/papers/zglactive.pdf): Greedily select queries from the unlabeled data to minimize the estimated expected classification error. GRF as basic model. This is also the most computationally expensive query framework since it needs to iterate over the whole unlabeled pool. (517 citations)
- [Active learning for logistic regression: an evaluation [2007, Machine Learning]](https://link.springer.com/article/10.1007/s10994-007-5019-5): Variance Reduction
- [An analysis of active learning strategies for sequence labeling tasks [2008, CEMNL]](https://www.aclweb.org/anthology/D08-1112.pdf): Expected gradient change in gradient based optimization.(659 citations)

## Representativeness-impart sampling

Previous introduced seldomly consider the data distributions.
So those strategies are more focusing on the decision boundary, and the representativeness of the data is neglected.
Therefore, many works take the representativeness of the data into account.
We note that there aren't many works only consider the representativeness of the data.
More commonly, the representativeness and informativeness are considered together to sample instances.

- Representativeness-impart sampling:
  - Cluster based sampling:
    - Pre-cluster
    - Hierarchical sampling
    - Cluster on other types of embedding
  - Density based sampling
    - **Information density**
    - **RALF**
    - **(*Batch) k-Center-Greedy (Core-set)**: Only consider the representativeness.
    - **Adversarial based**: The labeled and the unlabeled instances should hard to be distinguished.
  <!-- - Optimization based: Set an objective including the representativeness, then take the optimization result as the score of the unlabeled instances. -->
  - Unlabeled loss: Also take into account the performance on the unlabeled instances.
    - **QUIRE**
    - **ALDR+**

Works:
- [Active learning using pre-clustering [2004, ICML]](https://dl.acm.org/doi/abs/10.1145/1015330.1015349): (483 citations)
- [Hierarchical Sampling for Active Learning [2008, ICML]](https://dl.acm.org/doi/pdf/10.1145/1390156.1390183): Take into account both the uncertainty and the representativeness. The performance heavily depends on the quality of clustering results. (388 citations)
- [An analysis of active learning strategies for sequence labeling tasks [2008, CEMNL]](https://www.aclweb.org/anthology/D08-1112.pdf): **Information density** framework. The main idea is that informative instances should not only be those which are uncertain, but also those which are “representative” of the underlying distribution (i.e., inhabit dense regions of the input space).(659 citations)
- [Active Learning by Querying Informative and Representative Examples [2010, NIPS]](http://papers.nips.cc/paper/4176-active-learning-by-querying-informative-and-representative-examples): **QUIRE**. Optimization based. Not only consider the loss in the labeled data (uncertainty) but also consider the loss in the unlabeled data (representations, correct labels would leads small value of overall evaluation function.). This methods is very computationally expensive. (370 citations)
- [RALF: A Reinforced Active Learning Formulation for Object Class Recognition [2012, CVPR]](https://ieeexplore.ieee.org/abstract/document/6248108/): **RALF**. A time-varying combination of exploration and exploitation sampling criteria. Include graph density in the exploitation strategies. (59 citations)
- [Exploring Representativeness and Informativeness for Active Learning [2017, IEEE TRANSACTIONS ON CYBERNETICS]](https://ieeexplore.ieee.xilesou.top/abstract/document/7329991): Optimization based. The representativeness is measured by fully investigating the triple similarities that include the similarities between a query sample and the unlabeled set, between a query sample and the labeled set, and between any two candidate query samples. For representativeness, our goal is also to find the sample that makes the distribution discrepancy of unlabeled data and labeled data small. For informativeness, use BvSB. (85 citations)
- [Active learning for convolutional neural networks: A core-set approach [ICLR, 2018]](https://arxiv.org/abs/1708.00489):
  Core-set loss is simply the difference between average empirical loss over the set of points which have labels for and the average empirical loss over the entire dataset including unlabelled points. Optimize the upper bound of core-set loss could be considered as a k-center problem in practice. Doesn't need to know the out put of the current model.
- [Efﬁcient Active Learning by Querying Discriminative and Representative Samples and Fully Exploiting Unlabeled Data [2020, TNNLS]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9178457): **ALDR+**. **This paper also provide a new taxonomy in AL classification**, which includes three parts: criteria for querying samples, exploiting unlabeled data and acceleration. In this paper, they provide a method take all three parts into account.
- [Ask-n-Learn: Active Learning via Reliable Gradient Representations for Image Classification [2020]](https://arxiv.org/pdf/2009.14448.pdf): Use kmeans++ on the learned gradient embeddings to select instances.
- [Minimax Active Learning [2020]](https://arxiv.org/pdf/2012.10467.pdf): Develop a semi-supervised minimax entropy-based active learning algorithm that leverages both uncertainty and diversity in an adversarial manner.

## Learn how to sample / Learn how to active learn

All the mentioned sampling strategies is basing on a heuristic approach.
Their intuition is clear, but might perform differently in different datasets.
So some researchers purposed that we can learn a sampling strategy during the sampling process.

- Learn how to sample
  - Learn a strategy selection method
  - Learn a score function
  - Learn a AL policy (as a MDP process)

Works:
- [Active learning by learning [2015, AAAI]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9636): **ALBL**. A single human-designed philosophy is unlikely to work on all scenarios. Given an appropriate choice for the multi-armed bandit learner, take the importance-weighted-accuracy as reward function (an unbiased estimator for the test accuracy). It is possible to estimate the performance of different strategies on the fly. SVM as underlying classifier.(41 citations)
- [Learning active learning from data [2017, NIPS]](http://papers.nips.cc/paper/7010-learning-active-learning-from-data): **LAL**. Train a random forest regressor that predicts the expected error reduction for a candidate sample in a particular learning state. Previous works they cannot go beyond combining pre-existing hand-designed heuristics. Random forest as basic classifiers. (Not clear how to get test classiﬁcation loss l. It is not explained in both the paper and the code.)(73 citations)
- [Learning how to Active Learn: A Deep Reinforcement Learning Approach [2017, Arxiv]](https://arxiv.org/abs/1708.02383): **PAL**. Use RL to learn how to select instance. Even though the strategy is learned and applied in a stream manner, the stream is made by the data pool. So under my angle, it could be considered as a pool-based method. (92)
- [Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, ACL]](https://www.aclweb.org/anthology/P18-1174.pdf): Learn an AL policy using imitation learning, mapping situations to most informative query datapoints. (8 citations)
- [Learning Loss for Active Learning [2019, CVPR]](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html): Attach a small parametric module, named “loss prediction module,” to a target network, and learn it to predict target losses of unlabeled inputs. 
- [Learning to Rank for Active Learning: A Listwise Approach [2020]](https://arxiv.org/pdf/2008.00078.pdf): Have an additional loss prediction model to predict the loss of instances beside the classification model. Then the loss is calculated by the ranking instead of the ground truth loss of the classifier.

## Other works
There still are other works uses innovative heuristics.
It is a little bit hard to classify those works for now.
So we put these works under this section.
These works might be classified later.

Works:
- [Self-Paced Active Learning: Query the Right Thing at the Right Time [AAAI 2019]](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4445): 
  Borrow the idea in the self-paced learning that the learning from easy instances to hard instances would improve the performance.
- [Combination of Active Learning and Self-Paced Learning for Deep Answer Selection with Bayesian Neural Network [2020, ECAI]](http://ecai2020.eu/papers/449_paper.pdf):
  Just a combination.
  Not use the idea of the self-paced learning into AL.


# AL with Multi-class classification

Multi-class problem is really common in real life.
In multi-class classification, conventional methods use one-vs-all and one-vs-rest methods to tackle this problem.
Of course, several other methods could handle multi-class problem naturally, such as Neural Network.
(AL for neural networks is quite complex, so here in this section, the strategies are mostly for non-deep models. We'll have a whole section for AL in deep models.)
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
1. [Multi-Class Active Learning for Image Classification [CVPR, 2009]](https://ieeexplore.ieee.org/abstract/document/5206627): A comparison of **BvSB** and **Entropy** for active multi-classification. They use one-vs-one SVM to evaluate the probability for each class. (338 citations). Also see [Scalable Active Learning for Multiclass Image Classification [TPAMI, 2012]](https://ieeexplore.ieee.org/abstract/document/6127880/). (104 citations)
2. [An active learning-based SVM multi-class classification model [2015, Pattern Recognition]](https://www.sciencedirect.com/science/article/pii/S003132031400497X): Use ove-vs-rest SVM, and select from the three types of unclear region (CBA, CCA, CNA). Allow the addition of new classes. (65 citations)

# Batch mode classification

Batch mode selection is important in real life applications.
Although the non-batch AL methods could still select the top evaluated instances as a batch, they would contain too much overlap information.
So the non-batch mode selection would waste the budget in the batch selection case.

Even though we have mentioned some batch-mode works in the two previous section, we will still summarize the batch mode AL works here.
In the previous two sections, our taxonomy are mostly basing on how to evaluate instances instead of how to batch.
In this section, we will summarize several types of techniques on batch mode selection.

Different batch selection strategies have the same intuition which is try to diverse the instances in a single training batch.
However, they might achieve this goal in different approaches.

## Heuristic-diversity

Define a heuristic way to evaluate the diversity.

- Cluster-based: 
  The diversity was maintained by use the clustering methods. 
  The instances in the same batch are collected from the different clusters.
- Diversity of the induced models (model change):
  Evaluate the induced models for each unlabeled instance.
  The diversity of instances are revealed by the diversity of the models or model changes.
  For example, for neural networks, the diversity could measured by the diversity of vectors in gradient descent.

Works:
- [Representative sampling for text classiﬁcation using support vector machines [2003, ECIR]](https://link.springer.xilesou.top/chapter/10.1007/3-540-36618-0_28): 
  Select the cluster centers of the instances lying within the margin of a support vector machine. 
  Propose a representative sampling approach that selects the cluster centers of the instances lying within the margin of a support vector machine. (220)
- [Incorporating diversity in active learning with support vector machines [2003, ICML]](https://www.aaai.org/Papers/ICML/2003/ICML03-011.pdf): 
  Take the diversity of the selected instances into account, in addition to individual informativeness. 
  The diversity considers the angles between the induced hyperplanes (471)
- [A batch-mode active learning technique based on multiple uncertainty for SVM classifier [2011, Geoscience and Remote Sensing Letters]](https://ieeexplore.ieee.org/abstract/document/6092438/): 
  Batch-mode SVM-based method. 
  Not only consider the smallest distances to the decision hyperplanes but also take into account the distances to other hyperplanes. 
  Use kernel k-means to keep the diversity of the query batch. (43 citations)

## Optimization-based

Different from the heuristic-diversity methods, optimization-based methods define an optimization objective.
The instance selection could be directly revealed from the optimization result.

The optimization objective for the batch selection could be:
-  Directly minimize the loss or the expected variance after querying.
-  The constrain of distribution of the batch selection set and a corresponding target set. 

Works:
- [Discriminative batch mode active learning [2008, NIPS]](http://papers.nips.cc/paper/3295-discriminative-batch-mode-active-learning): 
  Optimization based. 
  Formulate batch mode active learning as an optimization problem that aims to learn a good classiﬁer directly. 
  The optimization selects the best set of unlabeled instances and their labels to produce a classifier that attains maximum likelihood on labels of the labeled instances while attaining minimum uncertainty on labels of the unlabeled instances. (248 citations)
- [Semisupervised SVM batch mode active learning with applications to image retrieval [2009, TOIS]](https://dlacm.xilesou.top/doi/abs/10.1145/1508850.1508854): 
  Optimization based. 
  Provide the min-max view of active learning. 
  Directly minimize the loss with the regularization. (310 citations)
- [Querying discriminative and representative samples for batch mode active learning [2015, TKDD]](https://dlacm.xilesou.top): 
  Optimization based. **BMDR**. 
  Query the most informative samples while preserving the source distribution as much as possible, thus identifying the most uncertain and representative queries. 
  Try to minimize the difference between two distributions (maximum mean discrepancy between iid. 
  Samples from the dataset and the actively selected samples). 
  **Also could be extended to multi-class AL.** (82 citations)

## Greedy Selection

This type of methods select instance in a greedy search way.
It could be considered as a conventional AL selection without updating the model in each iteration.
The model updating only process after the whole batch is collected.

Works:
1. [Batch mode active learning and its application to medical image classification [2006, ICML]](https://dlacm.xilesou.top/doi/abs/10.1145/1143844.1143897): 
   An approach that uses Fisher information matrix for instance selection. 
   Use logistic regression. 
   Efficiently identify the subset of unlabeled examples that can result in the largest reduction in the Fisher information based on the property of **submodular** functions. (367 citations)
2. [Active learning for convolutional neural networks: A core-set approach [ICLR, 2018]](https://arxiv.org/abs/1708.00489):
  This method use a k-center-greedy method.
  We need to note that this work doesn't need the feedback of the current model to guide selection. 
  So there is no training during the selecting process.
  Therefore this kind of works would be easily extended to batch mode. (128 citations)
3. [BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [2019, NIPS]](http://papers.nips.cc/paper/8925-batchbald-efficient-and-diverse-batch-acquisition-for-deep-bayesian-active-learning): 
   Jointly score points by estimating the mutual information between a joint of multiple data points and the model parameters. 
   BALD overestimates the joint mutual information. 
   **BatchBALD**, however, takes the overlap between variables into account and will strive to acquire a better cover of ω.(5 citations)




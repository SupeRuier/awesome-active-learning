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

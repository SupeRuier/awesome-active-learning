# Batch mode classification

Batch mode selection is important in pool-based AL.
In many real life cases, it is more efficient to select a number of instances to be annotated in each AL iteration.
Although the non-batch AL methods could still meet this requirement by selecting the top evaluated instances as a batch, they would contain too much overlap information.
So the non-batch mode selection would waste the budget compared to the batch selection case.
Batch mode AL requires that the information overlap of instances in a single query batch should be small enough.
Different batch selection strategies have the same intuition which is try to diverse the instances in a single training batch.
However, they might achieve this goal in different approaches.

| Intuition                                   | Techniques          |
| ------------------------------------------- | ------------------- |
| Diverse the instances in the selected batch | Heuristic-diversity |
|                                             | Optimization-based  |
|                                             | Greedy Selection    |
| Representativeness                          |                     |
| Directly learn from the trajectories        |                     |

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
- [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds [2020, ICLR]](https://arxiv.org/pdf/1906.03671.pdf):
  Cluster over the gradient vectors for the last fully connect layer.

## Optimization-based

Different from the heuristic-diversity methods, optimization-based methods define an optimization objective.
The instance selection could be directly revealed from the optimization result.

The optimization objective for the batch selection could be:
-  The loss or the expected variance after querying.
-  The discrepancy of distributions (the batch selection set and a corresponding target set). 

Works:
- [Discriminative batch mode active learning [2007, NeurIPS]](http://papers.nips.cc/paper/3295-discriminative-batch-mode-active-learning): 
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
  Try to minimize the difference between two distributions (maximum mean discrepancy between iid. samples from the dataset and the actively selected samples). 
  **Also could be extended to multi-class AL.** (82 citations)
- [Bayesian Batch Active Learning as Sparse Subset Approximation [2019， NeurIPS]](https://proceedings.neurips.cc/paper/2019/file/84c2d4860a0fc27bcf854c444fb8b400-Paper.pdf):
  The key idea is to re-cast batch construction as optimizing a sparse subset approximation to the log posterior induced by the full dataset.

## Greedy Selection

This type of methods select instance in a greedy search way.
In each active learning iteration, the strategies greedily select instances to maximum their diversity (in distribution or in other criteria).

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
3. [BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [2019, NeurIPS]](http://papers.nips.cc/paper/8925-batchbald-efficient-and-diverse-batch-acquisition-for-deep-bayesian-active-learning): 
   Jointly score points by estimating the mutual information between a joint of multiple data points and the model parameters. 
   BALD overestimates the joint mutual information. 
   **BatchBALD**, however, takes the overlap between variables into account and will strive to acquire a better cover of ω.(5 citations)
4. [Gone Fishing: Neural Active Learning with Fisher Embeddings [2021, Arxiv]](https://arxiv.org/pdf/2106.09675.pdf)

## Representativeness

Make sure the selected instances are more consistent to the true distribution.
Many works in the [representativeness-impart sampling](subfields/pb_classification.md#representativeness-impart-sampling) could be considered as this batch selection approach.

## Directly learn from the trajectories

Applying the non-batch selection multiple times would provide a trajectory with minimum information overlap.
So several works try to directly learn from these trajectories. 

- [Batch Active Learning via Coordinated Matching [2012, ICML]](https://arxiv.org/pdf/1206.6458.pdf):
  Optimize the batch selection strategy as a distribution on the trajectories of non-batch selections. (36 citations)
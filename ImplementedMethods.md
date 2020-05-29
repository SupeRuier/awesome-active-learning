# Our summarized methods

- Uncertainty sampling
  - Classification uncertainty
  - Classification margin
  - Classification entropy
- Disagreement sampling (QBC)
  - Disagreement sampling for classifiers
    - Vote entropy
    - Consensus entropy
    - Max disagreement
  - Disagreement sampling for regressors
    - Standard deviation sampling
- Acquisition functions
  - Probability of improvement
  - Expected improvement
    - Error Reduction: Expected Error Reduction (ICML 2001)
    - Variance Reduction [2007]: Active learning for logistic regression: an evaluation 
  - Upper confidence bound
- Density Weighted sampling: 
  - active learning using pre-clustering
  - Graph Density (CVPR 2012) 
  - RALF: A Reinforced Active Learning Formulation for Object Class Recognition
  - QUIRE [2010]:Active Learning by QUerying Informative and Representative Examples (QUIRE)
  - **k-Center-Greedy** 2017: A Geometric Approach to Active Learning for Convolutional Neural Networks  
- Cluster based sampling
- Representative based sampling
- Learn how to sanpling:
  - **ALBL** 2015: Hsu & Lin 2015, Active Learning by Learning. 
  - LAL (NIPS 2017), 
- Not sure
  - **Hierarchical cluster AL method** 2008: Hierarchical Sampling for Active Learning.
  - BMDR (KDD 2013), 
  - SPAL (AAAI 2019), 


# Google:
- **ALBL** 2015: Hsu & Lin 2015, Active Learning by Learning. 
- **RALF** 2012: [RALF: A Reinforced Active Learning Formulation for Object Class Recognition](https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf): Graph_density: Diversity promoting sampling method that uses graph density to determine most representative points.
- **Hierarchical cluster AL method** 2008: Hierarchical Sampling for Active Learning.
- **Informative and diverse**: Informative and diverse batch sampler that samples points with small margin while maintaining same distribution over clusters as entire training data. Batch is created by sorting datapoints by increasing margin and then growing the batch greedily.  A point is added to the batch if the result batch still respects the constraint that the cluster distribution of the batch will match the cluster distribution of the entire training set
- **k-Center-Greedy** 2017: A Geometric Approach to Active Learning for Convolutional Neural Networks 
- **MarginAL**: Uncertainty
- **mixture_of_samplers**: Mixture of base sampling strategies
- **represent_cluster_centers** 2003: Xu, et. al., Representative Sampling for Text Classification Using Support Vector Machines, 2003. Batch is created by clustering points within the margin of the classifier and choosing points closest to the k centroids.

# modAL
- Acquisition functions
  - Probability of improvement
  - Expected improvement
  - Upper confidence bound
- Uncertainty sampling
  - Classification uncertainty
  - Classification margin
  - Classification entropy
- Disagreement sampling
  - Disagreement sampling for classifiers
    - Vote entropy
    - Consensus entropy
    - Max disagreement
  - Disagreement sampling for regressors
    - Standard deviation sampling
- Ranked batch-mode sampling
- Information density

# libact

## Conventional
- QUIRE [2010]:Active Learning by QUerying Informative and Representative Examples (QUIRE)
- Variance Reduction [2007]: Active learning for logistic regression: an evaluation
- Uncertainty
  - least confidence method 
  - smallest margin method 
  - Entropy
- Query by committee [1992]: Query by committee. Seung, H. Sebastian, Manfred Opper, and Haim Sompolinsky
- hintSVM [2012]: Active Learning with Hinted Support Vector Machine. Active learning
           using hint information (2015).
- Density Weighted Uncertainty Sampling (DWUS) [2004]: Active learning using pre-clustering. Support binary case and LogisticRegression only.
- Density Weighted
- ALBL [2015]: Wei-Ning Hsu, and Hsuan-Tien Lin. "Active Learning by Learning." Twenty-Ninth AAAI Conference on Artificial Intelligence.

## Multi-class
- Active Learning with Cost Embedding (ALCE) [2016]: A Novel Uncertainty Sampling Algorithm for Cost-sensitive Multiclass Active Learning. ICDM 2016
- Expected Error Reduction
- Hierarchical Sampling for Active Learning (HS) [2008]: Sanjoy Dasgupta and Daniel Hsu. "Hierarchical sampling for active learning." ICML 2008.

## Multi-label
- Adaptive active learning [2013]: Active Learning with Multi-Label SVM Classification. IJCAI
- Binary Minimization [2006]: Brinker, Klaus. "On active learning in multi-label classification." From Data and Information Analysis to Knowledge Engineering. Springer Berlin Heidelberg, 2006. 206-213.
- Maximum loss reduction with Maximal Confidence (**MMC**) [2009]: Yang, Bishan, et al. "Effective multi-label active learning for text classification." Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2009.
- Multi-label Active Learning with Auxiliary Learner [2011]: Multi-label Active Learning with Auxiliary Learner

# Alipy 

## AL with Instance Selection:
- Uncertainty (SIGIR 1994), 
- Graph Density (CVPR 2012), 
- QUIRE (TPAMI 2014), 
- SPAL (AAAI 2019), 
- Query By Committee (ICML 1998), 
- Random, 
- BMDR (KDD 2013), 
- LAL (NIPS 2017), 
- Expected Error Reduction (ICML 2001)

## AL for Multi-Label Data
- AUDI (ICDM 2013) , 
- QUIRE (TPAMI 2014) , 
- Random, 
- MMC (KDD 2009), 
- Adaptive (IJCAI 2013)

## Querying Features
- AFASMC (KDD 2018) , 
- Stability (ICDM 2013

## AL with Different Costs: 
- HALC (IJCAI 2018) , 
- Random , 
- Cost performance

## AL with Noisy Oracles: 
- CEAL (IJCAI 2017) , 
- IEthresh (KDD 2009)

## AL with Novel Query Types: 
- AURO (IJCAI 2015)

## AL for Large Scale Tasks: 
- Subsampling


# PyTorch Active Learning

- Least Confidence sampling
- Margin of Confidence sampling
- Ratio of Confidence sampling
- Entropy (classification entropy)
- Model-based outlier sampling
- Cluster-based sampling
- Representative sampling
- Adaptive Representative sampling
- Active Transfer Learning for Uncertainty Sampling
- Active Transfer Learning for Representative Sampling
- Active Transfer Learning for Adaptive Sampling (ATLAS)

# Deep Active Learning

- Random Sampling
- Least Confidence [1]
- Margin Sampling [1]
- Entropy Sampling [1]
- Uncertainty Sampling with Dropout Estimation [2]
- Bayesian Active Learning Disagreement [2]
- K-Means Sampling [3]
- K-Centers Greedy [3]
- Core-Set [3]
- Adversarial - Basic Iterative Method
- Adversarial - DeepFool [4]
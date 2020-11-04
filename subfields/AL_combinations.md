# AL with other Research Problems

Active learning is also been used with other Learning/Research paradigms.
These works could be considered as the applications of AL in other research fields.
For the information of the real applications of AL in industry, please check [AL Applications](subfields/AL_applications.md).

*(Note that the works list in this pages are works I browsed in background reading. 
There might be more important works in the corresponding fields not listed here.
Besides, I can't ensure the works are representative in the fields.
So if you have any comments and recommendations, pls let me know.)*

For the works in this page, we divided them to two types.
Some of them are use AL to reduce the annotation cost.
Others try to improve the AL process with the knowledge in other fields.

- [AL with other Research Problems](#al-with-other-research-problems)
  - [Utilize AL](#utilize-al)
    - [Computer vision (CV)](#computer-vision-cv)
    - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
    - [Domain adaptation/Transfer learning](#domain-adaptationtransfer-learning)
    - [Anomaly Detection](#anomaly-detection)
    - [Graph data](#graph-data)
    - [Metric learning/Pairwise comparison/Similarity learning](#metric-learningpairwise-comparisonsimilarity-learning)
    - [One-shot learning](#one-shot-learning)
    - [Clustering](#clustering)
    - [Generative Adversarial Network](#generative-adversarial-network)
    - [De-noise](#de-noise)
    - [Causal Analysis](#causal-analysis)
    - [Positive and unlabeled (PU) learning](#positive-and-unlabeled-pu-learning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Human Learning](#human-learning)
  - [Improve AL](#improve-al)
    - [Reinforcement Learning](#reinforcement-learning-1)
    - [Meta-learning](#meta-learning)
    - [Quantum computing](#quantum-computing)
    - [GAN](#gan)


## Utilize AL

Reducing the labeling cost is a common need in many research fields.
So AL sometimes could be utilized in other research fields.

### Computer vision (CV)

CV is quite a wide conception.
Here we only post several subtypes in the fields.

Image segmentation/Semantic Segmentation：
- Geometry in active learning for binary and multi-class [2019, Computer vision and image understanding]
- Contextual Diversity for Active Learning [2020, Arxiv]
- Semi-supervised Active Learning for Instance Segmentation via Scoring Predictions

Object Detection: 
- Deep Active Learning for Remote Sensing Object Detection [2020, Arxiv]
- [Active Object Detection in Sonar Images [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9106398)
- [Importance of Self-Consistency in Active Learning for Semantic Segmentation [2020, Arxiv]](https://arxiv.org/pdf/2008.01860.pdf)

### Natural Language Processing (NLP)

NLP is also quite a wide conception.
Here we only post several subtypes in the fields.

Sentiment classification:
- Active learning for imbalanced sentiment classification [2012, EMNLP-CoNLL]

Named entity recognition: 
- Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, ACL]
- LTP: A New Active Learning Strategy for CRF-Based Named Entity Recognition [2020, Arxiv]

### Domain adaptation/Transfer learning

Normally when we use AL in domain adaptation, we can obtain several true labels of the unlabeled instances on source/target domain.
And for transfer learning, the concept is even wider.
Here we refer active transfer learning to any forms of transfer, eg. data or model.

Domain adaptation:
- Active learning for cross-domain sentiment classification [2013, IJCAI]
- Transfer Learning with Active Queries from Source Domain [2016, IJCAI]
- Active Sentiment Domain Adaptation [2017, ACL]
- Active Adversarial Domain Adaptation [2020, WACV]

Transfer learning:
- Accelerating active learning with transfer learning [2013, ICDM]
- Hierarchical Active Transfer Learning [2015, SIAM]
- Semi-Supervised Active Learning with Cross-Class Sample Transfer [2016, IJCAI]
- Active learning with cross-class knowledge transfer [2016, AAAI]
- Active learning with cross-class similarity transfer [2017, AAAI]
- Rapid Performance Gain through Active Model Reuse [IJCAI, 2019]


### Anomaly Detection

- Meta-AAD: Active Anomaly Detection with Deep Reinforcement Learning [2020, Arxiv]

### Graph data

Graph Embedding/Network representation learning:
- Active discriminative network representation learning [2018, IJCAI]

Graph node classification:
- Active learning for streaming networked data [2014, ACM International Conference on Conference on Information and Knowledge Management]
- MetAL: Active Semi-Supervised Learning on Graphs via Meta Learning [2020, Arxiv]

Graph transfer:
- [Graph Policy Network for Transferable Active Learning on Graphs [2020, Arxiv]](https://arxiv.org/pdf/2006.13463.pdf)
- [Active Domain Transfer on Network Embedding [2020, Proceedings of The Web Conference ]](https://arxiv.org/pdf/2007.11230.pdf)
- [Active Learning on Graphs with Geodesically Convex Classes [2020, MLG]](http://www.mlgworkshop.org/2020/papers/MLG2020_paper_40.pdf)

### Metric learning/Pairwise comparison/Similarity learning 

Works:
- [Active Ordinal Querying for Tuplewise Similarity Learning [2020 AAAI]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-CanalG.9650.pdf): Introduce AL to their new similarity methods (InfoTuple). Active Sampling for Pairwise Comparisons via Approximate Message Passing and Information Gain Maximization.
- Batch Decorrelation for Active Metric Learning [2020, IJCAI]

### One-shot learning

One/few-shot learning is to learn a model on a new dataset with one for few available instances.
AL could be used to select the instances in the new dataset.

Works:
- Active one-shot learning [2017, Arxiv]
- Augmented Memory Networks for Streaming-Based Active One-Shot Learning [2019, Arxiv]
- Active one-shot learning with Prototypical Networks [2019, ESANN]
- Active one-shot learning by a deep Q-network strategy [2020, Neurocomputing]


### Clustering

AL could support clustering by provide pairwise information.

Works:
- Semi-Supervised Selective Affinity Propagation Ensemble Clustering With Active Constraints [2020, IEEE Access]
- Active Learning for Constrained Document Clustering with Uncertainty Region [Complexity, 2020]: Must link & cannot link.
- Cautious Active Clustering [2020]
- Improving evolutionary constrained clustering using Active Learning [2020, Knowledge-Based Systems]

### Generative Adversarial Network

AL could reduce the number of needed instances to train a GAN.

Works:
- Learning Class-Conditional GANs with Active Sampling [2019, SIGKDD]

### De-noise

Works:
- Active Deep Learning to Tune Down the Noise in Labels [2018, KDD]

### Causal Analysis

Works:
- [Root Cause Analysis for Self-organizing Cellular Network: an Active Learning Approach](https://link.springer.com/article/10.1007/s11036-020-01589-1)
- [Active Invariant Causal Prediction: Experiment Selection through Stability](https://arxiv.org/pdf/2006.05690.pdf)

### Positive and unlabeled (PU) learning

A special case of binary classiﬁcation where a learner only has access to labeled positive examples and unlabeled examples.

Works:
- Class Prior Estimation in Active Positive and Unlabeled Learning [2020, IJCAI]


### Reinforcement Learning

Works:
- [SAMBA: Safe Model-Based & Active Reinforcement Learning [2020, Arxiv]](https://arxiv.org/pdf/2006.09436.pdf)

### Human Learning

Works:
- [Human Active Learning [2008, NIPS]](http://papers.nips.cc/paper/3456-human-active-learning)
- [Human Active Learning [2018]](https://www.intechopen.com/books/active-learning-beyond-the-future/human-active-learning)


## Improve AL

Instead of use AL to save labeling cost in other research fields, the ideas in other research fields could also be used to improve AL framework.

### Reinforcement Learning

Could be used as a query strategy.

Works:
- Active Learning by Learning [2015, AAAI]
- Learning how to Active Learn: A Deep Reinforcement Learning Approach [2017, Arxiv]
- Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, Annual Meeting of the Association for Computational Linguistics]

### Meta-learning

Could be used to develop query strategy (define query loss).

Works:
- Learning Loss for Active Learning [2019, CVPR]

### Quantum computing 

Works:
- Quantum speedup for pool-based active learning

### GAN 

Could be used in query synthesis.

Works:
- Generative Adversarial Active Learning
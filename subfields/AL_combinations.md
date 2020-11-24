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
    - [Ordinal Regression/Classification](#ordinal-regressionclassification)
    - [Generative Adversarial Network](#generative-adversarial-network)
    - [De-noise](#de-noise)
    - [Causal Analysis](#causal-analysis)
    - [Model selection](#model-selection)
    - [Positive and unlabeled (PU) learning](#positive-and-unlabeled-pu-learning)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Human Learning](#human-learning)
    - [Model interpretability](#model-interpretability)
    - [Sequence labeling](#sequence-labeling)
    - [Sample selection for optimization problem](#sample-selection-for-optimization-problem)
  - [Improve AL](#improve-al)
    - [Reinforcement Learning](#reinforcement-learning-1)
    - [Quantum computing](#quantum-computing)
    - [Generative Adversarial Network](#generative-adversarial-network-1)


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
- [Attention, Suggestion and Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation [2020, MICCAI]](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_1)
- [Deep Active Learning for Joint Classification & Segmentation with Weak Annotator [2020]](https://arxiv.org/pdf/2010.04889.pdf): Leverage the unlabeled images to improve model accuracy with less oracle-annotation. AL for segmentation of images selected for pixel-level annotation.
- [Difficulty-aware Active Learning for Semantic Segmentation [2020]](https://arxiv.org/pdf/2010.08705.pdf)

Object Detection: 
- Deep Active Learning for Remote Sensing Object Detection [2020, Arxiv]
- [Active Object Detection in Sonar Images [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9106398)
- [Importance of Self-Consistency in Active Learning for Semantic Segmentation [2020, Arxiv]](https://arxiv.org/pdf/2008.01860.pdf)

Image Captioning:
- Structural Semantic Adversarial Active Learning for Image Captioning [2020, ACMMM]

### Natural Language Processing (NLP)

NLP is also quite a wide conception.
Here we only post several subtypes in the fields.

Sentiment/text classification:
- Active learning for imbalanced sentiment classification [2012, EMNLP-CoNLL]
- [Deep Active Learning with Simulated Rationales for Text Classiﬁcation [2020, PRAI]](https://link.springer.com/chapter/10.1007/978-3-030-59830-3_32): Utilize auxiliary task to reduce the data scarce affect at the beginning of the AL process.

Named entity recognition: 
- Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, ACL]
- LTP: A New Active Learning Strategy for CRF-Based Named Entity Recognition [2020, Arxiv]

Semantic Parsing:
- [Uncertainty and Traffic-Aware Active Learning for Semantic Parsing [2020]](https://assets.amazon.science/af/ca/4c43ed0c4932a3a8365693e68420/uncertainty-and-traffic-aware-active-learning-for-semantic-parsing.pdf)

Classifier Pruning:
- [FIND: Human-in-the-Loop Debugging Deep Text Classifiers [2020, EMNLP]](https://www.aclweb.org/anthology/2020.emnlp-main.24.pdf): Visualize each extracted feature as a word cloud. Human decide wether to block the corresponding feature.

Neural Machine Translation:
- [Active Learning Approaches to Enhancing Neural Machine Translation [2020, EMNLP]](https://www.aclweb.org/anthology/2020.findings-emnlp.162.pdf): The first to do a large-scale study on actively training Transformer for NMT.

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
- Active Learning for Node Classiﬁcation: An Evaluation [2020, Entropy MDPI]

Link Prediction:
- Complex Query Answering with Neural Link Predictors [2020]

Node response prediction:
- [Meta-Active Learning for Node Response Prediction in Graphs [2020]](https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0%2C5&q=META-ACTIVE+LEARNING+FOR+NODE+RESPONSE+PREDICTION+IN+GRAPHS&btnG=)

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

AL could support clustering by provide supervised information.

Works:
- Semi-Supervised Selective Affinity Propagation Ensemble Clustering With Active Constraints [2020, IEEE Access]
- Active Learning for Constrained Document Clustering with Uncertainty Region [Complexity, 2020]: Must link & cannot link.
- Cautious Active Clustering [2020]
- Improving evolutionary constrained clustering using Active Learning [2020, Knowledge-Based Systems]

Review:
- Interactive clustering: a scoping review [2020, Artificial Intelligence Review]

### Ordinal Regression/Classification

Could be consider as a regression where the relative orders of the instances matter.

- [Active Learning for Imbalanced Ordinal Regression [2020, IEEE Access]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9208667)

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

### Model selection

As active model selection, both pool based and stream based. 
Normally there only is model selection without training.

- Online Active Model Selection for Pre-trained Classifiers [2020]

### Positive and unlabeled (PU) learning

A special case of binary classiﬁcation where a learner only has access to labeled positive examples and unlabeled examples.

Works:
- Class Prior Estimation in Active Positive and Unlabeled Learning [2020, IJCAI]


### Reinforcement Learning

Active exploration Works:
- Model-Based Active Exploration
- [SAMBA: Safe Model-Based & Active Reinforcement Learning [2020, Arxiv]](https://arxiv.org/pdf/2006.09436.pdf)

### Human Learning

Works:
- [Human Active Learning [2008, NIPS]](http://papers.nips.cc/paper/3456-human-active-learning)
- [Human Active Learning [2018]](https://www.intechopen.com/books/active-learning-beyond-the-future/human-active-learning)

### Model interpretability

- [ALEX: Active Learning based Enhancement of a Model’s EXplainability [2020, CIKM]](https://dl.acm.org/doi/pdf/10.1145/3340531.3417456)

### Sequence labeling

- [SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup [2020]](https://arxiv.org/pdf/2010.02322.pdf): Not only provide the selected instance, but also provide a generated sequence according to the selected one.

### Sample selection for optimization problem

The background is that simulation (for evaluation) is quite expensive in many optimization problem.
Utilize active sampling to reduce the optimization cost.

- [Building energy optimization using surrogate model and active sampling [2020]](https://www.tandfonline.com/doi/pdf/10.1080/19401493.2020.1821094)


## Improve AL

Instead of use AL to save labeling cost in other research fields, the ideas in other research fields could also be used to improve AL framework.

### Reinforcement Learning

Could be used as a query strategy.

Works:
- Active Learning by Learning [2015, AAAI]
- Learning how to Active Learn: A Deep Reinforcement Learning Approach [2017, Arxiv]
- Learning How to Actively Learn: A Deep Imitation Learning Approach [2018, Annual Meeting of the Association for Computational Linguistics]
- [Deep Reinforcement Active Learning for Medical Image Classiﬁcation [2020, MICCAI]](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_4): Take the prediction probability of the whole unlabeled set as the state. The action as the strategy is to get a rank of unlabeled set by a actor network. The reward is the different of prediction value and true label of the selected instances. Adopt a critic network with parameters θ cto approximate the Q-value function.

### Quantum computing 

Works:
- Quantum speedup for pool-based active learning

### Generative Adversarial Network

Could be used in query synthesis.

Works:
- Generative Adversarial Active Learning